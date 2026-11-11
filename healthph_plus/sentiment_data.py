"""Validate sentiment CSVs and preserve authored synthetic evaluation splits.

Model features use only ``text``; ``label`` supplies training targets.
Other columns are audit metadata.
Synthetic artifacts are stricter than legacy, explicitly selected annotations.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


LANGUAGES = ('en', 'tl', 'ceb', 'ilo', 'hil')
SPLITS = ('train', 'validation', 'test')
SYNTHETIC_COLUMNS = ('id', 'post', 'sentiment_label', 'language', 'code_switched',
                     'family_id', 'split', 'is_synthetic')
QUOTAS = {'train': {0: 420, 1: 280}, 'validation': {0: 90, 1: 60}, 'test': {0: 90, 1: 60}}


def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def duplicate_key(text):
    """A stricter audit key; never used to alter model text."""
    return normalize_text(text).casefold()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_sentiment_data(path, *, min_class_rows=10, enforce_v1_quotas=True):
    """Return validated posts with metadata and a JSON-serializable audit.

    Presence of any synthetic metadata column requires the complete contract.
    Legacy two-column datasets retain whitespace-only exact deduplication.
    """
    path = Path(path)
    raw = pd.read_csv(path, dtype=str, encoding='utf-8-sig', keep_default_na=False)
    missing = {'post', 'sentiment_label'} - set(raw)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    frame = raw.copy()
    frame['source_row'] = range(1, len(frame) + 1)
    frame['text'] = frame['post'].map(normalize_text)
    labels = frame['sentiment_label'].str.strip()
    if frame['text'].eq('').any() or not labels.isin(['0', '1']).all():
        raise ValueError('Posts must be nonblank and sentiment_label must be 0 or 1.')
    frame['label'] = labels.astype(int)
    metadata = set(SYNTHETIC_COLUMNS) - {'post', 'sentiment_label'}
    synthetic = bool(metadata & set(raw))
    if synthetic:
        missing = set(SYNTHETIC_COLUMNS) - set(raw)
        if missing:
            raise ValueError(f'Incomplete synthetic metadata: {sorted(missing)}')
        for column in ['id', 'family_id']:
            if (frame[column].str.strip().eq('').any()
                    or frame[column].ne(frame[column].str.strip()).any()):
                raise ValueError(f'{column} must be nonblank with no surrounding whitespace.')
        if not frame['id'].is_unique:
            raise ValueError('Duplicate synthetic IDs.')
        if not frame['language'].isin(LANGUAGES).all():
            raise ValueError('Unsupported synthetic language.')
        if not frame['split'].isin(SPLITS).all():
            raise ValueError('Invalid predefined split.')
        if not frame['code_switched'].isin(['true', 'false']).all():
            raise ValueError('code_switched must be true or false.')
        if not frame['is_synthetic'].eq('true').all():
            raise ValueError('is_synthetic must be true for synthetic artifacts.')
        if frame.loc[frame.language.eq('en'), 'code_switched'].ne('false').any():
            raise ValueError('English posts cannot have code_switched=true.')
        if frame.groupby('family_id')['split'].nunique().gt(1).any():
            raise ValueError('Family leakage across predefined splits.')
    keys = frame['text'].map(duplicate_key) if synthetic else frame['text']
    conflicts = frame.groupby(keys)['label'].nunique().gt(1)
    if conflicts.any():
        raise ValueError('Conflicting labels for normalized duplicate posts.')
    duplicates = keys.duplicated()
    if synthetic and duplicates.any():
        raise ValueError('Normalized duplicate synthetic posts; replace them within their quota.')
    clean = frame.loc[~duplicates].copy().reset_index(drop=True)
    clean['text_fingerprint'] = clean['text'].map(lambda text: hashlib.sha256(text.encode('utf-8')).hexdigest())
    counts = clean['label'].value_counts().reindex([0, 1], fill_value=0)
    if counts.lt(min_class_rows).any():
        raise ValueError(f'Need at least {min_class_rows} unique posts per class.')
    if synthetic:
        for split in SPLITS:
            for language in LANGUAGES:
                for label in [0, 1]:
                    cell = clean[(clean.split == split) & (clean.language == language) & (clean.label == label)]
                    if enforce_v1_quotas and len(cell) != QUOTAS[split][label]:
                        raise ValueError(f'Incorrect quota: {language}/{split}/{label}: {len(cell)}')
                    if enforce_v1_quotas:
                        mixed = 0 if language == 'en' else QUOTAS[split][label] // 5
                        if cell['code_switched'].eq('true').sum() != mixed:
                            raise ValueError(f'Incorrect code-switch quota: {language}/{split}/{label}')
    audit = {
        'dataset_path': str(path.resolve()), 'dataset_sha256': sha256_file(path),
        'raw_rows': len(raw), 'unique_rows': len(clean), 'duplicates_removed': int(duplicates.sum()),
        'conflicting_texts': 0, 'raw_class_counts': frame.label.value_counts().sort_index().to_dict(),
        'unique_class_counts': counts.to_dict(), 'evaluation_source': 'synthetic' if synthetic else 'unspecified_legacy',
        'split_strategy': 'predefined_family_disjoint' if synthetic else 'seeded_label_stratified',
        'normalization': 'collapse whitespace and strip; preserve casing and Unicode',
    }
    if synthetic:
        audit['counts_by_language_label_split'] = clean.groupby(['language', 'label', 'split']).size().rename('rows').reset_index().to_dict('records')
    return clean, audit


def split_sentiment_data(frame, *, seed=42, validation_size=0.15, test_size=0.15):
    if 'split' in frame:
        return tuple(frame.loc[frame['split'].eq(name)].reset_index(drop=True) for name in SPLITS)
    train_val, test = train_test_split(frame, test_size=test_size, stratify=frame.label, random_state=seed)
    train, validation = train_test_split(train_val, test_size=validation_size / (1 - test_size),
                                        stratify=train_val.label, random_state=seed)
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))


def audit_near_duplicates(frame, *, threshold=0.88):
    """Find close lexical variants, including cross-language code switching.

    This character TF-IDF audit does not establish semantic equivalence or
    guarantee that all paraphrases/translations have been identified.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not 0 < threshold <= 1:
        raise ValueError('Similarity threshold must be in (0, 1].')
    features = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), lowercase=True).fit_transform(frame.text)
    similarities = cosine_similarity(features, dense_output=False).tocoo()
    records = frame.to_dict('records')
    pairs = []
    for left, right, score in zip(similarities.row, similarities.col, similarities.data):
        if left >= right or score < threshold:
            continue
        a, b = records[left], records[right]
        pairs.append({'left': a.get('id', a['source_row']), 'right': b.get('id', b['source_row']),
                      'similarity': round(float(score), 6),
                      'same_family': a.get('family_id') == b.get('family_id') if 'family_id' in a else None,
                      'cross_split': a.get('split') != b.get('split') if 'split' in a else None})
    return {'method': 'char_wb TF-IDF cosine; character ngrams 3–5; lowercase audit only',
            'threshold': threshold, 'candidate_pairs': len(pairs),
            'cross_split_candidate_pairs': sum(p['cross_split'] is True for p in pairs),
            'pairs': pairs,
            'limitation': 'Lexical similarity cannot guarantee semantic paraphrase or translation isolation.'}


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--near-duplicates', action='store_true')
    args = parser.parse_args()
    frame, audit = load_sentiment_data(args.csv)
    if args.near_duplicates:
        audit['near_duplicate_audit'] = audit_near_duplicates(frame)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
