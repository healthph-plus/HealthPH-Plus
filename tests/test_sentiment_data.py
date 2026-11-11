"""Corpus contracts and an offline run of the actual sentiment notebook."""

import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import pytest

from healthph_plus.sentiment_data import audit_near_duplicates, load_sentiment_data, split_sentiment_data


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/training_data/synthetic_sentiment_v1.csv'


@pytest.fixture(scope='module')
def corpus():
    return load_sentiment_data(DATA)


def test_corpus_contract_and_frozen_splits(corpus):
    frame, audit = corpus
    assert audit['unique_rows'] == 5000
    assert audit['unique_class_counts'] == {0: 3000, 1: 2000}
    assert audit['evaluation_source'] == 'synthetic'
    assert frame.text.is_unique and frame.id.is_unique
    parts = split_sentiment_data(frame)
    assert [len(part) for part in parts] == [3500, 750, 750]
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        assert set(parts[first].family_id).isdisjoint(parts[second].family_id)
    for actual, repeated in zip(parts, split_sentiment_data(frame, seed=99)):
        assert actual.id.tolist() == repeated.id.tolist()
    manifest = json.loads(DATA.with_suffix('.manifest.json').read_text())
    quality = json.loads(DATA.with_suffix('.quality.json').read_text())
    assert manifest['dataset_sha256'] == audit['dataset_sha256'] == quality['dataset_sha256']
    assert len(quality['qualitative_review']['reviewed_ids']) == 250
    near = audit_near_duplicates(frame)
    assert near == quality['near_duplicate_audit']
    assert near['cross_split_candidate_pairs'] == 0
    assert all(pair['same_family'] for pair in near['pairs'])


@pytest.mark.parametrize(('column', 'value', 'message'), [
    ('language', 'xx', 'language'), ('split', 'dev', 'split'),
    ('code_switched', 'yes', 'code_switched'), ('is_synthetic', 'false', 'is_synthetic'),
    ('sentiment_label', '2', 'sentiment_label'), ('post', '  ', 'nonblank'),
    ('family_id', '', 'nonblank'), ('id', '', 'nonblank'),
])
def test_invalid_metadata_rejected(tmp_path, column, value, message):
    raw = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    raw.loc[0, column] = value
    path = tmp_path / 'bad.csv'
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_sentiment_data(path)


def test_duplicates_conflicts_and_family_leakage(tmp_path):
    raw = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    path = tmp_path / 'bad.csv'
    changed = raw.copy()
    changed.loc[1, 'id'] = changed.loc[0, 'id']
    changed.to_csv(path, index=False)
    with pytest.raises(ValueError, match='IDs'):
        load_sentiment_data(path)
    changed = raw.copy()
    other = changed.index[changed.split.ne(changed.loc[0, 'split'])][0]
    changed.loc[other, 'family_id'] = changed.loc[0, 'family_id']
    changed.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Family leakage'):
        load_sentiment_data(path)
    for conflicting in [False, True]:
        changed = raw.copy()
        changed.loc[1, 'post'] = '  ' + changed.loc[0, 'post'].upper() + '  '
        changed.loc[1, 'sentiment_label'] = str(1 - int(changed.loc[0, 'sentiment_label'])) if conflicting else changed.loc[0, 'sentiment_label']
        changed.to_csv(path, index=False)
        with pytest.raises(ValueError, match='Conflicting|duplicate'):
            load_sentiment_data(path)


def test_partial_metadata_and_wrong_quotas_rejected(tmp_path):
    raw = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    path = tmp_path / 'bad.csv'
    raw.drop(columns=['split']).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Incomplete'):
        load_sentiment_data(path)
    raw.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match='quota'):
        load_sentiment_data(path)
    native = raw.index[raw.language.eq('tl') & raw.code_switched.eq('false')][0]
    raw.loc[native, 'code_switched'] = 'true'
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match='code-switch quota'):
        load_sentiment_data(path)


def test_explicit_legacy_loading_preserves_unicode_and_literal_na(tmp_path):
    path = tmp_path / 'legacy.csv'
    posts = ['  Héllo 😷\n world  ', 'NA'] + [f'legacy calm {i}' for i in range(18)] + [f'legacy distressed {i}' for i in range(20)]
    raw = pd.DataFrame({'post': posts, 'sentiment_label': [0] * 20 + [1] * 20})
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    raw.to_csv(path, index=False)
    frame, audit = load_sentiment_data(path)
    assert frame.iloc[0].text == 'Héllo 😷 world'
    assert frame.iloc[1].text == 'NA'
    assert audit['duplicates_removed'] == 1
    assert audit['evaluation_source'] == 'unspecified_legacy'
    parts = split_sentiment_data(frame)
    assert [len(part) for part in parts] == [28, 6, 6]
    for left, right in zip(parts, split_sentiment_data(frame)):
        assert left.text.tolist() == right.text.tolist()
    raw.loc[len(raw) - 1, 'sentiment_label'] = 1
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Conflicting'):
        load_sentiment_data(path)


def test_notebook_offline_baseline_exports_and_reloads(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from healthph_plus import model_reporting

    def inexpensive_figure(fig, output_path, **kwargs):
        path = Path(output_path).with_suffix('')
        fig.savefig(path.with_suffix('.png'), dpi=35)
        fig.savefig(path.with_suffix('.svg'))
        plt.close(fig)

    monkeypatch.setattr(model_reporting, 'save_figure', inexpensive_figure)
    notebook_path = ROOT / 'notebooks/05_binary_sentiment_mbert.ipynb'
    notebook = json.loads(notebook_path.read_text())
    namespace = {'__name__': '__smoke__'}
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        ast.parse(source)
        if index == 3:
            source = source.replace("DEVICE_PREFERENCE = 'auto'", "DEVICE_PREFERENCE = 'cpu'")
            source = source.replace('RUN_MBERT = True', 'RUN_MBERT = False')
            source = source.replace('LOCAL_FILES_ONLY = False', 'LOCAL_FILES_ONLY = True')
        exec(compile(source, str(notebook_path), 'exec'), namespace)
        if index == 3:
            namespace['RUN_DIR'] = tmp_path / 'baseline_smoke'
        if index == 2:
            namespace['display'] = lambda *args: None
    model_dir = namespace['model_dirs']['tfidf_logreg']
    config = json.loads((model_dir / 'run_config.json').read_text())
    assert config['evaluation_source'] == 'synthetic'
    assert config['split_strategy'] == 'predefined_family_disjoint'
    manifest = pd.read_csv(model_dir / 'reports/split_manifest.csv')
    assert set(['language', 'family_id', 'is_synthetic', 'code_switched', 'id']) <= set(manifest)
    assert len(manifest) == 5000
    predictions = pd.read_csv(model_dir / 'reports/test_predictions.csv')
    assert len(predictions) == 750
    assert set(['language', 'family_id', 'is_synthetic']) <= set(predictions)
    by_language = pd.read_csv(model_dir / 'reports/metrics_by_language.csv')
    assert len(by_language) == 5
    assert set(by_language.rows) == {150}
    assert by_language.evaluation_source.eq('synthetic').all()
    assert namespace['baseline_model'].named_steps['classifier'].n_features_in_ > 1
    # The actual notebook reload cell checks prediction parity and saved thresholds.
    assert (model_dir / 'reports/evaluation_summary.csv').is_file()
    assert list((model_dir / 'reports').glob('*.png'))
