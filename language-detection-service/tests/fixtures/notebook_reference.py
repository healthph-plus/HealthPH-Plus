"""Frozen notebook logic before extraction; path argument added for isolated tests."""
import html
import re
import pandas as pd

def normalize_scraped_text(value):
    """Normalize scraped text while preserving the annotation cleaning contract."""
    if pd.isna(value):
        return ""

    text = html.unescape(html.unescape(str(value)))
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"^\s*rt\s+@\w+\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*$", r"\1", text)
    text = re.sub(r"https?://\S+|//www\.\S+|www\.\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"https?://\S+|//www\.\S+|www\.\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(^|\s)#+(?=\s|$)", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def load_language_keywords(filename, keyword_dir):
    keyword_candidates = [
        keyword_dir / filename,
        keyword_dir / filename,
    ]
    keyword_path = next((path for path in keyword_candidates if path.exists()), None)
    if keyword_path is None:
        raise FileNotFoundError(
            f'Could not find {filename}. Checked: '
            + ', '.join(str(path) for path in keyword_candidates)
        )

    keywords = (
        pd.read_csv(
            keyword_path,
            header=None,
            usecols=[0],
            names=['keyword'],
            dtype=str,
            encoding='utf-8-sig',
        )['keyword']
        .dropna()
        .str.strip()
    )
    keywords = sorted(
        {keyword for keyword in keywords if keyword},
        key=len,
        reverse=True,
    )
    if not keywords:
        raise ValueError(f'Language keyword list is empty: {keyword_path}')
    return keywords

def language_keyword_matches(text, keywords):
    # Match complete words/phrases case-insensitively. Flexible whitespace lets
    # phrases match across repeated spaces or line breaks in the original post.
    alternatives = [
        re.escape(keyword).replace(r'\ ', r'\s+')
        for keyword in keywords
    ]
    pattern = re.compile(
        r'(?<!\w)(?:' + '|'.join(alternatives) + r')(?!\w)',
        flags=re.IGNORECASE,
    )
    return text.fillna('').astype(str).str.contains(pattern, na=False)
