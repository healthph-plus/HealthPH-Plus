"""Text normalization extracted from preproccessor.ipynb."""
import html
import re


def normalize_text(text: str) -> str:
    """Normalize scraped text while preserving the annotation cleaning contract."""
    text = html.unescape(html.unescape(text))
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
