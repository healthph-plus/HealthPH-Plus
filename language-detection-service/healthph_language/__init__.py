"""Reusable language detection without notebook or dataframe dependencies."""

from .cleaning import normalize_text
from .detector import LanguageDetector

__all__ = ["LanguageDetector", "normalize_text"]
