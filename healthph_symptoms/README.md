# Keyword symptom extraction

The disease services extract symptoms from
[`docs/keywords/keywords_master.csv`](../docs/keywords/keywords_master.csv) with
case-insensitive literal matching. Extraction uses only the Python standard
library and requires no symptom model, training, or annotated dataset.

```python
from healthph_symptoms.keywords import KeywordSymptomExtractor

extractor = KeywordSymptomExtractor()
symptoms = extractor.extract("May ubo at lagnat ako. Walang sipon.")
```

Results include exact text, English normalization when available, original-text
offsets, and simple rule-based negation. `confidence: 1.0` denotes a literal
keyword hit. The existing `symptom_model_version` response field fingerprints
the keyword list, normalization mappings, and matching rules.

See the [keyword workflow](KEYWORD_WORKFLOW.md) for custom lists, service exports,
matching behavior, and tests. Try the
[keyword notebook](../notebooks/08_symptom_keywords.ipynb) for a runnable example.
