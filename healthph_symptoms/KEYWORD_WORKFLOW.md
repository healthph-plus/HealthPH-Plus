# Keyword symptom extraction

Both disease services extract symptoms using literal keyword matching.
No symptom NLP model, tokenizer, annotation review, or training is required.
Disease classification continues to use its existing saved model and thresholds.

## Keyword source

The source is `docs/keywords/keywords_master.csv` in the research repository.
Its columns are `disease,symptoms`; commas inside a quoted symptoms cell separate
keywords. Slashes separate alternative spellings. Matching searches every row
regardless of the disease classifier's selected labels, so shared symptoms appear
once per occurrence. Bare disease names (`covid`, `covid-19`, `tb`,
`tuberculosis`, `pneumonia`) and standalone `smell`/`taste` are excluded.

Matching ignores case, permits variable whitespace inside phrases, and requires
whole-word boundaries. Longer overlapping phrases take precedence, so `dry cough`
is one mention. Separate occurrences remain separate. Only listed keywords match;
unlisted inflections, synonyms, and spelling errors do not match automatically.

The existing `aliases.json` supplies English `normalized_symptom` values when
available. An unknown normalization is `null`; the matched text is retained.
Aliases do not add keywords to the search list.

## Use directly

From the research repository root:

```python
from healthph_symptoms.keywords import KeywordSymptomExtractor

extractor = KeywordSymptomExtractor()
print(extractor.extract("May ubo at lagnat ako."))
```

Both `/v1/disease/predict` and `/v1/disease/predict-batch` return these matches in
`symptoms` alongside existing disease predictions. Each mention has `text`,
`normalized_symptom`, `start_index`, `end_index`, `confidence`, and `negated`.
No matches returns `[]`.

Offsets are zero-based, end-exclusive Unicode code-point indices into the exact
submitted text. The matcher searches the full accepted text, including text after
the disease model's token limit. In JavaScript, slice `Array.from(text)` to use
these indices.

`confidence: 1.0` means a literal keyword matched; it is not a model probability.
`negated` uses simple cue rules within the current clause and preceding eight
words. For example, `walang ubo pero may lagnat` flags only `ubo` as negated.
The complete phrase `walang panlasa` is a loss-of-taste symptom. These rules do
not determine who experienced a symptom or whether it is current.
`symptom_model_version` retains its API name and fingerprints the effective
keywords, normalization mappings, and matching rules independently of the disease
model version.

## Standalone services and updates

Each standalone service includes `healthph_symptoms/keywords_master.csv`, generated
from the source CSV. Share the entire service folder plus its disease `model/`.
Symptom extraction needs only the bundled CSV and normalization aliases. Set
`HEALTHPH_SYMPTOM_KEYWORDS_PATH=/absolute/path/to/keywords.csv` to use a custom CSV,
or pass `symptom_keywords_path=...` to `DiseasePredictor`. Restart after updates.

After editing the source list or canonical runtime, refresh both service copies
from the research repository root:

```sh
.venv/bin/python -m healthph_symptoms.export --service-dir disease-detection-service --code-only
.venv/bin/python -m healthph_symptoms.export --service-dir xlmr-disease-detection-service --code-only
```

To export disease weights together with the keyword runtime:

```sh
.venv/bin/python -m healthph_symptoms.export \
  --service-dir disease-detection-service \
  --disease-source artifacts/poc_transformers/electra_local_finetuned

.venv/bin/python -m healthph_symptoms.export \
  --service-dir xlmr-disease-detection-service \
  --disease-source artifacts/disease_xlmr/SELECTED_FULL_RUN
```

Stop the service before export. Export stages and validates assets offline before
replacing the destination. Exports include the keyword runtime and disease weights.

## Notebook and checks

Try `notebooks/08_symptom_keywords.ipynb` in the research repository for a runnable
example. Run keyword and standalone-export tests from the repository root:

```sh
.venv/bin/python -m pytest tests/test_symptom_keywords.py tests/test_symptom_export.py -q
```

To measure disease inference with keyword extraction:

```sh
.venv/bin/python -m healthph_symptoms.benchmark --service-dir disease-detection-service --repeats 10
```
