# HealthPH+ XLM-RoBERTa disease detection

Standalone CPU inference service for the JavaScript HealthPH+ backend. It serves
the XLM-RoBERTa model trained in `notebooks/06_xlm_roberta_disease.ipynb` and detects
zero, one, or several disease signals: **AURI, PN, TB, COVID**, in that order.
It also extracts symptom mentions using literal keyword matching.

Keyword extraction needs no symptom NLP model or training. Follow the
[keyword workflow](SYMPTOM_WORKFLOW.md) for matching rules and keyword updates.

The initial disease bundle comes from `full_20261022T064326Z_fa1cd053`. It uses the saved
validation thresholds: AURI **0.81**, PN **0.72**, TB **0.74**, COVID **0.91**.
These outputs describe signals in text; scores are sigmoid model outputs, not
calibrated clinical probabilities.

## Share with backend developers

After combined export, share this entire folder, including `healthph_symptoms/`
and its complete Git-ignored `model/` directory. Disease assets occupy about
1.13 GB; keyword extraction adds a small CSV file. The folder contains the
Python packages, HTTP API, JavaScript client, dependency pins, and tests. A
code-only clone needs a separate copy of the complete model bundle.
Exclude `.venv/`, `__pycache__/`, `.pytest_cache/`, and `.xlmr-export-*/`.
No research notebook, dataset, checkpoints, or credentials are required to serve it.

## Setup and run

Run from inside `xlmr-disease-detection-service/`:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m healthph_xlmr.api
```

The default URL is `http://127.0.0.1:8003`; interactive API documentation is at
`http://127.0.0.1:8003/docs`. Configure the server with `--host` and `--port`.
One worker loads the model once on CPU in float32; forward passes are serialized.
The dependency pins match Python 3.14.6 on macOS arm64. Verify dependency
installation and performance separately on other deployment platforms.

Model files are resolved relative to this folder, independently of the working
directory. Set `HEALTHPH_XLMR_DISEASE_ARTIFACT_DIR` to select another complete bundle.
Keywords are included in `healthph_symptoms/keywords_master.csv`; set
`HEALTHPH_SYMPTOM_KEYWORDS_PATH` to select a custom CSV. The service uses local
model files only. Symptom extraction uses only the keyword CSV and aliases. Invalid disease
assets, a missing/malformed keyword list, or failed startup inference prevent startup.

ELECTRA continues to use its own service on port 8000. Both services expose the
same disease endpoints and response format; select the intended service URL.

## HTTP API

```sh
curl http://127.0.0.1:8003/v1/disease/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"May ubo at lagnat ako."}'
```

| Endpoint | Request / response |
| --- | --- |
| `POST /v1/disease/predict` | `{"text":"..."}` → one prediction |
| `POST /v1/disease/predict-batch` | `{"texts":["...","..."]}` → `{"results":[...]}` in input order |
| `GET /health/live` | `{"status":"ok"}` |
| `GET /health/ready` | `{"status":"ready","model_version":"sha256:...","symptom_model_version":"sha256:..."}` after the disease model and keywords load and warmup succeeds |

Each prediction contains:

| Field | Meaning |
| --- | --- |
| `labels` | Selected labels in AURI, PN, TB, COVID order; an empty array is valid |
| `probabilities` | Sigmoid score for each of the four labels |
| `thresholds_used` | Saved threshold for each label |
| `model_version` | SHA-256 identifier of inference assets and effective settings |
| `symptoms` | Extracted mentions; `[]` when none are detected |
| `symptom_model_version` | Independent SHA-256 identifier of effective keywords, normalization mappings, and matching rules |

A label is selected when its probability **>= its saved threshold**. The service
uses independent label thresholds, rather than argmax or a shared cutoff.
Adding symptom extraction preserves the disease scores, thresholds, and disease
model version. Each symptom contains `text`, nullable English `normalized_symptom`,
`start_index`, `end_index`, `confidence`, and `negated`. Repeated and negated
mentions are retained. Unknown normalizations are `null`. `confidence: 1.0` means a literal keyword matched; it is not a model probability.
Negation uses simple keyword cue rules. `negated: false` means explicit
negation was not detected; it does not establish a current symptom or identify
its experiencer.

Send the text to be analyzed. As in training, the disease classifier collapses whitespace
and trims it while preserving casing, Unicode, emojis, hashtags, and links. It
does not invoke the language service's cleaner. Previously cleaned text is
accepted and receives the same disease-model whitespace normalization. The symptom
extractor receives the exact submitted text before that normalization; its offsets
therefore refer to that exact text.

Python and HTTP calls accept nonblank strings of at most 10,000 Unicode code points before
normalization and batches of 1–32 texts. All inputs are validated before any batch
inference. Disease tokenization truncates to the saved `max_length`, initially
**128** tokens including special tokens, with dynamic padding for batches. Symptom
extraction covers the complete accepted input using case-insensitive whole-word
matching, flexible whitespace, and longest nonoverlapping phrases.

HTTP `422` means invalid input, including extra request fields; the entire batch
is rejected. HTTP `503` means the model is unavailable. HTTP `500` means inference
failed. Treat service errors as failed analyses, not empty-label predictions.
Extraction failures remain inference errors; the service does not substitute an
empty symptom list for a failure.

## JavaScript client

Use Node.js 24. Copy `examples/xlmr-disease-client.mjs` into your backend or import
it from this folder:

```js
import { createXlmrDiseaseClient } from './examples/xlmr-disease-client.mjs';

const client = createXlmrDiseaseClient();
try {
  const text = '😷 May ubo at lagnat ako.';
  const result = await client.predict(text);
  const { results } = await client.predictBatch(['ubo at lagnat', 'feeling okay']);
  console.log(result.labels, result.model_version, results);
  for (const symptom of result.symptoms) {
    const phrase = Array.from(text).slice(symptom.start_index, symptom.end_index).join('');
    console.log(phrase, symptom.normalized_symptom, symptom.negated);
  }
} catch (error) {
  console.error(error.code, error.status, error.message);
}
```

Set `HEALTHPH_XLMR_DISEASE_URL` or pass `serviceUrl` to override the URL. Pass
`timeoutMs` to override the 30-second default. The client checks response fields,
batch length, and label/threshold consistency. It exports `XlmrDiseaseClientError`
with codes `HTTP_ERROR`, `TIMEOUT`, `CONNECTION_ERROR`, and `INVALID_RESPONSE`.
Requests are not retried; HTTP errors include `status` and `details`.

Offsets are zero-based, end-exclusive Unicode code-point indices into the exact
submitted text, including whitespace and emoji. Use `Array.from(text)` as above;
JavaScript string slicing uses UTF-16 code units. When cleaned text is submitted,
offsets refer to that cleaned text rather than the original raw post.

```sh
node examples/xlmr-disease-client.mjs 'May ubo at lagnat ako.'
```

Browser applications should call the HealthPH backend. For a backend on another
machine, bind the service to a private interface using `--host` and set the
backend's service URL. Access control belongs to the backend and private network;
this service provides no authentication or public ingress.

## Python package

Run from this folder, or add its absolute path to `PYTHONPATH`:

```python
from healthph_xlmr import DiseasePredictor

predictor = DiseasePredictor()  # Create once and reuse.
result = predictor.predict("May ubo at lagnat ako.")
results = predictor.predict_many(["ubo at lagnat", "feeling okay"])
```

Pass `artifact_dir="/absolute/path/to/model"` to override the environment/default
directory. Python calls raise `ValueError` for invalid input or incompatible
metadata; model loading and inference failures propagate to the caller.

## Export and update the model

Continue disease training and evaluation in the existing XLM-RoBERTa notebook.
Keyword extraction requires no additional training. Stop any service using the
target directory before exporting.

From this service folder:

```sh
python -m healthph_xlmr.export \
  ../artifacts/disease_xlmr/full_20261022T064326Z_fa1cd053
```

Use `--target-dir /path/to/new/model` to prepare a bundle elsewhere. The source
must contain `run_config.json` and `best_model/`. Smoke runs are rejected.
The notebook's optional handoff cell uses this same keyword-based default.

To refresh the runtime and keyword list while exporting disease weights, run
from the research repository root:

```sh
.venv/bin/python -m healthph_symptoms.export \
  --service-dir xlmr-disease-detection-service \
  --disease-source artifacts/disease_xlmr/full_20261022T064326Z_fa1cd053
```

The exporters copy explicit inference assets into staging, load
the staged disease model and keyword extractor offline, and run a prediction before replacing the
destination. Validation or warmup failure leaves the previous bundle intact.
A failed replacement restores the previous bundle. If the filesystem also
prevents rollback, the error identifies the preserved backup for manual recovery.
Run one export at a time for a given destination.

```text
model/
  run_config.json
  label_order.json
  thresholds.json
  best_model/
    model.safetensors
    config.json
    tokenizer.json
    tokenizer_config.json
    # sentencepiece.bpe.model, special_tokens_map.json, added_tokens.json when present

```

Disease weights, tokenizer, labels/thresholds, and the bundled keyword runtime
belong to the complete service handoff. The
notebook's original `run_config.json` is retained; its training paths are metadata
and do not need to exist on the backend machine. Reports, datasets, optimizer
state, checkpoints, and `training_args.bin` are excluded.

For each backend handoff:

1. Send the complete exported `model/` directory.
2. Stop the service and keep its previous complete bundle for rollback.
3. Replace the complete bundle and restart the service.
4. Check `/health/ready` and a prediction; record both model versions.

The version changes when weights, tokenizer, labels, thresholds, or effective
inference settings change. Symptom changes affect `symptom_model_version`
independently and preserve the disease version. Moving an unchanged bundle or changing training-only
run metadata preserves its version. Updates take effect after restart; there is
no runtime reload endpoint. To roll back, stop the service, restore the previous
complete bundle, and restart.

## Tests

```sh
python -m pip install -r requirements-test.txt
python -m pytest
python -m pytest --run-model
node --test tests/xlmr-disease-client.test.mjs
```

The disease model suite compares predictions against a frozen notebook loader on CPU,
checks multilingual input, Unicode, normalization, batching, and truncation,
and runs a copied service outside the repository with an empty offline model
cache. It also checks version changes and new thresholds after restart.
Real-model tests require the local bundle and are enabled with `--run-model`;
missing artifacts fail those explicitly requested tests.
Disease parity tests explicitly inject a fake symptom extractor; keyword integration tests verify extraction without a symptom model.
Shared keyword tests check multilingual terms, negation, offsets, and custom lists.

After exporting the disease bundle and starting the service, include
JavaScript tests against it:

```sh
HEALTHPH_TEST_REAL_SERVICE=1 node --test tests/xlmr-disease-client.test.mjs
```
