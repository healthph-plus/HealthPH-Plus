# Disease inference integration

The JavaScript backend calls a local Python HTTP service. The service runs the saved ELECTRA classifier on CPU and a standard-library keyword symptom extractor, returning disease signals and symptom mentions for already cleaned post text. It does not clean text, retrain models, or write to a database.

Symptom extraction matches the bundled keyword CSV. No symptom model, annotation review, or training is required. See the [keyword workflow](../SYMPTOM_WORKFLOW.md).

## Install and start

Run from the standalone `disease-detection-service/` folder. The verified Python environment is **Python 3.14.6 on macOS arm64**. No checkout of the original research repository is needed.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m healthph_plus.disease_api
```

If this folder's `.venv` already exists, activate it and run the installation command without recreating it. The service defaults to `http://127.0.0.1:8000` and one worker. The disease model is loaded once and serializes its forward passes; keyword patterns compile once at startup. OpenAPI documentation is available at `/docs` and `/openapi.json`.

To use another artifact directory or port:

```sh
HEALTHPH_DISEASE_ARTIFACT_DIR=/absolute/path/to/combined/model \
  python -m healthph_plus.disease_api --port 8001
```

The default artifact directory is `model/` within this standalone folder, resolved relative to the Python package and independent of the working directory. An override points to the disease bundle root. Keywords are bundled at `healthph_symptoms/keywords_master.csv`; use `HEALTHPH_SYMPTOM_KEYWORDS_PATH` for a custom CSV.

Model loading always uses local files, with remote code disabled. To verify operation without Hub access explicitly:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m healthph_plus.disease_api
```

Package installation requires access to the Python package index; inference does not require Internet access.

## Transfer the model artifacts

The artifacts are ignored by Git, so a fresh clone needs a separate copy of the saved model bundle:

```text
model/
├── best_model/
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── label_order.json
└── thresholds.json
```

Transfer the complete disease `best_model/` directory, including additional tokenizer files, plus the generated `healthph_symptoms/` package with its keyword CSV. Training checkpoints, datasets, and symptom model weights are unnecessary. Invalid disease assets or a missing/malformed keyword list prevent startup.

To export from the research repository root:

```sh
.venv/bin/python -m healthph_symptoms.export \
  --service-dir disease-detection-service \
  --disease-source artifacts/poc_transformers/electra_local_finetuned
```

Stop the service first. Export stages disease assets and the shared keyword runtime, validates offline inference, then replaces the bundle. Validation failure preserves the previous bundle; replacement failure attempts rollback.

`model_version` is a SHA-256 fingerprint of the inference contract revision, the bundle's inference file names and contents, and the 128-token limit. It covers the complete non-hidden contents of `best_model/`, plus label order and thresholds. Moving an unchanged bundle preserves its version. Stop the service before replacing a bundle; transfer the matching files together, restart, and verify the version through `/health/ready`. Files are not hot-reloaded.

`symptom_model_version` independently fingerprints effective keyword phrases, normalization mappings, and matching rules. Adding or updating symptom artifacts does not change the disease model's version or scores.

## HTTP contract

| Endpoint | Purpose |
| --- | --- |
| `GET /health/live` | Returns `{"status":"ok"}` when the application can respond. |
| `GET /health/ready` | Returns `status: "ready"`, `model_version`, and `symptom_model_version` after the disease model and keywords load and startup inference succeeds; otherwise HTTP 503. |
| `POST /v1/disease/predict` | Classifies one text. |
| `POST /v1/disease/predict-batch` | Classifies 1–32 texts and preserves input order, including duplicates. |

Send `Content-Type: application/json`. Single requests contain only `text`; batch requests contain only `texts`. A text must be a string containing non-whitespace characters and at most 10,000 Unicode code points. The service preserves supplied text exactly before tokenization. The disease classifier truncates to 128 tokens, including special tokens, matching its training notebook. Symptom extraction matches the full accepted input against every disease row in the keyword CSV, regardless of the disease labels selected. Matching ignores case, allows flexible whitespace, and keeps the longest nonoverlapping phrases.

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/disease/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"feeling okay, no symptoms today"}'

curl --fail-with-body http://127.0.0.1:8000/v1/disease/predict-batch \
  -H 'Content-Type: application/json' \
  -d '{"texts":["ubo at lagnat","feeling okay, no symptoms today"]}'
```

A single result has this shape (illustrative values, not a trained symptom prediction):

```json
{
  "labels": [],
  "probabilities": {
    "AURI": 0.006843,
    "PN": 0.002060,
    "TB": 0.002587,
    "COVID": 0.003539
  },
  "thresholds_used": {
    "AURI": 0.11,
    "PN": 0.12,
    "TB": 0.11,
    "COVID": 0.36
  },
  "model_version": "sha256:62e3d6aa66ae57d2eedbb59ef2e7a7b3966317d80315f1acdfd71939966c6ddc",
  "symptoms": [],
  "symptom_model_version": "sha256:illustrative"
}
```

Batch responses are `{ "results": [<result>, <result>] }`, with the same fields in every result. Preserve the source post IDs in the backend and associate them by input index. Every result exposes all four scores and thresholds, in the fixed label order `AURI`, `PN`, `TB`, `COVID`. Labels are selected independently using sigmoid scores greater than or equal to their saved thresholds; scores do not sum to one. Multiple labels or an empty label list are valid results.

Every symptom contains `text`, nullable English `normalized_symptom`, `start_index`, `end_index`, `confidence`, and `negated`. Unmapped phrases retain their exact text with a `null` normalized name. Repeated occurrences remain separate, and negated mentions are returned. `symptoms: []` means no mentions were detected. `confidence: 1.0` means a literal keyword matched, not a model probability. Negation uses bounded keyword cue rules. `negated: false` only means explicit negation was not detected; it does not establish current symptoms, identify an experiencer, or attribute a symptom to a disease.

Offsets are zero-based, end-exclusive Unicode code-point indices into the **exact submitted text**, including leading whitespace and emoji. They refer to the cleaned text when that is what the backend submitted, rather than the original raw post. JavaScript UTF-16 string slicing uses different indices; use `Array.from(text)` as shown below.

Invalid JSON, missing/extra fields, blank/non-string text, excessive text length, or an invalid batch produces HTTP 422 with FastAPI's `detail` validation information. If one batch item is invalid, none are inferred. HTTP 503 means the model is unavailable, and HTTP 500 means inference failed. A failure must not be stored as a successful empty-label prediction.

## JavaScript backend usage

The dependency-free ESM client is [examples/disease-client.mjs](../examples/disease-client.mjs). It was verified with Node.js 24.21.0; use Node.js 24 to reproduce the tests.

```js
import { createDiseaseClient, DiseaseClientError } from './examples/disease-client.mjs';

const diseaseClient = createDiseaseClient({
  serviceUrl: process.env.HEALTHPH_DISEASE_URL ?? 'http://127.0.0.1:8000',
  timeoutMs: 30_000,
});

try {
  const text = '😷 ubo at lagnat';
  const result = await diseaseClient.predict(text);
  console.log(result.labels, result.probabilities, result.model_version);
  for (const symptom of result.symptoms) {
    const phrase = Array.from(text).slice(symptom.start_index, symptom.end_index).join('');
    console.log(phrase, symptom.normalized_symptom, symptom.confidence, symptom.negated);
  }

  const { results } = await diseaseClient.predictBatch([
    'ubo at lagnat',
    'feeling okay, no symptoms today',
  ]);
  console.log(results);
} catch (error) {
  if (!(error instanceof DiseaseClientError)) throw error;
  console.error(error.code, error.status, error.message);
  // Propagate or record a failed analysis in the backend; do not substitute [].
}
```

The client defaults to a 30-second timeout covering both response headers and body. Error codes are `HTTP_ERROR` (with `status` and server `details`), `TIMEOUT`, `CONNECTION_ERROR`, and `INVALID_RESPONSE`. It does not retry automatically. `serviceUrl` is the service origin; requests use absolute `/v1/disease/...` paths.

Run the example directly:

```sh
HEALTHPH_DISEASE_URL=http://127.0.0.1:8000 node examples/disease-client.mjs 'ubo at lagnat'
```

For a backend on another machine, bind to that machine's private interface using `--host <private-IP>` and set `HEALTHPH_DISEASE_URL` in the JavaScript backend. Access control belongs to the backend and private network; this service adds no authentication or public ingress. Browser clients should call the HealthPH backend. Run one service worker; increased backend concurrency queues at the model lock, so choose backend concurrency and timeout settings based on measured workloads.

## Tests and model status

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest --run-model -q -s
node --test tests/disease-client.test.mjs
```

The default Python suite skips real-checkpoint tests. `--run-model` requires the disease artifact bundle and fails if it is missing. Disease parity tests explicitly inject a fake symptom extractor; keyword integration tests verify real matching without symptom weights. The tests use a frozen copy of the original notebook's inference function in `tests/fixtures/notebook_inference.py`; no notebook or training data is required. They check scores within absolute tolerance `1e-6`, selected labels, Unicode, truncation, ordering, duplicates, and API responses. Unit tests cover invalid artifacts, threshold boundaries, startup/readiness, batch validation, inference failures, and serialization. Shared keyword tests verify multilingual matches, exact offsets, negation, custom lists, and standalone operation without NLP dependencies.

After exporting a reviewed combined bundle and starting the Python service, enable the actual HTTP integration test:

```sh
HEALTHPH_TEST_REAL_SERVICE=1 HEALTHPH_DISEASE_URL=http://127.0.0.1:8000 \
  node --test tests/disease-client.test.mjs
```

JavaScript tests also exercise HTTP failures, malformed responses, stalled-body timeouts, and connection refusal using temporary localhost servers.

Earlier disease-only validation on 2026-09-11 used Python 3.14.6, PyTorch 2.14.0, Transformers 5.16.1, and Node.js 24.21.0 on macOS arm64. Its approximately 23 ms warmed CPU batch measurement excluded symptom extraction, startup, and HTTP overhead. Measure current combined memory and latency with the keyword extractor; the [symptom workflow](../SYMPTOM_WORKFLOW.md) provides the benchmark command. Choose backend timeouts using those measurements.

The pinned Starlette test client currently emits two upstream deprecation warnings about `httpx` and `BlockingPortal`; they do not prevent the tests from passing.

The saved evaluation in the original research repository (`reports/poc_transformers/poc_metrics.json`, not required by this package) reports **test micro-F1 0.459**, below the notebook's **0.75** target. Integration preserves the checkpoint and its thresholds. Outputs represent disease signals in text; preparing this service does not establish improved model accuracy or validate its scores as clinical probabilities.
