# HealthPH+ sentiment analysis

Standalone CPU inference service for the JavaScript HealthPH+ backend. It serves
the mBERT model trained in `notebooks/05_binary_sentiment_mbert.ipynb` and predicts
**calm (0)** or **worried (1)**. These are the notebook's supplied categories.

## Share with backend developers

Share this entire folder, including the Git-ignored `model/` directory (about
714 MB). It contains the Python module, HTTP API, JavaScript client, dependencies,
and tests. Exclude `.venv/`, `__pycache__/`, and `.pytest_cache/`.
No notebook, training dataset, research repository, or credentials are needed.
A code-only clone requires a separate copy of the model bundle.

The initial bundle comes from run `20260928T014234Z_b10df0fd`. Serving it does not
resume training or select between architectures: this API serves mBERT only.

## Setup and run

Run from inside `sentiment-analysis-service/`:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m healthph_sentiment.api
```

Default URL: `http://127.0.0.1:8002`. Interactive documentation:
`http://127.0.0.1:8002/docs`. Use `--host` and `--port` to configure the server.
One worker loads the model once on CPU; forward passes are serialized.
The pinned runtime is tested on Python 3.14.6, macOS arm64. Other platforms need
their own dependency installation check.

Assets are resolved relative to this folder, independently of the working
directory. Set `HEALTHPH_SENTIMENT_ARTIFACT_DIR` to select another complete model
directory. The service uses local files only and never downloads model assets.
Missing files, invalid settings, incompatible weights, or a failed startup
prediction prevent server startup.

## HTTP API

```sh
curl http://127.0.0.1:8002/v1/sentiment/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Kinakabahan ako, my fever is not going away."}'
```

| Endpoint | Request / result |
| --- | --- |
| `POST /v1/sentiment/predict` | `{"text":"..."}` → one prediction |
| `POST /v1/sentiment/predict-batch` | `{"texts":["...","..."]}` → `{"results":[...]}` in input order |
| `GET /health/live` | `{"status":"ok"}` |
| `GET /health/ready` | `{"status":"ready","model_version":"sha256:..."}` after successful model warmup |

Each prediction contains:

| Field | Meaning |
| --- | --- |
| `predicted_label` | `0` for calm, `1` for worried |
| `predicted_label_name` | `calm` or `worried` |
| `worried_probability` | Two-class softmax probability for class 1 |
| `threshold_used` | Validation-tuned threshold from the saved bundle |
| `model_version` | SHA-256 identifier of inference artifacts and settings |

The decision is **worried when probability >= threshold**. The saved threshold
is used rather than a fixed 0.5 or argmax. The score is a model probability; the
API does not apply an additional calibration step.

Send posts to be analyzed. Only whitespace is collapsed and trimmed; casing,
Unicode, emojis, hashtags, and links are preserved, matching training. The
service does not invoke the language service's cleaner. If your upstream pipeline
has already cleaned a post, inference uses that supplied content.

HTTP and Python calls accept nonblank strings of at most 10,000 characters and
1–32 texts per batch. All inputs are validated before a batch runs. Tokenization
truncates to the bundle's `max_length` (256 in the initial bundle).

HTTP `422` means invalid input, including extra request fields; the entire batch
is rejected. HTTP `503` means the model is unavailable. HTTP `500` means inference
failed. Treat service errors as failed analyses, rather than calm predictions.

## JavaScript client

Use Node.js 24. Copy `examples/sentiment-client.mjs` to your backend or import it
from this folder:

```js
import { createSentimentClient } from './examples/sentiment-client.mjs';

const client = createSentimentClient();
try {
  const result = await client.predict('Kinakabahan ako, my fever is not going away.');
  const { results } = await client.predictBatch(['may sakit ako', 'okay na ako']);
  console.log(result.predicted_label_name, result.model_version, results);
} catch (error) {
  console.error(error.code, error.status, error.message);
}
```

Set `HEALTHPH_SENTIMENT_URL` or pass `serviceUrl` to change the URL. The default
timeout is 30 seconds; pass `timeoutMs` to override it. The client validates the
response fields and threshold/label consistency. Error codes are `HTTP_ERROR`,
`TIMEOUT`, `CONNECTION_ERROR`, and `INVALID_RESPONSE`. Requests are not retried.

Try the client directly: `node examples/sentiment-client.mjs 'may sakit ako'`.

## Python module

Run from this folder, or add its absolute path to `PYTHONPATH`:

```python
from healthph_sentiment import SentimentPredictor

predictor = SentimentPredictor()  # Create once and reuse.
result = predictor.predict("Kinakabahan ako, my fever is not going away.")
results = predictor.predict_many(["may sakit ako", "okay na ako"])
```

Pass `artifact_dir="/path/to/model"` to override the environment/default path.
Python calls raise `ValueError` for invalid input or incompatible artifacts.

## Update the model after training

Continue training and evaluation in the existing notebook. Its final optional
handoff cell defaults to `EXPORT_FOR_BACKEND = False`. After completing a run,
set it to `True` and run that cell to export that run's mBERT model into this
service's `model/` directory. Stop any service using the target directory first.
The cell does not change model selection, train again, or export TF-IDF.

The export helper copies only inference assets into a staging directory, loads
the staged model, and runs a prediction before replacing the existing bundle.
A validation or warmup failure leaves the previous bundle intact. A failed
replacement restores the previous bundle.

You can also export a completed saved run from this service folder:

```sh
python -m healthph_sentiment.export \
  ../artifacts/binary_sentiment/<run_id>/mbert
```

Use `--target-dir /path/to/new/model` to prepare a bundle elsewhere. The source
must be the exported `mbert/` directory, rather than `checkpoints/` or a whole run.

The bundle contains:

```text
model/
  run_config.json
  label_mapping.json
  threshold.json
  best_model/
    model.safetensors
    config.json
    tokenizer.json
    tokenizer_config.json
    # vocab.txt, special_tokens_map.json, added_tokens.json when present
```

Weights, tokenizer, label mapping, and threshold belong to the same export.
The original notebook's `run_config.json` is retained; its training paths are
metadata and do not need to exist on the backend machine. Checkpoints, optimizer
state, training arguments, datasets, and reports are excluded.

For subsequent backend handoffs:

1. Send the complete newly exported `model/` directory.
2. Backend developers stop the service and keep the previous bundle for rollback.
3. Replace the entire `model/` directory and restart the same service command.
4. Check `GET /health/ready` and a prediction. Record the returned `model_version`.

The version changes when weights, tokenizer, threshold, labels, or inference
settings change. Moving an unchanged bundle or changing training-only metadata
preserves its version. No runtime reload endpoint is provided.
For rollback, stop the service, restore the previous complete bundle, and restart.

## Tests

```sh
python -m pip install -r requirements-test.txt
python -m pytest
python -m pytest --run-model
node --test tests/sentiment-client.test.mjs
```

The model suite compares predictions with a frozen notebook loader, checks
multilingual inputs and truncation, and runs a copied service outside the research
repository with offline loading. It also verifies a changed threshold becomes
active with a new model version after restart.

To include JavaScript tests against a running service:

```sh
HEALTHPH_TEST_REAL_SERVICE=1 node --test tests/sentiment-client.test.mjs
```
