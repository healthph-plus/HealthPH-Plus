# HealthPH+ disease detection service

Local Python service for the JavaScript HealthPH backend. It uses the saved ELECTRA model on CPU to detect `AURI`, `PN`, `TB`, and `COVID` signals in already cleaned text.

## Share with developers

Share this entire folder, including `model/` (about 55 MB). It contains the service, JavaScript client, dependencies, tests, and documentation. No research notebooks, datasets, or credentials are needed. The model is ignored by Git, so a code-only clone needs a separate copy of `model/`. Exclude `.venv/`, `__pycache__/`, and `.pytest_cache/` when sharing.

## Start the service

Open a terminal inside this folder (`disease-detection-service/`), then run:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m healthph_plus.disease_api
```

If `.venv` already exists, skip its creation. The saved bundle belongs in `model/`, containing `best_model/`, `label_order.json`, and `thresholds.json`. Set `HEALTHPH_DISEASE_ARTIFACT_DIR` to use another bundle directory.

Base URL: `http://127.0.0.1:8000`. Interactive API docs: `http://127.0.0.1:8000/docs`. Use `--port 8001` to change the port.

## HTTP usage

```sh
curl http://127.0.0.1:8000/v1/disease/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"ubo at lagnat"}'
```

| Endpoint | Request / result |
| --- | --- |
| `POST /v1/disease/predict` | Send `{"text":"..."}`; receive one prediction. |
| `POST /v1/disease/predict-batch` | Send `{"texts":["...","..."]}`; receive `{"results":[...]}` in input order. |
| `GET /health/live` | Check that the service responds. |
| `GET /health/ready` | Check that the model is loaded and get its version. |

Each prediction contains `labels`, `probabilities`, `thresholds_used`, and `model_version`. Multiple labels or an empty `labels` array are valid.

## JavaScript usage

Use Node.js 24. In an `.mjs` file at this folder's root:

```js
import { createDiseaseClient } from './examples/disease-client.mjs';

const client = createDiseaseClient(); // localhost:8000; 30-second timeout
try {
  const result = await client.predict('ubo at lagnat');
  const { results } = await client.predictBatch(['ubo at lagnat', 'feeling okay']);
  console.log(result.labels, results);
} catch (error) {
  console.error(error.code, error.status, error.message);
}
```

Set `HEALTHPH_DISEASE_URL` in the backend to change the service URL. To try the client directly: `node examples/disease-client.mjs 'ubo at lagnat'`.

## Input and error handling

- Send already cleaned, non-blank text: at most 10,000 characters per text and 1–32 texts per batch. Tokenization truncates each text to 128 tokens.
- HTTP `422`: invalid input; the entire batch is rejected. HTTP `503`: model unavailable. HTTP `500`: inference failed.
- Client errors include `HTTP_ERROR`, `TIMEOUT`, `CONNECTION_ERROR`, and `INVALID_RESPONSE`. Treat them as failed analyses, not empty-label predictions.

See the [full integration guide](docs/disease_inference.md) for artifact transfer, tests, and the model's current evaluation results.
