# HealthPH+ disease detection service

Local Python service for the JavaScript HealthPH backend. It uses the saved ELECTRA model on CPU to detect `AURI`, `PN`, `TB`, and `COVID` signals in already cleaned text, and literal keyword matching to extract symptom mentions.

Symptom extraction uses the bundled keyword list. No symptom NLP model or training is required. See the [keyword workflow](SYMPTOM_WORKFLOW.md).

## Share with developers

After combined export, share this entire folder, including `healthph_symptoms/` and the complete `model/` directory. The ELECTRA disease assets occupy about 55 MB; the symptom keyword list adds only a small CSV file. No research notebooks, datasets, or credentials are needed for inference. Models are ignored by Git, so a code-only clone needs a separate copy of the complete bundle. Exclude `.venv/`, `__pycache__/`, and `.pytest_cache/` when sharing.

## Start the service

Open a terminal inside this folder (`disease-detection-service/`), then run:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m healthph_plus.disease_api
```

If `.venv` already exists, skip its creation. The saved bundle belongs in `model/`, containing `best_model/`, `label_order.json`, and `thresholds.json`. Keywords are included in `healthph_symptoms/keywords_master.csv`. Set `HEALTHPH_DISEASE_ARTIFACT_DIR` to use another complete bundle directory. Missing or incompatible disease assets or keyword files prevent startup. Symptom extraction uses only the keyword CSV and aliases.

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
| `GET /health/ready` | Check combined startup inference; receive `status`, `model_version`, and `symptom_model_version`. |

Each single/batch prediction contains `labels`, `probabilities`, `thresholds_used`, `model_version`, `symptoms`, and `symptom_model_version`. Multiple labels or an empty `labels` array are valid. Disease scores, thresholds, and the disease version remain unchanged by adding symptom extraction.

Each symptom contains `text`, nullable English `normalized_symptom`, `start_index`, `end_index`, `confidence`, and `negated`. Empty detections return `symptoms: []`; extraction failures remain errors. `confidence: 1.0` denotes a literal keyword match, not a model probability. `negated: false` means explicit negation was not detected; it does not establish a current symptom or identify its experiencer.

## JavaScript usage

Use Node.js 24. In an `.mjs` file at this folder's root:

```js
import { createDiseaseClient } from './examples/disease-client.mjs';

const client = createDiseaseClient(); // localhost:8000; 30-second timeout
try {
  const text = '😷 ubo at lagnat';
  const result = await client.predict(text);
  const { results } = await client.predictBatch(['ubo at lagnat', 'feeling okay']);
  console.log(result.labels, results);
  for (const symptom of result.symptoms) {
    const phrase = Array.from(text).slice(symptom.start_index, symptom.end_index).join('');
    console.log(phrase, symptom.normalized_symptom, symptom.negated);
  }
} catch (error) {
  console.error(error.code, error.status, error.message);
}
```

Set `HEALTHPH_DISEASE_URL` in the backend to change the service URL. To try the client directly: `node examples/disease-client.mjs 'ubo at lagnat'`.

## Input and error handling

- Send already cleaned, non-blank text: at most 10,000 Unicode code points per text and 1–32 texts per batch. Disease tokenization truncates each text to 128 tokens; symptom extraction covers the complete accepted input using case-insensitive whole-word matching, with flexible whitespace and longer phrases preferred.
- Symptom offsets are zero-based, end-exclusive Unicode code-point indices into the exact submitted text. Use `Array.from(text)` in JavaScript, as above, rather than UTF-16 string slicing.
- HTTP `422`: invalid input; the entire batch is rejected. HTTP `503`: model unavailable. HTTP `500`: inference failed.
- Client errors include `HTTP_ERROR`, `TIMEOUT`, `CONNECTION_ERROR`, and `INVALID_RESPONSE`. Treat them as failed analyses, not empty-label predictions.

See the [full integration guide](docs/disease_inference.md) for artifact transfer, tests, and the model's current evaluation results.

## Train or update the disease model

Optional training runs directly from this folder, without a notebook or the
research repository. Install the training extras in the service environment:

```sh
python -m pip install -r requirements-training.txt
python -m healthph_plus.train \
  --data /absolute/path/to/annotations.csv \
  --output-dir training-runs/my-run \
  --epochs 5 --batch-size 4 --learning-rate 2e-5
```

The CSV uses the existing annotation format, with four binary integers in
**AURI, PN, TB, COVID** order. Multiple diseases and all-zero examples are valid:

```csv
post,annotate
"May ubo ako.","[1, 0, 0, 0]"
"Feeling well.","[0, 0, 0, 0]"
```

This two-row example illustrates the format; a training dataset needs positive
and negative examples of every disease in each split. The command validates all
rows, removes consistent duplicates across input files, rejects conflicting
annotations, and makes reproducible multilabel train/validation/test splits
(80/10/10 by default). Whitespace variants belong to the same duplicate group;
ELECTRA receives the original submitted text, matching its inference service.

The current research CSVs contain conflicting duplicate annotations. Correct
those labels, or pass `--conflict-policy exclude` to remove every row in each
conflicting post group. The command prints excluded counts and records all
excluded fingerprints/source rows in `reports/data_quality.json`.
Long training posts are retained and tokenized with the service's 128-token limit;
the audit counts posts above the inference API's 10,000-character input limit.

Training starts from `model/best_model/`, with a fresh optimizer. Use
`--model-path /path/to/another/best_model` for another compatible local four-label
ELECTRA model/tokenizer. Model loading is offline. Repeat `--data` to combine
CSVs. Use `--validate-only` to save data audits and a split manifest without
loading weights. `--device cpu` forces CPU; `auto` uses the device selected by
PyTorch/Trainer. Run `python -m healthph_plus.train --help` for split fractions,
seed, gradient accumulation, early stopping, and other settings.

Each output directory must be new. Runs save `best_model/`, `label_order.json`,
`thresholds.json`, `run_config.json`, `selection.json`, and `status.json`.
`reports/` includes data quality and split audits, epoch logs, threshold searches,
validation/test metrics, per-disease precision/recall/F1 and confusion counts,
prediction scores, and a saved-model reload check. Reports identify posts by
fingerprint and source row; they do not copy post text. `status: complete` means
training, evaluation, and reload verification finished successfully.

Checkpoint selection and early stopping use validation micro-F1 at 0.5.
Per-disease thresholds are then tuned using validation F1; the held-out test split
is evaluated after those decisions are frozen. ELECTRA uses unweighted binary
cross-entropy. These commands continue existing weights: independent performance
estimates also require examples excluded from the initial model's earlier training.

After reviewing the completed run, stop the service and select its bundle on restart:

```sh
HEALTHPH_DISEASE_ARTIFACT_DIR=/absolute/path/to/training-runs/my-run \
  python -m healthph_plus.disease_api
```

Training leaves the current `model/` bundle intact. Keep `training-runs/`, datasets,
and checkpoints out of inference-only handoffs. Symptom keywords need no training.
The existing research export below can also package a completed CLI run by using
its output directory as `--disease-source`.

## Prepare a combined handoff

Run from the research repository root:

```sh
.venv/bin/python -m healthph_symptoms.export \
  --service-dir disease-detection-service \
  --disease-source artifacts/poc_transformers/electra_local_finetuned
```

Stop the service before export. The exporter stages disease weights, keyword data, and shared runtime code, then verifies offline inference before replacing the bundle. Set `HEALTHPH_SYMPTOM_KEYWORDS_PATH` to use a custom CSV. See the [keyword workflow](SYMPTOM_WORKFLOW.md) for updates and matching rules.
