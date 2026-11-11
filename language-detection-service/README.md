# HealthPH+ language detection

Copy this entire folder to give developers a local HTTP API and reusable Python
module. It reproduces the language detection in `notebooks/preproccessor.ipynb`.
No research datasets or disease model are required.

## Setup and run

From inside `language-detection-service/`:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m healthph_language.api
```

The pinned runtime was tested on Python 3.14.6, macOS arm64, CPU. FastText includes
a native extension; installing from source requires a C++ compiler (on macOS,
Xcode Command Line Tools). Other platforms require their own installation check.

Base URL: `http://127.0.0.1:8001`. Interactive API docs: `http://127.0.0.1:8001/docs`.
Use `--port 8002` or `--host` to configure the server.

### Models and keywords

Include `models/lid.176.bin` (about 125 MB), `models/lid.176.ftz` (about 916 KB),
and both CSV files in `keywords/` when sharing. Model binaries are ignored by Git:
a code-only clone requires copying them from the repository's original `models/`
directory or the developer handoff. Exclude virtual environments and test caches.

The detector prefers `.bin` and uses `.ftz` only when `.bin` is absent. Set
`HEALTHPH_LANGUAGE_MODEL_PATH` to select a model explicitly. In Python,
`LanguageDetector(model_path="/absolute/path/lid.176.ftz")` overrides that setting;
`keyword_dir=` selects another keyword directory. Bundled paths are resolved
relative to the module. No assets are downloaded automatically. A missing explicit
model, corrupt selected model, or missing/empty keyword list fails startup with an
asset error. A corrupt `.bin` does not silently fall back to `.ftz`.

## HTTP API

```sh
curl http://127.0.0.1:8001/v1/language/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"I am feeling better today."}'
```

| Endpoint | Request / result |
| --- | --- |
| `POST /v1/language/predict` | `{"text":"..."}` → one prediction |
| `POST /v1/language/predict-batch` | `{"texts":["...","..."]}` → `{"results":[...]}` in input order |
| `GET /health/live` | `{"status":"ok"}` |
| `GET /health/ready` | Ready status and the loaded model filename after a successful inference warmup |

Every prediction contains:

| Field | Meaning |
| --- | --- |
| `language` | Final language code after keyword overrides |
| `cleaned_text` | Text passed to FastText |
| `is_supported` | Whether the final code is one of `en`, `tl`, `ceb`, `fil`, `ilo`, `hil` |
| `prediction_source` | `fasttext` or `keyword_override` |
| `fasttext_language` | Original top FastText language code |
| `fasttext_confidence` | Score for **fasttext_language**, even when overridden; clamped to [0, 1] for numerical rounding |
| `fasttext_confidence_percentage` | The same model score multiplied by 100 and rounded to two decimal places, from 0 to 100 |

### Optional two-language predictions

Both prediction endpoints accept these optional fields:

| Request field | Default | Meaning |
| --- | --- | --- |
| `top_k` | `1` | Integer `1` or `2`; use `2` to request up to two ranked model candidates |
| `secondary_min_confidence` | `0.30` | Finite number from `0` to `1`, inclusive; minimum score for the second candidate |

```sh
curl http://127.0.0.1:8001/v1/language/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello po, can I ask kung saan ang clinic?","top_k":2,"secondary_min_confidence":0.30}'
```

With `top_k: 2`, each prediction includes a `fasttext_predictions` array. For
example, its candidates may look like this (scores rounded for illustration):

```json
{
  "fasttext_predictions": [
    {"language": "tl", "confidence": 0.60, "confidence_percentage": 60.0},
    {"language": "en", "confidence": 0.31, "confidence_percentage": 31.0}
  ]
}
```

The highest-scoring candidate is always included, even below the threshold. The
second is included only when its clamped score is **at least** the threshold.
Set the threshold to `0` to keep both candidates. Candidates can be any model
language, including unsupported languages; they are not restricted to English
and Tagalog. In batch requests, the options apply to every input text.

The array reports model predictions independently of keyword overrides. Existing
fields such as `language`, `is_supported`, and `prediction_source` keep their
current meaning. `top_k: 1` omits the array and ignores a valid secondary threshold;
the primary `fasttext_confidence_percentage` is always included. Invalid options
return HTTP `422` before inference. Thresholds still use the unrounded 0–1 scores.

Each candidate's `confidence_percentage` is its `confidence` multiplied by 100,
rounded to two decimal places. Percentages are JSON numbers: `60.0` means 60%.
Use `top_k: 2` and `secondary_min_confidence: 0` to receive both candidates,
including a low-scoring second language. Returned percentages need not sum to
100%, since other model languages can account for the remaining confidence.

The model has English (`en`) and Tagalog (`tl`) labels, with no separate Taglish
label or flag. Mixed text may produce other candidate pairs. Scores represent
model confidence, not the proportion of each language in the text, and are not
renormalized across the returned candidates.

### Cleaning, overrides, and input limits

Send original raw posts. The cleaner preserves the notebook's exact rules:
double HTML decoding, removal of links, mentions, hashtags, and non-ASCII
characters, whitespace normalization, and lowercasing. Curated complete words or
phrases are matched against **original text**, case-insensitively with flexible
whitespace. Hiligaynon wins if both keyword lists match. These inherited rules
include broad words such as `ubo`; they are heuristics, and an override has no
separate confidence score. There is no minimum score for the primary prediction.

Unsupported languages are returned with `is_supported: false`; callers decide
whether to filter them. `fil` and `tl` are not remapped. The compact model may
produce different predictions from `.bin`.

HTTP requests accept at most 10,000 characters per text and 1–32 texts per batch.
Blank, non-string, or text with no ASCII letters after cleaning is invalid
(including emoji-only, URL-only, and non-ASCII-only input). Invalid input rejects
the entire batch with HTTP `422`. HTTP `503` means the detector is not ready;
HTTP `500` means inference failed. Asset failures prevent server startup.

## Python usage

Run from this folder, or add its absolute path to `PYTHONPATH` in another project:

```python
from healthph_language import LanguageDetector, normalize_text

detector = LanguageDetector()  # Create once and reuse.
result = detector.predict("RT @person: I am feeling better today. https://example.com")
results = detector.predict_many(["may ubo ako", "sakit ti barukong"])
print(result["language"], result["cleaned_text"])

mixed = detector.predict("Hello po, can I ask kung saan ang clinic?", top_k=2)
mixed_batch = detector.predict_many(
    ["Hello po, can I ask kung saan ang clinic?", "feeling okay"],
    top_k=2, secondary_min_confidence=0.20,
)
print(mixed["fasttext_predictions"])
for candidate in mixed["fasttext_predictions"]:
    print(f"{candidate['language']}: {candidate['confidence_percentage']:.2f}%")
```

Both prediction methods accept keyword-only `top_k=1` and
`secondary_min_confidence=0.30` options with the same rules as HTTP. Python calls
raise `ValueError` for invalid inputs or options, including invalid options on
an empty batch. For offline notebook use,
`predict_many` accepts any list length (an empty list returns `[]`), processes
32 texts at a time, and has no text-length cap. All texts are validated before
inference. `normalize_text` accepts strings and does not validate the result;
the notebook adapts pandas missing values before calling it.

## JavaScript usage

Use Node.js 24. Copy `examples/language-client.mjs` into your backend or import it
from this folder:

```js
import { createLanguageClient } from './examples/language-client.mjs';

const client = createLanguageClient();
try {
  const result = await client.predict('may ubo ako');
  const { results } = await client.predictBatch(['sakit ti barukong', 'feeling okay']);
  console.log(result.language, results);

  const mixed = await client.predict('Hello po, can I ask kung saan ang clinic?', {
    topK: 2,
  });
  const mixedBatch = await client.predictBatch(['feeling okay', 'may sakit ako'], {
    topK: 2, secondaryMinConfidence: 0.20,
  });
  console.log(mixed.fasttext_predictions, mixedBatch.results);
  for (const candidate of mixed.fasttext_predictions) {
    console.log(`${candidate.language}: ${candidate.confidence_percentage.toFixed(2)}%`);
  }
} catch (error) {
  console.error(error.code, error.status, error.message);
}
```

Set `HEALTHPH_LANGUAGE_URL` or pass `serviceUrl` to change the URL. The default
timeout is 30 seconds; set `timeoutMs` to override it. Client error codes are
`HTTP_ERROR`, `TIMEOUT`, `CONNECTION_ERROR`, and `INVALID_RESPONSE`. Requests are
not retried. To try the client: `node examples/language-client.mjs 'may ubo ako'`.
The optional second argument to either prediction method is
`{ topK, secondaryMinConfidence }`. Unspecified options are omitted from request
JSON so the API defaults apply. The client validates candidate arrays and
requires them when `topK: 2` is requested. It validates percentage fields when
present and also accepts older service responses that omit them.

## Tests

```sh
python -m pip install -r requirements-test.txt
python -m pytest
python -m pytest --run-model
node --test tests/language-client.test.mjs
```

The model suite compares predictions and cleaning to a frozen notebook reference,
checks both model formats, and exercises a copied service outside the repository.
To also check the JavaScript client against a running service:

```sh
HEALTHPH_TEST_REAL_SERVICE=1 node --test tests/language-client.test.mjs
```
