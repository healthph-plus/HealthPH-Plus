
# 🏥 HealthPH+

**HealthPH+** is an AI-powered, multilingual public health surveillance system designed to detect disease signals, track misinformation, and analyze public sentiment using social media and participatory data across the Philippines.

It builds upon the original HealthPH initiative by integrating **multilingual NLP**, **machine learning**, and **real-time analytics** to support early outbreak detection and data-driven public health decision-making.

---

## 🚀 Overview

HealthPH+ leverages informal data sources such as social media posts, community reports, and environmental data to enhance traditional disease surveillance systems.

Key capabilities include:

- 🌐 Multilingual NLP (Filipino, Cebuano, Ilocano, Hiligaynon, English)
- 📊 Real-time disease surveillance and outbreak detection
- 🧠 AI-based sentiment analysis and misinformation tracking
- 🗺️ Geospatial visualization and heatmaps
- 📱 Web dashboard and mobile reporting application
- 📈 Time-series forecasting for outbreak prediction

The system aligns with the **One Health framework**, integrating human, environmental, and community data for holistic health monitoring.

---

## 🧠 Core Features

### 1. Data Pipeline
- Multi-source ingestion (Twitter/X, Facebook, Reddit, TikTok)
- Data cleaning, normalization, and language detection
- Multilingual annotation for:
  - Symptoms
  - Diseases
  - Sentiment
  - Misinformation

### 2. NLP & AI Models
- Transformer-based models (mBERT, XLM-R)
- Keyword-based symptom extraction
- Sentiment classification
- Misinformation detection
- Multilabel disease classification

### 3. Analytics & Visualization
- Outbreak heatmaps
- Sentiment trend analysis
- Misinformation hotspot tracking
- Model explainability (SHAP, LIME)

### 4. Applications
- **Web Dashboard** for analysts and policymakers
- **Mobile App** for citizen reporting and health literacy

---

## ⚙️ Tech Stack

- **App backend:** JavaScript
- **Disease inference service:** Python / FastAPI
- **Frontend:** ReactJS
- **Database:** MongoDB
- **ML/NLP:** Python, PyTorch
- **Visualization:** 
- **Mobile:** Flutter / Swift

---

## 📊 Data Sources

- Social Media Platforms:
  - Twitter/X
  - Facebook
  - Reddit
  - TikTok
  - Threads
- Community Reports
- Environmental Data:
  - Weather
  - Air Quality

---

## 🎯 Objectives

HealthPH+ aims to:

- Improve early detection of disease outbreaks
- Enhance multilingual public health surveillance
- Reduce misinformation spread
- Support data-driven decision-making for DOH and LGUs
- Increase inclusivity for underserved linguistic communities

---

## 🌍 Impact

- **Public Health:** Faster outbreak response and improved surveillance
- **Social:** Increased trust and community engagement
- **Economic:** Reduced healthcare costs through early intervention
- **Technological:** Advancement of AI in low-resource languages

---

## 🤝 Partners

- Department of Health (DOH)
- DOST-PCHRD
- Local Government Units (LGUs)
- Philippine Statistics Authority (PSA)
- Academic institutions across the Philippines

---

## 📌 Future Work

- Collect more data for  Ilocano and Hiligaynon
- Update model weights using updated dataset
- Integrate sentiment analysis for outbreak detection

---

## 💻 Initialization

1. Initialize virtual environment using ``` python -m venv .venv ```
2. Activate venv using ``` source .venv/bin/activate```
3. Install dependencies using ```pip install -r requirements.txt```

### Disease detection integration

The standalone service, JavaScript client, tests, and documentation are in
[`disease-detection-service/`](disease-detection-service/README.md). Share that folder
with backend developers and follow its README for setup and usage. Run service
commands from inside that folder. See its [integration guide](disease-detection-service/docs/disease_inference.md)
for the full API contract and validation instructions.

### XLM-RoBERTa disease detection integration

The standalone XLM-RoBERTa predictor, HTTP API, JavaScript client, and export
workflow are in [`xlmr-disease-detection-service/`](xlmr-disease-detection-service/README.md).
Share the entire folder, including its Git-ignored `model/` bundle. It uses the
same disease API contract as ELECTRA on its own default port, **8003**. The initial
bundle is run `full_20261022T064326Z_fa1cd053`. Training is available through the
standalone command below or `notebooks/06_xlm_roberta_disease.ipynb`; the notebook's
optional final export cell and the service's export command prepare complete model updates.

### Disease model training commands

Both disease services support offline fine-tuning from their local model bundles.
From the selected service folder, install `requirements-training.txt`, then run:

```sh
# Inside disease-detection-service/
python -m healthph_plus.train --data /path/to/annotations.csv --output-dir training-runs/my-run

# Inside xlmr-disease-detection-service/
python -m healthph_xlmr.train --data /path/to/annotations.csv --output-dir training-runs/my-run
```

CSVs use `post` and `annotate`, with binary label lists in AURI, PN, TB, COVID order.
Commands validate/deduplicate data, create reproducible splits, select checkpoints
and thresholds using validation data, evaluate the held-out test split, and verify
the saved bundle reloads. `--validate-only` checks data without training.
`--help` lists epochs, batch size, learning rate, device, and other options.
Conflicting duplicate annotations fail by default; `--conflict-policy exclude`
explicitly excludes and audits entire conflict groups in the current research CSVs.
Training writes a new bundle and reports without replacing the running service's
model. See each service README for setup, data requirements, and model activation.

The canonical runtime is [`healthph_disease_training/`](healthph_disease_training/).
After editing it, refresh the generated copies in both standalone services with
`python -m healthph_disease_training.vendor`. Run the offline training checks with
`.venv/bin/python -m pytest tests/test_disease_training.py -q`.

---

### Keyword symptom extraction

Both disease services now extract symptoms from
[`docs/keywords/keywords_master.csv`](docs/keywords/keywords_master.csv) using
case-insensitive keyword matching. No symptom NLP model or training is required.
Responses include exact matched text, source offsets, optional English names,
and simple rule-based negation alongside the existing disease predictions.
See the [keyword workflow](healthph_symptoms/KEYWORD_WORKFLOW.md) for usage,
custom keyword lists, and standalone exports. Try the
[keyword notebook](notebooks/08_symptom_keywords.ipynb) for a runnable example.

### Language detection integration

The reusable Python module, HTTP API, FastText models, and JavaScript client are in
[`language-detection-service/`](language-detection-service/README.md). Share the
entire folder, including its Git-ignored models, with developers. The preprocessing
notebook uses this same module for cleaning and language detection.

### Sentiment analysis integration

The mBERT inference module, HTTP API, JavaScript client, tests, and handoff
instructions are in [`sentiment-analysis-service/`](sentiment-analysis-service/README.md).
Share the entire folder, including its Git-ignored `model/` bundle, with backend
developers. Training stays in `notebooks/05_binary_sentiment_mbert.ipynb`; its
optional final export cell prepares future model updates. Backend developers
replace the complete model bundle and restart the service without changing
their API integration.

### Model performance reports

The ELECTRA, classic multilabel, mBERT sentiment, and XLM-RoBERTa notebooks use
shared reporting utilities in [`healthph_plus/model_reporting.py`](healthph_plus/model_reporting.py).
Run the notebooks with the project kernel from the repository root or `notebooks/`.
Figures appear inline and are exported as **300-DPI PNG and SVG**, alongside CSV
and JSON source tables in each model's report directory.

Transformer dashboards refresh at each validation event and include full-training-set
and validation accuracy, F1, precision, recall, and loss, plus learning rate. The
extra training evaluation adds runtime each epoch. Optimization loss is displayed
separately from evaluation-mode loss; weighted losses retain their objective names.
Checkpoint selection and early stopping use validation metrics only. Training
curves use threshold 0.5; final evaluations use the recorded validation-selected
thresholds. Disease accuracy means **exact match across all four labels**.

Classic models record Optuna trial metrics, best-so-far validation F1, final
train/validation/test scores, and unweighted probability log loss. Trials represent
independent fits rather than training epochs. Fresh detailed artifacts live under
`reports/classic_multilabel/runs/<run_id>/<model>/`; the existing summary CSV points
to those directories so analysis does not combine older files with a newer run.

All models export validation/test confusion matrices in raw-count and
row-normalized views. Cell annotations include counts and percentages within the
actual class, with `N/A` for missing classes. Per-label scores include support;
precision–recall and validation threshold/F1 plots accompany the matrices.
Inference demos display scores, decision thresholds, and warmed inference time
without changing prediction return formats. Unlabeled demos have no accuracy/F1
or loss because those measurements require ground truth.

To redraw an existing experiment without training or loading model weights,
explicitly select its report directory:

```bash
.venv/bin/python -m healthph_plus.render_model_reports \
  --report-dir artifacts/disease_xlmr/full_20261022T064326Z_fa1cd053/reports \
  --model-name XLM-RoBERTa --task multilabel
```

For sentiment use `--task binary`. For historical ELECTRA reports use
`--report-dir reports/poc_transformers --model-name electra_local_finetuned`.
For older classic reports select `--report-dir reports/classic_multilabel` and the
exact model name, such as `BinaryRelevance_LogReg`. Outputs default to the selected
directory's `visualizations/`; `--output-dir` selects another destination.
Missing historical training accuracy/F1 is labeled unavailable. No historical
training metrics or thresholds are reconstructed from unrelated artifacts.

Run the focused offline checks with:

```bash
.venv/bin/python -m pytest tests/test_model_reporting.py tests/test_training_monitor.py tests/test_classic_reporting.py -q
```

### Daily disease-post forecasting prototype

Run [`notebooks/07_time_series_forecasting.ipynb`](notebooks/07_time_series_forecasting.ipynb)
with the project `.venv` kernel from the repository root or `notebooks/`. It uses
the timestamped processed corpus and the complete local XLM-RoBERTa bundle in
`xlmr-disease-detection-service/model/`, explicitly pinned to
`full_20261022T064326Z_fa1cd053`. Use the disease service's inference dependencies
when preparing a new environment; no running HTTP service or retraining is needed.

The default experiment covers September 1–November 30, 2025 in Asia/Manila time.
It forecasts daily AURI, pneumonia, TB, and COVID post counts seven days ahead,
comparing zero, last-value, previous-week, and trailing-week-mean baselines. Three
weekly validation folds select each disease's model; two later weekly test folds
measure held-out error. The final December 1–7 forecast is a historical
demonstration, with unknown actuals preserved in its exports.

Reusable preparation, inference caching, aggregation, evaluation, and reporting
live in `healthph_plus/forecast_data.py`, `healthph_plus/forecasting.py`, and
`healthph_plus/forecast_reporting.py`. Completed prediction batches are cached
under `artifacts/forecasting/`; model versions and normalized-text hashes prevent
reuse across changed models or inputs. Isolated runs export audits, fingerprints,
labeled post IDs, daily counts, MAE/RMSE tables, predictions, and PNG/SVG charts
under `reports/forecasting/<run_id>/`. Reports exclude raw and cleaned post text.

These forecasts describe the sampled corpus. Collection gaps and changing keyword
searches are not zero disease activity. Days without eligible posts remain
unavailable and stop evaluation. The frozen classifier was trained after the
demonstration period without a chronological training cutoff; this experiment
does not establish prospective outbreak-prediction performance.

Run the focused offline checks without loading classifier weights:

```bash
.venv/bin/python -m pytest tests/test_forecast_data.py tests/test_forecasting.py tests/test_forecast_reporting.py -q
```

## 📄 License

This project is part of a DOST-funded research initiative. Licensing and usage terms will be defined upon release.
