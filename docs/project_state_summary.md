# HealthPH+ Project State Summary

Last updated: 2026-07-03

## Executive Summary

HealthPH+ is currently a research-oriented multilingual public health NLP project focused on collecting Philippine social media posts, preparing annotation/training datasets, and evaluating multilabel disease classifiers for AURI, pneumonia, tuberculosis, and COVID-19. The repository contains raw and processed datasets, language-specific keyword lists, annotation guidance, classic ML experiment outputs, a proof-of-concept ELECTRA transformer run, and scraper/modeling notebooks.

The strongest current classic baseline by test micro-F1 is `ClassifierChains_SGD` with `0.539` test micro-F1 and `0.415` exact match ratio. The available transformer proof of concept uses `google/electra-small-discriminator`; it reaches `0.633` validation micro-F1 but drops to `0.459` test micro-F1, suggesting more validation is needed before treating it as a production candidate.

## Repository Snapshot

- Branch: `main`
- Latest commit observed: `af95e7c` on 2026-06-25, `Update .gitignore`
- Uncommitted changes currently exist in:
  - `data/combined/combined_data.csv`
  - `data/processed/merged_data.csv`
  - `notebooks/02_classic_multilabel_classifiers.ipynb`
  - `notebooks/03_keyword_disease_annotator.ipynb`
  - `notebooks/eda.ipynb`
  - `scrapers/scraper_reddit.ipynb`
  - `scrapers/scraper_x.ipynb`

## Current Architecture and Scope

The README describes HealthPH+ as an AI-powered, multilingual public health surveillance system for disease signal detection, misinformation tracking, and sentiment analysis across Philippine social media and participatory data. The planned product scope includes:

- Multisource ingestion from Twitter/X, Facebook, Reddit, TikTok, and Threads.
- Multilingual NLP across Filipino, Cebuano, Ilocano, Hiligaynon, and English.
- Disease, symptom, sentiment, and misinformation analysis.
- Future web dashboard, mobile reporting, geospatial visualization, and forecasting features.

The codebase is currently weighted toward data preparation and modeling research. Backend, frontend, mobile app, database migrations, and dashboard implementation are not yet present as application code.

## Data Assets

### Raw and Processed Data

| Dataset | Rows | Columns | Notes |
|---|---:|---:|---|
| `data/combined/combined_data.csv` | 59,732 | 8 | Combined social data with `created_at`, `id`, `text`, `source`, and engagement counts. |
| `data/processed/merged_data.csv` | 40,482 | 14 | Main processed dataset following the legacy posts-like shape. |
| `data/processed/merged_data_2.csv` | 3,773 | 4 | Smaller processed dataset with `id`, `text`, `date`, `source`. |
| `data/processed/twitter_data_final.csv` | 1,197 | 4 | Twitter-specific processed output. |

Current `combined_data.csv` source distribution:

| Source | Rows |
|---|---:|
| Twitter/X | 28,933 |
| Reddit | 24,902 |
| Threads | 4,987 |
| TikTok | 810 |
| Facebook | 100 |

Current `merged_data.csv` source distribution:

| Source | Rows |
|---|---:|
| Reddit | 22,298 |
| Twitter/X | 14,102 |
| Threads | 3,526 |
| TikTok | 484 |
| Facebook | 72 |

Current `merged_data.csv` language distribution:

| Language | Rows |
|---|---:|
| `tl` | 19,541 |
| `en` | 19,118 |
| `ceb` | 1,761 |
| `ilo` | 62 |

No Hiligaynon rows are currently visible in `merged_data.csv`, despite Hiligaynon keyword assets being present.

### Training and Annotation Data

| Dataset | Rows | Columns | Notes |
|---|---:|---:|---|
| `data/training_data/annotation_ready.csv` | 30,264 | 4 | Exposes `cleaned_text`, `disease`, `misinformation`, `sentiment`; label fields are currently blank. |
| `data/training_data/annotation_ready.xlsx` | - | - | Spreadsheet version of the annotation-ready dataset. |
| `data/training_data/gold_standard.csv` | 21,968 | 2 | Existing gold-standard annotation source with `post` and `annotate`. |
| `data/training_data/training_1.csv` | 21,968 | 4 | Adds misinformation and sentiment fields to gold-standard-style data. |
| `data/training_data/training_2.csv` | 40,482 | 4 | Full processed training-style data with disease vectors, misinformation, and sentiment. |

`training_2.csv` currently has all misinformation and sentiment values set to `0`, so it should not be treated as a validated source for those tasks without review. The most common disease vectors in `training_2.csv` are:

| Disease Vector | Rows |
|---|---:|
| `[1,0,0,0]` | 10,980 |
| `[0,0,0,0]` | 10,471 |
| `[1,1,1,1]` | 7,335 |
| `[1,0,0,1]` | 2,575 |
| `[1,0,1,1]` | 2,036 |

The high count of `[1,1,1,1]` labels should be checked, because broad all-positive multilabel assignments can inflate training noise if they came from keyword matching rather than human validation.

## Annotation Workflow

The main annotation guide is `docs/cleaned_text_annotation_guide.md`. It defines:

- Disease label order: `[AURI, PN, TB, COVID]`
- Misinformation: binary `0` or `1`
- Sentiment: binary `0` or `1`, where `1` means clear negative or distressed sentiment
- Annotation should use only `cleaned_text`, not raw text

The helper module `healthph_plus/annotation_ready_csv.py` generates annotation-ready CSV files from `data/processed/merged_data.csv`. By default it writes `cleaned_text`, `disease`, `misinformation`, and `sentiment`, skips empty text, and leaves labels blank unless `--fill-default-labels` is supplied.

## Keyword Resources

Keyword resources are stored under `docs/keywords/` and include disease-specific files plus language-specific files:

| File | Rows | Notes |
|---|---:|---|
| `docs/keywords/keywords_master.csv` | 4 | Disease-to-symptoms master list. |
| `docs/keywords/by_language/english/english_keywords.csv` | 63 | English keywords. |
| `docs/keywords/by_language/filipino_keywords.csv` | 71 | Filipino keywords. |
| `docs/keywords/by_language/cebuano_keywords.csv` | 100 | Cebuano keywords. |
| `docs/keywords/by_language/ilocano_keywords.csv` | 75 | Ilocano keywords. |
| `docs/keywords/by_language/hiligaynon_keywords.csv` | 167 | Hiligaynon keywords. |

These assets support low-resource language coverage, but the current processed dataset is still dominated by Tagalog and English.

## Model and Experiment State

### Classic Multilabel Models

Classic multilabel reports are stored in `reports/classic_multilabel/`. The summary file compares Binary Relevance and Classifier Chains using SGD and Logistic Regression, with Optuna trials and tuned thresholds.

| Model | Validation Micro-F1 | Test Micro-F1 | Test Exact Match |
|---|---:|---:|---:|
| `ClassifierChains_SGD` | 0.714 | 0.539 | 0.415 |
| `BinaryRelevance_SGD` | 0.726 | 0.534 | 0.396 |
| `BinaryRelevance_LogReg` | 0.727 | 0.523 | 0.377 |
| `ClassifierChains_LogReg` | 0.715 | 0.520 | 0.420 |

The best test micro-F1 is `ClassifierChains_SGD`. The best exact match ratio in the table is `ClassifierChains_LogReg` at `0.420`, though its micro-F1 is slightly lower.

### Transformer Proof of Concept

Transformer artifacts and reports are available for an ELECTRA model:

- Base model: `google/electra-small-discriminator`
- Selected model key: `electra_local_finetuned`
- Artifact directory: `artifacts/poc_transformers/electra_local_finetuned/`
- Report directory: `reports/poc_transformers/`
- Label order: `AURI`, `PN`, `TB`, `COVID`

Reported aggregate performance:

| Split | Micro-F1 | Macro-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|
| Validation | 0.633 | 0.625 | 0.636 | 0.203 |
| Test | 0.459 | 0.455 | 0.374 | 0.281 |

Test per-label F1:

| Label | Test F1 |
|---|---:|
| AURI | 0.564 |
| PN | 0.444 |
| TB | 0.383 |
| COVID | 0.428 |

The transformer run currently underperforms the classic baselines on test micro-F1. The validation-to-test drop should be investigated before further deployment work.

## Database and Product Schema

`docs/data schema.md` defines a simple PostgreSQL-oriented schema with:

- Main `posts` table.
- Analysis tables for disease labels, symptoms, sentiment, and misinformation.
- Optional `daily_disease_metrics` aggregate table.
- Privacy rules and recommended indexes.

The schema document notes that `data/processed/merged_data.csv` still uses a legacy column shape and has not yet been regenerated for schema version 2.2.

## Notebooks and Scripts

Current notebook areas:

- `notebooks/eda.ipynb` for exploratory analysis.
- `notebooks/preproccessor.ipynb` for preprocessing work.
- `notebooks/01_electra_model.ipynb` for transformer modeling.
- `notebooks/02_classic_multilabel_classifiers.ipynb` for classic multilabel experiments.
- `notebooks/03_keyword_disease_annotator.ipynb` for keyword-based labeling.
- `notebooks/05_binary_sentiment_mbert.ipynb` for sentiment modeling.
- `scrapers/` notebooks for Reddit, Twitter/X, Facebook, TikTok, and Threads collection.

Reusable Python package code is currently limited. The main package file with substantial logic is `healthph_plus/annotation_ready_csv.py`.

## Dependencies

`requirements.txt` includes data science, modeling, and notebook dependencies such as:

- `pandas`, `numpy`, `scikit-learn`, `scipy`
- `torch`, `transformers`, `accelerate`, `safetensors`
- `optuna`
- `fasttext`
- `SQLAlchemy`, `alembic`
- `matplotlib`, `seaborn`
- `ipython`, `ipykernel`, `jupyter_client`

README setup instructions currently say `pip install requirements.txt`; the usual command should be `pip install -r requirements.txt`.

## Key Risks and Gaps

- The project is not yet an application implementation. FastAPI, React, MongoDB/PostgreSQL integration, dashboard, and mobile app features are planned but not present as production code.
- `merged_data.csv` is marked as legacy relative to schema version 2.2.
- `annotation_ready.csv` has blank labels and appears ready for manual annotation, not model training.
- `training_2.csv` has all misinformation and sentiment labels set to `0`, so those columns likely represent placeholders or weak labels.
- Hiligaynon keyword coverage exists, but processed data currently has no visible Hiligaynon rows.
- Ilocano processed coverage is very small at 62 rows.
- Transformer validation performance does not transfer well to test performance.
- Multiple important notebooks and data files have uncommitted local modifications.

## Recommended Next Steps

1. Decide which dataset is the source of truth for modeling: `gold_standard.csv`, `training_2.csv`, or a newly validated annotation export.
2. Audit keyword-generated disease labels, especially all-positive `[1,1,1,1]` rows.
3. Complete or sample-review `annotation_ready.csv` before using it as supervised training data.
4. Regenerate `merged_data.csv` or add an export that conforms to schema version 2.2.
5. Improve coverage for Ilocano and Hiligaynon data collection.
6. Compare classic and transformer models on the same validated train/validation/test split.
7. Move repeated notebook logic into package modules under `healthph_plus/`.
8. Fix README setup instructions to use `pip install -r requirements.txt`.
