# HealthPH+ Data Schema (Simple)
Version: 2.2
Database: PostgreSQL

---

# 1. Goal

Use one main `posts` table for source data.
Store NLP/model outputs in separate analysis tables.

---

# 2. Main Table: `posts`

| Field | Type | Notes |
|---|---|---|
| id | UUID (PK) | Internal record ID |
| source | VARCHAR(50) NOT NULL | reddit, twitter, facebook, tiktok, threads |
| external_post_id | VARCHAR NOT NULL | Platform post ID |
| text | TEXT NOT NULL | Raw post content |
| cleaned_text | TEXT | Cleaned text for NLP |
| language | VARCHAR(20) | tl, en, ceb, ilo, hil, etc. |
| date_posted | TIMESTAMPTZ | Original post time |
| date_collected | TIMESTAMPTZ | Ingestion time |
| view_count | BIGINT | Nullable view/impression count |
| like_count | BIGINT | Nullable like/reaction count |
| share_count | BIGINT | Nullable share/repost count |
| comment_count | BIGINT | Nullable comment/reply count |
| created_at | TIMESTAMPTZ NOT NULL DEFAULT NOW() | Record creation time |
| updated_at | TIMESTAMPTZ NOT NULL DEFAULT NOW() | Record update time |

Constraints:

- `UNIQUE (source, external_post_id)` to prevent duplicate ingestion.

---

# 3. Analysis Table: `post_disease_labels`

| Field | Type | Notes |
|---|---|---|
| id | UUID (PK) | Row ID |
| post_id | UUID NOT NULL | FK to `posts.id` |
| disease_label | VARCHAR(50) NOT NULL | COVID-19, TB, Pneumonia, AURI |
| confidence_score | FLOAT | 0.0 to 1.0 |
| is_human_validated | BOOLEAN DEFAULT FALSE | Manual QA flag |
| created_at | TIMESTAMPTZ NOT NULL DEFAULT NOW() | Creation time |

---

# 4. Analysis Table: `post_symptoms`

| Field | Type | Notes |
|---|---|---|
| id | UUID (PK) | Row ID |
| post_id | UUID NOT NULL | FK to `posts.id` |
| symptom | VARCHAR(100) NOT NULL | e.g. ubo, sipon, lagnat |
| start_index | INT | Character start offset |
| end_index | INT | Character end offset |
| extraction_confidence | FLOAT | 0.0 to 1.0 |
| created_at | TIMESTAMPTZ NOT NULL DEFAULT NOW() | Creation time |

---

# 5. Analysis Table: `post_sentiment`

| Field | Type | Notes |
|---|---|---|
| id | UUID (PK) | Row ID |
| post_id | UUID NOT NULL | FK to `posts.id` |
| sentiment_label | VARCHAR(20) | Positive, Negative, Neutral |
| sentiment_score | FLOAT | Polarity score |
| emotion_label | VARCHAR(50) | Fear, Anger, Trust, etc. |
| created_at | TIMESTAMPTZ NOT NULL DEFAULT NOW() | Creation time |

Constraint:

- `UNIQUE (post_id)` if only one sentiment result per post.

---

# 6. Analysis Table: `post_misinformation`

| Field | Type | Notes |
|---|---|---|
| id | UUID (PK) | Row ID |
| post_id | UUID NOT NULL | FK to `posts.id` |
| is_misinformation | BOOLEAN | True if flagged |
| misinformation_category | VARCHAR(100) | vaccine myth, fake cure, conspiracy |
| confidence_score | FLOAT | 0.0 to 1.0 |
| flagged_keywords | JSONB | Extracted evidence keywords |
| requires_review | BOOLEAN DEFAULT FALSE | Manual review queue |
| created_at | TIMESTAMPTZ NOT NULL DEFAULT NOW() | Creation time |

Constraint:

- `UNIQUE (post_id)` if only one misinformation result per post.

---

# 7. Optional Aggregate Table: `daily_disease_metrics`

Use only if you need fast dashboards/forecast features.

| Field | Type | Notes |
|---|---|---|
| id | UUID (PK) | Aggregate row ID |
| date | DATE NOT NULL | Metric date |
| disease_label | VARCHAR(50) NOT NULL | Target disease |
| post_count | INT NOT NULL | Number of posts |
| avg_sentiment_score | FLOAT | Average sentiment |
| misinformation_count | INT | Count of flagged posts |

Constraint:

- `UNIQUE (date, disease_label)`

---

# 8. Privacy Rules

- Store anonymized user IDs only.
- Do not store direct PII.
- Restrict access by role.
- Log data-access events at the application layer.

---

# 9. Recommended Indexes

- `posts(id)` already has an index because it is the primary key.
- `posts(date_posted)` for time-based queries.
- `post_disease_labels(post_id)` and `post_disease_labels(disease_label)` for disease drilldowns.
- `post_symptoms(post_id)` and `post_symptoms(symptom)` for symptom lookups.
- `post_sentiment(post_id)` for joins (also unique).
- `post_misinformation(post_id)` and `post_misinformation(is_misinformation)` for flagged-post filters.

---

# 10. Compatibility Note

- `data/processed/merged_data.csv` still uses the legacy column shape and has not yet been regenerated for schema version 2.2.
