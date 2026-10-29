"""Prepare sampled social posts and aggregate locally predicted disease signals.

These counts describe the supplied corpus, rather than disease incidence or a
complete platform feed. A calendar day without eligible posts remains missing.
The predictor is injected so preparation and cache checks do not import torch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

import pandas as pd


LABELS = ("AURI", "PN", "TB", "COVID")
_REQUIRED = ("source", "external_post_id", "date_posted", "cleaned_text")
_NORMALIZATION = "collapse-whitespace-preserve-case-unicode-v1"


def _normalize_text(text):
    return " ".join(text.split())


def _bounds(start_date, end_date, timezone):
    # Resolve the zone even when the two bounds have no timezone information.
    pd.Timestamp("2000-01-01", tz=timezone)
    bounds = []
    for value in (start_date, end_date):
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("start_date and end_date must be valid dates")
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(timezone).tz_localize(None)
        bounds.append(stamp.normalize())
    if bounds[0] > bounds[1]:
        raise ValueError("start_date must be on or before end_date")
    return bounds[0], bounds[1]


def _missing_spans(days):
    missing = days[days.eq(0)]
    if missing.empty:
        return []
    groups = missing.index.to_series().diff().ne(pd.Timedelta(days=1)).cumsum()
    return [
        {"start_date": str(group.index[0].date()),
         "end_date": str(group.index[-1].date()), "days": len(group)}
        for _, group in missing.groupby(groups.to_numpy())
    ]


def _coverage(posts, *, start=None, end=None):
    if start is None and not posts.empty:
        start, end = posts["date"].min(), posts["date"].max()
    if start is None:
        calendar = pd.DatetimeIndex([])
    else:
        calendar = pd.date_range(start, end, freq="D")
    counts = posts.groupby("date").size().reindex(calendar, fill_value=0)
    source_ranges = {}
    for source, group in posts.groupby("source", sort=True):
        source_ranges[source] = {
            "posts": len(group), "first_date": str(group.date.min().date()),
            "last_date": str(group.date.max().date()),
            "observed_days": int(group.date.nunique()),
        }
    monthly = posts.assign(month=posts["date"].dt.strftime("%Y-%m"))
    monthly_counts = [
        {"month": month, "post_count": len(group),
         "observed_days": int(group.date.nunique())}
        for month, group in monthly.groupby("month", sort=True)
    ]
    monthly_sources = [
        {"month": month, "source": source, "post_count": int(count)}
        for (month, source), count in monthly.groupby(["month", "source"]).size().items()
    ]
    return {
        "posts": len(posts), "calendar_days": len(calendar),
        "first_date": str(start.date()) if start is not None else None,
        "last_date": str(end.date()) if end is not None else None,
        "observed_days": int(counts.gt(0).sum()),
        "unobserved_days": int(counts.eq(0).sum()),
        "unobserved_spans": _missing_spans(counts),
        "source_counts": {str(k): int(v) for k, v in posts.source.value_counts().items()},
        "source_date_coverage": source_ranges,
        "monthly_counts": monthly_counts, "monthly_source_counts": monthly_sources,
    }


def prepare_posts(frame, *, start_date, end_date, timezone="Asia/Manila",
                  max_text_characters=10_000):
    """Return eligible posts, text-free exclusions, and a JSON-compatible audit.

    Naive source timestamps are interpreted as UTC, matching the source export.
    Dates are timezone-naive local calendar midnights. Durable duplicate keys
    with differing post times or normalized text raise instead of being counted.
    Other columns, including engagement snapshots and internal UUIDs, are kept
    from the first duplicate and do not define a different forecasting record.
    """
    missing = set(_REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required post columns: {', '.join(sorted(missing))}")
    if type(max_text_characters) is not int or max_text_characters < 1:
        raise ValueError("max_text_characters must be a positive integer")
    start, end = _bounds(start_date, end_date, timezone)
    work = frame.copy().reset_index(drop=True)
    work["source"] = work.source.astype(object)
    work["external_post_id"] = work.external_post_id.astype(object)
    times = pd.to_datetime(work.date_posted, format="mixed", errors="coerce", utc=True)
    local_dates = times.dt.tz_convert(timezone).dt.tz_localize(None).dt.normalize()
    selected_input = int(local_dates.between(start, end).sum())
    exclusion_rows, valid_indices, seen = [], [], {}

    def exclude(index, reason):
        source = work.at[index, "source"]
        external = work.at[index, "external_post_id"]
        exclusion_rows.append({
            "row_number": int(index) + 1,
            "source": str(source) if pd.notna(source) else None,
            "external_post_id": str(external) if pd.notna(external) else None,
            "reason": reason,
        })

    for index, row in work.iterrows():
        source, external, text = row.source, row.external_post_id, row.cleaned_text
        if not isinstance(source, str) or not source.strip():
            exclude(index, "missing_source")
            continue
        if pd.isna(external) or isinstance(external, bool) or not str(external).strip():
            exclude(index, "missing_external_post_id")
            continue
        source, external = source.strip().lower(), str(external).strip()
        work.at[index, "source"], work.at[index, "external_post_id"] = source, external
        if pd.isna(times.iloc[index]):
            exclude(index, "invalid_date")
            continue
        if not isinstance(text, str):
            exclude(index, "nonstring_text")
            continue
        if len(text) > max_text_characters:
            exclude(index, "text_too_long")
            continue
        text = _normalize_text(text)
        if not text:
            exclude(index, "blank_text")
            continue
        work.at[index, "cleaned_text"] = text
        key = source, external
        signature = times.iloc[index], text
        if key in seen:
            if seen[key] != signature:
                raise ValueError(f"Conflicting duplicate post identity at row {index + 1}")
            exclude(index, "duplicate_identity")
            continue
        seen[key] = signature
        valid_indices.append(index)

    work["date"] = local_dates
    valid = work.loc[valid_indices].copy()
    in_window = valid.date.between(start, end)
    for index in valid.index[~in_window]:
        exclude(index, "outside_window")
    eligible = valid.loc[in_window].reset_index(drop=True)
    exclusions = pd.DataFrame(exclusion_rows, columns=[
        "row_number", "source", "external_post_id", "reason"
    ]).sort_values("row_number", ignore_index=True)
    reasons = {str(k): int(v) for k, v in exclusions.reason.value_counts().items()}
    selected_rows = set(local_dates.index[local_dates.between(start, end)] + 1)
    selected_exclusions = exclusions.loc[exclusions.row_number.isin(selected_rows)]
    selected_reasons = {str(k): int(v) for k, v in selected_exclusions.reason.value_counts().items()}
    audit = {
        "input_posts": len(frame), "selected_input_posts": selected_input,
        "selected_excluded_posts": len(selected_exclusions),
        "selected_exclusion_counts": selected_reasons,
        "deduplicated_posts": len(valid),
        "duplicate_posts": reasons.get("duplicate_identity", 0),
        "excluded_posts": len(exclusions), "eligible_posts": len(eligible),
        "exclusion_counts": reasons,
        "window": {"start_date": str(start.date()), "end_date": str(end.date()),
                   "timezone": timezone},
        "text_normalization": _NORMALIZATION,
        "full_corpus": _coverage(valid),
        "selected_window": _coverage(eligible, start=start, end=end),
    }
    return eligible, exclusions, audit


def _scores(values, name):
    if not isinstance(values, Mapping) or set(values) != set(LABELS):
        raise ValueError(f"{name} must contain exactly {', '.join(LABELS)}")
    clean = {}
    for label in LABELS:
        value = values[label]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"Invalid {name} score for {label}")
        clean[label] = float(value)
    return clean


def _validate_result(result, model_version, thresholds):
    required = {"labels", "probabilities", "thresholds_used", "model_version"}
    if not isinstance(result, Mapping) or set(result) != required:
        raise ValueError("Prediction must contain labels, probabilities, thresholds_used, model_version")
    if result["model_version"] != model_version:
        raise ValueError("Prediction model_version does not match predictor")
    probabilities = _scores(result["probabilities"], "probabilities")
    used = _scores(result["thresholds_used"], "thresholds_used")
    if used != thresholds:
        raise ValueError("Prediction/cache thresholds do not match predictor thresholds")
    labels = result["labels"]
    if (not isinstance(labels, list) or any(not isinstance(x, str) for x in labels)
            or len(set(labels)) != len(labels) or not set(labels).issubset(LABELS)):
        raise ValueError("Prediction labels must be a unique list of disease labels")
    expected = [label for label in LABELS if probabilities[label] >= used[label]]
    if set(labels) != set(expected):
        raise ValueError("Prediction labels are inconsistent with probabilities and thresholds")
    return {"labels": expected, "probabilities": probabilities,
            "thresholds_used": used, "model_version": model_version}


def _cache_key(model_version, text_hash):
    return hashlib.sha256(json.dumps([model_version, text_hash]).encode()).hexdigest()


def _save_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(cache, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def label_posts(posts, predictor, *, cache_path, batch_size=32, progress=None):
    """Predict unique normalized texts with a resumable, text-free JSON cache.

    The predictor supplies ``predict_many(texts)``, ``model_version`` and the
    saved ``thresholds`` mapping. Any malformed or failed prediction halts the
    call. Each complete valid batch is committed atomically before continuing.
    ``progress(completed, total)`` counts unique normalized texts, including hits.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("batch_size must be an integer between 1 and 32")
    if "cleaned_text" not in posts:
        raise ValueError("posts must contain cleaned_text")
    version = getattr(predictor, "model_version", None)
    if not isinstance(version, str) or not version.strip():
        raise ValueError("predictor.model_version must be a non-empty string")
    thresholds = _scores(getattr(predictor, "thresholds", None), "predictor.thresholds")
    texts = []
    for text in posts.cleaned_text:
        if not isinstance(text, str) or not text.strip() or len(text) > 10_000:
            raise ValueError("label_posts requires valid non-empty texts of at most 10000 characters")
        texts.append(_normalize_text(text))
    unique = list(dict.fromkeys(texts))
    hashes = {text: hashlib.sha256(text.encode("utf-8")).hexdigest() for text in unique}
    path = Path(cache_path)
    cache = {"schema_version": 1, "text_normalization": _NORMALIZATION, "entries": {}}
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError("Cannot read prediction cache") from exc
        if (not isinstance(cache, dict) or cache.get("schema_version") != 1
                or cache.get("text_normalization") != _NORMALIZATION
                or not isinstance(cache.get("entries"), dict)):
            raise ValueError("Prediction cache has an incompatible schema")
    results, pending = {}, []
    for text in unique:
        digest = hashes[text]
        entry = cache["entries"].get(_cache_key(version, digest))
        if entry is None:
            pending.append(text)
            continue
        if (not isinstance(entry, dict) or set(entry) != {"text_hash", "result"}
                or entry["text_hash"] != digest):
            raise ValueError("Prediction cache entry has an invalid text hash")
        results[text] = _validate_result(entry["result"], version, thresholds)
    if progress is not None:
        progress(len(results), len(unique))
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        predictions = predictor.predict_many(batch)
        if not isinstance(predictions, (list, tuple)) or len(predictions) != len(batch):
            raise ValueError("Predictor returned the wrong number of predictions")
        validated = [_validate_result(result, version, thresholds) for result in predictions]
        # Validate the entire batch before changing either the cache or results.
        for text, result in zip(batch, validated):
            digest = hashes[text]
            cache["entries"][_cache_key(version, digest)] = {"text_hash": digest, "result": result}
            results[text] = result
        _save_cache(path, cache)
        if progress is not None:
            progress(len(results), len(unique))
    labeled = posts.copy()
    labeled["text_hash"] = [hashes[text] for text in texts]
    labeled["model_version"] = version
    for label in LABELS:
        labeled[label] = pd.Series(
            [int(label in results[text]["labels"]) for text in texts],
            index=labeled.index, dtype="int64")
        labeled[f"probability_{label}"] = pd.Series(
            [results[text]["probabilities"][label] for text in texts],
            index=labeled.index, dtype="float64")
    return labeled


def aggregate_daily(labeled_posts, *, start_date, end_date, timezone="Asia/Manila"):
    """Return the full daily label calendar and observed source-count calendar.

    ``post_count`` uses nullable integers: missing means there were no eligible
    posts on that date; zero means eligible posts existed but none had the label.
    Source counts are zero on an observed day where that source had no posts,
    and missing on a completely unobserved day.
    """
    required = {"source", *LABELS}
    if not required.issubset(labeled_posts.columns):
        raise ValueError("labeled_posts must contain source and four binary disease columns")
    start, end = _bounds(start_date, end_date, timezone)
    work = labeled_posts.copy()
    if "date_posted" in work:
        posted = pd.to_datetime(work.date_posted, utc=True, format="mixed", errors="coerce")
        work["date"] = posted.dt.tz_convert(timezone).dt.tz_localize(None).dt.normalize()
    elif "date" in work:
        work["date"] = pd.to_datetime(work.date, errors="coerce").dt.normalize()
    else:
        raise ValueError("labeled_posts must contain date_posted or local date")
    if work.date.isna().any():
        raise ValueError("labeled_posts contains invalid dates")
    for label in LABELS:
        if not pd.api.types.is_numeric_dtype(work[label]) or not work[label].isin([0, 1]).all():
            raise ValueError(f"{label} must contain numeric binary labels without missing values")
    if work.source.isna().any() or work.source.map(lambda x: not isinstance(x, str) or not x.strip()).any():
        raise ValueError("labeled_posts contains invalid sources")
    work = work.loc[work.date.between(start, end)]
    calendar = pd.date_range(start, end, freq="D", name="date")
    total = work.groupby("date").size().reindex(calendar, fill_value=0)
    sums = work.groupby("date")[list(LABELS)].sum().reindex(calendar, fill_value=0)
    sums.loc[total.eq(0)] = pd.NA
    series = (sums.reset_index().melt(id_vars="date", var_name="disease_label", value_name="post_count"))
    series["post_count"] = series.post_count.astype("Int64")
    series["eligible_post_count"] = series.date.map(total).astype("int64")
    series = series.sort_values(["date", "disease_label"], key=lambda col:
        col.map({label: i for i, label in enumerate(LABELS)}) if col.name == "disease_label" else col,
        ignore_index=True)
    sources = sorted(work.source.unique())
    source_index = pd.MultiIndex.from_product([calendar, sources], names=["date", "source"])
    source_counts = work.groupby(["date", "source"]).size().reindex(source_index, fill_value=0)
    daily_sources = source_counts.rename("post_count").reset_index()
    daily_sources["post_count"] = daily_sources.post_count.astype("Int64")
    unobserved = daily_sources.date.map(total).eq(0)
    daily_sources.loc[unobserved, "post_count"] = pd.NA
    return series, daily_sources
