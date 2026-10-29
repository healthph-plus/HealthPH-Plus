"""Temporal preparation, fail-closed inference caching and count semantics."""

import copy
import json

import pandas as pd
import pytest

from healthph_plus.forecast_data import LABELS, aggregate_daily, label_posts, prepare_posts


def posts_frame(rows=None):
    if rows is None:
        rows = [
            ("twitter", "one", "2025-08-31T16:00:00Z", "  May\tubo  "),
            ("reddit", "two", "2025-09-02T01:00:00Z", "Nothing to report"),
        ]
    return pd.DataFrame(rows, columns=["source", "external_post_id", "date_posted", "cleaned_text"])


def prepared(frame=None, **kwargs):
    return prepare_posts(posts_frame() if frame is None else frame,
                         start_date="2025-09-01", end_date="2025-09-03", **kwargs)


class Predictor:
    model_version = "sha256:test-model"
    thresholds = {"AURI": .2, "PN": .9, "TB": .1, "COVID": .5}

    def __init__(self):
        self.calls = []

    def predict_many(self, texts):
        self.calls.append(list(texts))
        return [self.result() for _ in texts]

    def result(self):
        # Multiple independently thresholded positives, including exact equality.
        return {"labels": ["AURI", "TB", "COVID"],
                "probabilities": {"AURI": .2, "PN": .8, "TB": .1, "COVID": .9},
                "thresholds_used": self.thresholds.copy(), "model_version": self.model_version}


def test_prepare_local_boundaries_normalization_duplicates_and_text_free_audit():
    raw = posts_frame([
        (" Twitter ", "one", "2025-08-31T16:00:00Z", "  May\tubo\nÉ  "),
        ("twitter", "one", "2025-09-01T00:00:00+08:00", "May ubo É"),
        ("reddit", "two", "2025-09-02T01:00:00Z", "No symptoms"),
        ("twitter", "three", "2025-09-03T16:00:00Z", "outside"),
        ("twitter", "four", "2025-08-31T15:59:59Z", "before"),
    ])
    original = raw.copy(deep=True)
    eligible, exclusions, audit = prepared(raw)
    pd.testing.assert_frame_equal(raw, original)
    assert eligible.external_post_id.tolist() == ["one", "two"]
    assert eligible.cleaned_text.tolist() == ["May ubo É", "No symptoms"]
    assert eligible.date.tolist() == [pd.Timestamp("2025-09-01"), pd.Timestamp("2025-09-02")]
    assert eligible.date.dt.tz is None
    assert set(exclusions.columns) == {"row_number", "source", "external_post_id", "reason"}
    assert exclusions.reason.tolist() == ["duplicate_identity", "outside_window", "outside_window"]
    assert audit["input_posts"] == 5
    assert audit["selected_input_posts"] == 3
    assert audit["selected_excluded_posts"] == 1
    assert audit["selected_exclusion_counts"] == {"duplicate_identity": 1}
    assert audit["deduplicated_posts"] == 4
    assert audit["duplicate_posts"] == 1
    assert audit["excluded_posts"] == 3
    assert audit["eligible_posts"] == 2
    assert audit["selected_window"]["unobserved_spans"] == [
        {"start_date": "2025-09-03", "end_date": "2025-09-03", "days": 1}]
    assert audit["full_corpus"]["source_counts"] == {"twitter": 3, "reddit": 1}
    assert audit["full_corpus"]["monthly_source_counts"] == [
        {"month": "2025-08", "source": "twitter", "post_count": 1},
        {"month": "2025-09", "source": "reddit", "post_count": 1},
        {"month": "2025-09", "source": "twitter", "post_count": 2}]
    json.dumps(audit, allow_nan=False)
    assert "May" not in exclusions.to_json()


def test_prepare_quality_exclusions_and_naive_utc_dates():
    raw = posts_frame([
        (None, "1", "2025-09-01", "valid"),
        ("reddit", None, "2025-09-01", "valid"),
        ("reddit", "3", "bad date", "valid"),
        ("reddit", "4", "2025-09-01", 123),
        ("reddit", "5", "2025-09-01", " \t "),
        ("reddit", "6", "2025-09-01", "abcdef"),
        ("reddit", "7", "2025-08-31 18:00:00", "okay"),
    ])
    eligible, exclusions, audit = prepared(raw, max_text_characters=5)
    assert eligible.external_post_id.tolist() == ["7"]
    assert eligible.date.iloc[0] == pd.Timestamp("2025-09-01")
    assert exclusions.reason.tolist() == ["missing_source", "missing_external_post_id", "invalid_date",
                                        "nonstring_text", "blank_text", "text_too_long"]
    assert audit["eligible_posts"] + audit["excluded_posts"] == len(raw)
    assert audit["selected_input_posts"] == 6
    assert audit["selected_excluded_posts"] == 5
    assert "invalid_date" not in audit["selected_exclusion_counts"]


def test_original_text_limit_excludes_whitespace_heavy_post_before_normalization(tmp_path):
    raw = posts_frame([("reddit", "large", "2025-09-01", "a" + " " * 10_000)])
    eligible, exclusions, _ = prepared(raw)
    assert eligible.empty
    assert exclusions.reason.tolist() == ["text_too_long"]
    predictor = Predictor()
    with pytest.raises(ValueError, match="at most 10000"):
        label_posts(raw, predictor, cache_path=tmp_path / "cache.json")
    assert predictor.calls == []


@pytest.mark.parametrize("changed", ["text", "date"])
def test_conflicting_durable_identity_halts(changed):
    rows = [("twitter", "one", "2025-09-01", "post"),
            ("twitter", "one", "2025-09-01", "post")]
    rows[1] = ("twitter", "one", "2025-09-02" if changed == "date" else "2025-09-01",
               "changed" if changed == "text" else "post")
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        prepared(posts_frame(rows))


def test_metadata_refresh_does_not_duplicate_post_and_cross_source_ids_are_distinct():
    raw = posts_frame([
        ("twitter", "one", "2025-09-01", "post"),
        ("twitter", "one", "2025-09-01", "post"),
        ("reddit", "one", "2025-09-01", "post"),
    ]).assign(id=["new-uuid", "old-uuid", "other-uuid"], like_count=[3, 2, 1])
    eligible, _, audit = prepared(raw)
    assert eligible.id.tolist() == ["new-uuid", "other-uuid"]
    assert audit["duplicate_posts"] == 1


def test_prepare_empty_and_input_errors():
    eligible, exclusions, audit = prepared(posts_frame([]))
    assert eligible.empty and exclusions.empty
    assert audit["full_corpus"]["first_date"] is None
    assert audit["selected_window"]["unobserved_days"] == 3
    json.dumps(audit, allow_nan=False)
    with pytest.raises(ValueError, match="Missing required"):
        prepared(posts_frame().drop(columns="date_posted"))
    with pytest.raises(ValueError, match="max_text_characters"):
        prepared(max_text_characters=0)
    with pytest.raises(ValueError, match="start_date"):
        prepare_posts(posts_frame(), start_date="2025-10-01", end_date="2025-09-01")


def test_independent_labels_unique_text_cache_and_progress(tmp_path):
    posts = posts_frame([
        ("twitter", "one", "2025-09-01", "  May\tubo "),
        ("reddit", "two", "2025-09-02", "May ubo"),
        ("reddit", "three", "2025-09-02", "Other text"),
    ])
    predictor, progress = Predictor(), []
    path = tmp_path / "cache.json"
    labeled = label_posts(posts, predictor, cache_path=path, batch_size=1,
                          progress=lambda completed, total: progress.append((completed, total)))
    assert predictor.calls == [["May ubo"], ["Other text"]]
    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert labeled[list(LABELS)].values.tolist() == [[1, 0, 1, 1]] * 3
    assert labeled.probability_PN.tolist() == [.8] * 3
    assert labeled.text_hash.iloc[0] == labeled.text_hash.iloc[1]
    assert labeled.text_hash.iloc[1] != labeled.text_hash.iloc[2]
    pd.testing.assert_frame_equal(labeled[posts.columns], posts)
    cached = json.loads(path.read_text())
    assert len(cached["entries"]) == 2
    assert "May ubo" not in path.read_text() and "Other text" not in path.read_text()
    predictor.calls.clear()
    progress.clear()
    again = label_posts(posts, predictor, cache_path=path,
                        progress=lambda completed, total: progress.append((completed, total)))
    assert predictor.calls == [] and progress == [(2, 2)]
    pd.testing.assert_frame_equal(again, labeled)


def test_model_text_cache_invalidation_and_same_version_threshold_change_halts(tmp_path):
    posts, _, _ = prepared()
    predictor, path = Predictor(), tmp_path / "cache.json"
    label_posts(posts, predictor, cache_path=path)
    changed = posts.copy()
    changed.loc[0, "cleaned_text"] = "Changed post"
    predictor.calls.clear()
    label_posts(changed, predictor, cache_path=path)
    assert predictor.calls == [["Changed post"]]
    predictor.model_version = "new-model"
    predictor.calls.clear()
    label_posts(changed, predictor, cache_path=path)
    assert predictor.calls == [["Changed post", "Nothing to report"]]
    predictor.thresholds = {**predictor.thresholds, "AURI": .3}
    predictor.calls.clear()
    with pytest.raises(ValueError, match="thresholds do not match"):
        label_posts(changed, predictor, cache_path=path)
    assert predictor.calls == []


def test_successful_batches_survive_later_inference_failure_and_resume(tmp_path):
    posts, _, _ = prepared()
    predictor, path = Predictor(), tmp_path / "cache.json"
    original = predictor.predict_many

    def fail_second(texts):
        if len(predictor.calls) == 1:
            raise RuntimeError("offline model failure")
        return original(texts)

    predictor.predict_many = fail_second
    with pytest.raises(RuntimeError, match="offline model failure"):
        label_posts(posts, predictor, cache_path=path, batch_size=1)
    assert len(json.loads(path.read_text())["entries"]) == 1
    predictor.predict_many = original
    predictor.calls.clear()
    progress = []
    labeled = label_posts(posts, predictor, cache_path=path, batch_size=1,
                          progress=lambda n, total: progress.append((n, total)))
    assert len(labeled) == 2
    assert predictor.calls == [["Nothing to report"]]
    assert progress == [(1, 2), (2, 2)]


def malformed_predictions():
    result = Predictor().result()
    yield "missing response key", {key: value for key, value in result.items() if key != "labels"}
    yield "wrong version", {**result, "model_version": "wrong"}
    yield "wrong probability keys", {**result, "probabilities": {"AURI": .2}}
    yield "nan score", {**result, "probabilities": {**result["probabilities"], "PN": float("nan")}}
    yield "out of range", {**result, "probabilities": {**result["probabilities"], "PN": 1.2}}
    yield "boolean score", {**result, "probabilities": {**result["probabilities"], "PN": True}}
    yield "wrong thresholds", {**result, "thresholds_used": {**result["thresholds_used"], "PN": .5}}
    yield "missing threshold", {**result, "thresholds_used": {"AURI": .2}}
    yield "inconsistent labels", {**result, "labels": ["PN"]}
    yield "unknown label", {**result, "labels": ["AURI", "TB", "COVID", "unknown"]}
    yield "duplicate labels", {**result, "labels": ["AURI", "TB", "COVID", "COVID"]}


@pytest.mark.parametrize("description,bad", list(malformed_predictions()), ids=lambda x: str(x)[:25])
def test_malformed_prediction_halts_without_partial_batch_cache(tmp_path, description, bad):
    posts, _, _ = prepared()
    predictor, path = Predictor(), tmp_path / "cache.json"
    predictor.predict_many = lambda texts: [predictor.result(), copy.deepcopy(bad)]
    with pytest.raises(ValueError):
        label_posts(posts, predictor, cache_path=path)
    assert not path.exists(), description


def test_wrong_prediction_count_cache_corruption_and_bad_configuration(tmp_path):
    posts, _, _ = prepared()
    predictor, path = Predictor(), tmp_path / "cache.json"
    predictor.predict_many = lambda texts: []
    with pytest.raises(ValueError, match="wrong number"):
        label_posts(posts, predictor, cache_path=path)
    with pytest.raises(ValueError, match="between 1 and 32"):
        label_posts(posts, predictor, cache_path=path, batch_size=33)
    path.write_text("{incomplete json")
    with pytest.raises(ValueError, match="Cannot read"):
        label_posts(posts, predictor, cache_path=path)
    path.write_text('{"schema_version": 900, "entries": {}}')
    with pytest.raises(ValueError, match="incompatible schema"):
        label_posts(posts, predictor, cache_path=path)


def test_corrupt_cached_prediction_is_validated_before_reuse(tmp_path):
    posts, _, _ = prepared()
    predictor, path = Predictor(), tmp_path / "cache.json"
    label_posts(posts, predictor, cache_path=path)
    cache = json.loads(path.read_text())
    next(iter(cache["entries"].values()))["result"]["labels"] = []
    path.write_text(json.dumps(cache))
    predictor.calls.clear()
    with pytest.raises(ValueError, match="inconsistent"):
        label_posts(posts, predictor, cache_path=path)
    assert predictor.calls == []


def test_empty_labeling_and_batch_limit(tmp_path):
    empty, _, _ = prepared(posts_frame([]))
    predictor = Predictor()
    labeled = label_posts(empty, predictor, cache_path=tmp_path / "empty.json")
    assert labeled.empty and predictor.calls == []
    assert set(LABELS).issubset(labeled.columns)
    many = posts_frame([( "reddit", str(i), "2025-09-01", f"Unique {i}") for i in range(65)])
    label_posts(many, predictor, cache_path=tmp_path / "many.json")
    assert [len(batch) for batch in predictor.calls] == [32, 32, 1]


def test_daily_counts_multilabel_absence_missing_day_and_source_breakdown():
    posts = posts_frame([
        ("twitter", "one", "2025-08-31T16:00:00Z", "one"),
        ("reddit", "two", "2025-09-01T01:00:00Z", "two"),
        ("reddit", "three", "2025-09-03T01:00:00Z", "three"),
    ])
    posts[list(LABELS)] = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]]
    daily, sources = aggregate_daily(posts, start_date="2025-09-01", end_date="2025-09-03")
    assert list(daily.columns) == ["date", "disease_label", "post_count", "eligible_post_count"]
    assert len(daily) == 12
    assert daily.disease_label.tolist() == list(LABELS) * 3
    day1 = daily[daily.date.eq("2025-09-01")].set_index("disease_label")
    assert day1.post_count.to_dict() == {"AURI": 2, "PN": 0, "TB": 0, "COVID": 1}
    assert day1.eligible_post_count.eq(2).all()
    day2 = daily[daily.date.eq("2025-09-02")]
    assert day2.post_count.isna().all() and day2.eligible_post_count.eq(0).all()
    day3 = daily[daily.date.eq("2025-09-03")].set_index("disease_label")
    assert day3.post_count.to_dict() == {"AURI": 0, "PN": 0, "TB": 1, "COVID": 0}
    assert str(daily.post_count.dtype) == "Int64"
    assert len(sources) == 6
    assert sources.loc[sources.date.eq("2025-09-02"), "post_count"].isna().all()
    assert sources.loc[sources.date.eq("2025-09-03") & sources.source.eq("twitter"), "post_count"].item() == 0


def test_aggregation_supports_local_dates_empty_inputs_and_other_timezone():
    posts = posts_frame().assign(AURI=1, PN=0, TB=0, COVID=0)
    daily, _ = aggregate_daily(posts, start_date="2025-08-31", end_date="2025-09-02", timezone="UTC")
    assert daily.loc[daily.date.eq("2025-08-31") & daily.disease_label.eq("AURI"), "post_count"].item() == 1
    local, _, _ = prepared()
    local = local.drop(columns="date_posted").assign(AURI=1, PN=0, TB=0, COVID=0)
    by_local, _ = aggregate_daily(local, start_date="2025-09-01", end_date="2025-09-03")
    assert by_local.loc[by_local.date.eq("2025-09-01"), "eligible_post_count"].eq(1).all()
    daily, sources = aggregate_daily(local.iloc[:0], start_date="2025-09-01", end_date="2025-09-03")
    assert len(daily) == 12 and daily.post_count.isna().all()
    assert daily.eligible_post_count.eq(0).all() and sources.empty


@pytest.mark.parametrize("bad", [pd.NA, float("nan"), 2, "0"])
def test_aggregation_rejects_missing_or_nonbinary_labels(bad):
    posts = posts_frame().assign(AURI=1, PN=0, TB=0, COVID=0)
    posts["AURI"] = pd.Series([bad, 1], dtype="object")
    with pytest.raises(ValueError, match="numeric binary"):
        aggregate_daily(posts, start_date="2025-09-01", end_date="2025-09-03")
