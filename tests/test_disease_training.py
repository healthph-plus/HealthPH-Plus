"""Offline data checks and real tiny-model training through both standalone CLIs."""

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from healthph_disease_training.data import LABELS, load_annotations, parse_labels, split_annotations
from healthph_disease_training.runner import make_parser, preflight, score_predictions, tune_thresholds
from healthph_disease_training.vendor import RUNTIME_FILES

ROOT = Path(__file__).resolve().parents[1]
SERVICES = {"electra": ("disease-detection-service", "healthph_plus"),
            "xlm-roberta": ("xlmr-disease-detection-service", "healthph_xlmr")}


def write_data(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["post", "annotate"])
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("invalid", ["[1,0,0]", "[1,0,0,0,1]", "[True,0,0,0]", "[1.0,0,0,0]",
                                      "(1,0,0,0)", "[1,2,0,0]", "__import__('os')", "", None])
def test_labels_reject_invalid_or_executable_values(invalid):
    with pytest.raises(ValueError):
        parse_labels(invalid)


def test_data_preserves_inference_preprocessing_and_deduplicates_across_files(tmp_path):
    one = write_data(tmp_path / "one.csv", [("  Ubo\n at lagnat 😷  ", "[1,0,0,0]")])
    two = write_data(tmp_path / "two.csv", [("Ubo at lagnat 😷", "[1,0,0,0]"), ("NA", "[0,0,0,0]")])
    for architecture in SERVICES:
        rows, audit = load_annotations([one, two], architecture)
        assert len(rows) == 2 and audit["duplicate_rows_removed"] == 1
        assert rows[1]["text"] == "NA"
        assert rows[0]["text"] == ("Ubo at lagnat 😷" if architecture == "xlm-roberta" else "  Ubo\n at lagnat 😷  ")
        assert rows[0]["source_rows"] == [{"file_index": 0, "csv_row": 2}, {"file_index": 1, "csv_row": 2}]
    write_data(two, [("Ubo at lagnat 😷", "[0,1,0,0]")])
    with pytest.raises(ValueError, match="conflicting annotations"):
        load_annotations([one, two], "electra")


@pytest.mark.parametrize("rows, message", [([], "empty"), ([(" ", "[0,0,0,0]")], "blank")])
def test_invalid_posts_fail_early(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        load_annotations([write_data(tmp_path / "data.csv", rows)], "electra")


def test_long_training_posts_are_retained_and_audited(tmp_path):
    path = write_data(tmp_path / "long.csv", [("x" * 10001, "[1,0,0,0]")])
    rows, audit = load_annotations([path], "electra")
    assert len(rows[0]["text"]) == 10001
    assert audit["retained_posts_over_service_character_limit"] == 1


def test_exclude_conflicts_removes_entire_groups_and_audits_source_rows(tmp_path):
    one = write_data(tmp_path / "one.csv", [("same post", "[1,0,0,0]"), ("same  post", "[1,0,0,0]"),
                                           ("keep", "[0,0,0,0]")])
    two = write_data(tmp_path / "two.csv", [(" same post ", "[0,1,0,0]"), ("same post", "[1,0,0,0]"),
                                           ("keep", "[0,0,0,0]")])
    rows, audit = load_annotations([one, two], "xlm-roberta", "exclude")
    assert [row["text"] for row in rows] == ["keep"]
    assert audit["conflicting_posts"] == 1 and audit["conflicting_rows_excluded"] == 4
    assert len(audit["excluded_conflicts"][0]["source_rows"]) == 4
    assert audit["duplicate_rows_removed"] == 1
    assert audit["raw_rows"] == audit["retained_posts"] + audit["duplicate_rows_removed"] + audit["conflicting_rows_excluded"]


@pytest.fixture
def dataset(tmp_path):
    # Every combination appears ten times, under distinct posts, so all splits have support.
    return write_data(tmp_path / "annotations.csv", [
        (f"Ubo lagnat sample {repeat} pattern {pattern}", str([(pattern >> bit) & 1 for bit in range(4)]))
        for repeat in range(10) for pattern in range(16)
    ])


def test_splits_are_reproducible_disjoint_and_supported(dataset):
    rows, _ = load_annotations([dataset], "electra")
    first = split_annotations(rows, 0.2, 0.2, 7)
    assert first == split_annotations(rows, 0.2, 0.2, 7)
    sets = [set(row["text_fingerprint"] for row in part) for part in first.values()]
    assert sum(map(len, sets)) == len(set.union(*sets)) == len(rows)
    for part in first.values():
        counts = np.asarray([row["labels"] for row in part]).sum(axis=0)
        assert np.all((counts > 0) & (counts < len(part)))
    with pytest.raises(ValueError, match="positive and negative"):
        split_annotations([{**row, "labels": [0, 0, 0, 0]} for row in rows])
    with pytest.raises(ValueError, match="fractions"):
        split_annotations(rows, 0.6, 0.5)


def test_threshold_ties_and_metrics_at_boundary():
    truth = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    thresholds, search = tune_thresholds(truth, truth.astype(float))
    assert thresholds.tolist() == [0.5] * 4
    assert len(search) == 4 * 91
    summary, labels = score_predictions(truth, np.where(truth, 0.5, 0.49), thresholds)
    assert summary["micro_f1"] == summary["exact_match_accuracy"] == 1.0
    assert all(row["support"] == row["tp"] == 1 and row["fp"] == row["fn"] == 0 for row in labels)
    # Thresholds must change when validation evidence calls for a lower cutoff.
    tuned, _ = tune_thresholds(truth, np.where(truth, 0.3, 0.2))
    assert tuned.tolist() == [0.3] * 4


def test_output_cannot_replace_source_or_active_bundle(tmp_path, dataset):
    root = tmp_path / "service"
    parser = make_parser("electra", root)
    for output in (root / "model", root / "model/new-run", root):
        args = parser.parse_args(["--data", str(dataset), "--output-dir", str(output)])
        with pytest.raises(ValueError, match="overlap"):
            preflight(args, root)
    output = tmp_path / "existing"
    output.mkdir()
    args = parser.parse_args(["--data", str(dataset), "--output-dir", str(output)])
    with pytest.raises(ValueError, match="new directory"):
        preflight(args, root)


def tiny_model(path, architecture):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import (ElectraConfig, ElectraForSequenceClassification, PreTrainedTokenizerFast,
                              XLMRobertaConfig, XLMRobertaForSequenceClassification, set_seed)

    set_seed(3)
    vocab = {token: i for i, token in enumerate(["<s>", "<pad>", "</s>", "<unk>", "Ubo", "lagnat", "sample", "pattern"])}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>",
                                       unk_token="<unk>", pad_token="<pad>", model_max_length=128)
    config_kwargs = dict(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=12, max_position_embeddings=132, num_labels=4, pad_token_id=1,
                         bos_token_id=0, eos_token_id=2, problem_type="multi_label_classification",
                         id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)})
    if architecture == "electra":
        model = ElectraForSequenceClassification(ElectraConfig(embedding_size=8, **config_kwargs))
    else:
        model = XLMRobertaForSequenceClassification(XLMRobertaConfig(**config_kwargs))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


@pytest.mark.parametrize("architecture", SERVICES)
def test_standalone_cli_trains_reloadable_bundle_offline(tmp_path, dataset, architecture):
    service_name, package = SERVICES[architecture]
    service = tmp_path / service_name
    service.mkdir()
    # Copy only code, proving training needs neither the research checkout nor notebook modules.
    for name in (package, "healthph_symptoms", "healthph_disease_training"):
        shutil.copytree(ROOT / service_name / name, service / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in RUNTIME_FILES:
        assert (service / "healthph_disease_training" / name).read_bytes() == (ROOT / "healthph_disease_training" / name).read_bytes()
    source = service / "model/best_model"
    tiny_model(source, architecture)
    before = hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest()
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "HEALTHPH_SYMPTOM_KEYWORDS_PATH"}}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(tmp_path / "empty-cache"),
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1")
    base_command = [sys.executable, "-m", f"{package}.train", "--data", str(dataset)]
    validation_output = tmp_path / "validation-only"
    checked = subprocess.run(base_command + ["--validate-only", "--model-path", str(tmp_path / "absent-model"),
                             "--output-dir", str(validation_output)], cwd=service, env=env, capture_output=True, text=True, timeout=60)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert json.loads((validation_output / "status.json").read_text())["status"] == "validated"
    assert not (validation_output / "best_model").exists()
    output = tmp_path / "trained"
    result = subprocess.run(base_command + ["--epochs", "1", "--batch-size", "16", "--gradient-accumulation-steps", "2",
                            "--device", "cpu", "--output-dir", str(output)], cwd=service, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    assert hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest() == before
    assert hashlib.sha256((output / "best_model/model.safetensors").read_bytes()).hexdigest() != before
    run = json.loads((output / "run_config.json").read_text())
    assert run["test_used_for_selection"] is False and run["label_order"] == list(LABELS)
    metrics = json.loads((output / "reports/metrics.json").read_text())
    assert set(metrics) == {"validation", "test"}
    assert all(0 <= metrics[split]["micro_f1"] <= 1 for split in metrics)
    assert json.loads((output / "reports/reload_verification.json").read_text())["probabilities_match"]
    # Exercise the actual service loader, plus the existing standalone XLM-R export command.
    loader = "healthph_plus.disease_inference" if architecture == "electra" else "healthph_xlmr.inference"
    script = (
        f"from {loader} import DiseasePredictor; import sys; "
        "p=DiseasePredictor(sys.argv[1]); r=p.predict('Ubo lagnat'); "
        "assert len(r['probabilities']) == 4 and isinstance(r['symptoms'],list)"
    )
    loaded = subprocess.run([sys.executable, "-c", script, str(output)], cwd=service, env=env,
                            capture_output=True, text=True, timeout=60)
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
    if architecture == "xlm-roberta":
        exported = subprocess.run([sys.executable, "-m", "healthph_xlmr.export", str(output),
                                   "--target-dir", str(tmp_path / "exported")], cwd=service, env=env,
                                  capture_output=True, text=True, timeout=60)
        assert exported.returncode == 0, exported.stdout + exported.stderr
    else:
        exported = subprocess.run([sys.executable, "-m", "healthph_symptoms.export", "--disease-source", str(output),
                                   "--service-dir", str(service), "--target-dir", str(tmp_path / "exported")],
                                  cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert exported.returncode == 0, exported.stdout + exported.stderr


def test_invalid_training_settings_do_not_create_run(tmp_path, dataset):
    from healthph_disease_training.runner import train

    output = tmp_path / "run"
    args = make_parser("electra", tmp_path / "service").parse_args(
        ["--data", str(dataset), "--output-dir", str(output), "--learning-rate", "nan"])
    with pytest.raises(ValueError, match="learning-rate"):
        train(args, "electra", tmp_path / "service")
    assert not output.exists()
