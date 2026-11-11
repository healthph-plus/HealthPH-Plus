"""Offline fine-tuning, validation-based selection, and inference bundle creation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import uuid

from .data import LABELS, MAX_LENGTH, NORMALIZATION, load_annotations, split_annotations


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    import csv

    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_predictions(labels, probabilities, thresholds):
    import numpy as np
    from sklearn.metrics import (accuracy_score, hamming_loss, multilabel_confusion_matrix,
                                 precision_recall_fscore_support)

    predicted = (np.asarray(probabilities, dtype=float) >= np.asarray(thresholds, dtype=float)).astype(int)
    summary = {"exact_match_accuracy": float(accuracy_score(labels, predicted)),
               "hamming_loss": float(hamming_loss(labels, predicted))}
    for average in ("micro", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted, average=average, zero_division=0,
        )
        summary.update({f"{average}_precision": float(precision), f"{average}_recall": float(recall),
                        f"{average}_f1": float(f1)})
    precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, zero_division=0)
    matrices = multilabel_confusion_matrix(labels, predicted)
    per_label = [{"label": label, "precision": float(precision[i]), "recall": float(recall[i]),
                  "f1": float(f1[i]), "support": int(support[i]),
                  **dict(zip(("tn", "fp", "fn", "tp"), map(int, matrices[i].ravel())))}
                 for i, label in enumerate(LABELS)]
    return summary, per_label


def tune_thresholds(labels, probabilities):
    import numpy as np
    from sklearn.metrics import f1_score

    # Try 0.5 first, then the nearest candidate, preferring the lower value in ties.
    grid = sorted((i / 100 for i in range(5, 96)), key=lambda t: (round(abs(t - 0.5), 2), t))
    selected, search = [], []
    for index, label in enumerate(LABELS):
        best_score, best_threshold = -1.0, 0.5
        for threshold in grid:
            score = float(f1_score(labels[:, index], probabilities[:, index] >= threshold, zero_division=0))
            search.append({"label": label, "threshold": threshold, "validation_f1": score})
            if score > best_score:
                best_score, best_threshold = score, threshold
        selected.append(best_threshold)
    return np.asarray(selected, dtype=float), search


def make_parser(architecture, service_root):
    parser = argparse.ArgumentParser(description=f"Fine-tune the local {architecture} disease model offline")
    parser.add_argument("--data", action="append", required=True, help="CSV with post/annotate; repeat to combine files")
    parser.add_argument("--model-path", type=Path, default=service_root / "model/best_model",
                        help="Local Hugging Face model/tokenizer directory; defaults to the service bundle")
    parser.add_argument("--output-dir", type=Path, help="New run directory; defaults to training-runs/<unique ID>")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--conflict-policy", choices=("reject", "exclude"), default="reject",
                        help="Reject conflicting annotations, or exclude every row of each conflicting post group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=architecture == "xlm-roberta")
    parser.add_argument("--validate-only", action="store_true", help="Validate/split data and save audits without loading weights")
    return parser


def preflight(args, service_root):
    for name in ("epochs", "batch_size", "gradient_accumulation_steps", "early_stopping_patience"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and nonnegative")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be in [0, 2**32)")
    source = args.model_path.expanduser().resolve()
    run_id = f"full_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    output = (args.output_dir or service_root / "training-runs" / run_id).expanduser().resolve()
    for protected in (source, (service_root / "model").resolve()):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Training output must not overlap source weights or the service model directory")
    if output.exists():
        raise ValueError("--output-dir must be a new directory; previous runs are never overwritten")
    return source, output


def source_fingerprint(source):
    digest = hashlib.sha256()
    for path in sorted(source.iterdir()):
        if path.is_file() and (path.suffix in {".json", ".safetensors", ".model", ".txt"}):
            digest.update(path.name.encode() + b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def train(args, architecture, service_root):
    import numpy as np

    service_root = Path(service_root).resolve()
    source, output = preflight(args, service_root)
    records, audit = load_annotations(args.data, architecture, args.conflict_policy)
    parts = split_annotations(records, args.validation_size, args.test_size, args.seed)
    print(f"Data: {audit['raw_rows']} rows; {audit['retained_posts']} retained posts; "
          f"{audit['duplicate_rows_removed']} duplicate rows removed; "
          f"{audit['conflicting_rows_excluded']} conflicting rows excluded", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    reports = output / "reports"
    reports.mkdir()
    audit["splits"] = {
        name: {"rows": len(rows), "positive_counts": dict(zip(LABELS, map(int, np.sum(
            [row["labels"] for row in rows], axis=0))))} for name, rows in parts.items()
    }
    audit["seed"] = args.seed
    audit["requested_split_fractions"] = {"train": 1 - args.validation_size - args.test_size,
                                          "validation": args.validation_size, "test": args.test_size}
    write_json(reports / "data_quality.json", audit)
    write_csv(reports / "split_manifest.csv", (
        {"split": name, "text_fingerprint": row["text_fingerprint"],
         "source_rows": json.dumps(row["source_rows"]), **dict(zip(LABELS, row["labels"]))}
        for name, rows in parts.items() for row in rows
    ))
    if args.validate_only:
        write_json(output / "status.json", {"status": "validated", "model_trained": False})
        return output
    write_json(output / "status.json", {"status": "training"})
    try:
        _fit(args, architecture, source, output, parts, audit)
    except BaseException as exc:
        write_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise
    write_json(output / "status.json", {"status": "complete"})
    return output


def _fit(args, architecture, source, output, parts, audit):
    import numpy as np
    import torch
    from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                              TrainingArguments, set_seed)

    set_seed(args.seed)
    config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if (config.model_type != architecture or config.num_labels != 4
            or config.id2label != dict(enumerate(LABELS))
            or config.label2id != {label: i for i, label in enumerate(LABELS)}
            or config.problem_type != "multi_label_classification"):
        raise ValueError("--model-path must contain a compatible four-label disease model and tokenizer")
    positions = config.max_position_embeddings
    capacity = positions - config.pad_token_id - 1 if architecture == "xlm-roberta" else positions
    if capacity < MAX_LENGTH:
        raise ValueError("Model position capacity is below the service's 128-token limit")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False, use_fast=True)
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        source, local_files_only=True, trust_remote_code=False, use_safetensors=True, output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Source weights do not match the model configuration: {loading}")
    if not tokenizer.is_fast or len(tokenizer) != model.get_input_embeddings().num_embeddings:
        raise ValueError("Model requires a matching fast tokenizer")
    if architecture == "xlm-roberta":
        if any(getattr(tokenizer, key) != getattr(config, key) for key in ("bos_token_id", "eos_token_id", "pad_token_id")):
            raise ValueError("Tokenizer special tokens do not match XLM-R model configuration")
        if tokenizer.model_max_length < MAX_LENGTH:
            raise ValueError("Tokenizer does not support the service's 128-token limit")
    model.float()

    class DiseaseDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.encodings = tokenizer([row["text"] for row in rows], truncation=True,
                                       max_length=MAX_LENGTH, padding=False)
            self.labels = [row["labels"] for row in rows]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {**{key: values[index] for key, values in self.encodings.items()},
                    "labels": self.labels[index]}

    labels = {name: np.asarray([row["labels"] for row in rows]) for name, rows in parts.items()}
    counts = labels["train"].sum(axis=0)
    positive_weights = ((len(parts["train"]) - counts) / counts if architecture == "xlm-roberta" else np.ones(4))

    class DiseaseTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.model_accepts_loss_kwargs = False
            self.loss_is_scaled_for_ga = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            model_inputs = dict(inputs)
            truth = model_inputs.pop("labels").float()
            result = model(**model_inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                result.logits.float(), truth.to(result.logits.device),
                pos_weight=torch.as_tensor(positive_weights, device=result.logits.device, dtype=torch.float32),
            )
            return (loss, result) if return_outputs else loss

    def probabilities(prediction):
        logits = prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        result = torch.sigmoid(torch.as_tensor(logits).float()).numpy().astype(float)
        if result.ndim != 2 or result.shape[1] != 4 or not np.isfinite(result).all():
            raise ValueError("Model returned invalid disease predictions")
        return result

    total_steps = math.ceil(math.ceil(len(parts["train"]) / args.batch_size) / args.gradient_accumulation_steps) * args.epochs
    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, warmup_steps=math.ceil(total_steps * 0.1),
        optim="adamw_torch", fp16=False, bf16=False,
        gradient_checkpointing=args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="micro_f1", greater_is_better=True,
        save_total_limit=1, save_only_model=True, dataloader_num_workers=0, dataloader_pin_memory=False,
        report_to="none", push_to_hub=False, seed=args.seed, data_seed=args.seed, use_cpu=args.device == "cpu",
    )
    reports = output / "reports"
    run_config = {
        "run_id": output.name, "run_mode": "full", "smoke_only": False,
        "training_entrypoint": "standalone CLI", "model_type": architecture,
        "source_model": str(source), "source_model_sha256": source_fingerprint(source),
        "label_order": list(LABELS), "max_length": MAX_LENGTH,
        "normalization": NORMALIZATION if architecture == "xlm-roberta" else "preserve submitted text",
        "datasets": audit["sources"], "input_column": "post", "target_column": "annotate",
        "checkpoint_metric": "validation micro_f1 at threshold 0.5",
        "threshold_objective": "per-label validation F1; ties closest to 0.5, then lower threshold",
        "test_used_for_selection": False, "positive_weights": dict(zip(LABELS, map(float, positive_weights))),
        "loss": "positive-weighted BCE" if architecture == "xlm-roberta" else "BCE",
        "split_algorithm": "MultilabelStratifiedShuffleSplit; test first, then validation",
        "split_fractions": {"train": 1 - args.validation_size - args.test_size,
                            "validation": args.validation_size, "test": args.test_size},
        "settings": {key: value for key, value in vars(args).items() if key not in {"data", "model_path", "output_dir"}},
        "device": str(training_args.device),
        "versions": {name: version(name) for name in ("torch", "transformers", "accelerate", "scikit-learn", "iterative-stratification")},
    }
    write_json(output / "run_config.json", run_config)
    trainer = DiseaseTrainer(
        model=model, args=training_args,
        train_dataset=DiseaseDataset(parts["train"]), eval_dataset=DiseaseDataset(parts["validation"]),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer), processing_class=tokenizer,
        compute_metrics=lambda prediction: score_predictions(prediction.label_ids, probabilities(prediction), [0.5] * 4)[0],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.train(resume_from_checkpoint=False)
    if trainer.state.best_model_checkpoint is None:
        raise RuntimeError("Training did not produce a validation-selected checkpoint")
    validation = trainer.predict(trainer.eval_dataset, metric_key_prefix="validation")
    validation_probabilities = probabilities(validation)
    thresholds, search = tune_thresholds(labels["validation"], validation_probabilities)
    write_csv(reports / "validation_threshold_search.csv", search)
    write_json(output / "thresholds.json", dict(zip(LABELS, thresholds.tolist())))
    write_json(output / "label_order.json", list(LABELS))
    trainer.save_model(str(output / "best_model"))
    tokenizer.save_pretrained(output / "best_model")
    write_json(output / "selection.json", {
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_validation_micro_f1_at_0_5": trainer.state.best_metric,
        "test_used_for_selection": False,
    })
    # Only now evaluate the test split, with the model and thresholds already frozen.
    test = trainer.predict(DiseaseDataset(parts["test"]), metric_key_prefix="test")
    metrics = {}
    for name, prediction, scores in (("validation", validation, validation_probabilities), ("test", test, probabilities(test))):
        metrics[name], per_label = score_predictions(labels[name], scores, thresholds)
        metrics[name]["loss"] = float(prediction.metrics[f"{name}_loss"])
        write_csv(reports / f"{name}_per_label.csv", per_label)
        write_csv(reports / f"{name}_predictions.csv", (
            {"text_fingerprint": row["text_fingerprint"], **{
                f"{label}_{field}": value
                for j, label in enumerate(LABELS)
                for field, value in (("truth", row["labels"][j]), ("score", float(scores[i, j])),
                                     ("predicted", int(scores[i, j] >= thresholds[j])))
            }} for i, row in enumerate(parts[name])
        ))
    write_json(reports / "metrics.json", metrics)
    write_json(reports / "training_log.json", trainer.state.log_history)
    write_csv(reports / "training_log.csv", trainer.state.log_history)

    # Release optimizer/model memory before checking the saved model on CPU.
    trainer.model.to("cpu")
    del trainer, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    reloaded = AutoModelForSequenceClassification.from_pretrained(
        output / "best_model", local_files_only=True, trust_remote_code=False, use_safetensors=True,
    ).float().eval()
    reloaded_tokenizer = AutoTokenizer.from_pretrained(output / "best_model", local_files_only=True, trust_remote_code=False)
    with torch.inference_mode():
        encoded = reloaded_tokenizer([row["text"] for row in parts["validation"][:2]], padding=True,
                                     truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        saved_scores = torch.sigmoid(reloaded(**encoded).logits).numpy()
    np.testing.assert_allclose(saved_scores, validation_probabilities[:2], atol=1e-5, rtol=1e-4)
    write_json(reports / "reload_verification.json", {"probabilities_match": True,
               "max_probability_difference": float(np.max(np.abs(saved_scores - validation_probabilities[:2])))})


def main(architecture, service_root, argv=None):
    parser = make_parser(architecture, Path(service_root))
    args = parser.parse_args(argv)
    try:
        output = train(args, architecture, service_root)
    except ImportError as exc:
        parser.exit(2, f"Missing training dependency: {exc}. Install requirements-training.txt.\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Training error: {exc}\n")
    print(f"{'Validated data' if args.validate_only else 'Trained disease bundle'}: {output}")
