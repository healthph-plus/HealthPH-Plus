import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    PreTrainedTokenizerFast,
    ElectraConfig, ElectraForSequenceClassification, XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
)

from healthph_symptoms import export
from healthph_symptoms.keywords import KeywordSymptomExtractor, MASTER_KEYWORDS

ROOT = Path(__file__).resolve().parents[1]
DISEASE_LABELS = ("AURI", "PN", "TB", "COVID")


def prepare_service(tmp_path, family):
    service_name = "disease-detection-service" if family == "electra" else "xlmr-disease-detection-service"
    package_name = "healthph_plus" if family == "electra" else "healthph_xlmr"
    service = tmp_path / service_name
    shutil.copytree(ROOT / service_name / package_name, service / package_name,
                    ignore=shutil.ignore_patterns("__pycache__"))
    return service


def prepare_disease(tmp_path, family):
    source = tmp_path / "disease_source"
    source.mkdir()
    kwargs = dict(
        vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, problem_type="multi_label_classification",
        id2label=dict(enumerate(DISEASE_LABELS)),
        label2id={s: i for i, s in enumerate(DISEASE_LABELS)},
    )
    if family == "electra":
        model = ElectraForSequenceClassification(ElectraConfig(embedding_size=8, **kwargs))
    else:
        model = XLMRobertaForSequenceClassification(XLMRobertaConfig(max_position_embeddings=514, **kwargs))
        (source / "run_config.json").write_text(json.dumps({
            "run_mode": "full", "smoke_only": False, "label_order": list(DISEASE_LABELS),
            "normalization": "collapse whitespace and strip; preserve casing and Unicode", "max_length": 128,
        }))
    model.save_pretrained(source / "best_model")
    # A disease tokenizer fixture needs no symptom model or symptom dataset.
    vocab = {word: i for i, word in enumerate([
        "<s>", "<pad>", "</s>", "<unk>", "ubo", "sipon", "wala", "at", "okay", "😷",
    ])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
        pad_token="<pad>", unk_token="<unk>", model_max_length=512,
    )
    tokenizer.save_pretrained(source / "best_model")
    (source / "label_order.json").write_text(json.dumps(list(DISEASE_LABELS)))
    (source / "thresholds.json").write_text(json.dumps(dict.fromkeys(DISEASE_LABELS, .5)))
    return source


@pytest.mark.parametrize("family", ["electra", "xlmr"])
def test_export_and_relocated_standalone_api_work_offline(tmp_path, family):
    service = prepare_service(tmp_path, family)
    disease = prepare_disease(tmp_path, family)
    # Refreshing a handoff removes obsolete runtime modules.
    (service / "healthph_symptoms").mkdir()
    (service / "healthph_symptoms/inference.py").write_text("raise RuntimeError('obsolete extractor')")
    target = export.export_service_bundle(service, disease)
    assert target == service / "model"
    assert not (service / "healthph_symptoms/inference.py").exists()
    assert (service / "healthph_symptoms/GENERATED.txt").is_file()
    assert (service / "SYMPTOM_WORKFLOW.md").is_file()
    relocated = tmp_path / "independent" / "service"
    shutil.copytree(service, relocated)
    module = "healthph_plus.disease_api" if family == "electra" else "healthph_xlmr.api"
    script = (
        "import importlib,json,sys; sys.path.insert(0,sys.argv[1]); "
        "from fastapi.testclient import TestClient; "
        "m=importlib.import_module(sys.argv[2]); "
        "p=m.DiseasePredictor(sys.argv[1]+'/model'); "
        "client=TestClient(m.create_app(lambda:p)); client.__enter__(); "
        "r=client.post('/v1/disease/predict',json={'text':'ubo at sipon'}); "
        "assert r.status_code==200,r.text; "
        "assert [s['text'] for s in r.json()['symptoms']]==['ubo','sipon']; "
        "assert client.get('/health/ready').json()['symptom_model_version']; "
        "client.__exit__(None,None,None); print('offline combined API passed')"
    )
    result = subprocess.run([sys.executable, "-I", "-c", script, str(relocated), module],
                            capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "offline combined API passed" in result.stdout
    if family == "electra":
        measured = subprocess.run([
            sys.executable, "-m", "healthph_symptoms.benchmark", "--service-dir", str(relocated),
            "--repeats", "2",
        ], capture_output=True, text=True, timeout=60, cwd=ROOT)
        assert measured.returncode == 0, measured.stderr
        report = json.loads(measured.stdout)
        assert report["device"] == "cpu" and report["repeats"] == 2
        assert 0 <= report["batch_latency_p50_seconds"] <= report["batch_latency_p95_seconds"]
        assert report["whole_process_peak_rss_mib"] > 0
        assert report["symptom_model_version"] == KeywordSymptomExtractor().model_version


def test_missing_disease_assets_never_touch_existing_handoff(tmp_path):
    service = prepare_service(tmp_path, "electra")
    target = service / "model"
    target.mkdir()
    (target / "keep.txt").write_text("existing bundle")
    with pytest.raises(ValueError, match="best_model directory"):
        export.export_service_bundle(service, tmp_path / "absent_disease")
    assert (target / "keep.txt").read_text() == "existing bundle"
    assert not (service / "healthph_symptoms").exists()
    assert not list(service.glob(".symptom-export-*"))


def test_failed_combined_warmup_preserves_previous_model_and_runtime(tmp_path, monkeypatch):
    service = prepare_service(tmp_path, "electra")
    disease = prepare_disease(tmp_path, "electra")
    target = service / "model"
    target.mkdir()
    (target / "keep").write_text("previous")

    def fail(*args):
        assert (args[1] / "best_model/model.safetensors").is_file()
        raise ValueError("bad combined forward")

    monkeypatch.setattr(export, "warmup_service_bundle", fail)
    with pytest.raises(ValueError, match="bad combined forward"):
        export.export_service_bundle(service, disease)
    assert (target / "keep").read_text() == "previous"
    assert not (service / "healthph_symptoms").exists()
    assert not list(service.glob(".symptom-export-*"))


def test_export_rejects_targets_that_could_replace_service_code(tmp_path):
    service = prepare_service(tmp_path, "electra")
    for target in (service, service.parent, service / "healthph_plus", export.PACKAGE_DIR):
        with pytest.raises(ValueError, match="Target must be"):
            export.export_service_bundle(service, tmp_path / "disease", target_dir=target)
    assert (service / "healthph_plus/disease_api.py").is_file()


def test_failed_replacement_restores_previous_directory(tmp_path, monkeypatch):
    target, temporary = tmp_path / "model", tmp_path / "staged"
    target.mkdir()
    (target / "old").write_text("previous")
    stage = temporary / "bundle"
    stage.mkdir(parents=True)
    (stage / "new").write_text("new")
    original = export.os.replace

    def fail(source, destination):
        if source == stage:
            raise OSError("replace failed")
        return original(source, destination)

    monkeypatch.setattr(export.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        export._replace_directory(stage, target)
    assert (target / "old").read_text() == "previous"


@pytest.mark.parametrize("existing_runtime", [False, True])
def test_model_switch_failure_also_rolls_back_runtime_and_workflow(tmp_path, monkeypatch, existing_runtime):
    service = prepare_service(tmp_path, "electra")
    disease = prepare_disease(tmp_path, "electra")
    target = service / "model"
    target.mkdir()
    (target / "old").write_text("previous model")
    if existing_runtime:
        (service / "healthph_symptoms").mkdir()
        (service / "healthph_symptoms/previous.py").write_text("previous runtime")
        (service / "SYMPTOM_WORKFLOW.md").write_text("previous docs")
    original = export.os.replace

    def fail(source, destination):
        if source.name == "bundle":
            raise OSError("model switch failed")
        return original(source, destination)

    monkeypatch.setattr(export, "warmup_service_bundle", lambda *args: {})
    monkeypatch.setattr(export.os, "replace", fail)
    with pytest.raises(OSError, match="model switch failed"):
        export.export_service_bundle(service, disease)
    assert (target / "old").read_text() == "previous model"
    if existing_runtime:
        assert (service / "healthph_symptoms/previous.py").read_text() == "previous runtime"
        assert (service / "SYMPTOM_WORKFLOW.md").read_text() == "previous docs"
        assert not (service / "healthph_symptoms/inference.py").exists()
    else:
        assert not (service / "healthph_symptoms").exists()
        assert not (service / "SYMPTOM_WORKFLOW.md").exists()
    assert not list(service.glob(".symptom-code-*"))
    assert not list(service.glob(".symptom-export-*"))


def test_both_vendored_runtimes_match_canonical_source():
    for service in ("disease-detection-service", "xlmr-disease-detection-service"):
        for name in export.RUNTIME_FILES:
            assert (ROOT / service / "healthph_symptoms" / name).read_bytes() == (export.PACKAGE_DIR / name).read_bytes()

        assert (ROOT / service / "healthph_symptoms/keywords_master.csv").read_bytes() == MASTER_KEYWORDS.read_bytes()
