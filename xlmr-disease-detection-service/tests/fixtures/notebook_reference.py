"""Frozen inference definitions from notebooks/06_xlm_roberta_disease.ipynb.

Copied before service extraction. No training cells, datasets, or service code
are imported. Real-model parity tests load the saved bundle independently.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def normalize_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('text must be a non-empty string')
    return re.sub(r'\s+', ' ', text).strip()

def select_device(preference='auto'):
    if preference not in {'auto', 'mps', 'cpu'}:
        raise ValueError('DEVICE_PREFERENCE must be auto, mps, or cpu')
    available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if preference == 'mps' and not available:
        raise RuntimeError('MPS is unavailable in this kernel. Select cpu or use an MPS-enabled kernel.')
    return torch.device('mps' if available and preference != 'cpu' else 'cpu')


def load_disease_predictor(run_dir, device='auto', allow_smoke=False):
    run_dir = Path(run_dir).expanduser().resolve()
    config = json.loads((run_dir / 'run_config.json').read_text(encoding='utf-8'))
    labels = json.loads((run_dir / 'label_order.json').read_text(encoding='utf-8'))
    threshold_map = json.loads((run_dir / 'thresholds.json').read_text(encoding='utf-8'))
    expected_labels = ['AURI', 'PN', 'TB', 'COVID']
    if labels != expected_labels or config.get('label_order') != labels or set(threshold_map) != set(labels):
        raise ValueError('Saved labels/threshold keys do not match the four disease labels')
    values = [threshold_map[label] for label in labels]
    if any(type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError('Saved thresholds must be finite numbers in [0, 1]')
    if config.get('normalization') != 'collapse whitespace and strip; preserve casing and Unicode':
        raise ValueError('Unsupported saved text normalization')
    max_length = config.get('max_length')
    if type(max_length) is not int or not 2 <= max_length <= 512:
        raise ValueError('Invalid saved token limit')
    if config.get('run_mode') not in {'full', 'smoke'} or type(config.get('smoke_only')) is not bool:
        raise ValueError('Missing or invalid run-mode metadata')
    if config['smoke_only'] != (config['run_mode'] == 'smoke'):
        raise ValueError('Inconsistent run-mode metadata')
    if config['smoke_only'] and not allow_smoke:
        raise ValueError('This is a smoke bundle, not a fully trained model; pass allow_smoke=True for diagnostics')
    target_device = select_device(str(device))
    saved_tokenizer = AutoTokenizer.from_pretrained(run_dir / 'best_model', local_files_only=True, trust_remote_code=False)
    saved_model = AutoModelForSequenceClassification.from_pretrained(
        run_dir / 'best_model', local_files_only=True, trust_remote_code=False,
    ).float().to(target_device)
    if (saved_model.config.id2label != dict(enumerate(labels))
            or saved_model.config.label2id != {label: i for i, label in enumerate(labels)}
            or saved_model.config.problem_type != 'multi_label_classification'):
        raise ValueError('Saved model configuration disagrees with bundle label mapping')
    saved_model.eval()
    threshold_array = np.array(values)

    def predict(text):
        normalized = normalize_text(text)
        inputs = saved_tokenizer(normalized, truncation=True, max_length=max_length, return_tensors='pt')
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
        with torch.inference_mode():
            scores = torch.sigmoid(saved_model(**inputs).logits.float())[0].cpu().numpy()
        if scores.shape != (4,) or not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise RuntimeError('Model returned invalid disease probabilities')
        return {
            'labels': [label for i, label in enumerate(labels) if scores[i] >= threshold_array[i]],
            'probabilities': {label: float(scores[i]) for i, label in enumerate(labels)},
            'thresholds_used': {label: float(threshold_array[i]) for i, label in enumerate(labels)},
        }

    return predict
