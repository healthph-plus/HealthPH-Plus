"""Frozen loader from sentiment notebook cells 2 and 18, before API extraction.

Test reference only; TF-IDF is not supported by the shipped service.
"""
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def select_local_device(preference='auto'):
    if preference not in {'auto', 'cuda', 'mps', 'cpu'}:
        raise ValueError("Device must be 'auto', 'cuda', 'mps', or 'cpu'.")
    available = {
        'cuda': torch.cuda.is_available(),
        'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cpu': True,
    }
    if preference == 'auto':
        return torch.device(next(name for name in ('cuda', 'mps', 'cpu') if available[name]))
    if not available[preference]:
        raise RuntimeError(f'{preference} is unavailable in this kernel. Use auto or cpu.')
    return torch.device(preference)

def load_sentiment_predictor(model_dir, device=None):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text())
    labels = {int(key): value for key, value in
              json.loads((model_dir / 'label_mapping.json').read_text())['labels'].items()}
    threshold = float(json.loads((model_dir / 'threshold.json').read_text())['worried_threshold'])
    if labels != {0: 'calm', 1: 'worried'} or not 0 <= threshold <= 1:
        raise ValueError('Saved label mapping or threshold does not match this notebook.')
    expected_normalization = 'collapse whitespace and strip; preserve casing and Unicode'
    if config['text_normalization'] != expected_normalization:
        raise ValueError('Unsupported saved normalization contract.')
    target_device = select_local_device(str(device) if device is not None else 'auto')
    if config['model_key'] == 'tfidf_logreg':
        saved_pipeline = joblib.load(model_dir / 'pipeline.joblib')
        positive_column = list(saved_pipeline.classes_).index(1)
    elif config['model_key'] == 'mbert':
        saved_tokenizer = AutoTokenizer.from_pretrained(model_dir / 'best_model', local_files_only=True)
        saved_model = AutoModelForSequenceClassification.from_pretrained(
            model_dir / 'best_model', local_files_only=True,
        ).to(target_device)
        if saved_model.config.id2label != labels:
            raise ValueError('Transformer label mapping differs from the saved label mapping.')
        saved_model.eval()
    else:
        raise ValueError(f"Unknown saved model type: {config['model_key']}")

    def predict(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('text must be a non-empty string')
        text = re.sub(r'\s+', ' ', text).strip()
        if config['model_key'] == 'tfidf_logreg':
            probability = float(saved_pipeline.predict_proba([text])[0, positive_column])
        else:
            encoded = saved_tokenizer(
                text, truncation=True, max_length=config['max_length'], return_tensors='pt',
            ).to(target_device)
            with torch.inference_mode():
                probability = float(torch.softmax(saved_model(**encoded).logits.float(), dim=-1)[0, 1].cpu())
        label = int(probability >= threshold)
        return {'predicted_label': label, 'predicted_label_name': labels[label],
                'worried_probability': probability, 'threshold_used': threshold}

    return predict