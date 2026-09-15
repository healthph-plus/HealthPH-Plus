"""Frozen inference reference from notebooks/01_electra_model.ipynb.

Model, tokenizer, labels, thresholds and torch are supplied by the parity test.
No training cells or data are included.
"""

def predict_disease_labels(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('text must be a non-empty string')

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt',
    )

    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits

    proba = torch.sigmoid(logits).cpu().numpy()[0]

    probabilities = {label: float(proba[i]) for i, label in enumerate(LABELS)}
    labels = [label for i, label in enumerate(LABELS) if proba[i] >= thresholds[label]]

    return {
        'labels': labels,
        'probabilities': probabilities,
        'thresholds_used': thresholds,
    }
