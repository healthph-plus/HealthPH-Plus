"""Consistent, artifact-backed notebook reports; importing this module needs no torch."""

from __future__ import annotations

import math
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    hamming_loss, multilabel_confusion_matrix, precision_recall_curve,
    precision_recall_fscore_support,
)


def save_table(frame, output_path):
    """Save the same data as CSV and JSON (missing measurements become JSON null)."""
    path = Path(output_path).with_suffix('')
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix('.csv'), index=False)
    frame.to_json(path.with_suffix('.json'), orient='records', indent=2, double_precision=15, date_format='iso')


def save_figure(fig, output_path, *, show=True, display_handle=None):
    path = Path(output_path).with_suffix('')
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'svg.fonttype': 'none'}):
        fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    if show:
        from IPython.display import display
        if display_handle is None:
            display_handle = display(fig, display_id=True)
        else:
            display_handle.update(fig)
    plt.close(fig)
    return display_handle


def extract_training_history(log_history, *, train_examples=None, validation_examples=None):
    """Normalize sparse Trainer logs without interpolating or inventing measurements."""
    aliases = {'accuracy': 'accuracy', 'exact_match_accuracy': 'accuracy'}
    metrics = {'loss', 'micro_f1', 'macro_f1', 'worried_f1', 'micro_precision',
               'micro_recall', 'macro_precision', 'macro_recall', 'worried_precision',
               'worried_recall', 'hamming_loss', 'balanced_accuracy', 'average_precision'}
    rows = []
    for record in log_history:
        for key, value in record.items():
            if value is None or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
                continue
            if key in {'loss', 'learning_rate'}:
                split, metric, examples = 'optimization', key, None
            elif key.startswith('train_eval_'):
                split, metric = 'train', key[len('train_eval_'):]
                examples = record.get('train_eval_examples', train_examples)
            elif key.startswith('eval_'):
                split, metric = 'validation', key[len('eval_'):]
                examples = record.get('eval_examples', record.get('eval_rows', validation_examples))
            else:
                continue
            metric = aliases.get(metric, metric)
            if metric not in metrics | {'accuracy', 'learning_rate'}:
                continue
            rows.append({'split': split, 'step': record.get('step'), 'epoch': record.get('epoch'),
                         'examples': examples, 'metric': metric, 'value': float(value),
                         'threshold': 0.5 if split != 'optimization' and metric != 'loss' else None})
    return pd.DataFrame(rows, columns=['split', 'step', 'epoch', 'examples', 'metric', 'value', 'threshold'])


def plot_training_history(log_history, output_path, *, model_name, loss_name='Loss',
                          best_step=None, train_examples=None, validation_examples=None,
                          task='multilabel', show=True, display_handle=None):
    history = extract_training_history(log_history, train_examples=train_examples,
                                       validation_examples=validation_examples)
    save_table(history, Path(output_path).parent / 'training_history')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    primary = 'worried' if task == 'binary' else 'micro'
    panels = [('loss', loss_name), ('accuracy', 'Accuracy' if task == 'binary' else 'Exact-match accuracy'),
              ('f1', 'F1'), ('precision', 'Worried precision' if task == 'binary' else 'Micro precision'),
              ('recall', 'Worried recall' if task == 'binary' else 'Micro recall'),
              ('learning_rate', 'Learning rate')]
    for ax, (metric, title) in zip(axes.flat, panels):
        wanted = [f'{primary}_f1', 'macro_f1'] if metric == 'f1' else [
            f'{primary}_{metric}' if metric in {'precision', 'recall'} else metric]
        for split, color in [('optimization', '#737373'), ('train', '#0072B2'), ('validation', '#D55E00')]:
            for field in wanted:
                data = history[(history['split'] == split) & (history['metric'] == field)].dropna(subset=['step'])
                if data.empty:
                    continue
                label = 'Optimization (logged batch mean)' if split == 'optimization' and field == 'loss' else split.title()
                if metric == 'f1':
                    label += ' ' + field.replace('_', ' ')
                if split != 'optimization' and metric == 'loss':
                    label += ' (evaluation mode)'
                ax.plot(data['step'], data['value'], color=color,
                        linestyle='--' if field == 'macro_f1' else '-',
                        marker='o' if split != 'optimization' else None, markersize=4,
                        linewidth=1.5, alpha=0.75 if split == 'optimization' else 1, label=label)
        if best_step is not None:
            ax.axvline(best_step, color='#009E73', linestyle=':', linewidth=1.5, label='Selected checkpoint')
        ax.set(title=title, xlabel='Optimizer step', ylabel=title)
        if metric not in {'loss', 'learning_rate'}:
            ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        if not ax.lines or (best_step is not None and len(ax.lines) == 1):
            ax.text(0.5, 0.5, 'Not recorded', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.legend(fontsize=8)
    note = 'Threshold 0.5; epoch and example counts are saved in training_history.csv.'
    if history[(history['split'] == 'train') & (history['metric'] == 'accuracy')].empty:
        note += '\nHistorical training accuracy/F1 unavailable; requires a monitored training run.'
    fig.suptitle(f'{model_name}: training and validation\n{note}', fontsize=12)
    return save_figure(fig, output_path, show=show, display_handle=display_handle)


def classification_metrics(y_true, probabilities, thresholds, labels, *, task='multilabel'):
    """Return final metrics, per-class metrics, 2x2 matrices, and thresholded predictions."""
    truth, scores = np.asarray(y_true), np.asarray(probabilities, dtype=float)
    if task not in {'binary', 'multilabel'}:
        raise ValueError('task must be binary or multilabel')
    expected = (len(truth),) if task == 'binary' else (len(truth), len(labels))
    if (truth.shape != expected or scores.shape != expected or not len(truth)
            or not np.isin(truth, [0, 1]).all() or not np.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()):
        raise ValueError('Expected nonempty binary labels and matching finite probabilities in [0, 1]')
    cutoffs = np.asarray(thresholds, dtype=float)
    if (cutoffs.shape != (() if task == 'binary' else (len(labels),))
            or not np.isfinite(cutoffs).all() or not ((cutoffs >= 0) & (cutoffs <= 1)).all()):
        raise ValueError('Thresholds must match the task and lie in [0, 1]')
    if task == 'binary' and len(labels) != 2:
        raise ValueError('Binary reports require two class names in class order')
    predictions = (scores >= cutoffs).astype(int)
    clipped = np.clip(scores, np.finfo(float).eps, 1 - np.finfo(float).eps)
    summary = {'rows': len(truth), 'accuracy' if task == 'binary' else 'exact_match_accuracy':
               float(accuracy_score(truth, predictions)),
               'log_loss': float(-(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped)).mean())}
    if task == 'binary':
        p, r, f, support = precision_recall_fscore_support(truth, predictions, labels=[0, 1], zero_division=0)
        matrices = np.array([confusion_matrix(truth, predictions, labels=[0, 1])])
        summary.update(worried_precision=float(p[1]), worried_recall=float(r[1]),
                       worried_f1=float(f[1]), macro_f1=float(f.mean()))
        class_scores = np.column_stack([1 - scores, scores])
        class_truth = np.column_stack([truth == 0, truth == 1])
        # Class 0 is the complement of the worried decision, not a second >= rule.
        class_cutoffs = [None, float(cutoffs)]
    else:
        summary.update(hamming_loss=float(hamming_loss(truth, predictions)),
                       label_accuracy=float(1 - hamming_loss(truth, predictions)))
        for average in ['micro', 'macro']:
            p, r, f, _ = precision_recall_fscore_support(truth, predictions, average=average, zero_division=0)
            summary.update({f'{average}_precision': float(p), f'{average}_recall': float(r), f'{average}_f1': float(f)})
        p, r, f, support = precision_recall_fscore_support(truth, predictions, average=None, zero_division=0)
        matrices = multilabel_confusion_matrix(truth, predictions)
        class_scores, class_truth, class_cutoffs = scores, truth, cutoffs
    per_label = pd.DataFrame({'label': labels, 'precision': p, 'recall': r, 'f1': f,
                              'support': support.astype(int), 'threshold': class_cutoffs})
    per_label['average_precision'] = [
        float(average_precision_score(class_truth[:, i], class_scores[:, i])) if support[i] else None
        for i in range(len(labels))]
    return summary, per_label, matrices, predictions


def confusion_cells(matrices, labels):
    rows = []
    for label, matrix in zip(labels, matrices):
        matrix = np.asarray(matrix)
        for (actual, predicted), count in np.ndenumerate(matrix):
            support = int(matrix[actual].sum())
            rows.append({'label': label, 'actual': actual, 'predicted': predicted,
                         'outcome': [['TN', 'FP'], ['FN', 'TP']][actual][predicted],
                         'count': int(count), 'actual_support': support,
                         'row_percentage': 100 * int(count) / support if support else None})
    return pd.DataFrame(rows)


def plot_confusion_matrices(matrices, labels, output_path, *, title,
                            class_names=('Absent', 'Present'), show=True):
    matrices = np.asarray(matrices)
    if matrices.shape != (len(labels), 2, 2) or (matrices < 0).any():
        raise ValueError('Expected one nonnegative 2x2 matrix per panel')
    cells = confusion_cells(matrices, labels)
    save_table(cells, output_path)
    columns = min(2, len(labels))
    rows = math.ceil(len(labels) / columns)
    for normalized in [False, True]:
        fig, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 4.8 * rows),
                                 squeeze=False, layout='constrained')
        vmax = 1 if normalized else max(1, int(matrices.max()))
        for ax, label, matrix in zip(axes.flat, labels, matrices):
            support = matrix.sum(axis=1)
            rates = np.divide(matrix, support[:, None], out=np.full((2, 2), np.nan), where=support[:, None] != 0)
            data = rates if normalized else matrix
            cmap = plt.colormaps['Blues'].with_extremes(bad='#eeeeee')
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
            for (i, j), count in np.ndenumerate(matrix):
                rate = rates[i, j]
                percent = f'{rate:.1%}' if np.isfinite(rate) else 'N/A'
                color = 'white' if np.isfinite(data[i, j]) and data[i, j] / vmax > 0.55 else '#162536'
                ax.text(j, i, f'{[["TN", "FP"], ["FN", "TP"]][i][j]}\n{int(count):,}  ({percent})',
                        ha='center', va='center', fontsize=12, color=color)
            ax.set(title=f'{label} · n={int(matrix.sum()):,}', xlabel='Predicted', ylabel='Actual',
                   xticks=[0, 1], xticklabels=class_names, yticks=[0, 1],
                   yticklabels=[f'{name}\n(n={int(n):,})' for name, n in zip(class_names, support)])
            ax.grid(False)
        for ax in list(axes.flat)[len(labels):]:
            ax.set_visible(False)
        fig.colorbar(im, ax=list(axes.flat)[:len(labels)], shrink=0.8,
                     label='Fraction of actual class' if normalized else 'Examples')
        fig.suptitle(f'{title}\nCounts and % within each actual class; N/A = no actual examples', fontsize=13)
        path = Path(output_path)
        if normalized:
            path = path.with_name(path.stem + '_normalized.png')
        save_figure(fig, path, show=show)
    return cells


def plot_per_label_metrics(table, output_path, *, title, show=True):
    splits = list(table['split'].unique())
    fig, axes = plt.subplots(len(splits), 1, figsize=(11, 3.5 * len(splits)), squeeze=False, layout='constrained')
    for ax, split in zip(axes.flat, splits):
        data = table[table['split'] == split]
        long = data.melt(id_vars=['label', 'support'], value_vars=['precision', 'recall', 'f1'],
                         var_name='metric', value_name='score')
        sns.barplot(data=long, x='label', y='score', hue='metric', palette='colorblind', ax=ax)
        ax.set(title=split.title(), xlabel='', ylabel='Score', ylim=(0, 1.08))
        ax.set_xticks(range(len(data)), [f'{r.label}\npositive n={r.support:,}' for r in data.itertuples()])
        ax.legend(loc='upper right', ncol=3, fontsize=8)
    fig.suptitle(title)
    save_figure(fig, output_path, show=show)


def plot_metric_comparison(table, output_path, *, title, task='multilabel', loss_name=None, show=True):
    keys = (['accuracy', 'worried_f1', 'macro_f1', 'worried_precision', 'worried_recall'] if task == 'binary'
            else ['exact_match_accuracy', 'micro_f1', 'macro_f1', 'micro_precision', 'micro_recall'])
    keys = [k for k in keys if k in table and table[k].notna().any()]
    losses = [k for k in ['log_loss', 'objective_loss'] if k in table and table[k].notna().any()]
    fig, axes = plt.subplots(1, 2 if losses else 1, figsize=(15 if losses else 10, 5),
                             squeeze=False, layout='constrained')
    data = table.copy()
    data['series'] = data['model'].astype(str) + ' / ' + data['split'] if 'model' in data else data['split']
    for ax, fields, is_loss in [(axes.flat[0], keys, False)] + ([(axes.flat[1], losses, True)] if losses else []):
        long = data.melt(id_vars=['series'], value_vars=fields, var_name='metric', value_name='value').dropna(subset=['value'])
        long['metric'] = long['metric'].replace({'exact_match_accuracy': 'Exact-match\naccuracy',
            'objective_loss': loss_name or 'Model objective loss', 'log_loss': 'Unweighted\nlog loss'}).str.replace('_', ' ')
        sns.barplot(data=long, x='metric', y='value', hue='series', palette='colorblind', ax=ax)
        ax.set(xlabel='', ylabel='Loss (lower is better)' if is_loss else 'Score', title='Loss' if is_loss else 'Final split metrics')
        if not is_loss:
            ax.set_ylim(0, 1.08)
        ax.tick_params(axis='x', rotation=15)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    save_figure(fig, output_path, show=show)


def plot_precision_recall(splits, labels, output_path, *, title, task='multilabel', show=True):
    fig, axes = plt.subplots(math.ceil(len(labels) / 2), min(2, len(labels)),
                             figsize=(11, 4 * math.ceil(len(labels) / 2)), squeeze=False, layout='constrained')
    curves = []
    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        for split, (truth, scores) in splits.items():
            truth, scores = np.asarray(truth), np.asarray(scores)
            y = (truth == i).astype(int) if task == 'binary' else truth[:, i]
            p = ((1 - scores) if i == 0 else scores) if task == 'binary' else scores[:, i]
            if not y.sum():
                ax.plot([], [], label=f'{split}: no positives (AP unavailable)')
                continue
            precision, recall, thresholds = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            ax.step(recall, precision, where='post', label=f'{split} (AP={ap:.3f}, n+={int(y.sum())})')
            curves.extend({'split': split, 'label': label, 'recall': float(r), 'precision': float(v),
                           'threshold': float(t) if np.isfinite(t) else None}
                          for r, v, t in zip(recall, precision, np.r_[thresholds, np.nan]))
        ax.set(title=label, xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.03))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(labels):]:
        ax.set_visible(False)
    fig.suptitle(title)
    save_table(pd.DataFrame(curves, columns=['split', 'label', 'recall', 'precision', 'threshold']), output_path)
    save_figure(fig, output_path, show=show)


def plot_threshold_f1(truth, scores, labels, thresholds, output_path, *, title, task='multilabel', show=True):
    truth, scores = np.asarray(truth), np.asarray(scores)
    names = [labels[1]] if task == 'binary' else labels
    cutoffs = np.atleast_1d(thresholds)
    fig, axes = plt.subplots(math.ceil(len(names) / 2), min(2, len(names)),
                             figsize=(11 if len(names) > 1 else 7, 3.8 * math.ceil(len(names) / 2)),
                             squeeze=False, layout='constrained')
    rows = []
    for i, (ax, label, cutoff) in enumerate(zip(axes.flat, names, cutoffs)):
        y = truth if task == 'binary' else truth[:, i]
        p = scores if task == 'binary' else scores[:, i]
        grid = np.unique(np.r_[np.linspace(0, 1, 101), cutoff])
        f1 = [f1_score(y, p >= t, zero_division=0) for t in grid]
        ax.plot(grid, f1, color='#0072B2')
        ax.axvline(cutoff, color='#D55E00', linestyle='--', label=f'Selected: {cutoff:.3f}')
        ax.set(title=f'{label} · positive n={int(y.sum()):,}', xlabel='Threshold', ylabel='Validation F1',
               xlim=(0, 1), ylim=(0, 1.03))
        ax.legend(fontsize=8)
        rows.extend({'label': label, 'threshold': float(t), 'f1': float(f), 'selected': bool(t == cutoff)} for t, f in zip(grid, f1))
    fig.suptitle(title)
    save_table(pd.DataFrame(rows), output_path)
    save_figure(fig, output_path, show=show)
    return pd.DataFrame(rows)


def evaluation_report(splits, labels, thresholds, output_dir, *, model_name,
                      task='multilabel', losses=None, loss_name=None, show=True,
                      confusion_names=None, threshold_context='Validation-tuned thresholds'):
    """Report supplied predictions only; never fits models or chooses thresholds."""
    output_dir = Path(output_dir)
    summary_rows, label_tables, cell_tables, prediction_tables = [], [], [], {}
    for split, (truth, scores) in splits.items():
        metrics, per_label, matrices, predictions = classification_metrics(truth, scores, thresholds, labels, task=task)
        if losses and losses.get(split) is not None:
            metrics['objective_loss'] = float(losses[split])
        summary_rows.append({'model': model_name, 'split': split, **metrics})
        label_tables.append(per_label.assign(model=model_name, split=split))
        frame = pd.DataFrame()
        if task == 'binary':
            frame = pd.DataFrame({'true_label': truth, 'probability': scores, 'predicted_label': predictions,
                                  'threshold': float(thresholds)})
        else:
            for i, label in enumerate(labels):
                frame[f'true_{label}'] = np.asarray(truth)[:, i]
                frame[f'probability_{label}'] = np.asarray(scores)[:, i]
                frame[f'predicted_{label}'] = predictions[:, i]
                frame[f'threshold_{label}'] = thresholds[i]
        prediction_tables[split] = frame
        save_table(frame, output_dir / f'evaluation_{split}_predictions')
        if split != 'train':
            name = (confusion_names or {}).get(split, f'{split}_confusion_matrices.png')
            cells = plot_confusion_matrices(matrices, [model_name] if task == 'binary' else labels,
                output_dir / name, title=f'{model_name}: {split} · {threshold_context}',
                class_names=labels if task == 'binary' else ('Absent', 'Present'), show=show)
            cell_tables.append(cells.assign(model=model_name, split=split))
    summary = pd.DataFrame(summary_rows)
    per_label = pd.concat(label_tables, ignore_index=True)
    cells = pd.concat(cell_tables, ignore_index=True) if cell_tables else pd.DataFrame()
    save_table(summary, output_dir / 'evaluation_summary')
    save_table(per_label, output_dir / 'evaluation_per_label')
    save_table(cells, output_dir / 'evaluation_confusion_matrices')
    (output_dir / 'evaluation_context.json').write_text(json.dumps({
        'model_name': model_name, 'task': task, 'labels': list(labels),
        'thresholds': np.asarray(thresholds, dtype=float).tolist(),
        'threshold_context': threshold_context, 'loss_name': loss_name,
    }, indent=2, allow_nan=False))
    plot_metric_comparison(summary, output_dir / 'evaluation.png', title=f'{model_name}: final evaluation',
                           task=task, loss_name=loss_name, show=show)
    plot_per_label_metrics(per_label, output_dir / 'per_label_scores.png', title=f'{model_name}: {threshold_context}', show=show)
    held_out = {k: v for k, v in splits.items() if k != 'train'}
    if held_out:
        plot_precision_recall(held_out, labels, output_dir / 'precision_recall.png',
                              title=f'{model_name}: precision–recall', task=task, show=show)
    if 'validation' in splits:
        plot_threshold_f1(*splits['validation'], labels, thresholds, output_dir / 'threshold_f1.png',
                          title=f'{model_name}: validation threshold sensitivity', task=task, show=show)
    return {'summary': summary, 'per_label': per_label, 'confusion': cells, 'predictions': prediction_tables}


def plot_optuna_history(trials, output_path, *, title, show=True, display_handle=None):
    """Plot completed trials in trial-number order, including legacy objective-only logs."""
    data = trials.copy().sort_values('number')
    data = data[data['value'].notna()]
    if 'state' in data:
        data = data[data['state'] == 'COMPLETE']
    save_table(trials, Path(output_path).parent / 'optuna_trials')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    axes[0].plot(data['number'], data['value'], 'o-', label='Trial validation micro-F1')
    axes[0].plot(data['number'], data['value'].cummax(), '--', label='Best so far')
    for ax, metric in zip(axes[1:], ['accuracy', 'log_loss']):
        for split in ['train', 'validation']:
            field = f'user_attrs_{split}_{metric}'
            if field in data:
                ax.plot(data['number'], data[field], 'o-', label=split.title())
        if not ax.lines:
            ax.text(0.5, 0.5, 'Not recorded in this run', ha='center', transform=ax.transAxes)
        ax.set_title('Exact-match accuracy' if metric == 'accuracy' else 'Unweighted log loss')
    axes[0].set_title('Validation F1 and search progress')
    for ax in axes:
        ax.set_xlabel('Optuna trial (independent model fits)')
        if ax.lines:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(title + '\nPer-trial validation-tuned thresholds; these are not training epochs')
    return save_figure(fig, output_path, show=show, display_handle=display_handle)


def timed_prediction(predict, text, *, device='cpu', warmup=True):
    """Return the original prediction unchanged and separately measured elapsed ms."""
    def synchronize():
        kind = str(device).split(':')[0]
        if kind in {'cuda', 'mps'}:
            import torch
            if kind == 'cuda':
                torch.cuda.synchronize(device)
            else:
                torch.mps.synchronize()
    if warmup:
        predict(text)
    synchronize()
    start = time.perf_counter()
    result = predict(text)
    synchronize()
    return result, (time.perf_counter() - start) * 1000


def plot_prediction(result, output_path, *, title, elapsed_ms, text=None, show=True):
    binary = 'worried_probability' in result
    if binary:
        labels = ['calm', 'worried']
        scores = [1 - result['worried_probability'], result['worried_probability']]
        thresholds = [None, result['threshold_used']]
        selected = [i == result['predicted_label'] for i in range(2)]
    else:
        labels = list(result['probabilities'])
        scores = [result['probabilities'][label] for label in labels]
        thresholds = [result['thresholds_used'][label] for label in labels]
        selected = [label in result['labels'] for label in labels]
    fig, ax = plt.subplots(figsize=(9, 2 + len(labels) * 0.6), layout='constrained')
    ax.barh(labels, scores, color=['#0072B2' if value else '#a9b6bf' for value in selected])
    for i, (score, cutoff) in enumerate(zip(scores, thresholds)):
        ax.text(min(score + 0.015, 0.94), i, f'{score:.3f}', va='center', fontsize=10)
        if cutoff is not None:
            ax.scatter(cutoff, i, marker='|', s=400, linewidths=2.5, color='#D55E00',
                       label='Decision threshold' if i == (1 if binary else 0) else None)
    ax.set(xlim=(0, 1.06), xlabel='Model score (0–1)', title=f'{title}\nWarmed inference: {elapsed_ms:.2f} ms')
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=9)
    save_figure(fig, output_path, show=show)
    table = pd.DataFrame({'label': labels, 'score': scores, 'threshold': thresholds,
                          'selected': selected, 'elapsed_ms': elapsed_ms, 'text': text})
    save_table(table, output_path)
    return table
