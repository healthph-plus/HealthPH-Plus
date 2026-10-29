"""Regenerate figures from an explicitly selected experiment, without loading models.

Example: python -m healthph_plus.render_model_reports --report-dir /run/reports
    --model-name XLM-RoBERTa --task multilabel
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .model_reporting import (
    evaluation_report, plot_confusion_matrices, plot_optuna_history,
    plot_precision_recall, plot_training_history,
)


def render_saved_reports(report_dir, *, model_name, task='multilabel', output_dir=None,
                         labels=None, loss_name=None, show=False):
    """Read only this directory; never discover a latest run or fall back to siblings.

    New self-contained prediction tables are preferred. Older tables are used only
    when they contain their own labels/scores/thresholds. Counts can still be
    rendered from older prediction tables that lack recorded decision thresholds.
    """
    source = Path(report_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    output = Path(output_dir).expanduser().resolve() if output_dir else source / 'visualizations'
    output.mkdir(parents=True, exist_ok=True)
    evaluation_context_path = source / 'evaluation_context.json'
    evaluation_context = json.loads(evaluation_context_path.read_text()) if evaluation_context_path.is_file() else {}
    if evaluation_context and evaluation_context['task'] != task:
        raise ValueError('The selected task disagrees with the saved evaluation context.')
    labels = labels or evaluation_context.get('labels') or (
        ['calm', 'worried'] if task == 'binary' else ['AURI', 'PN', 'TB', 'COVID'])
    context_path = source / 'training_context.json'
    context = json.loads(context_path.read_text()) if context_path.is_file() else {}
    config_path = source.parent / 'run_config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    objective_name = loss_name or evaluation_context.get('loss_name') or context.get('loss_name') or config.get('loss') or (
        'Class-weighted cross-entropy' if task == 'binary' and config.get('model_key') == 'mbert' else 'Recorded model objective loss')
    if config.get('smoke_only'):
        model_name += ' · SMOKE DIAGNOSTIC ONLY'
    messages, logs = [], None
    if (source / 'training_log.json').is_file():
        logs = json.loads((source / 'training_log.json').read_text())
    elif (source / 'training_log.csv').is_file():
        logs = pd.read_csv(source / 'training_log.csv').replace({np.nan: None}).to_dict('records')
    elif (source / 'poc_metrics.json').is_file():
        logs = json.loads((source / 'poc_metrics.json').read_text()).get('training', {}).get('log_history', [])
    if logs is not None:
        best_step = None
        selection_path = source.parent / 'selection.json'
        selection = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
        checkpoint = context.get('best_checkpoint') or selection.get('best_checkpoint') or config.get('best_checkpoint')
        match = re.search(r'checkpoint-(\d+)$', checkpoint or '')
        best_step = int(match.group(1)) if match else None
        plot_training_history(logs, output / 'training_curves.png', model_name=model_name,
                              loss_name=objective_name, best_step=best_step,
                              train_examples=context.get('train_examples'), validation_examples=context.get('validation_examples'),
                              task=task, show=show)
        messages.append('Redrew recorded training history; missing metrics remain unavailable.')

    splits, cutoffs = {}, None
    for split in ['train', 'validation', 'test']:
        candidates = [source / f'evaluation_{split}_predictions.csv', source / f'{split}_predictions.csv',
                      source / f'{model_name}_{split}_predictions.csv']
        prediction_file = next((p for p in candidates if p.is_file()), None)
        if prediction_file is None:
            continue
        frame = pd.read_csv(prediction_file)
        if task == 'binary':
            truth_key = 'true_label' if 'true_label' in frame else 'label'
            score_key = 'probability' if 'probability' in frame else 'worried_probability'
            cutoff_key = 'threshold' if 'threshold' in frame else 'threshold_used'
            truth, scores = frame[truth_key].to_numpy(), frame[score_key].to_numpy()
            recorded = frame[cutoff_key].unique() if cutoff_key in frame else []
            current = float(recorded[0]) if len(recorded) == 1 else None
        else:
            truth = frame[[f'true_{label}' for label in labels]].to_numpy()
            score_keys = [f'probability_{label}' if f'probability_{label}' in frame else f'proba_{label}' for label in labels]
            scores = frame[score_keys].to_numpy()
            threshold_keys = [f'threshold_{label}' for label in labels]
            current = [float(frame[key].iloc[0]) for key in threshold_keys] if (
                all(key in frame and frame[key].nunique() == 1 for key in threshold_keys)) else None
        if current is None:
            if task == 'multilabel' and all(f'pred_{label}' in frame for label in labels):
                from sklearn.metrics import multilabel_confusion_matrix
                predictions = frame[[f'pred_{label}' for label in labels]].to_numpy()
                plot_confusion_matrices(multilabel_confusion_matrix(truth, predictions), labels,
                    output / f'{split}_confusion_matrices.png', title=f'{model_name}: recorded {split} predictions', show=show)
                plot_precision_recall({split: (truth, scores)}, labels, output / f'{split}_precision_recall.png',
                                      title=model_name, task=task, show=show)
            messages.append(f'{split}: recorded thresholds unavailable; no thresholds inferred from other files.')
            continue
        if cutoffs is not None and not np.array_equal(cutoffs, current):
            raise ValueError('Prediction files disagree about frozen thresholds; select a single coherent experiment.')
        cutoffs = current
        splits[split] = truth, scores
    if splits:
        losses = {}
        summary_path = source / 'evaluation_summary.csv'
        if summary_path.is_file():
            summary = pd.read_csv(summary_path)
            if 'objective_loss' in summary:
                losses = {row['split']: row['objective_loss'] for row in summary.to_dict('records')
                          if pd.notna(row['objective_loss'])}
        evaluation_report(splits, labels, cutoffs, output, model_name=model_name, task=task,
                          losses=losses, loss_name=objective_name, show=show,
                          threshold_context=evaluation_context.get('threshold_context', 'Recorded frozen thresholds'))
        messages.append('Redrew evaluation from self-contained prediction records.')

    # ELECTRA's historical report has counts but no full prediction records.
    if not splits and task == 'multilabel':
        path = source / f'{model_name}_validation_confusion_matrix.csv'
        if path.is_file():
            counts = pd.read_csv(path).set_index('label').reindex(labels)
            matrices = [np.array([[row.tn, row.fp], [row.fn, row.tp]]) for row in counts.itertuples()]
            plot_confusion_matrices(matrices, labels, output / 'validation_confusion_matrices.png',
                                    title=f'{model_name}: recorded validation predictions', show=show)
            messages.append('Redrew saved validation confusion counts.')
    trials_path = source / 'optuna_trials.csv'
    if not trials_path.is_file():
        trials_path = source / f'{model_name}_optuna_trials.csv'
    if trials_path.is_file():
        plot_optuna_history(pd.read_csv(trials_path), output / 'optuna_progress.png', title=model_name, show=show)
        messages.append('Redrew saved Optuna trials.')
    if not messages:
        raise ValueError(f'No supported saved metrics or prediction tables in {source}')
    return messages


def main():
    # Command-line report generation must work on servers and macOS without a GUI.
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', required=True, type=Path)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--task', choices=['multilabel', 'binary'], default='multilabel')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--loss-name')
    args = parser.parse_args()
    for message in render_saved_reports(**vars(args)):
        print(message)


if __name__ == '__main__':
    main()
