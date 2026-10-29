"""Numerical and artifact checks for the notebook reporting layer."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

from healthph_plus import model_reporting as reporting
from healthph_plus.render_model_reports import render_saved_reports


LABELS = ['AURI', 'PN', 'TB', 'COVID']


def test_sparse_history_preserves_measured_steps_and_missing_values():
    logs = [
        {'step': 2, 'epoch': 0.5, 'loss': 0.8, 'learning_rate': 0.01},
        {'step': 4, 'epoch': 1, 'eval_loss': 0.7, 'eval_exact_match_accuracy': 0.75,
         'eval_micro_f1': 0.6, 'eval_runtime': 2},
        {'step': 4, 'epoch': 1, 'train_eval_loss': 0.5, 'train_eval_accuracy': 0.8,
         'train_eval_micro_f1': 0.9, 'train_eval_examples': 12},
        {'step': 4, 'train_loss': 0.81, 'train_runtime': 5},
    ]
    history = reporting.extract_training_history(logs, validation_examples=4)
    assert len(history) == 8
    assert set(history['split']) == {'optimization', 'train', 'validation'}
    assert history.query("split == 'train'")['examples'].tolist() == [12, 12, 12]
    assert history.query("split == 'validation'")['examples'].tolist() == [4, 4, 4]
    assert history.query("metric == 'accuracy'")['value'].tolist() == [0.75, 0.8]
    assert history.query("metric == 'loss'")['threshold'].isna().all()
    assert set(history.query("metric == 'micro_f1'")['step']) == {4}
    assert reporting.extract_training_history(logs[:2]).query("split == 'train'").empty


def test_multilabel_known_predictions_and_row_percentages():
    truth = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    scores = np.array([[0.9, 0.8, 0.1, 0.9], [0.8, 0.9, 0.1, 0.1],
                       [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
    summary, per_label, matrices, pred = reporting.classification_metrics(truth, scores, [0.5] * 4, LABELS)
    assert summary['exact_match_accuracy'] == 0.25
    assert summary['hamming_loss'] == 3 / 16
    np.testing.assert_array_equal(matrices[0], [[1, 1], [1, 1]])
    assert per_label['support'].tolist() == [2, 1, 0, 1]
    cells = reporting.confusion_cells(matrices, LABELS)
    assert cells.query("label == 'TB' and actual == 1")['row_percentage'].isna().all()
    assert cells.query("label == 'AURI'")['row_percentage'].tolist() == [50] * 4
    assert np.isfinite(summary['log_loss'])


@pytest.mark.parametrize('truth', [np.eye(4, dtype=int), np.zeros((3, 4), dtype=int)])
def test_perfect_and_all_negative_predictions(truth):
    summary, per_label, matrices, pred = reporting.classification_metrics(truth, truth.astype(float), [0.5] * 4, LABELS)
    assert summary['exact_match_accuracy'] == 1
    assert summary['log_loss'] < 1e-12
    assert summary['hamming_loss'] == 0
    np.testing.assert_array_equal(pred, truth)
    assert not matrices[:, 0, 1].any() and not matrices[:, 1, 0].any()
    if truth.any():
        assert summary['micro_f1'] == 1
    else:
        assert summary['micro_f1'] == 0
        assert per_label['average_precision'].isna().all()


def test_binary_threshold_ties_and_unseen_class():
    summary, per_label, matrices, pred = reporting.classification_metrics(
        np.array([0, 1, 1]), np.array([0.1, 0.5, 0.9]), 0.5, ['calm', 'worried'], task='binary')
    assert summary['worried_f1'] == summary['accuracy'] == 1
    assert per_label['support'].tolist() == [1, 2]
    assert pd.isna(per_label.loc[0, 'threshold'])
    assert per_label.loc[1, 'threshold'] == 0.5
    np.testing.assert_array_equal(matrices, [[[1, 0], [0, 2]]])
    summary, per_label, matrices, _ = reporting.classification_metrics(
        np.zeros(3), np.zeros(3), 0.5, ['calm', 'worried'], task='binary')
    assert summary['macro_f1'] == 0.5
    assert np.isnan(per_label.loc[1, 'average_precision'])


@pytest.mark.parametrize('scores,cutoffs', [([[float('nan')] * 4], [0.5] * 4),
                                          ([[0.1] * 4], [2] * 4),
                                          ([[0.1] * 3], [0.5] * 4)])
def test_invalid_scores_rejected(scores, cutoffs):
    with pytest.raises(ValueError):
        reporting.classification_metrics(np.zeros((1, 4)), scores, cutoffs, LABELS)


def test_complete_report_exports_and_offline_regeneration(tmp_path):
    truth = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    scores = 0.1 + 0.8 * truth
    report = reporting.evaluation_report(
        {'validation': (truth, scores), 'test': (truth, scores)}, LABELS, [0.5] * 4,
        tmp_path / 'original', model_name='synthetic', show=False)
    assert len(report['confusion']) == 32
    for stem in ['evaluation', 'validation_confusion_matrices', 'validation_confusion_matrices_normalized',
                 'test_confusion_matrices', 'per_label_scores', 'threshold_f1', 'precision_recall']:
        assert (tmp_path / 'original' / (stem + '.png')).stat().st_size > 1000
        assert (tmp_path / 'original' / (stem + '.svg')).stat().st_size > 1000
    threshold_rows = pd.read_csv(tmp_path / 'original/threshold_f1.csv')
    assert threshold_rows.query('selected')['threshold'].tolist() == [0.5] * 4
    assert json.loads((tmp_path / 'original/evaluation_summary.json').read_text())[0]['micro_f1'] == 1
    messages = render_saved_reports(tmp_path / 'original', model_name='synthetic', output_dir=tmp_path / 'redrawn')
    assert any('self-contained' in text for text in messages)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / 'original/evaluation_summary.csv'),
                                  pd.read_csv(tmp_path / 'redrawn/evaluation_summary.csv'))


def test_reject_mixed_threshold_artifacts(tmp_path):
    for split, cutoff in [('validation', 0.4), ('test', 0.6)]:
        frame = pd.DataFrame({f'true_{label}': [0] for label in LABELS})
        for label in LABELS:
            frame[f'probability_{label}'] = 0.1
            frame[f'threshold_{label}'] = cutoff
        frame.to_csv(tmp_path / f'evaluation_{split}_predictions.csv', index=False)
    with pytest.raises(ValueError, match='disagree'):
        render_saved_reports(tmp_path, model_name='inconsistent')


def test_historical_curves_do_not_backfill_training_f1(tmp_path):
    logs = [{'loss': 0.5, 'step': 1, 'epoch': 1}, {'eval_micro_f1': 0.7, 'step': 1, 'epoch': 1}]
    (tmp_path / 'training_log.json').write_text(json.dumps(logs))
    render_saved_reports(tmp_path, model_name='historical')
    history = pd.read_csv(tmp_path / 'visualizations/training_history.csv')
    assert history.query("split == 'train'").empty
    assert (tmp_path / 'visualizations/training_curves.svg').is_file()


def test_timed_prediction_preserves_result_and_warms_once(tmp_path):
    calls = []
    result = {'labels': ['AURI'], 'probabilities': dict(zip(LABELS, [0.8, 0.1, 0.1, 0.1])),
              'thresholds_used': dict.fromkeys(LABELS, 0.5)}
    def predict(text):
        calls.append(text)
        return result
    actual, elapsed = reporting.timed_prediction(predict, 'example')
    assert actual is result
    assert calls == ['example', 'example']
    assert elapsed >= 0
    table = reporting.plot_prediction(actual, tmp_path / 'prediction.png', title='Demo',
                                       elapsed_ms=elapsed, show=False)
    assert table['selected'].tolist() == [True, False, False, False]
    assert set(actual) == {'labels', 'probabilities', 'thresholds_used'}
