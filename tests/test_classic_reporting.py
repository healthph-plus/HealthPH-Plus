"""Run the actual classic notebook search/fitting cell on a small offline dataset."""

import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, multilabel_confusion_matrix, precision_recall_fscore_support
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from optuna.samplers import TPESampler

from healthph_plus import model_reporting as reporting


def test_classic_notebook_preserves_objective_and_saves_detail(tmp_path, monkeypatch):
    def inexpensive_export(fig, output_path, **kwargs):
        # Actual raster/vector exports are covered by test_model_reporting.
        plt.close(fig)
    monkeypatch.setattr(reporting, 'save_figure', inexpensive_export)
    features, labels = make_multilabel_classification(n_samples=80, n_features=12,
        n_classes=4, n_labels=2, allow_unlabeled=True, random_state=42)
    features = csr_matrix(features)
    namespace = dict(globals())
    namespace.update({key: getattr(reporting, key) for key in [
        'classification_metrics', 'evaluation_report', 'save_table', 'plot_optuna_history',
        'plot_per_label_metrics', 'plot_confusion_matrices', 'save_figure']})
    namespace['display'] = lambda *args: None
    namespace.update(SEED=42, LABELS=['AURI', 'PN', 'TB', 'COVID'],
        N_TRIALS=2, OPTUNA_TIMEOUT_SECONDS=None, OUTPUT_DIR=tmp_path,
        CLASSIC_RUN_DIR=tmp_path / 'runs/current', RUN_ID='current',
        X_train=features[:48], y_train=labels[:48], X_val=features[48:64], y_val=labels[48:64],
        X_test=features[64:], y_test=labels[64:], test_df=pd.DataFrame({'text': [f'Post {i}' for i in range(16)]}))
    path = Path(__file__).resolve().parents[1] / 'notebooks/02_classic_multilabel_classifiers.ipynb'
    notebook = json.loads(path.read_text())
    for index in [6, 7, 8, 9]:
        exec(compile(''.join(notebook['cells'][index]['source']), str(path), 'exec'), namespace)
    summary = pd.read_csv(tmp_path / 'multilabel_evaluation_summary.csv')
    assert len(summary) == 4
    for row in summary.to_dict('records'):
        directory = Path(row['report_dir'])
        trials = pd.read_csv(directory / 'optuna_trials.csv')
        assert len(trials) == 2
        np.testing.assert_allclose(trials['value'], trials['user_attrs_validation_micro_f1'])
        assert not any('test_' in column for column in trials)
        assert {'user_attrs_train_log_loss', 'user_attrs_validation_accuracy'}.issubset(trials.columns)
        scores = pd.read_csv(directory / 'evaluation_summary.csv').set_index('split')
        assert set(scores.index) == {'train', 'validation', 'test'}
        assert row['test_micro_f1'] == scores.loc['test', 'micro_f1']
        assert len(pd.read_csv(directory / 'evaluation_per_label.csv')) == 12
        assert len(pd.read_csv(directory / 'evaluation_confusion_matrices.csv')) == 32
        assert (directory / 'evaluation_test_predictions.json').is_file()
        legacy_confusion = pd.read_csv(directory / f"{row['model']}_confusion_matrix.csv")
        assert {'label', 'tn', 'fp', 'fn', 'tp'}.issubset(legacy_confusion.columns)
    # Exercise all restored analysis cells against the current run references.
    for index in [11, 12, 13, 14, 15, 16]:
        exec(compile(''.join(notebook['cells'][index]['source']), str(path), 'exec'), namespace)
    assert len(namespace['detail_dirs']) == 4
