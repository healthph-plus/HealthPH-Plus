"""Offline training parity, including the weighted Trainer classes in the notebooks."""

import ast
import copy
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import (BertConfig, BertForSequenceClassification, EarlyStoppingCallback,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)

from healthph_plus.training_monitor import FullTrainingMetricsMixin, preserved_evaluation_state
from healthph_plus import training_monitor


ROOT = Path(__file__).resolve().parents[1]


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, rows, binary):
        self.rows, self.binary = rows, binary

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        return {'input_ids': torch.tensor([1, index % 8 + 2, 2]), 'attention_mask': torch.ones(3, dtype=torch.long),
                'labels': torch.tensor(index % 2) if self.binary else
                          torch.tensor([index % 2, 1, 0, (index + 1) % 2], dtype=torch.float32)}


class ValidationCounter(TrainerCallback):
    def __init__(self):
        self.evaluations = 0

    def on_evaluate(self, args, state, control, **kwargs):
        self.evaluations += 1


def notebook_trainer(filename, class_name):
    notebook = json.loads((ROOT / 'notebooks' / filename).read_text())
    namespace = {'Trainer': Trainer, 'torch': torch, 'FullTrainingMetricsMixin': FullTrainingMetricsMixin}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
                return namespace[class_name]
    raise AssertionError(class_name)


@pytest.mark.parametrize('architecture', ['electra', 'mbert', 'xlmr'])
def test_monitored_training_matches_unmonitored_training(tmp_path, monkeypatch, architecture):
    # Renderer exports are tested separately; avoid producing 18 redundant dashboards here.
    monkeypatch.setattr(training_monitor, 'plot_training_history', lambda *a, **k: None)
    binary = architecture == 'mbert'
    if architecture == 'electra':
        monitored_class = type('MonitoredTrainer', (FullTrainingMetricsMixin, Trainer), {})
        baseline_class, extras = Trainer, {}
    else:
        name, cls = ('05_binary_sentiment_mbert.ipynb', 'WeightedLossTrainer') if binary else (
            '06_xlm_roberta_disease.ipynb', 'WeightedMultilabelTrainer')
        monitored_class = notebook_trainer(name, cls)
        # Same actual weighted-loss methods, with only the monitoring mixin removed.
        baseline_class = type('UnmonitoredWeightedTrainer', (Trainer,), {
            '__init__': lambda self, *a, **kw: Trainer.__init__(self, *a, **kw),
            'compute_loss': monitored_class.compute_loss,
        })
        extras = {'class_weights': torch.tensor([1.0, 3.0])} if binary else {
            'positive_weights': torch.tensor([1.0, 2.0, 3.0, 4.0])}
    set_seed(23)
    config = BertConfig(vocab_size=16, hidden_size=8, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=12, num_labels=2 if binary else 4,
                        problem_type='single_label_classification' if binary else 'multi_label_classification')
    initial = BertForSequenceClassification(config)
    outcomes = []
    for monitored in [False, True]:
        counter = ValidationCounter()
        kwargs = dict(model=copy.deepcopy(initial),
            args=TrainingArguments(output_dir=str(tmp_path / str(monitored)), use_cpu=True,
                num_train_epochs=4, per_device_train_batch_size=2, per_device_eval_batch_size=2,
                eval_strategy='epoch', save_strategy='epoch', logging_strategy='epoch',
                learning_rate=1e-3, load_best_model_at_end=True, metric_for_best_model='score',
                greater_is_better=True, save_total_limit=1, report_to='none', seed=23,
                dataloader_pin_memory=False, disable_tqdm=True),
            train_dataset=TinyDataset(8, binary), eval_dataset=TinyDataset(4, binary),
            compute_metrics=lambda prediction: {'score': 0.5, 'accuracy': 0.5},
            callbacks=[counter, EarlyStoppingCallback(early_stopping_patience=1)])
        if monitored:
            kwargs.update(extras, monitor_dir=tmp_path / 'reports', monitor_name=architecture, monitor_show=False)
            trainer = monitored_class(**kwargs)
        else:
            trainer = baseline_class(**kwargs)
            for key, value in extras.items():
                setattr(trainer, key, value)
            if extras:
                trainer.model_accepts_loss_kwargs = False
        trainer.train()
        outcomes.append((trainer, counter))
    baseline, baseline_counter = outcomes[0]
    monitored, monitored_counter = outcomes[1]
    assert baseline_counter.evaluations == monitored_counter.evaluations == 2
    assert baseline.state.global_step == monitored.state.global_step == 8
    assert baseline.state.best_metric == monitored.state.best_metric == 0.5
    assert Path(baseline.state.best_model_checkpoint).name == Path(monitored.state.best_model_checkpoint).name
    base_losses = [r['loss'] for r in baseline.state.log_history if 'loss' in r]
    observed_losses = [r['loss'] for r in monitored.state.log_history if 'loss' in r]
    np.testing.assert_array_equal(base_losses, observed_losses)
    for key, tensor in baseline.model.state_dict().items():
        torch.testing.assert_close(tensor, monitored.model.state_dict()[key], rtol=0, atol=0)
    rows = [r for r in monitored.state.log_history if 'train_eval_examples' in r]
    assert [r['train_eval_examples'] for r in rows] == [8, 8]
    assert [r['step'] for r in rows] == [4, 8]
    assert not any('test_' in key for r in monitored.state.log_history for key in r)
    assert (tmp_path / 'reports/training_log.json').is_file()
    before = len(rows)
    monitored.evaluate()
    assert len([r for r in monitored.state.log_history if 'train_eval_examples' in r]) == before


def test_preserve_rng_modes_and_control_on_failure():
    from types import SimpleNamespace
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    model.train()
    model[1].eval()
    trainer = SimpleNamespace(model=model, args=SimpleNamespace(device=torch.device('cpu')),
                              control=SimpleNamespace(should_save=True, should_training_stop=False))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    with pytest.raises(RuntimeError):
        with preserved_evaluation_state(trainer):
            random.random(); np.random.rand(); torch.rand(1)
            model.eval()
            trainer.control.should_save = False
            raise RuntimeError('diagnostic failed')
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(1), expected[2])
    assert model.training and not model[1].training
    assert trainer.control.should_save
