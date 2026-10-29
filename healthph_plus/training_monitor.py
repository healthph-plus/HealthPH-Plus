"""A Trainer mixin that measures full-training-set metrics at validation events."""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import torch

from .model_reporting import plot_training_history, save_table


@contextmanager
def preserved_evaluation_state(trainer):
    """Diagnostics must not change training RNG, module modes, or callback control."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mps_rng = torch.mps.get_rng_state() if trainer.args.device.type == 'mps' else None
    control = copy.copy(trainer.control)
    modes = {module: module.training for module in trainer.model.modules()}
    handler = getattr(trainer, 'callback_handler', None)
    eval_dataloader = getattr(handler, 'eval_dataloader', None)
    try:
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)
        for module, mode in modes.items():
            module.training = mode
        if handler is not None:
            handler.eval_dataloader = eval_dataloader
        trainer.control = control


class FullTrainingMetricsMixin:
    """Place before Trainer in the MRO, including custom weighted-loss Trainers.

    The extra predict pass emits on_predict, never on_evaluate. Only validation
    metrics are returned to Trainer's checkpoint, scheduler, and early-stop logic.
    """

    def __init__(self, *args, monitor_dir, monitor_name, monitor_loss_name='Loss',
                 monitor_task='multilabel', monitor_show=True, **kwargs):
        self.monitor_dir = Path(monitor_dir)
        self.monitor_name = monitor_name
        self.monitor_loss_name = monitor_loss_name
        self.monitor_task = monitor_task
        self.monitor_show = monitor_show
        self._monitor_display = None
        self._monitor_active = False
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                   metric_key_prefix=metric_key_prefix)
        if self.is_in_train and metric_key_prefix == 'eval' and not self._monitor_active:
            self._monitor_active = True
            try:
                with preserved_evaluation_state(self):
                    output = super().predict(self.train_dataset, ignore_keys=ignore_keys,
                                             metric_key_prefix='train_eval')
                    self.log({**output.metrics, 'train_eval_examples': len(self.train_dataset)})
                    del output
                    self.save_training_report()
            finally:
                self._monitor_active = False
        return metrics

    def save_training_report(self, output_path=None):
        """Also call after train() to mark the finally selected checkpoint."""
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        history = self.state.log_history
        save_table(pd.DataFrame(history), self.monitor_dir / 'training_log')
        # Keep Trainer's sparse native records rather than the rectangular table.
        (self.monitor_dir / 'training_log.json').write_text(json.dumps(history, indent=2, allow_nan=False))
        (self.monitor_dir / 'training_context.json').write_text(json.dumps({
            'model_name': self.monitor_name, 'task': self.monitor_task,
            'loss_name': self.monitor_loss_name, 'threshold': 0.5,
            'train_examples': len(self.train_dataset), 'validation_examples': len(self.eval_dataset),
            'best_checkpoint': self.state.best_model_checkpoint,
        }, indent=2, allow_nan=False))
        match = re.search(r'checkpoint-(\d+)$', self.state.best_model_checkpoint or '')
        self._monitor_display = plot_training_history(
            history, output_path or self.monitor_dir / 'training_curves.png',
            model_name=self.monitor_name, loss_name=self.monitor_loss_name,
            best_step=int(match.group(1)) if match else None,
            train_examples=len(self.train_dataset), validation_examples=len(self.eval_dataset),
            task=self.monitor_task, show=self.monitor_show, display_handle=self._monitor_display)
