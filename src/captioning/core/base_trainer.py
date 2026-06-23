import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


def make_summary_writer(logdir_prefix: str, log_name: str):
    data_path = os.path.join(os.getcwd(), "runs")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + "_" + log_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)

    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    return SummaryWriter(logdir, flush_secs=1, max_queue=1)


class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        max_steps: int,
        eval_every_n_steps: int,
        logger: SummaryWriter = None,
        scheduler: _LRScheduler = None,
        config: dataclass = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_steps = max_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.logger = logger
        self.scheduler = scheduler
        self.config = config

        self.gradient_accumulation_steps = getattr(
            config, "gradient_accumulation_steps", 1
        )

    @abstractmethod
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, Any]:
        pass

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        batch = [item.to(self.device) for item in batch]
        metrics = self.training_step(batch)
        loss = metrics["loss"] / self.gradient_accumulation_steps
        loss.backward()
        if (self._step + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        out_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                out_metrics[key] = value.item()
            else:
                out_metrics[key] = value
        return out_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_metrics = {}
        for batch in self.val_loader:
            batch = [item.to(self.device) for item in batch]
            metrics = self.validation_step(batch)

            for key, value in metrics.items():
                if key not in val_metrics:
                    val_metrics[key] = []
                val_metrics[key].append(
                    value.item() if isinstance(value, torch.Tensor) else value
                )

        return {key: sum(values) / len(values) for key, values in val_metrics.items()}

    def fit(self):
        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.max_steps, unit="step", desc="Training", mininterval=0)

        for self._step in range(self.max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Train step
            train_metrics = self.train_step(batch)

            pbar.set_description(
                f"Training Step: {self._step + 1}/{self.max_steps} | Loss: {train_metrics['loss']:.3f}"
            )

            # log metrics
            if self.logger is not None:
                for key, value in train_metrics.items():
                    self.logger.add_scalar(
                        f"{self.config.exp_name}/train_{key}", value, self._step
                    )

                # TODO: log the "base" learning rate
                self.logger.add_scalar(
                    f"{self.config.exp_name}/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    self._step,
                )

            # Validation
            if self._step % self.eval_every_n_steps == 0:
                pbar.set_description(
                    f"Validating Step: {self._step + 1}/{self.max_steps}"
                )
                val_metrics = self.validate()
                if self.logger is not None:
                    for key, value in val_metrics.items():
                        self.logger.add_scalar(
                            f"{self.config.exp_name}/val_{key}", value, self._step
                        )
                yield self._step, train_metrics, val_metrics

            pbar.update(1)

        # Final validation
        val_metrics = self.validate()
        if self.logger is not None:
            pbar.set_description("Final Validation")
            for key, value in val_metrics.items():
                self.logger.add_scalar(
                    f"{self.config.exp_name}/val_{key}", value, self.max_steps
                )

        pbar.close()
        yield self.max_steps, train_metrics, val_metrics
