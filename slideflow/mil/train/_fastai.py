import torch
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Union, Tuple
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging import version
from fastai.metrics import Metric
from fastai.vision.all import (
    DataLoaders, Learner, SaveModelCallback, CSVLogger, EarlyStoppingCallback
)
from lifelines.utils import concordance_index
from slideflow import log
from slideflow.model import torch_utils
from .._params import TrainerConfig

# -----------------------------------------------------------------------------

def train(learner, config, callbacks=None):
    """Train an attention-based multi-instance learning model with FastAI.

    Args:
        learner (``fastai.learner.Learner``): FastAI learner.
        config (``TrainerConfig``): Trainer and model configuration.

    Keyword args:
        callbacks (list(fastai.Callback)): FastAI callbacks. Defaults to None.
    """
    cbs = [
        SaveModelCallback(fname=f"best_valid", monitor=config.save_monitor),
        CSVLogger(),
    ]
    if callbacks:
        cbs += callbacks
    if config.fit_one_cycle:
        if config.lr is None:
            lr = learner.lr_find().valley
            log.info(f"Using auto-detected learning rate: {lr}")
        else:
            lr = config.lr
        learner.fit_one_cycle(n_epoch=config.epochs, lr_max=lr, cbs=cbs)
    else:
        if config.lr is None:
            lr = learner.lr_find().valley
            log.info(f"Using auto-detected learning rate: {lr}")
        else:
            lr = config.lr
        learner.fit(n_epoch=config.epochs, lr=lr, wd=config.wd, cbs=cbs)
    return learner

class CIndex(Metric):
    @property
    def name(self):
        return 'cindex'

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds = []
        self.times = []
        self.events = []

    def accumulate(self, learn):
        preds = learn.pred.detach().cpu().view(-1) 
        times, events = learn.yb[0]
        times = times.detach().cpu().view(-1)
        events = events.detach().cpu().view(-1)
        
        if torch.isnan(preds).any() or torch.isnan(times).any() or torch.isnan(events).any():
            print("Skipping batch due to NaN values.")
            return
        
        self.preds.append(preds)
        self.times.append(times)
        self.events.append(events)

    @property
    def value(self):
        preds = torch.cat(self.preds)
        times = torch.cat(self.times)
        events = torch.cat(self.events)
        return concordance_index(times, -preds, events)


from lifelines.utils import concordance_index
from fastai.learner import Metric
import torch

class PooledMultiIntervalCIndex(Metric):
    def __init__(self, split_points):
        """
        Parameters:
        -----------
        split_points : list of float
            Time cut-points to define K intervals.
            For example, [2.0, 5.0] defines 3 intervals: (0–2], (2–5], (5+]
        """
        self.split_points = split_points
        self.reset()

    @property
    def name(self):
        return 'cindex_pooled'

    def reset(self):
        self.preds = []
        self.times = []
        self.events = []

    def accumulate(self, learn):
        preds = learn.pred.detach().cpu()   # shape: [N, K]
        times, events = learn.yb[0]
        times = times.detach().cpu().view(-1)
        events = events.detach().cpu().view(-1)

        if torch.isnan(preds).any() or torch.isnan(times).any() or torch.isnan(events).any():
            print("Skipping batch due to NaN values.")
            return

        self.preds.append(preds)
        self.times.append(times)
        self.events.append(events)

    @property
    def value(self):
        preds = torch.cat(self.preds)     # shape: (N, K)
        times = torch.cat(self.times)
        events = torch.cat(self.events)

        if preds.ndim == 1:
            return concordance_index(times, -preds, events)

        # For each patient, pick the log-risk score from the interval their time falls into
        split_points = [-float('inf')] + self.split_points + [float('inf')]
        interval_preds = torch.empty_like(times, dtype=torch.float)

        for i in range(len(split_points) - 1):
            left, right = split_points[i], split_points[i + 1]
            mask = (times > left) & (times <= right)
            interval_preds[mask] = preds[mask, i]

        return concordance_index(times.numpy(), -interval_preds.numpy(), events.numpy())



class SampleWeightedMultiIntervalCIndex(Metric):
    def __init__(self, split_points):
        """
        Parameters:
        -----------
        split_points : list of float
            Time cut-points to define K intervals.
            For example, [60] defines two intervals: (0–60], (60+]
        """
        self.split_points = split_points
        self.reset()

    @property
    def name(self):
        return 'cindex_weighted'

    def reset(self):
        self.preds = []
        self.times = []
        self.events = []

    def accumulate(self, learn):
        preds = learn.pred.detach().cpu()  # shape: [N, K]
        times, events = learn.yb[0]
        times = times.detach().cpu().view(-1)
        events = events.detach().cpu().view(-1)

        if torch.isnan(preds).any() or torch.isnan(times).any() or torch.isnan(events).any():
            print("Skipping batch due to NaN values.")
            return

        self.preds.append(preds)
        self.times.append(times)
        self.events.append(events)

    @property
    def value(self):
        preds = torch.cat(self.preds)     # shape: (N, K)
        times = torch.cat(self.times)
        events = torch.cat(self.events)

        if preds.ndim == 1:
            return concordance_index(times, -preds, events)

        # Initialize
        split_points = [-float('inf')] + self.split_points + [float('inf')]
        total_n = len(times)
        weighted_sum = 0.0

        for k in range(len(split_points) - 1):
            left, right = split_points[k], split_points[k + 1]
            mask = (times > left) & (times <= right)
            n = torch.sum(mask).item()

            if n < 2 or torch.sum(events[mask]) == 0:
                continue

            ci = concordance_index(times[mask].numpy(), -preds[mask, k].numpy(), events[mask].numpy())
            weighted_sum += ci * n

        weighted_cindex = weighted_sum / total_n
        return weighted_cindex
        
class MLTRCIndex(Metric):
    def __init__(self, split_points):
        self.split_points = split_points
        self.reset()

    @property
    def name(self):
        return 'cindex_weighted'

    def reset(self):
        self.preds = []
        self.times = []
        self.events = []

    def accumulate(self, learn):
        preds = learn.pred.detach().cpu()
        times, events = learn.yb[0]

        times = times.detach().cpu().view(-1)
        events = events.detach().cpu().view(-1)

        self.preds.append(preds)
        self.times.append(times)
        self.events.append(events)

    @property
    def value(self):
        preds = torch.cat(self.preds)
        times = torch.cat(self.times)
        events = torch.cat(self.events)

        # Cox case
        if preds.ndim == 1 or preds.shape[1] == 1:
            return concordance_index(times.numpy(), -preds.view(-1).numpy(), events.numpy())

        # MTLR case
        hazards = torch.sigmoid(preds)
        survival = torch.cumprod(1 - hazards, dim=1)

        # compute bin widths from split points
        splits = torch.tensor(self.split_points, dtype=torch.float32)
        edges = torch.cat([torch.tensor([0.0]), splits])
        dt = torch.diff(torch.cat([edges, edges[-1:] + 1]))

        dt = dt[:survival.shape[1]]

        expected_time = torch.sum(survival * dt, dim=1)

        risk = -expected_time

        return concordance_index(times.numpy(), risk.numpy(), events.numpy())
        
        
class MTLRLoss(torch.nn.Module):
    def __init__(self, split_points):
        super().__init__()
        self.split_points = torch.tensor(split_points)
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, target):
        times, events = target
        times = times.view(-1)
        events = events.view(-1)

        device = output.device
        splits = self.split_points.to(device)

        n_bins = len(splits) + 1
        y = torch.zeros((len(times), n_bins), device=device)

        for i, (t, e) in enumerate(zip(times, events)):
            idx = torch.bucketize(t, splits)

            if e == 1:
                y[i, :idx+1] = 1
            else:
                y[i, :idx] = 1

        loss = self.bce(output, y)
        return loss.mean()
                        
def train_survival(learner, config, callbacks=None):
    """Train an attention-based MIL survival model with FastAI.
    Args:
        learner (``fastai.learner.Learner``): FastAI learner.
        config (``TrainerConfigFastAI``): Trainer and model configuration.
    Keyword args:
        callbacks (list(fastai.Callback)): FastAI callbacks. Defaults to None.
    """

    cbs = [
        EarlyStoppingCallback(monitor=learner.metrics[0].name, comp=np.greater, patience=100),
        SaveModelCallback(monitor=learner.metrics[0].name, comp=np.greater, fname=f"best_valid"),
        CSVLogger(),
    ]

    if callbacks:
        cbs += callbacks
    if config.lr is None:
        lr = learner.lr_find().valley
        log.info(f"Using auto-detected learning rate: {lr}")
    else:
        lr = config.lr
    if config.fit_one_cycle:
        learner.fit_one_cycle(n_epoch=config.epochs, lr_max=lr, cbs=cbs)
    else:
        learner.fit(n_epoch=config.epochs, lr=lr, wd=config.wd, cbs=cbs)
    return learner

# -----------------------------------------------------------------------------

def build_learner(
    config: TrainerConfig,
    bags: List[str],
    targets: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for training an MIL model.

    Args:
        config (``TrainerConfig``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:
        fastai.learner.Learner, (int, int): FastAI learner and a tuple of the
            number of input features and output classes.

    """
    log.debug("Building FastAI learner")

    # Prepare device.
    device = torch_utils.get_device(device)

    # Prepare data.
    # Set oh_kw to a dictionary of keyword arguments for OneHotEncoder,
    # using the argument sparse=False if the sklearn version is <1.2
    # and sparse_output=False if the sklearn version is >=1.2.
    if version.parse(sklearn_version) < version.parse("1.2"):
        oh_kw = {"sparse": False}
    else:
        oh_kw = {"sparse_output": False}

    if config.is_classification():
        encoder = OneHotEncoder(**oh_kw).fit(unique_categories.reshape(-1, 1))
    else:
        encoder = None

    # Build the dataloaders.
    train_dl = config.build_train_dataloader(
        bags[train_idx],
        targets[train_idx],
        encoder=encoder,
        dataloader_kwargs=dict(
            num_workers=1,
            device=device,
            pin_memory=True,
            **dl_kwargs
        )
    )
    val_dl = config.build_val_dataloader(
        bags[val_idx],
        targets[val_idx],
        encoder=encoder,
        dataloader_kwargs=dict(
            shufle=False,
            num_workers=8,
            persistent_workers=True,
            device=device,
            pin_memory=False,
            **dl_kwargs
        )
    )

    # Prepare model.
    batch = train_dl.one_batch()
    n_in, n_out = config.inspect_batch(batch)
    model = config.build_model(n_in, n_out).to(device)

    if hasattr(model, 'relocate'):
        model.relocate()

    # Loss should weigh inversely to class occurences.
    if config.is_classification() and config.weighted_loss:
        counts = pd.value_counts(targets[train_idx])
        weights = counts.sum() / counts
        weights /= weights.sum()
        weights = torch.tensor(
            list(map(weights.get, encoder.categories_[0])), dtype=torch.float32
        ).to(device)
        loss_kw = {"weight": weights}
    else:
        loss_kw = {}
    loss_func = config.loss_fn(**loss_kw)

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    learner = Learner(dls, model, loss_func=loss_func, metrics=config.get_metrics(), path=outdir)

    return learner, (n_in, n_out)


def build_learner_survival(
    config: TrainerConfig,
    bags: List[str],
    targets: npt.NDArray,
    events: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    clin_vars: Optional[npt.NDArray] = None,
    risk = None,
    split_times = None,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for training an MIL model.

    Args:
        config (``TrainerConfig``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:
        fastai.learner.Learner, (int, int): FastAI learner and a tuple of the
            number of input features and output classes.

    """
    log.debug("Building FastAI learner")

    # Prepare device.
    device = torch_utils.get_device(device)

    # Prepare data.
    # Set oh_kw to a dictionary of keyword arguments for OneHotEncoder,
    # using the argument sparse=False if the sklearn version is <1.2
    # and sparse_output=False if the sklearn version is >=1.2.
    if version.parse(sklearn_version) < version.parse("1.2"):
        oh_kw = {"sparse": False}
    else:
        oh_kw = {"sparse_output": False}

    if config.is_classification():
        encoder = OneHotEncoder(**oh_kw).fit(unique_categories.reshape(-1, 1))
    else:
        encoder = None

    # Build the dataloaders.
    train_clin_vars = None
    val_clin_vars = None
    if not (clin_vars is None):
        print("Training model with clin variables")
        train_clin_vars = clin_vars[train_idx]
        val_clin_vars = clin_vars[val_idx]
    if not (risk is None):
      print("Training with risk tertiles")
      risk = risk[train_idx]
    train_dl = config.build_train_dataloader_surv(
        bags[train_idx],
        targets[train_idx],
        events[train_idx],
        clin_vars = train_clin_vars,
        risk = risk,
        encoder=encoder,
        dataloader_kwargs=dict(
            num_workers=1,
            device=device,
            pin_memory=True,
            **dl_kwargs
        )
    )
    val_dl = config.build_val_dataloader_surv(
        bags[val_idx],
        targets[val_idx],
        events[val_idx],
        clin_vars = val_clin_vars,
        encoder=encoder,
        dataloader_kwargs=dict(
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
            device=device,
            pin_memory=False,
            **dl_kwargs
        )
    )

    # Prepare model.
    #batch = train_dl.one_batch()
    
    if config.tertile:
          batch = next(iter(train_dl))
    else:
          batch = train_dl.one_batch()    
    if not (clin_vars is None):
        n_in, n_out = config.inspect_batch(batch[1:])
        if not config.n_intermediate:
            n_intermediate = len(clin_vars[0])
        mil_model = config.build_model(n_in, config.n_intermediate).to(device)
        model = ClinicalFusionModel(mil_model = mil_model,
                                    num_clin = len(clin_vars[0]), 
                                    num_path = config.n_intermediate,
                                    hidden_dim = len(clin_vars[0]) + config.n_intermediate,
                                    fusion_type = config.fusion_type,
                                    n_hidden = config.n_hidden,
                                    n_out = n_out).to(device)
    else:
        print(batch)
        n_in, n_out = config.inspect_batch(batch)
        if split_times:
            n_out = len(split_times) + 1
        model = config.build_model(n_in, n_out).to(device)
    if hasattr(model, 'relocate'):
        model.relocate()

    # Loss should weigh inversely to class occurences.
    if config.is_classification() and config.weighted_loss:
        counts = pd.value_counts(targets[train_idx])
        weights = counts.sum() / counts
        weights /= weights.sum()
        weights = torch.tensor(
            list(map(weights.get, encoder.categories_[0])), dtype=torch.float32
        ).to(device)
        loss_kw = {"weight": weights}
    else:
        loss_kw = {}
    loss_func = config.loss_fn(**loss_kw)

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    if config.survival == "mtlr":
      learner = Learner(dls, model, loss_func = MLTRLoss(split_times), metrics = [MLTRCIndex(split_times)], path = outdir)
    else:   
      if split_times:
          learner = Learner(dls, model, loss_func=loss_func, metrics=[SampleWeightedMultiIntervalCIndex(split_times)], path=outdir)
      else:
          learner = Learner(dls, model, loss_func=loss_func, metrics=[CIndex()], path=outdir)

    return learner, (n_in, n_out)
    
    
    
    
    
import torch
import torch.nn as nn
from slideflow.model.torch_utils import get_device

class ClinicalFusionModel(nn.Module):
    """
    Wraps an existing MIL model (like Attention_MIL) and merges the
    additional variables into the final prediction pipeline.
    """

    def __init__(
        self,
        mil_model: nn.Module,
        num_clin: int,
        num_path: int,
        hidden_dim: int,
        fusion_type: str = 'concat',
        n_hidden = None,
        n_out: int = 2,  # or 1 for regression
    ):
        super().__init__()
        self.mil_model = mil_model   # e.g. an instance of Attention_MIL or Transformer
        self.fusion_type = fusion_type
        self.num_clin = num_clin
        self.num_path = num_path
        # If bilinear, create a matrix for bilinear interaction
        if fusion_type == 'bilinear':
            # map (hdim x hdim) -> final hdim
            self.fusion = nn.Bilinear(num_clin, num_path, hidden_dim)
        elif fusion_type == 'concat':
            self.fusion = nn.Linear(num_clin + num_path, hidden_dim)
        else:
            raise ValueError("Unrecognized fusion type")

        # Final classifier
        if n_hidden:
            self.classifier = n_hidden
        else:
                
            self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_out)
            )

    def forward(self, clin_vars, *args, return_attention=False, **kwargs):
        """
        For a typical MIL pipeline, the model expects either:
          - attention_mil: (bags, lens, [optionally more])
          - transformer: (x, coords=..., register_hook=..., return_attention=...)
        So we pass everything except the extra vars to the MIL model, 
        then handle them here.
        """
        #features, clin_vars = batch
        mil_out = self.mil_model(*args, return_attention=return_attention, **kwargs)

        if return_attention:
            logits_mil, attention = mil_out
        else:
            logits_mil = mil_out
            attention = None
        
        if self.fusion_type == 'concat':
            fused = torch.cat([logits_mil, clin_vars], dim=-1)
            # Then reduce to hidden_dim again if you want
            fused = self.fusion(fused)
        elif self.fusion_type == 'bilinear':
            fused = self.fusion(logits_mil, clin_vars)
        else:
            raise ValueError("Unrecognized fusion type")

        # 6) Final classification/regression
        logits = self.classifier(fused)

        final_out = logits

        if return_attention:
            return final_out, attention
        else:
            return final_out

    def relocate(self):
        """Move model to GPU. Required for FastAI compatibility."""
        device = get_device()
        self.to(device)
        self.mil_model = self.mil_model.to(device)

    def plot(*args, **kwargs):
        pass
 
    def calculate_attention(self, **kwargs):
        return self.mil_model.calculate_attention(kwargs)
