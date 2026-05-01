# Slideflow-GPL - Add-ons for the deep learning library Slideflow
# Copyright (C) 2024 James Dolezal
#
# This file is part of Slideflow-GPL.
#
# Slideflow-GPL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Slideflow-GPL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Slideflow-GPL. If not, see <https://www.gnu.org/licenses/>.

import slideflow as sf
from typing import Union, List, Tuple, Optional, TYPE_CHECKING
from slideflow import log, errors, Dataset
from slideflow.mil import MILModelConfig, TrainerConfig

if TYPE_CHECKING:
    import torch

# -----------------------------------------------------------------------------

class CLAMModelConfig(MILModelConfig):

    valid_models = ['clam_sb', 'clam_mb', 'mil_fc_mc', 'mil_fc']

    def __init__(
        self,
        model: str = 'clam_sb',
        *,
        model_size: str = 'small',
        bag_loss: str = 'ce',
        bag_weight: float = 0.7,
        dropout: bool = False,
        opt: str = 'adam',
        inst_loss: str = 'ce',
        no_inst_cluster: bool = False,
        B: int = 8,
        model_kwargs: Optional[dict] = None,
        validate: bool = True,
        **kwargs
    ):
        """Model configuration for CLAM models.

        These configuration options are identical to the options in the
        `original CLAM paper <https://arxiv.org/abs/2004.09666>`_.

        Keyword args:
            model (str): Model. Either ``'clam_sb'``, ``'clam_mb'``,
                ``'mil_fc'``, or ``'mil_fc_mc'``. Defaults to ``'clam_sb'``.
            model_size (str): Size of the model. Available sizes include:

                ``clam_sb``

                .. list-table::
                    :header-rows: 0

                    * - small
                      - [1024, 512, 256]
                    * - big
                      - [1024, 512, 384]
                    * - multiscale
                      - [2048, 512, 256]
                    * - xception
                      - [2048, 256, 128]
                    * - xception_multi
                      - [1880, 128, 64]
                    * - xception_3800
                      - [3800, 512, 256]

                ``clam_mb``

                .. list-table::
                    :header-rows: 0

                    * - small
                      - [1024, 512, 256]
                    * - big
                      - [1024, 512, 384]
                    * - multiscale
                      - [2048, 512, 256]

                ``mil_fc``

                .. list-table::
                    :header-rows: 0

                    * - small
                      - [1024, 512]

                ``mil_fc_mc``

                .. list-table::
                    :header-rows: 0

                    * - small
                      - [1024, 512]

            bag_loss (str): Primary loss function. Either 'ce' or 'svm'.
                If 'ce', the model loss function is a cross entropy loss.
                If 'svm', the model loss is topk.SmoothTop1SVM.
                Defaults to 'ce'.
            bag_weight (float): Weight of the bag loss. The total loss is
                defined0 as ``W * loss + (1 - W) * instance_loss``, where
                ``W`` is the bag weight. Defaults to 0.7
            dropout (bool): Add dropout (p=0.25) after the attention layers.
                Defaults to False.
            opt (str): Optimizer. Either 'adam' (Adam optimizer) or 'sgd'
                (Stochastic Gradient Descent). Defaults to 'adam'.
            inst_loss (str): Instance loss function. Either 'ce' or 'svm'.
                If 'ce', the instance loss is a cross entropy loss.
                If 'svm', the loss is topk.SmoothTop1SVM.
                Defaults to 'ce'.
            no_inst_cluster (bool): Disable instance-level clustering.
                Defaults to False.
            B (int): Number of positive/negative patches to sample for
                instance-level training. Defaults to 8.
            validate (bool): Validate the hyperparameter configuration.
                Defaults to True.

        """

        for argname, argval in dict(locals()).items():
            if argname not in ('kwargs', 'validate'):
                setattr(self, argname, argval)
        if kwargs and validate:
            raise errors.UnrecognizedHyperparameterError("Unrecognized parameters: {}".format(
                ', '.join(list(kwargs.keys()))
            ))
        elif kwargs:
            log.warning("Ignoring unrecognized parameters: {}".format(
                ', '.join(list(kwargs.keys()))
            ))

    @property
    def model_fn(self):
        from .model import CLAM_MB, CLAM_SB, MIL_fc_mc, MIL_fc
        model_dict = {
            'clam_sb': CLAM_SB,
            'clam_mb': CLAM_MB,
            'mil_fc_mc': MIL_fc_mc,
            'mil_fc': MIL_fc
        }
        return model_dict[self.model]

    @property
    def loss_fn(self):
        from .legacy.utils import loss_utils
        if self.bag_loss == 'ce':
            if self.model.startswith('clam'):
                return loss_utils.CrossEntropyWithInstanceLoss
            else:
                return loss_utils.CrossEntropyLoss
        else:
            raise ValueError("Unrecognized bag loss: {}".format(self.bag_loss))

    @property
    def model_type(self):
        return 'classification'

    def get_metrics(self):
        from .legacy.utils import loss_utils
        return [loss_utils.RocAuc()]

    def build_model(self, n_in, n_out, **kwargs):
        if isinstance(self.model_size, str):
            config_size = self.model_fn.sizes[self.model_size]
        else:
            config_size = self.model_size
        model_size = [n_in] + config_size[1:]
        return self.model_fn(size=model_size, n_classes=n_out, **kwargs)

    def verify_trainer(self, trainer):
        if hasattr(trainer, 'batch_size') and trainer.batch_size > 1:
            log.info(
                "CLAM models do not support batch sizes > 1; setting batch_size to 1."
            )
            trainer.batch_size = 1

    def inspect_batch(self, batch) -> Tuple[int, int]:
        """Inspect a batch to determine the input and output dimensions.."""
        bags, targets, _ = batch[0]
        n_in = bags.shape[-1]
        n_out = targets.shape[-1]
        return n_in, n_out

    def _verify_eval_params(self, **kwargs):
        """Verify evaluation parameters."""
        super()._verify_eval_params(**kwargs)

        if kwargs.get('uq'):
            raise ValueError(
                "Cannot calculate uncertainty quantification using CLAM models."
            )

    def _build_dataloader(
        self,
        bags,
        targets,
        encoder,
        *,
        dataset_kwargs = None,
        dataloader_kwargs = None
    ) -> "torch.utils.DataLoader":
        from fastai.vision.all import DataLoader
        from .data import build_clam_dataset

        dataset_kwargs = dataset_kwargs or dict()
        dataloader_kwargs = dataloader_kwargs or dict()

        dataset = build_clam_dataset(bags, targets, encoder=encoder, **dataset_kwargs)
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        return dataloader

    def predict(self, model, bags, attention=False, device=None, **kwargs):
        """Generate CLAM predictions for a list of bags."""
        from .inference import run_inference

        self._verify_eval_params(**kwargs)
        return run_inference(model, bags, attention=attention)

    def batched_predict(self, *args, **kwargs):
        """CLAM models do not support batched predictions with batch_size > 1.

        Thus, this method is equivalent to :meth:`predict`, which generates
        predictions for each bag individually.

        """
        return self.predict(*args, **kwargs)

# -----------------------------------------------------------------------------

class LegacyCLAMTrainerConfig(TrainerConfig):

    tag = 'legacy_clam'

    def __init__(
        self,
        *,
        num_splits: int = 1,   # Unused; kept for backwards compatibility
        k: int = 3,
        k_start: int = -1,
        k_end: int = -1,
        max_epochs: int = 20,
        lr: float = 1e-4,
        reg: float = 1e-5,
        label_frac: float = 1,
        weighted_sample: bool = False,
        log_data: bool = False,
        testing: bool = False,
        early_stopping: bool = False,
        subtyping: bool = False,
        seed: int = 1,
        results_dir: Optional[str] = None,  # Unused; kept for compatibility
        n_classes: Optional[int] = None,
        split_dir=None,
        data_root_dir=None,
        micro_average=False,
        **kwargs
    ):
        """Training configuration for the legacy CLAM trainer.

        This configures the legacy CLAM trainer. The FastAI trainer is
        preferred for all models, including CLAM.

        The configuration options for the legacy CLAM trainer are identical to
        the options in the `original CLAM paper <https://arxiv.org/abs/2004.09666>`_.

        Keyword args:
            k (int): Number of cross-fold splits. Defaults to 3.
            k_start (int): Starting cross-fold. Defaults to first cross-fold.
            k_end (int): Ending cross-fold. Defaults to ending after last
                cross-fold is done.
            max_epochs (int): Number of epochs to train. Defaults to 20.
            lr (float): Learning rate. Defaults to 1e-4.
            reg (float): Weight decay. Defaults to 1e-5.
            weighted_sample (bool): Equally sample from all outcome classes.
                Defaults to False.
            log_data (bool): Log to tensorboard. Defaults to False.
            early_stopping (bool): Stop the training if validation loss doesn't
                improve after 5 epochs. Will not trigger early stopping
                until epoch 50. Defaults to False.
            subtyping (bool): Whether this is a subtyping problem.
                Defaults to False.
            seed (int): Set the random seed. Defaults to 1.
            n_classes (int): Number of outcome classes. Defaults to None.
            micro_average (bool): Use micro averaging when calculate AUROC.
            **kwargs: All additional keyword arguments are passed to
                :class:`slideflow.mil.CLAMModelConfig`.
        """
        for argname, argval in dict(locals()).items():
            if argname != 'kwargs':
                setattr(self, argname, argval)
        self.model_config = CLAMModelConfig(**kwargs)

    def _to_clam_args(self):
        """Convert into CLAM_Args format (legacy support)."""
        from .legacy import CLAM_Args
        all_kw = self.to_dict()
        all_kw.update(self.model_config.to_dict())
        all_kw['model_type'] = all_kw['model']
        all_kw['drop_out'] = all_kw['dropout']
        del all_kw['model']
        del all_kw['dropout']
        del all_kw['model_kwargs']
        return CLAM_Args(**all_kw)

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        outcomes: Union[str, List[str]],
        bags: Union[str, List[str]],
        *,
        outdir: str = 'mil',
        exp_label: Optional[str] = None,
        **kwargs
    ):
        from .legacy.trainer import train_clam

        # Prepare output directory
        outdir = self.prepare_training(outcomes, exp_label, outdir)

        # Use training data as validation if no validation set is provided
        if val_dataset is None:
            sf.log.info(
                "Training without validation; metrics will be calculated on training data."
            )
            val_dataset = train_dataset

        return train_clam(
            self,
            train_dataset,
            val_dataset,
            outcomes,
            bags,
            outdir=outdir,
            **kwargs
        )

    def _verify_eval_params(self, **kwargs):
        """Verify evaluation parameters."""
        super()._verify_eval_params(**kwargs)

        if kwargs.get('aggregation_level') == 'patient':
            raise ValueError(
                "Cannot aggregate bags by patient using the legacy CLAM trainer."
            )
        if kwargs.get('uq'):
            raise ValueError(
                "Cannot calculate uncertainty quantification using the legacy CLAM trainer."
            )

