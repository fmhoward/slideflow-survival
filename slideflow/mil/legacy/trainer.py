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

"""Legacy trainer for CLAM models."""

import os
import numpy as np
import slideflow as sf
from os.path import join, exists
from typing import Union, List
from slideflow import Dataset, log
from slideflow.util import path_to_name
from os.path import join

from slideflow.mil.train import _log_mil_params
from slideflow.mil.eval import predict_mil, generate_attention_heatmaps
from slideflow.mil.utils import _export_attention

from ..config import LegacyCLAMTrainerConfig

# -----------------------------------------------------------------------------

def train_clam(
    config: LegacyCLAMTrainerConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *,
    outdir: str = 'mil',
    attention_heatmaps: bool = False,
    **heatmap_kwargs
) -> None:
    """Train a CLAM model from layer activations exported with
    :meth:`slideflow.project.generate_features_for_clam`.

    See :ref:`mil` for more information.

    Args:
        train_dataset (:class:`slideflow.Dataset`): Training dataset.
        val_dataset (:class:`slideflow.Dataset`): Validation dataset.
        outcomes (str): Outcome column (annotation header) from which to
            derive category labels.
        bags (str): Either a path to directory with \*.pt files, or a list
            of paths to individual \*.pt files. Each file should contain
            exported feature vectors, with each file containing all tile
            features for one patient.

    Keyword args:
        outdir (str): Directory in which to save model and results.
        exp_label (str): Experiment label, used for naming the subdirectory
            in the ``outdir`` folder, where training history
            and the model will be saved.
        clam_args (optional): Namespace with clam arguments, as provided
            by :func:`slideflow.clam.get_args`.
        attention_heatmaps (bool): Generate attention heatmaps for slides.
            Defaults to False.
        interpolation (str, optional): Interpolation strategy for smoothing
            attention heatmaps. Defaults to 'bicubic'.
        cmap (str, optional): Matplotlib colormap for heatmap. Can be any
            valid matplotlib colormap. Defaults to 'inferno'.
        norm (str, optional): Normalization strategy for assigning heatmap
            values to colors. Either 'two_slope', or any other valid value
            for the ``norm`` argument of ``matplotlib.pyplot.imshow``.
            If 'two_slope', normalizes values less than 0 and greater than 0
            separately. Defaults to None.

    Returns:
        None

    """
    from .datasets import CLAM_Dataset
    from . import train

    # Set up results directory
    if outdir:
        results_dir = join(outdir, 'results')
        if not exists(results_dir):
            os.makedirs(results_dir)

    # Set up labels.
    labels, unique_train = train_dataset.labels(outcomes, format='name', use_float=False)
    val_labels, unique_val = val_dataset.labels(outcomes, format='name', use_float=False)
    labels.update(val_labels)
    unique_labels = np.unique(unique_train + unique_val)
    label_dict = dict(zip(unique_labels, range(len(unique_labels))))

    # Prepare CLAM arguments.
    clam_args = config._to_clam_args()
    clam_args.results_dir = results_dir
    clam_args.n_classes = len(unique_labels)

    # Set up bags.
    if isinstance(bags, str):
        train_bags = train_dataset.get_bags(bags)
        val_bags = val_dataset.get_bags(bags)
    else:
        train_bags = val_bags = bags

    # Write slide/bag manifest
    if outdir:
        sf.util.log_manifest(
            [b for b in train_bags],
            [b for b in val_bags],
            labels=labels,
            filename=join(outdir, 'slide_manifest.csv'),
        )

    # Set up datasets.
    train_mil_dataset = CLAM_Dataset(
        train_bags,
        annotations=train_dataset.filtered_annotations,
        label_col=outcomes,
        label_dict=label_dict
    )
    val_mil_dataset = CLAM_Dataset(
        val_bags,
        annotations=val_dataset.filtered_annotations,
        label_col=outcomes,
        label_dict=label_dict
    )

    # Get base CLAM args/settings if not provided.
    num_features = train_mil_dataset.detect_num_features()
    if isinstance(clam_args.model_size, str):
        model_size = config.model_fn.sizes[clam_args.model_size]
    else:
        model_size = clam_args.model_size
    if model_size[0] != num_features:
        _old_size = model_size[0]
        model_size[0] = num_features
        clam_args.model_size = model_size
        log.warn(
            f"First dimension of model size ({_old_size}) does not "
            f"match features ({num_features}). Updating model size to "
            f"{clam_args.model_size}. "
        )

    # Save clam settings
    if outdir:
        sf.util.write_json(clam_args.to_dict(), join(outdir, 'experiment.json'))

    # Save MIL settings
    _log_mil_params(config, outcomes, unique_labels, bags, num_features, clam_args.n_classes, outdir)

    # Run CLAM
    datasets = (train_mil_dataset, val_mil_dataset, val_mil_dataset)
    model, results, test_auc, val_auc, test_acc, val_acc = train(
        datasets, 0, clam_args
    )

    # Generate validation predictions
    df, attention = predict_mil(
        model,
        config,
        dataset=val_dataset,
        outcomes=outcomes,
        bags=bags,
        attention=True
    )
    if outdir:
        pred_out = join(outdir, 'results', 'predictions.parquet')
        df.to_parquet(pred_out)
        log.info(f"Predictions saved to [green]{pred_out}[/]")

    # Print categorical metrics, including per-category accuracy
    outcome_name = outcomes if isinstance(outcomes, str) else '-'.join(outcomes)
    df.rename(
        columns={c: f"{outcome_name}-{c}" for c in df.columns if c != 'slide'},
        inplace=True
    )
    sf.stats.metrics.classification_metrics(df, level='slide', data_dir=outdir)

    # Attention heatmaps
    if isinstance(bags, str):
        val_bags = val_dataset.get_bags(bags)
    else:
        val_bags = np.array(bags)

    # Export attention to numpy arrays
    if attention and outdir:
        _export_attention(
            join(outdir, 'attention'),
            attention,
            [path_to_name(b) for b in val_bags]
        )

    # Save attention heatmaps
    if attention and attention_heatmaps and outdir:
        assert len(val_bags) == len(attention)
        generate_attention_heatmaps(
            outdir=join(outdir, 'heatmaps'),
            dataset=val_dataset,
            bags=val_bags,
            attention=attention,
            **heatmap_kwargs
        )
