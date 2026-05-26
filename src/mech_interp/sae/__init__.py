"""Sparse autoencoder primitives for mechanistic interpretability experiments.

Provides a Top-K SAE (Gao et al., 2024) trained on captured model activations,
along with a small analysis pass that surfaces per-feature top-activating prompts.
"""

from mech_interp.sae.analysis import FeatureAnalysis, FeatureRecord, compute_feature_analysis
from mech_interp.sae.model import TopKSAE
from mech_interp.sae.trainer import TrainingHistory, save_sae_weights, train_top_k_sae

__all__ = [
    "FeatureAnalysis",
    "FeatureRecord",
    "TopKSAE",
    "TrainingHistory",
    "compute_feature_analysis",
    "save_sae_weights",
    "train_top_k_sae",
]
