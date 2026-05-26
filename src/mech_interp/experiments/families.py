from __future__ import annotations

from enum import StrEnum


class ExperimentFamily(StrEnum):
    POLYSEMANTICITY = "polysemanticity"
    POLYSEMANTICITY_SAE = "polysemanticity_sae"
    SUPERPOSITION = "superposition"
    CIRCUIT_PATCHING = "circuit_patching"
    CROSS_MODEL_REPRESENTATION_PROBE = "cross_model_representation_probe"
    ACDC_LITE = "acdc_lite"
    ACDC_EDGE = "acdc_edge"
    REFUSAL_DIRECTION = "refusal_direction"
    SAE_CROSS_MODEL = "sae_cross_model"
    DIRECT_LOGIT_ATTRIBUTION = "direct_logit_attribution"
    SPARSE_PROBING = "sparse_probing"
    ATTRIBUTION_PATCHING = "attribution_patching"
    CROSSCODER = "crosscoder"


class ExperimentBackend(StrEnum):
    TRANSFORMER_LENS = "transformerlens"
    NNSIGHT = "nnsight"
    MLX = "mlx"


SUPPORTED_EXPERIMENT_FAMILIES = tuple(family.value for family in ExperimentFamily)
SUPPORTED_EXPERIMENT_BACKENDS = tuple(backend.value for backend in ExperimentBackend)
