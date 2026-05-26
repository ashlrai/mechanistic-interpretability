from __future__ import annotations

from enum import StrEnum


class ExperimentFamily(StrEnum):
    POLYSEMANTICITY = "polysemanticity"
    POLYSEMANTICITY_SAE = "polysemanticity_sae"
    SUPERPOSITION = "superposition"
    CIRCUIT_PATCHING = "circuit_patching"
    CROSS_MODEL_REPRESENTATION_PROBE = "cross_model_representation_probe"
    ACDC_LITE = "acdc_lite"


class ExperimentBackend(StrEnum):
    TRANSFORMER_LENS = "transformerlens"
    NNSIGHT = "nnsight"
    MLX = "mlx"


SUPPORTED_EXPERIMENT_FAMILIES = tuple(family.value for family in ExperimentFamily)
SUPPORTED_EXPERIMENT_BACKENDS = tuple(backend.value for backend in ExperimentBackend)
