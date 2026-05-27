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
    CAA_STEERING = "caa_steering"
    LOGIT_LENS = "logit_lens"
    CAUSAL_SCRUBBING = "causal_scrubbing"


class ExperimentBackend(StrEnum):
    TRANSFORMER_LENS = "transformerlens"
    NNSIGHT = "nnsight"
    MLX = "mlx"
    HUGGINGFACE = "huggingface"


# Experiment families that work with the HuggingFace backend.
HF_SUPPORTED_FAMILIES: frozenset[str] = frozenset({"activation_capture", "circuit_patching"})

# Families that require TL (gradient cache, SAE training, cross-model probing, etc.)
HF_UNSUPPORTED_FAMILIES: frozenset[str] = frozenset(
    {
        "polysemanticity_sae",
        "sae_cross_model",
        "crosscoder",
        "attribution_patching",  # needs run_with_grad_cache
        "cross_model_representation_probe",
        "acdc_edge",
        "acdc_lite",
    }
)


SUPPORTED_EXPERIMENT_FAMILIES = tuple(family.value for family in ExperimentFamily)
SUPPORTED_EXPERIMENT_BACKENDS = tuple(backend.value for backend in ExperimentBackend)
