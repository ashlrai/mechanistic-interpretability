"""Steering-vector registry — pre-extracted directions for apply-without-training.

Each entry in STEERING_REGISTRY points to a safetensors file that contains a
single normalised direction tensor under the key ``"direction"``.  Pair with
``load_steering_vector`` to get a ``(Tensor, metadata_dict)`` tuple ready for
insertion into TransformerLens hooks.

Extraction provenance:
  refusal-qwen-2.5-1.5b-l10  -- copied from artifacts/run-000070/direction.safetensors
  sentiment-gpt2-medium-l8    -- extracted by scripts/extract_sentiment_direction.py
  helpfulness-qwen-2.5-0.5b-l8 -- extracted by scripts/extract_helpfulness_direction.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

# ---------------------------------------------------------------------------
# Descriptor dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SteeringVectorDescriptor:
    name: str
    """Canonical registry key."""
    model_name: str
    """TransformerLens / HuggingFace model name."""
    hook_site: str
    """TransformerLens hook point where the vector is added."""
    direction_norm: float
    """L2 norm of the raw (un-normalised) direction before normalisation."""
    description: str
    license: str
    source_run_id: int | None = None
    """Local run that produced this vector (if any)."""
    source_paper: str | None = None
    hf_repo: str | None = None
    """Optional HF repo to download from if local_path is absent."""
    local_path: Path | None = None
    """Path to the .safetensors file, relative to the project root."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STEERING_REGISTRY: dict[str, SteeringVectorDescriptor] = {
    "refusal-qwen-2.5-1.5b-l10": SteeringVectorDescriptor(
        name="refusal-qwen-2.5-1.5b-l10",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        hook_site="blocks.10.hook_resid_post",
        direction_norm=11.73,
        description=(
            "Refusal direction extracted from Qwen2.5-1.5B-Instruct via Arditi/RepE "
            "mean-difference on 5 harmful vs 5 harmless prompts. "
            "Extraction quality (projection margin) 4.11 — high. "
            "From investigation #1 (run 70). "
            "Use coefficient -3 to suppress refusal, +3 to amplify it."
        ),
        license="research-only",
        source_run_id=70,
        source_paper="Arditi et al. 2024",
        local_path=Path("data/steering/refusal_qwen_2.5_1.5b_l10.safetensors"),
    ),
    "sentiment-gpt2-medium-l8": SteeringVectorDescriptor(
        name="sentiment-gpt2-medium-l8",
        model_name="gpt2-medium",
        hook_site="blocks.8.hook_resid_pre",
        direction_norm=15.36,  # raw norm; see sidecar JSON for extraction_quality
        description=(
            "Sentiment direction extracted from gpt2-medium via mean-difference "
            "on 10 positive vs 10 negative movie-review prompts at layer 8. "
            "Use positive coefficients to steer toward positive sentiment, "
            "negative coefficients toward negative. "
            "Extracted by scripts/extract_sentiment_direction.py."
        ),
        license="research-only",
        source_paper="Zou et al. 2023",
        local_path=Path("data/steering/sentiment_gpt2_medium_l8.safetensors"),
    ),
    "helpfulness-qwen-2.5-0.5b-l8": SteeringVectorDescriptor(
        name="helpfulness-qwen-2.5-0.5b-l8",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        hook_site="blocks.8.hook_resid_post",
        direction_norm=5.54,  # raw norm; see sidecar JSON for extraction_quality
        description=(
            "Helpfulness direction extracted from Qwen2.5-0.5B-Instruct via "
            "mean-difference on 8 helpful vs 8 evasive response prompts at layer 8. "
            "Use positive coefficients to steer toward more helpful completions. "
            "Extracted by scripts/extract_helpfulness_direction.py."
        ),
        license="research-only",
        source_paper="Arditi et al. 2024",
        local_path=Path("data/steering/helpfulness_qwen_2.5_0.5b_l8.safetensors"),
    ),
}


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------


def load_steering_vector(
    name: str,
    *,
    device: str = "cpu",
    base_dir: Path | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Load a steering vector by registry name.

    Parameters
    ----------
    name:
        Registry key (e.g. ``"refusal-qwen-2.5-1.5b-l10"``).
    device:
        Torch device string.
    base_dir:
        Project root to resolve relative ``local_path`` entries.  Defaults to
        the directory three levels above this file (the project root).

    Returns
    -------
    (direction, metadata)
        ``direction`` is a 1-D unit-norm float32 Tensor of shape ``(d_model,)``.
        ``metadata`` contains all registry fields plus any sidecar JSON data.
    """
    if name not in STEERING_REGISTRY:
        available = ", ".join(sorted(STEERING_REGISTRY))
        raise KeyError(f"Unknown steering vector '{name}'. Available: {available}.")

    descriptor = STEERING_REGISTRY[name]

    # Resolve path relative to project root
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent.parent  # src/mech_interp/steering -> root

    if descriptor.local_path is not None:
        resolved = base_dir / descriptor.local_path
    else:
        resolved = None

    if resolved is None or not resolved.exists():
        # Fall back to HuggingFace Hub download when local file is absent
        if descriptor.hf_repo is not None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface_hub is required to download steering vectors. "
                    "Run: uv sync --extra interp"
                ) from exc
            weights_filename = (
                descriptor.local_path.name
                if descriptor.local_path is not None
                else "direction.safetensors"
            )
            local_file = hf_hub_download(
                repo_id=descriptor.hf_repo,
                filename=weights_filename,
            )
            resolved = Path(local_file)
        elif resolved is None:
            raise ValueError(
                f"Steering vector '{name}' has no local_path and no hf_repo set. "
                "Set local_path or run the extraction script."
            )
        else:
            raise FileNotFoundError(
                f"Steering vector file not found: {resolved}\n"
                f"Run the extraction script to produce it, or check that "
                f"data/steering/ was committed to the repository."
            )

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to load steering vectors. "
            "Run: uv sync --extra interp"
        ) from exc

    state = load_file(str(resolved))
    if "direction" not in state:
        raise ValueError(
            f"Safetensors file '{resolved}' does not contain a 'direction' key. "
            f"Available keys: {sorted(state.keys())}"
        )

    import torch
    direction: Tensor = state["direction"].to(torch.float32).to(device)

    # Load sidecar JSON if present
    sidecar_path = resolved.with_suffix(".safetensors.json")
    sidecar: dict[str, Any] = {}
    if sidecar_path.exists():
        import json
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    metadata: dict[str, Any] = {
        "name": descriptor.name,
        "model_name": descriptor.model_name,
        "hook_site": descriptor.hook_site,
        "direction_norm": descriptor.direction_norm,
        "description": descriptor.description,
        "license": descriptor.license,
        "source_run_id": descriptor.source_run_id,
        "source_paper": descriptor.source_paper,
        "local_path": str(descriptor.local_path),
        **sidecar,
    }

    return direction, metadata
