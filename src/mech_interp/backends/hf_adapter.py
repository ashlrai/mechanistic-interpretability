"""HuggingFace universal causal-LM backend.

Loads *any* ``AutoModelForCausalLM`` checkpoint and wires up activation
capture / patching via PyTorch ``register_forward_hook``.  This is the
opt-in fallback for models that are not in TransformerLens's
``OFFICIAL_MODEL_NAMES``.

Usage (YAML)::

    backend: huggingface
    parameters:
      model_name: meta-llama/Llama-3.2-1B
      device: cpu          # optional, default auto
      trust_remote_code: false  # optional

Site names follow the same ``blocks.L.hook_*`` convention as TransformerLens.
Pass ``architecture`` (or set ``backend_config.architecture``) to select the
translation map; if omitted the backend auto-detects from ``config.model_type``.

Raw HF dotted paths (e.g. ``model.layers.10.mlp``) are accepted as-is.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from mech_interp.backends.hf_site_translator import resolve_architecture, translate_hook_site
from mech_interp.types import (
    ActivationPatchRequest,
    ActivationPatchSiteResult,
    CrossModelProbeRequest,
    CrossModelProbeResult,
)


def _require_transformers() -> Any:
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "Install 'transformers' with `uv sync --extra interp` before using "
            "the HuggingFace backend."
        ) from exc


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "Install 'torch' with `uv sync --extra interp` before using "
            "the HuggingFace backend."
        ) from exc


# ---------------------------------------------------------------------------
# Helper: resolve a dotted module path on a model
# ---------------------------------------------------------------------------

def _get_module(model: Any, dotted_path: str) -> Any:
    """Retrieve a submodule by dotted path, e.g. ``model.layers.10.mlp``.

    SECURITY: rejects dunder attributes (``__class__``, ``__globals__``, etc.)
    so a malicious experiment YAML cannot walk from a model into arbitrary
    Python attributes via the hook-site translator. Also caps the path depth
    at 12 to bound walk cost.
    """
    parts = dotted_path.split(".")
    if len(parts) > 12:
        raise ValueError(
            f"Hook-site path '{dotted_path}' has {len(parts)} components; "
            "maximum 12 to bound traversal cost."
        )
    obj = model
    for part in parts:
        if part.startswith("__") or part.endswith("__"):
            raise ValueError(
                f"Hook-site path '{dotted_path}' contains a dunder component "
                f"('{part}'); these are rejected to prevent attribute-walk attacks."
            )
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Helper: tokenize + forward pass
# ---------------------------------------------------------------------------

def _encode(tokenizer: Any, prompts: list[str], device: Any) -> Any:
    _require_torch()
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {k: v.to(device) for k, v in enc.items()}


def _logits_at_position(logits: Any, position: int) -> list[float]:
    """Extract a single sequence position from (batch, seq, vocab) logits."""
    if logits.ndim == 3:
        selected = logits[0, position, :]
    elif logits.ndim == 2:
        selected = logits[position, :]
    else:
        selected = logits
    return [float(v) for v in selected.detach().cpu().tolist()]


# ---------------------------------------------------------------------------
# Main backend
# ---------------------------------------------------------------------------

class HuggingFaceBackend:
    """Universal HuggingFace causal-LM instrumented backend.

    Satisfies the ``InstrumentedModelBackend`` protocol via forward hooks.
    """

    name = "huggingface"

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "auto",
        trust_remote_code: bool = False,
        architecture: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self._architecture_override = architecture
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._architecture: str | None = None  # resolved after load()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        transformers = _require_transformers()
        torch = _require_torch()

        # Resolve device.
        if self.device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = self.device

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }

        try:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs
            )
        except OSError as exc:
            # SECURITY: do NOT silently retry with trust_remote_code=True. If the
            # user didn't opt in, a malicious HF repo could ship a modeling_*.py
            # that executes arbitrary code at load time. Raise a clear error
            # pointing the user at the security implication so they can opt in
            # consciously by setting trust_remote_code: true in their YAML.
            if not self.trust_remote_code:
                raise RuntimeError(
                    f"Loading '{self.model_name}' failed and the model repo "
                    "likely requires custom code (trust_remote_code=True). "
                    "We do NOT auto-enable this because it allows arbitrary "
                    "code execution from the HuggingFace repo. If you trust "
                    "this specific repo, re-run with `trust_remote_code: true` "
                    "in your experiment YAML's backend parameters."
                ) from exc
            raise

        self.model = self.model.to(resolved_device)
        self.model.eval()

        # Tokenizer uses the SAME trust_remote_code setting as the model — do
        # not escalate independently for the same reason as above.
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect architecture.
        if self._architecture_override:
            self._architecture = resolve_architecture(self._architecture_override)
        else:
            model_type = getattr(
                getattr(self.model, "config", None), "model_type", ""
            ) or ""
            self._architecture = resolve_architecture(model_type)

        self._resolved_device = resolved_device

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load()

    @property
    def _device(self) -> Any:
        torch = _require_torch()
        dev = getattr(self, "_resolved_device", "cpu")
        return torch.device(dev)

    # ------------------------------------------------------------------
    # Site → module resolution
    # ------------------------------------------------------------------

    def _resolve_site(self, site: str) -> tuple[Any, str]:
        """Return (module, io) for a TL or raw HF site name."""
        assert self._architecture is not None, "call load() first"
        hf_path, io = translate_hook_site(site, self._architecture)
        module = _get_module(self.model, hf_path)
        return module, io

    # ------------------------------------------------------------------
    # capture_activations
    # ------------------------------------------------------------------

    def capture_activations(
        self, prompts: list[str], sites: list[str]
    ) -> dict[str, Any]:
        self._ensure_loaded()
        torch = _require_torch()

        captured: dict[str, Any] = {}
        handles: list[Any] = []

        for site in sites:
            try:
                module, io = self._resolve_site(site)
            except (KeyError, ValueError, AttributeError):
                continue  # site not translatable — skip silently (reported as missing)

            # Capture the first element of the hook output (some modules return
            # tuples; we want the primary tensor).
            def _make_hook(key: str, capture_input: bool) -> Any:
                def _hook(
                    mod: Any,
                    inp: Any,
                    out: Any,
                ) -> None:
                    value = inp[0] if capture_input else out
                    if isinstance(value, tuple):
                        value = value[0]
                    captured[key] = value.detach().cpu()

                return _hook

            handle = module.register_forward_hook(
                _make_hook(site, capture_input=(io == "input"))
            )
            handles.append(handle)

        assert self.model is not None
        try:
            inputs = _encode(self.tokenizer, prompts, self._device)
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        return captured

    # ------------------------------------------------------------------
    # run_with_hooks
    # ------------------------------------------------------------------

    def run_with_hooks(
        self,
        prompts: list[str] | str,
        fwd_hooks: list[tuple[str, Callable[..., Any]]],
    ) -> Any:
        """Run a forward pass with temporary hooks; return logits tensor."""
        self._ensure_loaded()
        torch = _require_torch()

        if isinstance(prompts, str):
            prompts = [prompts]

        handles: list[Any] = []
        for site, hook_fn in fwd_hooks:
            try:
                module, io = self._resolve_site(site)
            except (KeyError, ValueError, AttributeError):
                continue

            def _make_wrapper(fn: Callable[..., Any], capture_input: bool) -> Any:
                def _hook(mod: Any, inp: Any, out: Any) -> Any:
                    value = inp[0] if capture_input else out
                    if isinstance(value, tuple):
                        primary, rest = value[0], value[1:]
                        result = fn(primary)
                        if result is not None:
                            return (result, *rest)
                        return value
                    result = fn(value)
                    return result if result is not None else out

                return _hook

            handle = module.register_forward_hook(
                _make_wrapper(hook_fn, capture_input=(io == "input"))
            )
            handles.append(handle)

        assert self.model is not None
        try:
            inputs = _encode(self.tokenizer, prompts, self._device)
            with torch.no_grad():
                output = self.model(**inputs)
            logits = output.logits
        finally:
            for h in handles:
                h.remove()

        return logits

    # ------------------------------------------------------------------
    # run_with_cache
    # ------------------------------------------------------------------

    def run_with_cache(
        self,
        prompts: list[str] | str,
        names_filter: Callable[[str], bool] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return (logits, cache_dict) — equivalent to TL's run_with_cache."""
        self._ensure_loaded()
        torch = _require_torch()

        if isinstance(prompts, str):
            prompts = [prompts]

        # Enumerate all named modules and select those passing the filter.
        candidate_sites: list[tuple[str, Any, str]] = []  # (site_key, module, io)
        for name, module in self.model.named_modules():  # type: ignore[union-attr]
            if names_filter is None or names_filter(name):
                candidate_sites.append((name, module, "output"))

        captured: dict[str, Any] = {}
        handles: list[Any] = []

        for site_key, module, _io in candidate_sites:
            def _make_hook(key: str) -> Any:
                def _hook(mod: Any, inp: Any, out: Any) -> None:
                    value = out[0] if isinstance(out, tuple) else out
                    captured[key] = value.detach().cpu()

                return _hook

            handle = module.register_forward_hook(_make_hook(site_key))
            handles.append(handle)

        assert self.model is not None
        try:
            inputs = _encode(self.tokenizer, prompts, self._device)
            with torch.no_grad():
                output = self.model(**inputs)
            logits = output.logits
        finally:
            for h in handles:
                h.remove()

        return logits, captured

    # ------------------------------------------------------------------
    # run_activation_patching
    # ------------------------------------------------------------------

    def run_activation_patching(
        self,
        request: ActivationPatchRequest,
    ) -> list[ActivationPatchSiteResult]:
        """Clean-to-corrupted activation patching via forward hooks."""
        self._ensure_loaded()

        from mech_interp.analysis import logit_diff_recovery  # local to avoid cycle

        results: list[ActivationPatchSiteResult] = []

        for pair in request.prompt_pairs:
            # Tokenize clean and corrupted to get token ids for correct/incorrect tokens.
            assert self.tokenizer is not None, "call load() before run_activation_patching()"
            correct_ids = self.tokenizer.encode(
                pair.correct_token, add_special_tokens=False
            )
            incorrect_ids = self.tokenizer.encode(
                pair.incorrect_token, add_special_tokens=False
            )
            correct_token_id = correct_ids[-1] if correct_ids else 0
            incorrect_token_id = incorrect_ids[-1] if incorrect_ids else 0

            # Collect base sites (strip per-head qualifiers — HF doesn't have them).
            base_sites: list[str] = list(request.hook_sites)

            # Run clean pass to get clean activations.
            clean_captured: dict[str, Any] = self.capture_activations(
                [pair.clean_prompt], base_sites
            )

            # Run corrupted pass to get corrupted logits (baseline).
            corrupted_logits = self.run_with_hooks([pair.corrupted_prompt], [])
            clean_logits = self.run_with_hooks([pair.clean_prompt], [])

            for hook_site in request.hook_sites:
                if hook_site not in clean_captured:
                    continue
                clean_act = clean_captured[hook_site]
                patch_pos = pair.patch_position

                def _patch_fn(
                    activation: Any,
                    _clean: Any = clean_act,
                    _pos: int = patch_pos,
                ) -> Any:
                    patched = activation.clone()
                    # Patch the requested position (or last if -1).
                    pos = _pos if _pos >= 0 else activation.shape[1] - 1
                    _clean_dev = _clean.to(activation.device)
                    src_pos = _pos if _pos >= 0 else _clean.shape[1] - 1
                    patched[:, pos, ...] = _clean_dev[:, src_pos, ...]
                    return patched

                patched_logits = self.run_with_hooks(
                    [pair.corrupted_prompt],
                    [(hook_site, _patch_fn)],
                )

                tgt_pos = pair.target_position
                recovery = logit_diff_recovery(
                    clean_logits=_logits_at_position(clean_logits, tgt_pos),
                    corrupted_logits=_logits_at_position(corrupted_logits, tgt_pos),
                    patched_logits=_logits_at_position(patched_logits, tgt_pos),
                    correct_token_index=correct_token_id,
                    incorrect_token_index=incorrect_token_id,
                )

                # Activation norm.
                act_norm: float | None = None
                try:
                    pos_idx = patch_pos if patch_pos >= 0 else clean_act.shape[1] - 1
                    act_norm = float(clean_act[:, pos_idx, ...].norm().item())
                except (AttributeError, IndexError, RuntimeError):
                    pass

                results.append(
                    ActivationPatchSiteResult(
                        pair_id=pair.id,
                        hook_site=hook_site,
                        clean_logit_diff=recovery.clean_logit_diff,
                        corrupted_logit_diff=recovery.corrupted_logit_diff,
                        patched_logit_diff=recovery.patched_logit_diff,
                        recovery_fraction=recovery.recovery_fraction,
                        activation_norm=act_norm,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # run_intervention (protocol requirement — stub)
    # ------------------------------------------------------------------

    def run_intervention(
        self, prompt: str, interventions: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Use run_with_hooks for fine-grained interventions on the HF backend."
        )

    # ------------------------------------------------------------------
    # run_cross_model_probe — not supported
    # ------------------------------------------------------------------

    def run_cross_model_probe(
        self,
        request: CrossModelProbeRequest,
    ) -> list[CrossModelProbeResult]:
        raise NotImplementedError(
            "cross_model_representation_probe requires the TransformerLens backend "
            "(two models must be loaded with TL's HookedTransformer). "
            "Switch to backend: transformerlens for this experiment family."
        )
