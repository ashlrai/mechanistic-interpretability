"""Pluggable feature labeler for SAE features.

Assigns human-readable labels to SAE features based on their top-activating prompts.
Three concrete implementations are provided:

- ``HeuristicFeatureLabeler``: token-overlap heuristic, no deps, deterministic.
- ``OllamaFeatureLabeler``: calls a local Ollama instance via HTTP.
- ``AnthropicFeatureLabeler``: calls Anthropic API (requires ANTHROPIC_API_KEY env var).

All labelers implement the ``FeatureLabeler`` ABC and can be swapped at runtime.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any


class OptionalDependencyError(ImportError):
    """Raised when an optional labeler dependency is not installed."""


class FeatureLabeler(ABC):
    """Abstract base class for SAE feature labelers."""

    @abstractmethod
    def label(
        self,
        feature_index: int,
        top_prompts: list[str],
        max_activation: float,
    ) -> str:
        """Return a short human-readable label for this feature.

        Parameters
        ----------
        feature_index:
            The integer index of the SAE feature.
        top_prompts:
            Up to 5 prompts where this feature fires most strongly.
        max_activation:
            Peak activation value observed for this feature.
        """


# ---------------------------------------------------------------------------
# Heuristic labeler — no deps, deterministic
# ---------------------------------------------------------------------------


class HeuristicFeatureLabeler(FeatureLabeler):
    """Token-overlap heuristic for offline / test use.

    Finds the most common content words across top-activating prompts and
    forms a short label from them.  No external dependencies required.
    """

    STOP_WORDS: frozenset[str] = frozenset(
        {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "about", "and",
            "or", "but", "not", "that", "this", "it", "i", "you", "he", "she",
            "we", "they", "what", "which", "who", "when", "where", "how",
        }
    )

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

    def label(
        self,
        feature_index: int,
        top_prompts: list[str],
        max_activation: float,
    ) -> str:
        if not top_prompts:
            return f"feature_{feature_index} (no data)"

        counter: Counter[str] = Counter()
        for prompt in top_prompts:
            words = prompt.lower().split()
            counter.update(w.strip(".,;:!?\"'()[]") for w in words if w not in self.STOP_WORDS)

        top_words = [w for w, _ in counter.most_common(self.top_k) if len(w) > 2]
        if not top_words:
            return f"feature_{feature_index} (sparse)"
        return " / ".join(top_words)


# ---------------------------------------------------------------------------
# Ollama labeler — local HTTP, optional dep
# ---------------------------------------------------------------------------

_OLLAMA_SYSTEM_PROMPT = (
    "You are a concise neuroscience assistant labeling features of a Sparse Autoencoder "
    "trained on a language model. Given a list of text snippets that maximally activate "
    "a feature, respond with a SHORT (2-6 word) human-readable label describing what "
    "concept or pattern this feature detects. Reply with ONLY the label, no explanation."
)

_OLLAMA_USER_TEMPLATE = (
    "Feature {feature_index} (max activation: {max_activation:.4f})\n"
    "Top activating prompts:\n{prompts}\n\nLabel:"
)


class OllamaFeatureLabeler(FeatureLabeler):
    """Calls a local Ollama instance to label SAE features.

    Parameters
    ----------
    host:
        Base URL of the Ollama server (default: ``http://localhost:11434``).
    model:
        Ollama model to use (default: ``llama3.2:3b``).
    system_prompt:
        Override the default system prompt template.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        system_prompt: str = _OLLAMA_SYSTEM_PROMPT,
        timeout: float = 30.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout

    def label(
        self,
        feature_index: int,
        top_prompts: list[str],
        max_activation: float,
    ) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise OptionalDependencyError(
                "OllamaFeatureLabeler requires 'httpx'. "
                "It is already a core dependency — ensure it is installed."
            ) from exc

        prompts_text = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(top_prompts))
        user_msg = _OLLAMA_USER_TEMPLATE.format(
            feature_index=feature_index,
            max_activation=max_activation,
            prompts=prompts_text,
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }

        try:
            response = httpx.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            label_text = data["message"]["content"].strip()
            # Normalise: strip quotes/trailing periods, truncate
            label_text = label_text.strip("\"'.").strip()
            return label_text[:100] if label_text else f"feature_{feature_index}"
        except Exception as exc:  # noqa: BLE001
            return f"feature_{feature_index} (ollama error: {type(exc).__name__})"


# ---------------------------------------------------------------------------
# Anthropic labeler — optional dep + API key
# ---------------------------------------------------------------------------

_ANTHROPIC_SYSTEM_PROMPT = _OLLAMA_SYSTEM_PROMPT
_ANTHROPIC_USER_TEMPLATE = _OLLAMA_USER_TEMPLATE


class AnthropicFeatureLabeler(FeatureLabeler):
    """Calls the Anthropic API to label SAE features.

    Only constructed when ``ANTHROPIC_API_KEY`` is set in the environment.
    Raises ``OptionalDependencyError`` if the ``anthropic`` SDK is not installed.

    Parameters
    ----------
    api_key:
        Anthropic API key (defaults to ``ANTHROPIC_API_KEY`` env var).
    model:
        Anthropic model to use (default: ``claude-haiku-4-5``).
    system_prompt:
        Override the default system prompt.
    max_tokens:
        Maximum tokens in the response.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5",
        system_prompt: str = _ANTHROPIC_SYSTEM_PROMPT,
        max_tokens: int = 32,
    ) -> None:
        import os

        try:
            import anthropic as _anthropic_module  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise OptionalDependencyError(
                "AnthropicFeatureLabeler requires the 'anthropic' SDK. "
                "Install with: pip install anthropic"
            ) from exc

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "AnthropicFeatureLabeler requires ANTHROPIC_API_KEY to be set."
            )
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self._api_key = resolved_key

    def label(
        self,
        feature_index: int,
        top_prompts: list[str],
        max_activation: float,
    ) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise OptionalDependencyError(
                "AnthropicFeatureLabeler requires the 'anthropic' SDK."
            ) from exc

        prompts_text = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(top_prompts))
        user_msg = _ANTHROPIC_USER_TEMPLATE.format(
            feature_index=feature_index,
            max_activation=max_activation,
            prompts=prompts_text,
        )

        try:
            client = anthropic.Anthropic(api_key=self._api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            label_text = message.content[0].text.strip("\"'.").strip()
            return label_text[:100] if label_text else f"feature_{feature_index}"
        except Exception as exc:  # noqa: BLE001
            return f"feature_{feature_index} (anthropic error: {type(exc).__name__})"


# ---------------------------------------------------------------------------
# Batch labeling entry point
# ---------------------------------------------------------------------------


def label_run_features(
    run_artifact_dir: Path,
    labeler: FeatureLabeler,
    *,
    max_features: int = 50,
) -> dict[int, str]:
    """Read ``feature_analysis.json``, label live features, return ``{index: label}``.

    Writes ``feature_labels.json`` next to ``feature_analysis.json`` and returns
    the mapping from feature index to label string.

    Parameters
    ----------
    run_artifact_dir:
        Directory containing ``feature_analysis.json`` (e.g. ``artifacts/run-000001/``).
    labeler:
        Concrete labeler instance to use.
    max_features:
        Cap on the number of live features to label (ordered by max_activation desc).
    """
    feature_analysis_path = run_artifact_dir / "feature_analysis.json"
    if not feature_analysis_path.is_file():
        raise FileNotFoundError(f"feature_analysis.json not found in {run_artifact_dir}")

    raw: Any = json.loads(feature_analysis_path.read_text(encoding="utf-8"))
    feature_list: list[dict[str, Any]] = []
    if isinstance(raw, dict) and isinstance(raw.get("features"), list):
        feature_list = raw["features"]
    elif isinstance(raw, list):
        feature_list = raw

    # Filter to live features, sort by max_activation desc, cap.
    live = [f for f in feature_list if isinstance(f, dict) and not f.get("dead", False)]
    live.sort(key=lambda f: float(f.get("max_activation", 0.0)), reverse=True)
    live = live[:max_features]

    labels: dict[int, str] = {}
    for feature in live:
        idx = int(feature.get("feature_index", 0))
        top_prompt_entries = feature.get("top_prompts") or []
        top_prompts: list[str] = []
        for entry in top_prompt_entries:
            if isinstance(entry, dict):
                top_prompts.append(str(entry.get("prompt", "")))
            elif isinstance(entry, str):
                top_prompts.append(entry)
        max_act = float(feature.get("max_activation", 0.0))
        labels[idx] = labeler.label(idx, top_prompts, max_act)

    output_path = run_artifact_dir / "feature_labels.json"
    output_path.write_text(
        json.dumps(
            {"feature_labels": {str(k): v for k, v in labels.items()}},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return labels
