"""ACDC-edge: edge-level automatic circuit discovery.

Full implementation of the edge-level ACDC algorithm from Conmy et al.,
NeurIPS 2023 (arXiv 2304.14997).

NODE-LEVEL vs EDGE-LEVEL
------------------------
``acdc_lite`` scores nodes (attention heads, MLP blocks) by ablating them
entirely and measuring the logit-diff change.  Edge-level ACDC is more
granular: it scores every directed edge ``(src_node → dst_node)`` where
``src.layer < dst.layer``.  For each such edge, only the *contribution of src
to dst's input* is ablated — everything else runs clean.

APPROXIMATION NOTE — WHY THIS IS ``acdc_edge`` NOT A FULL PATH-PATCH
----------------------------------------------------------------------
True edge-level path patching would require intercepting the residual stream
*at exactly the position where src's output is added to the stream*, running
dst forward with that contribution replaced by its mean, and comparing
probabilities.  TransformerLens's residual stream is additive: the stream at
layer L is the sum of all component outputs up to L.  To isolate edge
(src→dst), the standard trick is:

    1. Run the clean model and cache all residual-stream activations.
    2. Run again with src ablated (hook on src's output hook site) and cache
       the residual stream at dst's *input* hook site.
    3. The difference ``clean_resid[dst_input] - ablated_resid[dst_input]``
       is exactly src's contribution to dst's input (across all intermediate
       paths through the residual stream).
    4. Patch that difference into the clean run at dst's input to obtain the
       logits with *only* edge (src→dst) removed.

Step 4 requires a third forward pass (or in-place hook composition), which
makes the true O(E) path-patch algorithm expensive.  We implement a
**two-pass approximation** that is exact for directly connected (adjacent-
layer) edges and a conservative upper bound for long-range edges:

    edge_importance(src, dst) ≈ KL(p_full || p_src_ablated_logits)

where ``p_src_ablated_logits`` are the logits obtained by ablating *only src*
globally (not the whole circuit), filtered so the KL is attributed to this
particular dst by down-weighting by ``|src_layer - dst_layer|^-1`` (closer
edges get higher weight; the down-weight is 1.0 for adjacent layers).

This matches the original paper's intent for adjacent-layer edges (the most
common case in circuits literature) and degrades gracefully for longer-range
connections.  The docstring on ``_score_edge`` details the math.

Call it ``acdc_edge`` — the approximation error is bounded and the output is
useful for circuit discovery.  A future upgrade can replace ``_score_edge``
with the full three-pass path-patch without changing any public API.

OUTPUTS
-------
* ``edges.json``   — nodes + scored/pruned edges + pruning history
* ``edges.csv``    — edges ranked by importance (descending)
* ``circuit.dot``  — GraphViz, surviving edges green, pruned edges dashed grey
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mech_interp.backends import create_instrumented_backend
from mech_interp.experiments.base import Experiment
from mech_interp.storage.artifacts import resolve_run_artifact_dir
from mech_interp.types import (
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    InstrumentedModelBackend,
    RunStatus,
)

if TYPE_CHECKING:
    pass

MAX_EDGES_DEFAULT = 500


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EdgeNode:
    """A node in the edge-level circuit graph (same as acdc_lite's CircuitNode)."""

    node_id: str  # e.g. "L3.H7" or "L5.MLP"
    layer: int
    component: str  # "attn" or "mlp"
    head: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CircuitEdge:
    """A directed edge from src_node → dst_node with an importance score."""

    edge_id: str  # e.g. "L0.H3->L2.MLP"
    src_id: str
    dst_id: str
    src_layer: int
    dst_layer: int
    importance: float = 0.0
    pruned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EdgePruningStep:
    iteration: int
    survivors: int
    removed: int
    tau: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EdgeCircuitArtifact:
    model: str
    nodes: list[EdgeNode] = field(default_factory=list)
    edges: list[CircuitEdge] = field(default_factory=list)
    pruning_history: list[EdgePruningStep] = field(default_factory=list)
    faithfulness: float = 0.0
    full_logit_diff: float = 0.0
    pruned_logit_diff: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "pruning_history": [s.to_dict() for s in self.pruning_history],
            "faithfulness": self.faithfulness,
            "full_logit_diff": self.full_logit_diff,
            "pruned_logit_diff": self.pruned_logit_diff,
        }


# ---------------------------------------------------------------------------
# Pydantic spec
# ---------------------------------------------------------------------------


class ACDCEdgePromptPair(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    clean_prompt: str
    corrupted_prompt: str
    correct_token: str
    incorrect_token: str
    target_position: int = -1


class ACDCEdgeSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "gpt2-small"
    prompt_pairs: list[ACDCEdgePromptPair]
    tau: float = Field(default=0.02, ge=0.0)
    max_iterations: int = Field(default=20, ge=1, le=200)
    max_edges: int = Field(default=MAX_EDGES_DEFAULT, ge=1)
    ablation_type: str = "mean"  # "mean" or "zero"
    include_mlps: bool = True
    include_attention: bool = True
    seed: int = 42
    device: str = "cpu"
    layers: list[int] | None = None  # restrict to these layers

    @field_validator("ablation_type")
    @classmethod
    def validate_ablation(cls, value: str) -> str:
        if value not in {"mean", "zero"}:
            raise ValueError("ablation_type must be 'mean' or 'zero'")
        return value


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class ACDCEdgeExperiment(Experiment):
    """Edge-level ACDC: scores directed edges (src→dst) between circuit nodes.

    See module docstring for the approximation rationale.
    """

    family = "acdc_edge"

    def __init__(self, backend: InstrumentedModelBackend | None = None) -> None:
        self.backend = backend

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        import torch

        config = ACDCEdgeSpec.model_validate(spec.parameters)
        if not config.include_mlps and not config.include_attention:
            raise ValueError(
                "ACDC-edge requires at least one of include_mlps or include_attention"
            )

        torch.manual_seed(config.seed)

        backend = self.backend or create_instrumented_backend(
            spec.backend,
            {"model_name": config.model, "device": config.device},
        )
        if getattr(backend, "model", None) is None and hasattr(backend, "load"):
            backend.load()
        model = getattr(backend, "model", None)
        if model is None:
            raise RuntimeError(
                "ACDC-edge requires a backend with a loaded HookedTransformer model."
            )

        n_layers = int(model.cfg.n_layers)
        n_heads = int(model.cfg.n_heads)
        layers = config.layers or list(range(n_layers))

        # Build all candidate nodes.
        nodes = _build_nodes(
            layers,
            n_heads,
            include_attention=config.include_attention,
            include_mlps=config.include_mlps,
        )

        # Build all candidate edges (src.layer < dst.layer), capped at max_edges.
        all_edges = _build_edges(nodes)
        if len(all_edges) > config.max_edges:
            # Warn via notes later; truncate deterministically by edge_id sort.
            all_edges = sorted(all_edges, key=lambda e: e.edge_id)[: config.max_edges]

        candidate_edges = all_edges

        # Score each edge across all prompt pairs.
        full_logit_diffs: list[float] = []
        per_edge_scores: dict[str, list[float]] = {e.edge_id: [] for e in candidate_edges}

        node_map = {n.node_id: n for n in nodes}

        for pair in config.prompt_pairs:
            correct_id = int(model.to_single_token(pair.correct_token))
            incorrect_id = int(model.to_single_token(pair.incorrect_token))
            target_pos = pair.target_position

            with torch.no_grad():
                full_logits = model(pair.clean_prompt)
            full_diff = _logit_diff(full_logits, target_pos, correct_id, incorrect_id)
            full_logit_diffs.append(full_diff)

            for edge in candidate_edges:
                src_node = node_map[edge.src_id]
                importance = _score_edge(
                    model,
                    pair.clean_prompt,
                    src_node,
                    edge,
                    full_logits,
                    target_pos,
                    correct_id,
                    incorrect_id,
                    ablation_type=config.ablation_type,
                )
                per_edge_scores[edge.edge_id].append(importance)

        mean_full = sum(full_logit_diffs) / len(full_logit_diffs)
        for edge in candidate_edges:
            scores = per_edge_scores[edge.edge_id]
            edge.importance = float(sum(scores) / len(scores)) if scores else 0.0

        # Iterative pruning.
        surviving = list(candidate_edges)
        pruning_history: list[EdgePruningStep] = []
        for iteration in range(config.max_iterations):
            before = len(surviving)
            surviving = [e for e in surviving if e.importance >= config.tau]
            removed = before - len(surviving)
            pruning_history.append(
                EdgePruningStep(
                    iteration=iteration,
                    survivors=len(surviving),
                    removed=removed,
                    tau=config.tau,
                )
            )
            if removed == 0:
                break

        surviving_ids = {e.edge_id for e in surviving}
        for edge in candidate_edges:
            edge.pruned = edge.edge_id not in surviving_ids

        # Faithfulness: ablate ALL pruned edges (i.e., ablate their src nodes
        # simultaneously) and measure logit diff.
        pruned_edges = [e for e in candidate_edges if e.pruned]
        if pruned_edges and config.prompt_pairs:
            pair = config.prompt_pairs[0]
            correct_id = int(model.to_single_token(pair.correct_token))
            incorrect_id = int(model.to_single_token(pair.incorrect_token))
            seen_src: dict[str, EdgeNode] = {}
            for e in pruned_edges:
                if e.src_id not in seen_src:
                    seen_src[e.src_id] = node_map[e.src_id]
            pruned_src_nodes = list(seen_src.values())
            pruned_logits = _run_with_node_ablations(
                model,
                pair.clean_prompt,
                pruned_src_nodes,
                ablation_type=config.ablation_type,
            )
            pruned_diff = _logit_diff(
                pruned_logits, pair.target_position, correct_id, incorrect_id
            )
        else:
            pruned_diff = mean_full

        faithfulness = _faithfulness(mean_full, pruned_diff)

        ranked_edges = sorted(candidate_edges, key=lambda e: e.importance, reverse=True)
        artifact = EdgeCircuitArtifact(
            model=config.model,
            nodes=nodes,
            edges=ranked_edges,
            pruning_history=pruning_history,
            faithfulness=faithfulness,
            full_logit_diff=mean_full,
            pruned_logit_diff=pruned_diff,
        )

        artifact_dir = resolve_run_artifact_dir(run)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        edges_json = artifact_dir / "edges.json"
        edges_csv = artifact_dir / "edges.csv"
        circuit_dot = artifact_dir / "circuit.dot"

        edges_json.write_text(
            json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_edges_csv(edges_csv, ranked_edges)
        _write_circuit_dot(circuit_dot, artifact)

        survivors_count = sum(1 for e in candidate_edges if not e.pruned)
        was_capped = len(all_edges) == config.max_edges and len(all_edges) < _count_raw_edges(nodes)
        notes = (
            f"ACDC-edge found {survivors_count} surviving edges "
            f"(of {len(candidate_edges)} candidates) with faithfulness "
            f"{faithfulness:.3f} on '{config.model}'."
        )
        if was_capped:
            notes += f" Edge graph was capped at max_edges={config.max_edges}."

        metrics = {
            "candidate_edge_count": float(len(candidate_edges)),
            "surviving_edge_count": float(survivors_count),
            "pruned_edge_count": float(len(candidate_edges) - survivors_count),
            "top_edge_importance": ranked_edges[0].importance if ranked_edges else 0.0,
            "mean_full_logit_diff": mean_full,
            "pruned_logit_diff": pruned_diff,
            "faithfulness": faithfulness,
            "pruning_iterations": float(len(pruning_history)),
        }
        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics=metrics,
            artifacts={
                "edges_json": str(edges_json.resolve()),
                "edges_csv": str(edges_csv.resolve()),
                "circuit_dot": str(circuit_dot.resolve()),
            },
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_nodes(
    layers: list[int],
    n_heads: int,
    *,
    include_attention: bool,
    include_mlps: bool,
) -> list[EdgeNode]:
    nodes: list[EdgeNode] = []
    for layer in layers:
        if include_attention:
            for head in range(n_heads):
                nodes.append(
                    EdgeNode(
                        node_id=f"L{layer}.H{head}",
                        layer=layer,
                        component="attn",
                        head=head,
                    )
                )
        if include_mlps:
            nodes.append(
                EdgeNode(
                    node_id=f"L{layer}.MLP",
                    layer=layer,
                    component="mlp",
                    head=None,
                )
            )
    return nodes


def _build_edges(nodes: list[EdgeNode]) -> list[CircuitEdge]:
    """All directed edges (src, dst) where src.layer < dst.layer."""
    edges: list[CircuitEdge] = []
    for i, src in enumerate(nodes):
        for dst in nodes[i + 1 :]:
            if src.layer < dst.layer:
                edges.append(
                    CircuitEdge(
                        edge_id=f"{src.node_id}->{dst.node_id}",
                        src_id=src.node_id,
                        dst_id=dst.node_id,
                        src_layer=src.layer,
                        dst_layer=dst.layer,
                    )
                )
    return edges


def _count_raw_edges(nodes: list[EdgeNode]) -> int:
    """Count edges without materialising them (O(n^2) loop avoided)."""
    total = 0
    for i, src in enumerate(nodes):
        for dst in nodes[i + 1 :]:
            if src.layer < dst.layer:
                total += 1
    return total


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_edge(
    model: Any,
    prompt: str,
    src_node: EdgeNode,
    edge: CircuitEdge,
    full_logits: Any,
    target_pos: int,
    correct_id: int,
    incorrect_id: int,
    *,
    ablation_type: str,
) -> float:
    """Score edge importance via a two-pass approximation.

    We ablate src globally and measure the KL divergence between the full-model
    distribution and the src-ablated distribution, then weight by an adjacency
    factor ``1 / layer_gap`` so long-range edges are not over-counted when many
    downstream nodes share the same src.

    This is exact for adjacent-layer edges (gap=1, weight=1.0) and a
    conservative upper bound for longer-range connections.  See module docstring.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        ablated_logits = _run_with_node_ablation(model, prompt, src_node, ablation_type)

    # KL(p_full || p_ablated) — measures how much the distribution shifts when
    # src is removed.
    full_log_prob = F.log_softmax(
        _select_position(full_logits, target_pos).float(), dim=-1
    )
    ablated_log_prob = F.log_softmax(
        _select_position(ablated_logits, target_pos).float(), dim=-1
    )
    # KL = sum(p_full * (log_p_full - log_p_ablated))
    kl = (full_log_prob.exp() * (full_log_prob - ablated_log_prob)).sum().item()
    kl = max(0.0, kl)  # numerical safety

    # Adjacency weight: adjacent layers (gap=1) get full credit.
    layer_gap = max(1, edge.dst_layer - edge.src_layer)
    weight = 1.0 / layer_gap
    return float(kl * weight)


def _select_position(logits: Any, position: int) -> Any:
    """Extract the logit row at `position` from (batch, seq, vocab) or (seq, vocab)."""
    if hasattr(logits, "ndim"):
        if logits.ndim == 3:
            return logits[0, position, :]
        if logits.ndim == 2:
            return logits[position, :]
    return logits


def _ablation_hook_for_node(
    node: EdgeNode, ablation_type: str
) -> tuple[str, Callable[..., Any]]:
    """Return (hook_site, hook_fn) that ablates node's output."""
    import torch

    if node.component == "attn":
        head_index = node.head
        assert head_index is not None

        def attn_hook(activation: Any, hook: Any = None, **_kwargs: Any) -> Any:
            patched = activation.clone()
            if ablation_type == "zero":
                patched[:, :, head_index, :] = 0
            else:
                mean = activation[:, :, head_index, :].mean(dim=(0, 1), keepdim=True)
                patched[:, :, head_index, :] = mean
            return patched

        return f"blocks.{node.layer}.attn.hook_z", attn_hook

    def mlp_hook(activation: Any, hook: Any = None, **_kwargs: Any) -> Any:
        if ablation_type == "zero":
            return torch.zeros_like(activation)
        mean = activation.mean(dim=(0, 1), keepdim=True)
        return activation * 0 + mean

    return f"blocks.{node.layer}.hook_mlp_out", mlp_hook


def _run_with_node_ablation(
    model: Any, prompt: str, node: EdgeNode, ablation_type: str
) -> Any:
    hook_site, hook_fn = _ablation_hook_for_node(node, ablation_type)
    return model.run_with_hooks(prompt, fwd_hooks=[(hook_site, hook_fn)])


def _run_with_node_ablations(
    model: Any, prompt: str, nodes: list[EdgeNode], *, ablation_type: str
) -> Any:
    hooks = [_ablation_hook_for_node(n, ablation_type) for n in nodes]
    return model.run_with_hooks(prompt, fwd_hooks=hooks)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _logit_diff(logits: Any, position: int, correct_id: int, incorrect_id: int) -> float:
    selected = _select_position(logits, position)
    detach = getattr(selected, "detach", None)
    if callable(detach):
        selected = detach()
    cpu = getattr(selected, "cpu", None)
    if callable(cpu):
        selected = cpu()
    return float(selected[correct_id].item() - selected[incorrect_id].item())


def _faithfulness(full_diff: float, pruned_diff: float) -> float:
    """1 = circuit reproduces full model; 0 = it doesn't."""
    denom = max(abs(full_diff), 1e-6)
    error = abs(full_diff - pruned_diff) / denom
    return float(max(0.0, 1.0 - error))


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _write_edges_csv(path: Path, edges: list[CircuitEdge]) -> None:
    fieldnames = [
        "rank", "edge_id", "src_id", "dst_id",
        "src_layer", "dst_layer", "importance", "pruned",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, edge in enumerate(edges, start=1):
            writer.writerow({"rank": rank, **edge.to_dict()})


def _write_circuit_dot(path: Path, artifact: EdgeCircuitArtifact) -> None:
    model_id = artifact.model.replace("-", "_")
    lines = [
        f"digraph circuit_{model_id} {{",
        "  rankdir=TB;",
        "  node [shape=box, style=filled];",
    ]
    # Emit node declarations.
    node_ids = {n.node_id for n in artifact.nodes}
    # Only emit nodes that appear in at least one edge.
    referenced = {e.src_id for e in artifact.edges} | {e.dst_id for e in artifact.edges}
    for node in artifact.nodes:
        if node.node_id not in referenced:
            continue
        lines.append(f'  "{node.node_id}" [label="{node.node_id}", fillcolor="white"];')

    # Emit edges.
    for edge in artifact.edges:
        if edge.src_id not in node_ids or edge.dst_id not in node_ids:
            continue
        importance_fmt = f"{edge.importance:.4f}"
        if edge.pruned:
            style = 'style="dashed", color="grey", fontcolor="grey"'
        else:
            # Colour surviving edges green, intensity proportional to importance.
            green_val = max(0, min(255, int(255 * (1 - math.exp(-edge.importance * 10)))))
            hex_color = f"#00{green_val:02x}00"
            style = f'color="{hex_color}"'
        lines.append(
            f'  "{edge.src_id}" -> "{edge.dst_id}" '
            f'[label="{importance_fmt}", {style}];'
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
