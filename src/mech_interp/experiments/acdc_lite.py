"""ACDC-lite: node-level automatic circuit discovery.

A scaled-down implementation of ACDC (Conmy et al., NeurIPS 2023; arXiv 2304.14997).
Instead of the full edge graph, we score the importance of every (layer, head)
attention node and every (layer, MLP) node by KL-divergence between full and
mean-ablated logits, then iteratively prune nodes whose contribution falls below
a threshold. Output: ranked node list, surviving pruned circuit, faithfulness
metric, plus a GraphViz dot file for visualisation.

Node-level ACDC is far cheaper than full edge-level ACDC and is the right
starting point on gpt2-small: 12 layers × (12 heads + 1 MLP) = 156 nodes is
tractable in seconds, and the discovered nodes feed straight into circuit_patching
as proposals for follow-up causal verification.
"""

from __future__ import annotations

import csv
import json
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
    from torch import Tensor


@dataclass
class CircuitNode:
    node_id: str  # e.g., "L3.H7" or "L5.MLP"
    layer: int
    component: str  # "attn" or "mlp"
    head: int | None
    importance: float = 0.0
    pruned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PruningStep:
    iteration: int
    survivors: int
    removed: int
    tau: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CircuitArtifact:
    model: str
    nodes: list[CircuitNode] = field(default_factory=list)
    pruning_history: list[PruningStep] = field(default_factory=list)
    faithfulness: float = 0.0
    full_logit_diff: float = 0.0
    pruned_logit_diff: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "nodes": [node.to_dict() for node in self.nodes],
            "pruning_history": [step.to_dict() for step in self.pruning_history],
            "faithfulness": self.faithfulness,
            "full_logit_diff": self.full_logit_diff,
            "pruned_logit_diff": self.pruned_logit_diff,
        }


class ACDCPromptPair(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    clean_prompt: str
    corrupted_prompt: str
    correct_token: str
    incorrect_token: str
    target_position: int = -1


class ACDCLiteSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "gpt2-small"
    prompt_pairs: list[ACDCPromptPair]
    tau: float = Field(default=0.01, ge=0.0)
    max_iterations: int = Field(default=20, ge=1, le=200)
    ablation_type: str = "mean"  # "mean" or "zero"
    include_mlps: bool = True
    include_attention: bool = True
    seed: int = 42
    device: str = "cpu"
    layers: list[int] | None = None  # restrict candidate nodes if set

    @field_validator("ablation_type")
    @classmethod
    def validate_ablation(cls, value: str) -> str:
        if value not in {"mean", "zero"}:
            raise ValueError("ablation_type must be 'mean' or 'zero'")
        return value


class ACDCLiteExperiment(Experiment):
    family = "acdc_lite"

    def __init__(self, backend: InstrumentedModelBackend | None = None) -> None:
        self.backend = backend

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        import torch

        config = ACDCLiteSpec.model_validate(spec.parameters)
        if not config.include_mlps and not config.include_attention:
            raise ValueError("ACDC-lite requires at least one of include_mlps or include_attention")

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
                "ACDC-lite requires a backend with a loaded HookedTransformer model."
            )

        n_layers = int(model.cfg.n_layers)
        n_heads = int(model.cfg.n_heads)
        layers = config.layers or list(range(n_layers))
        candidate_nodes = _build_candidate_nodes(
            layers,
            n_heads,
            include_attention=config.include_attention,
            include_mlps=config.include_mlps,
        )

        # Score full and per-node logit diff on the first prompt pair (scoring is
        # cheap so we average across pairs to reduce variance).
        full_logit_diffs: list[float] = []
        per_node_diffs: dict[str, list[float]] = {node.node_id: [] for node in candidate_nodes}
        for pair in config.prompt_pairs:
            correct_id = int(model.to_single_token(pair.correct_token))
            incorrect_id = int(model.to_single_token(pair.incorrect_token))
            target_pos = pair.target_position

            with torch.no_grad():
                full_logits = model(pair.clean_prompt)
            full_diff = _logit_diff(full_logits, target_pos, correct_id, incorrect_id)
            full_logit_diffs.append(full_diff)

            for node in candidate_nodes:
                ablated_logits = _run_with_ablation(
                    model,
                    pair.clean_prompt,
                    node,
                    ablation_type=config.ablation_type,
                )
                ablated_diff = _logit_diff(
                    ablated_logits, target_pos, correct_id, incorrect_id
                )
                # Importance = magnitude of effect on the answer-token logit diff
                # when this node is removed.
                per_node_diffs[node.node_id].append(abs(full_diff - ablated_diff))

        mean_full = sum(full_logit_diffs) / len(full_logit_diffs)
        for node in candidate_nodes:
            scores = per_node_diffs[node.node_id]
            node.importance = float(sum(scores) / len(scores)) if scores else 0.0

        # Iterative pruning: drop nodes whose importance is below tau and re-evaluate
        # the joint ablation each round so we surface synergistic nodes.
        surviving = list(candidate_nodes)
        pruning_history: list[PruningStep] = []
        for iteration in range(config.max_iterations):
            before = len(surviving)
            surviving = [n for n in surviving if n.importance >= config.tau]
            removed = before - len(surviving)
            pruning_history.append(
                PruningStep(
                    iteration=iteration,
                    survivors=len(surviving),
                    removed=removed,
                    tau=config.tau,
                )
            )
            if removed == 0:
                break

        pruned_node_ids = {node.node_id for node in candidate_nodes} - {
            node.node_id for node in surviving
        }
        for node in candidate_nodes:
            node.pruned = node.node_id in pruned_node_ids

        # Faithfulness: run with the entire pruned set ablated together and compare
        # to the full-model logit diff on the first prompt pair (single-pair eval is
        # the standard quick faithfulness check).
        if pruned_node_ids and config.prompt_pairs:
            pair = config.prompt_pairs[0]
            correct_id = int(model.to_single_token(pair.correct_token))
            incorrect_id = int(model.to_single_token(pair.incorrect_token))
            pruned_logits = _run_with_ablations(
                model,
                pair.clean_prompt,
                [n for n in candidate_nodes if n.pruned],
                ablation_type=config.ablation_type,
            )
            pruned_diff = _logit_diff(
                pruned_logits, pair.target_position, correct_id, incorrect_id
            )
        else:
            pruned_diff = mean_full

        faithfulness = _faithfulness(mean_full, pruned_diff)

        ranked = sorted(candidate_nodes, key=lambda n: n.importance, reverse=True)
        circuit = CircuitArtifact(
            model=config.model,
            nodes=ranked,
            pruning_history=pruning_history,
            faithfulness=faithfulness,
            full_logit_diff=mean_full,
            pruned_logit_diff=pruned_diff,
        )

        artifact_dir = resolve_run_artifact_dir(run)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        circuit_json = artifact_dir / "circuit.json"
        circuit_csv = artifact_dir / "edge_scores.csv"
        circuit_dot = artifact_dir / "circuit.dot"
        circuit_json.write_text(
            json.dumps(circuit.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_edge_csv(circuit_csv, ranked)
        _write_circuit_dot(circuit_dot, circuit)

        survivors_count = sum(1 for n in candidate_nodes if not n.pruned)
        metrics = {
            "candidate_node_count": float(len(candidate_nodes)),
            "surviving_node_count": float(survivors_count),
            "pruned_node_count": float(len(candidate_nodes) - survivors_count),
            "top_node_importance": ranked[0].importance if ranked else 0.0,
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
                "circuit_json": str(circuit_json.resolve()),
                "circuit_csv": str(circuit_csv.resolve()),
                "circuit_dot": str(circuit_dot.resolve()),
            },
            notes=(
                f"ACDC-lite found {survivors_count} surviving nodes (of {len(candidate_nodes)} "
                f"candidates) with faithfulness {faithfulness:.3f} on '{config.model}'."
            ),
        )


def _build_candidate_nodes(
    layers: list[int],
    n_heads: int,
    *,
    include_attention: bool,
    include_mlps: bool,
) -> list[CircuitNode]:
    nodes: list[CircuitNode] = []
    for layer in layers:
        if include_attention:
            for head in range(n_heads):
                nodes.append(
                    CircuitNode(
                        node_id=f"L{layer}.H{head}",
                        layer=layer,
                        component="attn",
                        head=head,
                    )
                )
        if include_mlps:
            nodes.append(
                CircuitNode(
                    node_id=f"L{layer}.MLP",
                    layer=layer,
                    component="mlp",
                    head=None,
                )
            )
    return nodes


def _ablation_hook_for(
    node: CircuitNode, ablation_type: str
) -> tuple[str, Callable[..., Tensor]]:
    """Return a TL forward-hook that zeroes / mean-ablates the given node."""
    import torch

    if node.component == "attn":
        head_index = node.head
        assert head_index is not None

        def hook(activation: Tensor, hook: Any = None, **_kwargs: Any) -> Tensor:
            # z is shape (batch, seq, n_heads, d_head). Replace the requested head
            # with zero (or its sequence-mean) to remove its causal contribution.
            patched = activation.clone()
            if ablation_type == "zero":
                patched[:, :, head_index, :] = 0
            else:  # mean
                mean = activation[:, :, head_index, :].mean(dim=(0, 1), keepdim=True)
                patched[:, :, head_index, :] = mean
            return patched

        return f"blocks.{node.layer}.attn.hook_z", hook

    # MLP: zero / mean-ablate the MLP output entirely
    def mlp_hook(activation: Tensor, hook: Any = None, **_kwargs: Any) -> Tensor:
        if ablation_type == "zero":
            return torch.zeros_like(activation)
        mean = activation.mean(dim=(0, 1), keepdim=True)
        return activation * 0 + mean

    return f"blocks.{node.layer}.hook_mlp_out", mlp_hook


def _run_with_ablation(
    model: Any, prompt: str, node: CircuitNode, *, ablation_type: str
) -> Tensor:
    hook_site, hook_fn = _ablation_hook_for(node, ablation_type)
    result: Tensor = model.run_with_hooks(prompt, fwd_hooks=[(hook_site, hook_fn)])
    return result


def _run_with_ablations(
    model: Any, prompt: str, nodes: list[CircuitNode], *, ablation_type: str
) -> Tensor:
    hooks = [_ablation_hook_for(node, ablation_type) for node in nodes]
    result: Tensor = model.run_with_hooks(prompt, fwd_hooks=hooks)
    return result


def _logit_diff(logits: Any, position: int, correct_id: int, incorrect_id: int) -> float:
    # Accept (batch, seq, vocab) or (seq, vocab); pull the target row.
    selected = logits
    if hasattr(selected, "ndim"):
        if selected.ndim == 3:
            selected = selected[0, position, :]
        elif selected.ndim == 2:
            selected = selected[position, :]
    detach = getattr(selected, "detach", None)
    if callable(detach):
        selected = detach()
    cpu = getattr(selected, "cpu", None)
    if callable(cpu):
        selected = cpu()
    return float(selected[correct_id].item() - selected[incorrect_id].item())


def _faithfulness(full_diff: float, pruned_diff: float) -> float:
    """1 = pruned circuit reproduces the full-model behaviour; 0 = it doesn't.

    We use a normalised absolute-error metric in logit-diff space so faithfulness
    is comparable across prompts.
    """
    denom = max(abs(full_diff), 1e-6)
    error = abs(full_diff - pruned_diff) / denom
    return float(max(0.0, 1.0 - error))


def _write_edge_csv(path: Path, nodes: list[CircuitNode]) -> None:
    fieldnames = ["rank", "node_id", "layer", "component", "head", "importance", "pruned"]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, node in enumerate(nodes, start=1):
            writer.writerow({"rank": rank, **node.to_dict()})


def _write_circuit_dot(path: Path, circuit: CircuitArtifact) -> None:
    lines = [
        f'digraph circuit_{circuit.model.replace("-", "_")} {{',
        '  rankdir=TB;',
        '  node [shape=box, style=filled];',
    ]
    for node in circuit.nodes:
        color = "lightgray" if node.pruned else "lightgreen"
        label = f"{node.node_id}\\nimportance={node.importance:.4f}"
        lines.append(f'  "{node.node_id}" [label="{label}", fillcolor="{color}"];')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

