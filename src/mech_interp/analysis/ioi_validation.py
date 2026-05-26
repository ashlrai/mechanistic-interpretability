"""IOI circuit validation utilities.

Compares a set of discovered edges/nodes against the canonical Indirect Object
Identification (IOI) circuit heads identified by Wang et al., 2022:
  "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"
  https://arxiv.org/abs/2211.00593

The canonical head groups (layer.head notation for GPT-2 small):

- Name-mover heads:    9.6,  9.9, 10.0
- S-inhibition heads:  7.3,  8.6
- Backup name-movers: 10.7, 11.10
- Induction heads:     5.5,  5.8,  5.9
- Duplicate token:     0.1,  3.0

This module is used by the integration test and can be called from the cockpit
or a ``mech validate-circuit`` CLI command.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Canonical IOI head groups (Wang et al., 2022)
# ---------------------------------------------------------------------------

# Each group maps to a set of (layer, head) tuples.
# Node IDs in acdc_edge use the format "L{layer}.H{head}".
CANONICAL_IOI_HEADS: dict[str, set[tuple[int, int]]] = {
    "name_mover": {(9, 6), (9, 9), (10, 0)},
    "s_inhibition": {(7, 3), (8, 6)},
    "backup_name_mover": {(10, 7), (11, 10)},
    "induction": {(5, 5), (5, 8), (5, 9)},
    "duplicate_token": {(0, 1), (3, 0)},
}


def _node_id_to_layer_head(node_id: str) -> tuple[int, int] | None:
    """Parse 'L7.H3' → (7, 3).  Returns None for MLP nodes or unrecognised formats."""
    # Expected format: "L{layer}.H{head}" or "L{layer}.H{head}->{...}"
    # Strip edge suffixes if present (e.g. from edge_id).
    part = node_id.split("->")[0].strip()
    if not part.startswith("L"):
        return None
    try:
        layer_part, comp_part = part[1:].split(".", 1)
        if not comp_part.startswith("H"):
            return None
        return int(layer_part), int(comp_part[1:])
    except (ValueError, IndexError):
        return None


def _collect_surviving_nodes(edges_json: dict[str, Any]) -> set[tuple[int, int]]:
    """Extract the set of (layer, head) tuples for all surviving edge endpoints."""
    surviving: set[tuple[int, int]] = set()
    for edge in edges_json.get("edges", []):
        if not isinstance(edge, dict):
            continue
        if edge.get("pruned", True):
            continue
        for field in ("src_id", "dst_id"):
            lh = _node_id_to_layer_head(str(edge.get(field, "")))
            if lh is not None:
                surviving.add(lh)
    return surviving


def compare_to_canonical_ioi(edges_json: dict[str, Any]) -> dict[str, Any]:
    """Compare discovered circuit edges against canonical IOI head groups.

    Parameters
    ----------
    edges_json:
        Parsed ``edges.json`` artifact dict (as produced by ``ACDCEdgeExperiment``).

    Returns
    -------
    dict with keys:
        ``surviving_nodes``       — list of (layer, head) tuples in the circuit.
        ``canonical_groups_hit``  — dict of group_name → list of matched heads.
        ``groups_with_any_hit``   — count of canonical groups with ≥1 match.
        ``total_canonical_heads`` — total number of canonical heads across all groups.
        ``recall``                — fraction of canonical heads found.
        ``precision``             — fraction of surviving attn nodes that are canonical.
        ``faithfulness``          — faithfulness value from edges_json (0.0 if absent).
    """
    surviving = _collect_surviving_nodes(edges_json)

    # All canonical heads flattened.
    all_canonical: set[tuple[int, int]] = set()
    for heads in CANONICAL_IOI_HEADS.values():
        all_canonical |= heads

    groups_hit: dict[str, list[tuple[int, int]]] = {}
    for group_name, heads in CANONICAL_IOI_HEADS.items():
        matched = sorted(surviving & heads)
        groups_hit[group_name] = matched

    groups_with_any_hit = sum(1 for v in groups_hit.values() if v)

    total_canonical = len(all_canonical)
    true_positives = len(surviving & all_canonical)
    recall = true_positives / total_canonical if total_canonical else 0.0
    precision = true_positives / len(surviving) if surviving else 0.0

    faithfulness = float(edges_json.get("faithfulness", 0.0))

    return {
        "surviving_nodes": sorted(surviving),
        "canonical_groups_hit": {
            k: [list(lh) for lh in v] for k, v in groups_hit.items()
        },
        "groups_with_any_hit": groups_with_any_hit,
        "total_canonical_heads": total_canonical,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "faithfulness": faithfulness,
    }
