#!/usr/bin/env python3
"""Grade IOI circuit reproduction runs against Wang et al. 2022 canonical heads.

Usage:
  uv run --extra interp python scripts/grade_ioi.py --run-id <id>
  uv run --extra interp python scripts/grade_ioi.py --edges-json <path>
  uv run --extra interp python scripts/grade_ioi.py --list-runs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mech_interp.analysis.ioi_validation import compare_to_canonical_ioi


def _find_edges_json(run_id: int) -> Path | None:
    """Search standard artifact dirs for edges.json for a given run ID."""
    candidates = [
        Path("artifacts") / str(run_id) / "edges.json",
        Path(".mech_interp") / "artifacts" / str(run_id) / "edges.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Walk all artifact dirs
    for base in [Path("artifacts"), Path(".mech_interp/artifacts")]:
        if base.exists():
            for found in base.rglob("edges.json"):
                if f"/{run_id}/" in str(found) or str(found).startswith(str(base / str(run_id))):
                    return found
    return None


def _find_circuit_json(run_id: int) -> Path | None:
    """Search for circuit.json (acdc_lite output) for a given run ID."""
    candidates = [
        Path("artifacts") / str(run_id) / "circuit.json",
        Path(".mech_interp") / "artifacts" / str(run_id) / "circuit.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    for base in [Path("artifacts"), Path(".mech_interp/artifacts")]:
        if base.exists():
            for found in base.rglob("circuit.json"):
                if f"/{run_id}/" in str(found):
                    return found
    return None


def grade_edges_json(path: Path) -> dict:
    """Load and grade an edges.json file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return compare_to_canonical_ioi(data), data


def grade_circuit_json(path: Path) -> dict:
    """Convert acdc_lite circuit.json to edges.json format for grading."""
    data = json.loads(path.read_text(encoding="utf-8"))
    # acdc_lite nodes → synthesize fake edge list so compare_to_canonical_ioi can parse them
    surviving_nodes = [n for n in data.get("nodes", []) if not n.get("pruned", True)]
    fake_edges = []
    for node in surviving_nodes:
        node_id = node["node_id"]
        # Create a self-edge so the node appears as both src and dst
        fake_edges.append({
            "src_id": node_id,
            "dst_id": node_id,
            "pruned": False,
        })
    fake_data = {
        "edges": fake_edges,
        "faithfulness": data.get("faithfulness", 0.0),
    }
    return compare_to_canonical_ioi(fake_data), data


def print_report(result: dict, raw: dict, title: str = "IOI Validation Report") -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"\nFaithfulness:  {result['faithfulness']:.4f}")
    print(f"Recall:        {result['recall']:.4f}  ({int(result['recall']*result['total_canonical_heads'])}/{result['total_canonical_heads']} canonical heads)")
    print(f"Precision:     {result['precision']:.4f}")
    print(f"Groups hit:    {result['groups_with_any_hit']}/5")

    print("\nPer-group recall:")
    groups = result["canonical_groups_hit"]
    canonical_sizes = {
        "name_mover": 3,
        "s_inhibition": 2,
        "backup_name_mover": 2,
        "induction": 3,
        "duplicate_token": 2,
    }
    for group, hits in groups.items():
        size = canonical_sizes.get(group, "?")
        hit_strs = [f"L{h[0]}.H{h[1]}" for h in hits] if hits else ["none"]
        print(f"  {group:22s}: {len(hits)}/{size}  {hit_strs}")

    print(f"\nSurviving canonical nodes: {result['surviving_nodes']}")

    # Top-10 surviving edges (for acdc_edge)
    edges = [e for e in raw.get("edges", []) if not e.get("pruned", True)]
    edges.sort(key=lambda e: e.get("importance", 0), reverse=True)
    if edges:
        print(f"\nTop-10 surviving edges (by importance):")
        for i, e in enumerate(edges[:10], 1):
            src = e.get("src_id", "?")
            dst = e.get("dst_id", "?")
            imp = e.get("importance", 0)
            print(f"  {i:2d}. {src} -> {dst}  importance={imp:.5f}")

    # For acdc_lite nodes
    nodes = [n for n in raw.get("nodes", []) if not n.get("pruned", True)]
    nodes.sort(key=lambda n: n.get("importance", 0), reverse=True)
    if nodes and not edges:
        print(f"\nTop-10 surviving nodes (by importance):")
        for i, n in enumerate(nodes[:10], 1):
            nid = n.get("node_id", "?")
            imp = n.get("importance", 0)
            print(f"  {i:2d}. {nid}  importance={imp:.5f}")

    print()


def list_runs() -> None:
    """List recent runs to help find the right run IDs."""
    import subprocess
    result = subprocess.run(
        ["uv", "run", "--extra", "interp", "mech", "runs"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr[:500])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade IOI run against canonical heads.")
    parser.add_argument("--run-id", type=int, help="Run ID to look up edges.json for")
    parser.add_argument("--lite-run-id", type=int, help="acdc_lite run ID to look up circuit.json for")
    parser.add_argument("--edges-json", type=str, help="Direct path to edges.json")
    parser.add_argument("--circuit-json", type=str, help="Direct path to circuit.json (acdc_lite)")
    parser.add_argument("--list-runs", action="store_true", help="List recent runs and exit")
    args = parser.parse_args()

    if args.list_runs:
        list_runs()
        sys.exit(0)

    if args.edges_json:
        path = Path(args.edges_json)
        result, raw = grade_edges_json(path)
        print_report(result, raw, f"acdc_edge — {path}")

    elif args.circuit_json:
        path = Path(args.circuit_json)
        result, raw = grade_circuit_json(path)
        print_report(result, raw, f"acdc_lite — {path}")

    elif args.run_id:
        path = _find_edges_json(args.run_id)
        if path is None:
            print(f"ERROR: Could not find edges.json for run {args.run_id}", file=sys.stderr)
            sys.exit(1)
        result, raw = grade_edges_json(path)
        print_report(result, raw, f"acdc_edge run {args.run_id} — {path}")

    elif args.lite_run_id:
        path = _find_circuit_json(args.lite_run_id)
        if path is None:
            print(f"ERROR: Could not find circuit.json for run {args.lite_run_id}", file=sys.stderr)
            sys.exit(1)
        result, raw = grade_circuit_json(path)
        print_report(result, raw, f"acdc_lite run {args.lite_run_id} — {path}")

    else:
        parser.print_help()
