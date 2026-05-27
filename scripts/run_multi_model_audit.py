#!/usr/bin/env python3
"""Run the 4-stage refusal audit pipeline across multiple models.

Usage
-----
    # Try the priority list and run the first 3 that load:
    uv run --extra interp python scripts/run_multi_model_audit.py

    # Run specific models by slug:
    uv run --extra interp python scripts/run_multi_model_audit.py \\
        --models qwen2_0_5b qwen25_0_5b qwen25_3b

    # Dry-run: show plan without executing:
    uv run --extra interp python scripts/run_multi_model_audit.py --dry-run

Idempotent: stages that already have a successful run for the given model slug
are skipped (detected by spec name lookup in the store).

Output
------
For each completed model audit, writes:
  docs/investigations/refusal_audit_<slug>.json
  docs/investigations/refusal_audit_<slug>.md

A cross-model summary is appended to:
  docs/investigations/refusal_audit_multi_model.md
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Priority-ordered candidate models — (slug, hf_name, n_layers)
# ---------------------------------------------------------------------------
PRIORITY_CANDIDATES: list[tuple[str, str, int]] = [
    ("qwen2_0_5b",   "Qwen/Qwen2-0.5B-Instruct",               24),
    ("qwen25_0_5b",  "Qwen/Qwen2.5-0.5B-Instruct",             24),
    ("qwen25_3b",    "Qwen/Qwen2.5-3B-Instruct",               36),
    ("llama32_1b",   "meta-llama/Llama-3.2-1B-Instruct",       16),
    ("phi3_mini",    "microsoft/Phi-3-mini-4k-instruct",        32),
    ("gemma2_2b",    "google/gemma-2-2b-it",                    26),
    ("stablelm_3b",  "stabilityai/stablelm-tuned-alpha-3b",     32),
]

# Stage definitions: (family, yaml_prefix)
STAGES: list[tuple[str, str]] = [
    ("refusal_direction",       "refusal_direction"),
    ("caa_steering",            "caa_steering"),
    ("refusal_circuit",         "refusal_circuit"),
    ("causal_scrubbing",        "causal_scrubbing_refusal"),
]

REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DOCS_DIR = REPO_ROOT / "docs" / "investigations"


class AuditRunIDs(NamedTuple):
    refusal_run: int
    caa_run: int
    circuit_run: int
    scrub_run: int


def probe_model_load(hf_name: str, timeout: int = 120) -> bool:
    """Return True if HookedTransformer.from_pretrained loads without error."""
    probe_script = f"""
import warnings
warnings.filterwarnings("ignore")
try:
    from transformer_lens import HookedTransformer
    m = HookedTransformer.from_pretrained(
        "{hf_name}",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        device="cpu",
        dtype="float32",
    )
    print("OK", m.cfg.n_layers)
except Exception as e:
    print("FAIL", str(e)[:200])
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_script],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        stdout = result.stdout.strip()
        if stdout.startswith("OK"):
            print(f"  [probe] {hf_name}: loaded ({stdout})")
            return True
        print(f"  [probe] {hf_name}: FAILED — {stdout or result.stderr[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [probe] {hf_name}: timeout after {timeout}s")
        return False
    except Exception as exc:
        print(f"  [probe] {hf_name}: error — {exc}")
        return False


def spec_name_for(family_prefix: str, slug: str) -> str:
    """Return the experiment spec name as written by gen_audit_yamls.py."""
    return f"{family_prefix.replace('_', '-')}-{slug.replace('_', '-')}"


def get_run_id_for_spec(slug: str, family_prefix: str) -> int | None:
    """Look up the most recent successful run ID for a spec name via mech CLI."""
    spec_name = spec_name_for(family_prefix, slug)
    try:
        result = subprocess.run(
            ["uv", "run", "--group", "dev", "mech", "runs",
             "--status", "completed", "--spec", spec_name, "--limit", "1"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=30,
        )
        # Output is a Rich table; look for a run-ID integer on the first data row.
        for line in result.stdout.splitlines():
            parts = line.split()
            for p in parts:
                if p.isdigit():
                    return int(p)
    except Exception:
        pass
    return None


def run_stage(slug: str, family_prefix: str, dry_run: bool = False) -> int | None:
    """Run a single experiment stage; return run ID or None on failure."""
    spec_name = spec_name_for(family_prefix, slug)

    # Idempotency check
    existing = get_run_id_for_spec(slug, family_prefix)
    if existing is not None:
        print(f"    [skip] {spec_name} already completed (run {existing})")
        return existing

    yaml_name = f"{family_prefix}_{slug}.yaml"
    yaml_path = EXPERIMENTS_DIR / yaml_name
    if not yaml_path.exists():
        print(f"    [error] YAML not found: {yaml_path}")
        return None

    print(f"    [run] {spec_name} …")
    if dry_run:
        print(f"    [dry] would run: mech run --name {spec_name}")
        return -1  # sentinel for dry-run

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["uv", "run", "--extra", "interp", "mech", "run", "--name", spec_name],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=7200,  # 2-hour per-stage timeout
        )
        elapsed = time.monotonic() - t0
        print(f"    [done] {spec_name} in {elapsed:.0f}s (exit={proc.returncode})")
        if proc.returncode != 0:
            print(f"    [stderr] {proc.stderr[-500:]}")
            return None
        # Parse run ID from table output
        for line in proc.stdout.splitlines():
            parts = line.split()
            for p in parts:
                if p.isdigit() and int(p) > 0:
                    return int(p)
    except subprocess.TimeoutExpired:
        print(f"    [timeout] {spec_name} exceeded 2h")
    except Exception as exc:
        print(f"    [error] {spec_name}: {exc}")
    return None


def compile_audit(slug: str, run_ids: AuditRunIDs, dry_run: bool = False) -> Path | None:
    """Run mech audit-refusal to compile the 4-run report."""
    output_stem = str(DOCS_DIR / f"refusal_audit_{slug}")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  [compile] mech audit-refusal for {slug} …")
    if dry_run:
        print(
            f"  [dry] would run: mech audit-refusal "
            f"--refusal-run {run_ids.refusal_run} "
            f"--caa-run {run_ids.caa_run} "
            f"--circuit-run {run_ids.circuit_run} "
            f"--scrub-run {run_ids.scrub_run} "
            f"--output {output_stem}"
        )
        return Path(output_stem + ".md")

    result = subprocess.run(
        [
            "uv", "run", "--extra", "interp", "mech", "audit-refusal",
            "--refusal-run", str(run_ids.refusal_run),
            "--caa-run", str(run_ids.caa_run),
            "--circuit-run", str(run_ids.circuit_run),
            "--scrub-run", str(run_ids.scrub_run),
            "--output", output_stem,
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    if result.returncode != 0:
        print(f"  [error] audit-refusal failed: {result.stderr[-300:]}")
        return None
    md_path = Path(output_stem + ".md")
    if md_path.exists():
        print(f"  [report] {md_path}")
    return md_path if md_path.exists() else None


def run_model_audit(
    slug: str,
    hf_name: str,
    n_layers: int,
    dry_run: bool,
    skip_probe: bool,
) -> AuditRunIDs | None:
    """Full 4-stage pipeline for one model. Returns run IDs or None on failure."""
    print(f"\n{'='*60}")
    print(f"  MODEL: {hf_name}  (L={n_layers}, slug={slug})")
    print(f"{'='*60}")

    if not skip_probe:
        if not probe_model_load(hf_name):
            print(f"  [skip] {hf_name} failed to load — skipping")
            return None

    run_ids: list[int | None] = []
    family_prefixes = ["refusal_direction", "caa_steering", "refusal_circuit", "causal_scrubbing_refusal"]

    for i, ((_family, _), prefix) in enumerate(zip(STAGES, family_prefixes)):
        print(f"\n  Stage {i+1}/4: {prefix}")
        rid = run_stage(slug, prefix, dry_run=dry_run)
        run_ids.append(rid)
        if rid is None:
            print(f"  [abort] Stage {i+1} failed — aborting audit for {slug}")
            return None

    if dry_run:
        print(f"  [dry] all 4 stages planned for {slug}")
        return AuditRunIDs(-1, -1, -1, -1)

    assert all(r is not None for r in run_ids)
    return AuditRunIDs(
        refusal_run=run_ids[0],  # type: ignore[arg-type]
        caa_run=run_ids[1],      # type: ignore[arg-type]
        circuit_run=run_ids[2],  # type: ignore[arg-type]
        scrub_run=run_ids[3],    # type: ignore[arg-type]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="SLUG",
        help="Model slugs to run (from catalogue). Default: first 3 that load.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=3,
        help="Stop after this many models complete successfully (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without executing any runs.",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip model-load probe (trust the YAML catalogue).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List candidate models and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for slug, hf, nl in PRIORITY_CANDIDATES:
            print(f"  {slug:20s}  L={nl:3d}  {hf}")
        return

    # Resolve which models to attempt
    if args.models:
        slug_map = {s: (h, n) for s, h, n in PRIORITY_CANDIDATES}
        candidates = []
        for s in args.models:
            if s not in slug_map:
                print(f"[warn] Unknown slug {s!r} — skipping")
                continue
            candidates.append((s, slug_map[s][0], slug_map[s][1]))
    else:
        candidates = list(PRIORITY_CANDIDATES)

    print(f"Multi-model refusal audit — attempting up to {args.max_models} models")
    print(f"Candidate queue: {[s for s,_,_ in candidates]}\n")

    completed: list[tuple[str, str, AuditRunIDs]] = []
    failed: list[str] = []

    for slug, hf_name, n_layers in candidates:
        if len(completed) >= args.max_models:
            print(f"\n[done] Reached target of {args.max_models} models — stopping.")
            break

        ids = run_model_audit(
            slug=slug,
            hf_name=hf_name,
            n_layers=n_layers,
            dry_run=args.dry_run,
            skip_probe=args.skip_probe,
        )
        if ids is None:
            failed.append(slug)
            continue

        if not args.dry_run:
            compile_audit(slug, ids, dry_run=False)

        completed.append((slug, hf_name, ids))

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(completed)} completed, {len(failed)} failed/skipped")
    if completed:
        print("Completed:")
        for slug, hf, ids in completed:
            print(f"  {slug}: runs {ids.refusal_run}/{ids.caa_run}/{ids.circuit_run}/{ids.scrub_run}")
    if failed:
        print(f"Failed/skipped: {failed}")

    if not args.dry_run and len(completed) > 0:
        print("\nNext step — run cross-model analysis:")
        slugs_str = " ".join(s for s, _, _ in completed)
        print(f"  uv run --extra interp python -m mech_interp.analysis.refusal_audit_multi \\")
        print(f"    --slugs {slugs_str} --output docs/investigations/refusal_audit_multi_model.md")


if __name__ == "__main__":
    main()
