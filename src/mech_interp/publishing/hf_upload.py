"""HuggingFace Hub publishing pipeline.

Bundles SAE weights, steering vectors, and investigation documents into
self-contained packages that can be uploaded to a HuggingFace repo and later
loaded without cloning this repository.

Usage (--dry-run never touches HF)::

    mech publish-sae --run-id 51 --repo myorg/sae-gpt2-medium --dry-run
    mech publish-steering --vector sentiment-gpt2-medium-l8 --repo myorg/sv-sentiment --dry-run
    mech publish-investigation --slug sae_replication_crisis --repo myorg/inv-sae-rep --dry-run
    mech publish-all --user myorg --dry-run
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Optional HuggingFace Hub imports — present at module scope so tests can patch them.
# Both names are None when huggingface_hub is not installed; upload_bundle
# raises RuntimeError before they are used.
_HfApi: Any
_hf_create_repo: Any
try:
    from huggingface_hub import HfApi as _HfApi
    from huggingface_hub import create_repo as _hf_create_repo
except ImportError:  # pragma: no cover
    _HfApi = None
    _hf_create_repo = None

# Expose at the names that tests patch
HfApi = _HfApi
hf_create_repo = _hf_create_repo

# ---------------------------------------------------------------------------
# Bundle dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HubArtifactBundle:
    name: str
    """Canonical key matching the local registry name (or slug for investigations)."""
    kind: str
    """One of: 'sae' | 'steering' | 'investigation'."""
    local_paths: list[Path]
    """Ordered list of files that will be uploaded. First entry is always the README."""
    metadata: dict[str, Any]
    """Sidecar metadata written to bundle_metadata.json alongside the README."""
    license: str


# ---------------------------------------------------------------------------
# README generators
# ---------------------------------------------------------------------------


# HuggingFace model-card YAML validates `license:` against a controlled
# vocabulary; any non-recognised value (e.g. "research-only") 400s the
# validate-yaml endpoint and aborts the upload. Map known SPDX-ish ids
# through, fold everything else to "other" (HF's escape hatch), and keep the
# human-readable original in the card body via `license_name`.
_HF_LICENSE_TAGS = {
    "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "gpl-3.0", "gpl-2.0",
    "lgpl-3.0", "agpl-3.0", "cc-by-4.0", "cc-by-sa-4.0", "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0", "cc0-1.0", "cc", "openrail", "bigscience-openrail-m",
    "creativeml-openrail-m", "llama2", "llama3", "gemma", "other", "unknown",
}


def _hf_license_frontmatter(raw: str | None) -> str:
    """Return YAML frontmatter line(s) for a license HF will accept."""
    value = (raw or "other").strip().lower()
    if value in _HF_LICENSE_TAGS:
        return f"license: {value}"
    # Non-standard (e.g. research-only): use `other` + preserve the real name.
    safe_name = (raw or "other").strip()
    return f'license: other\nlicense_name: {safe_name}'


def _readme_sae(bundle_name: str, metadata: dict[str, Any], repo_id: str) -> str:
    """Generate HuggingFace-flavoured README for an SAE bundle."""
    spec = metadata.get("spec", {})
    result = metadata.get("result", {})
    env = metadata.get("environment", {})
    metrics = result.get("metrics", {})
    params = spec.get("parameters", {})

    model_name = env.get("model_name", params.get("model", "unknown"))
    hook_site = params.get("hook_site", "unknown")
    n_features = params.get("n_features", "?")
    k = params.get("k", "?")
    run_id = result.get("run_id", "?")
    ev = metrics.get("explained_variance", None)
    ev_str = f"{ev:.4f}" if isinstance(ev, float) else "?"
    live = metrics.get("live_features", "?")
    n_tok = metrics.get("n_tokens", "?")
    description = spec.get("description", "A Top-K Sparse Autoencoder trained locally.")

    return textwrap.dedent(f"""\
        ---
        {_hf_license_frontmatter(metadata.get("license"))}
        tags:
          - mechanistic-interpretability
          - sparse-autoencoder
          - {model_name}
        library_name: mech-interpretability
        ---

        # {bundle_name}

        {description}

        ## Model card

        | Field | Value |
        |---|---|
        | Model | `{model_name}` |
        | Hook site | `{hook_site}` |
        | Features | {n_features} (k={k}) |
        | Live features | {live} |
        | Explained variance | {ev_str} |
        | Training tokens | {n_tok} |
        | Source run | run-{run_id:06d} if known |

        ## How to use

        ```bash
        # Install
        pip install mech-interpretability  # or: uv sync --extra interp
        ```

        ```python
        from mech_interp.sae.registry import load_pretrained_sae

        sae, config = load_pretrained_sae("{bundle_name}")
        ```

        Or via the CLI:

        ```bash
        mech load-sae --name {bundle_name}
        ```

        ## Provenance

        - Source run: `run-{run_id:06d}` (see `environment.json` in this repo)
        - Python `{env.get("python_version", "?")}`, \
torch `{env.get("package_versions", {}).get("torch", "?")}`
        - Platform: `{repo_id}`

        ## Files

        | File | Description |
        |---|---|
        | `sae_weights.safetensors` | Encoder + decoder weights |
        | `sae_config.json` | Hyperparameters and training metadata |
        | `feature_analysis.json` | Per-feature top-activating prompts |
        | `environment.json` | Full environment provenance snapshot |
        | `bundle_metadata.json` | Machine-readable bundle manifest |
    """)


def _readme_steering(bundle_name: str, metadata: dict[str, Any], repo_id: str) -> str:
    """Generate HuggingFace-flavoured README for a steering vector bundle."""
    desc = metadata.get("description", "A pre-extracted steering vector.")
    model_name = metadata.get("model_name", "unknown")
    hook_site = metadata.get("hook_site", "unknown")
    direction_norm = metadata.get("direction_norm", "?")
    source_paper = metadata.get("source_paper", "—")
    source_run_id = metadata.get("source_run_id", None)
    license_ = metadata.get("license", "research-only")

    source_run_str = f"run-{source_run_id:06d}" if isinstance(source_run_id, int) else "—"
    direction_norm_str = f"{direction_norm:.4f}" if isinstance(direction_norm, float) else str(direction_norm)  # noqa: E501

    return textwrap.dedent(f"""\
        ---
        {_hf_license_frontmatter(license_)}
        tags:
          - mechanistic-interpretability
          - steering-vector
          - {model_name}
        library_name: mech-interpretability
        ---

        # {bundle_name}

        {desc}

        ## Vector card

        | Field | Value |
        |---|---|
        | Model | `{model_name}` |
        | Hook site | `{hook_site}` |
        | Direction norm (raw) | {direction_norm_str} |
        | Source paper | {source_paper} |
        | Source run | {source_run_str} |

        ## How to use

        ```bash
        mech apply-steering --vector {bundle_name} --coefficient 3.0 --prompt "Your prompt here"
        ```

        ```python
        from mech_interp.steering.registry import load_steering_vector

        direction, metadata = load_steering_vector("{bundle_name}")
        ```

        ## Provenance

        - Extraction method: mean-difference (Arditi/RepE)
        - Source: {source_paper}
        - Platform repo: `{repo_id}`

        ## Files

        | File | Description |
        |---|---|
        | `direction.safetensors` | Unit-norm direction tensor under key `"direction"` |
        | `direction.safetensors.json` | Extraction metadata sidecar |
        | `bundle_metadata.json` | Machine-readable bundle manifest |
    """)


def _readme_investigation(bundle_name: str, metadata: dict[str, Any], repo_id: str) -> str:
    """Generate HuggingFace-flavoured README for an investigation bundle."""
    title = metadata.get("title", bundle_name)
    description = metadata.get("description", "A mechanistic interpretability investigation.")
    license_ = metadata.get("license", "CC-BY-4.0")

    return textwrap.dedent(f"""\
        ---
        {_hf_license_frontmatter(license_)}
        tags:
          - mechanistic-interpretability
          - investigation
          - research
        library_name: mech-interpretability
        ---

        # {title}

        {description}

        ## How to use

        Read the investigation writeup in `investigation.md`.

        Load associated artifacts programmatically:

        ```bash
        mech load-investigation --slug {bundle_name}
        ```

        ## Provenance

        - Source: `docs/investigations/{bundle_name}.md` in the platform repo
        - Platform: `{repo_id}`

        ## Files

        | File | Description |
        |---|---|
        | `investigation.md` | Full investigation writeup |
        | `bundle_metadata.json` | Machine-readable bundle manifest |
    """)


# ---------------------------------------------------------------------------
# Bundle builders
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # src/mech_interp/publishing -> root


def build_sae_bundle(
    run_id: int,
    *,
    artifact_root: Path | None = None,
    license: str = "research-only",
) -> HubArtifactBundle:
    """Walk a SAE run's artifact directory into a publishable bundle.

    Includes sae_weights.safetensors, sae_config.json, feature_analysis.json,
    environment.json, and a generated README.md.

    Parameters
    ----------
    run_id:
        Local run ID (e.g. 51).
    artifact_root:
        Root artifacts directory. Defaults to ``<project_root>/artifacts``.
    license:
        SPDX license identifier or custom string for the README frontmatter.
    """
    root = artifact_root or (_PROJECT_ROOT / "artifacts")
    run_dir = root / f"run-{run_id:06d}"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run artifact directory not found: {run_dir}")

    # Required files
    weights = run_dir / "sae_weights.safetensors"
    sae_config = run_dir / "sae_weights.safetensors.json"
    environment = run_dir / "environment.json"
    manifest = run_dir / "manifest.json"
    result_json = run_dir / "result.json"
    spec_json = run_dir / "spec.json"
    feature_analysis = run_dir / "feature_analysis.json"

    local_paths: list[Path] = []
    for p in [weights, sae_config, environment, feature_analysis]:
        if p.exists():
            local_paths.append(p)

    if not local_paths:
        raise FileNotFoundError(f"No SAE artifacts found in {run_dir}")

    # Build metadata dict from available JSON files
    metadata: dict[str, Any] = {"run_id": run_id, "kind": "sae", "license": license}
    for name, path in [
        ("spec", spec_json),
        ("result", result_json),
        ("environment", environment),
        ("manifest", manifest),
        ("sae_config", sae_config),
    ]:
        if path.exists():
            try:
                metadata[name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

    spec = metadata.get("spec", {})
    params = spec.get("parameters", {})
    model_name = metadata.get("environment", {}).get("model_name", params.get("model", "unknown"))
    run_name = f"sae-run{run_id}-{model_name.replace('/', '-')}"

    return HubArtifactBundle(
        name=run_name,
        kind="sae",
        local_paths=local_paths,
        metadata=metadata,
        license=license,
    )


def build_steering_bundle(
    name: str,
    *,
    base_dir: Path | None = None,
) -> HubArtifactBundle:
    """Look up a registered steering vector and bundle its safetensors + sidecar.

    Parameters
    ----------
    name:
        Registry key (e.g. ``"sentiment-gpt2-medium-l8"``).
    base_dir:
        Project root to resolve relative ``local_path`` entries.
    """
    from mech_interp.steering.registry import STEERING_REGISTRY

    if name not in STEERING_REGISTRY:
        available = ", ".join(sorted(STEERING_REGISTRY))
        raise KeyError(f"Unknown steering vector '{name}'. Available: {available}.")

    descriptor = STEERING_REGISTRY[name]
    if descriptor.local_path is None:
        raise ValueError(
            f"Steering vector '{name}' has no local_path set. "
            "Cannot bundle without a local file."
        )

    resolved_base = base_dir or _PROJECT_ROOT
    safetensors_path = resolved_base / descriptor.local_path
    sidecar_path = safetensors_path.with_suffix(".safetensors.json")

    if not safetensors_path.exists():
        raise FileNotFoundError(
            f"Steering vector file not found: {safetensors_path}\n"
            "Run the extraction script to produce it."
        )

    local_paths: list[Path] = [safetensors_path]
    if sidecar_path.exists():
        local_paths.append(sidecar_path)

    # Build metadata from descriptor + sidecar
    metadata: dict[str, Any] = {
        "name": descriptor.name,
        "kind": "steering",
        "model_name": descriptor.model_name,
        "hook_site": descriptor.hook_site,
        "direction_norm": descriptor.direction_norm,
        "description": descriptor.description,
        "license": descriptor.license,
        "source_run_id": descriptor.source_run_id,
        "source_paper": descriptor.source_paper,
    }
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            metadata.update(sidecar)
        except (OSError, json.JSONDecodeError):
            pass

    return HubArtifactBundle(
        name=name,
        kind="steering",
        local_paths=local_paths,
        metadata=metadata,
        license=descriptor.license,
    )


def _parse_investigation_header(md_path: Path, slug: str) -> tuple[str, str]:
    """Extract ``(title, description)`` from a markdown investigation doc.

    Title is the first ``# heading``; description is the first non-heading,
    non-frontmatter paragraph (truncated to 200 chars). Falls back to the slug.
    """
    title = slug.replace("_", " ").title()
    description = f"Investigation: {title}"
    try:
        content = md_path.read_text(encoding="utf-8")
    except OSError:
        return title, description

    title_found = False
    in_frontmatter = False
    for line in content.splitlines():
        stripped = line.strip()
        if not title_found and stripped.startswith("# "):
            title = stripped[2:].strip()
            description = f"Investigation: {title}"
            title_found = True
            continue
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        if stripped and not stripped.startswith("#"):
            description = stripped[:200]
            break
    return title, description


def build_investigation_bundle(
    slug: str,
    *,
    docs_dir: Path | None = None,
) -> HubArtifactBundle:
    """Bundle an investigation writeup + any associated publication artifacts.

    Looks for:
    - ``docs/investigations/<slug>.md``  (required)
    - ``docs/publications/<slug>/``  (optional; included if present)

    Parameters
    ----------
    slug:
        Investigation slug matching the docs filename (e.g. ``"sae_replication_crisis"``).
    docs_dir:
        Override project docs directory. Defaults to ``<project_root>/docs``.
    """
    resolved_docs = docs_dir or (_PROJECT_ROOT / "docs")
    investigation_md = resolved_docs / "investigations" / f"{slug}.md"

    if not investigation_md.exists():
        raise FileNotFoundError(
            f"Investigation not found: {investigation_md}\n"
            f"Available slugs: check docs/investigations/"
        )

    local_paths: list[Path] = [investigation_md]

    # Include publication artifacts directory contents if present.
    # Accept either ``<slug>_artifacts/`` or plain ``<slug>/``.
    pub_dir: Path | None = next(
        (
            candidate
            for candidate in (
                resolved_docs / "publications" / f"{slug}_artifacts",
                resolved_docs / "publications" / slug,
            )
            if candidate.exists() and candidate.is_dir()
        ),
        None,
    )
    if pub_dir is not None:
        local_paths.extend(sorted(p for p in pub_dir.iterdir() if p.is_file()))

    title, description = _parse_investigation_header(investigation_md, slug)

    metadata: dict[str, Any] = {
        "slug": slug,
        "kind": "investigation",
        "title": title,
        "description": description,
        "license": "CC-BY-4.0",
        "has_publication_artifacts": pub_dir is not None,
    }

    return HubArtifactBundle(
        name=slug,
        kind="investigation",
        local_paths=local_paths,
        metadata=metadata,
        license="CC-BY-4.0",
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_bundle(
    bundle: HubArtifactBundle,
    *,
    repo_id: str,
    create_repo: bool = True,
    dry_run: bool = False,
    token: str | None = None,
) -> str:
    """Upload a bundle to a HuggingFace Hub repository.

    Parameters
    ----------
    bundle:
        The bundle produced by one of the ``build_*`` functions.
    repo_id:
        Full HF repo id, e.g. ``"myorg/sae-gpt2-medium"``.
    create_repo:
        Create the repository if it does not exist.
    dry_run:
        Print what would be uploaded without calling the Hub API.
    token:
        HuggingFace auth token. Falls back to the cached login token from
        ``huggingface-cli login`` when ``None``.

    Returns
    -------
    str
        The HuggingFace repo URL (real or simulated for dry-run).
    """
    import tempfile

    repo_url = f"https://huggingface.co/{repo_id}"

    if dry_run:
        _print_dry_run(bundle, repo_id, repo_url)
        return repo_url

    if HfApi is None or hf_create_repo is None:
        raise RuntimeError(
            "huggingface_hub is required for publishing. "
            "Run: uv sync --extra interp"
        )

    api = HfApi(token=token)

    if create_repo:
        hf_create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            token=token,
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _stage_bundle(bundle, repo_id, tmp_dir)
        api.upload_folder(
            folder_path=str(tmp_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {bundle.kind} bundle: {bundle.name}",
            token=token,
        )

    return repo_url


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_size(size_bytes: int) -> str:
    """Human-readable byte size with one decimal place."""
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _staged_name(bundle_kind: str, src_path: Path) -> str:
    """Return the in-repo filename for *src_path*, normalising steering / investigation files."""
    if bundle_kind == "steering":
        if src_path.suffix == ".safetensors":
            return "direction.safetensors"
        if src_path.name.endswith(".safetensors.json"):
            return "direction.safetensors.json"
    elif bundle_kind == "investigation" and src_path.suffix == ".md":
        return "investigation.md"
    return src_path.name


def _stage_bundle(bundle: HubArtifactBundle, repo_id: str, dest: Path) -> None:
    """Copy bundle files into a staging directory and write README + metadata."""
    import shutil

    (dest / "README.md").write_text(_generate_readme(bundle, repo_id), encoding="utf-8")
    (dest / "bundle_metadata.json").write_text(
        json.dumps(bundle.metadata, indent=2, default=str),
        encoding="utf-8",
    )

    for src_path in bundle.local_paths:
        if not src_path.exists():
            continue
        shutil.copy2(src_path, dest / _staged_name(bundle.kind, src_path))


_README_GENERATORS = {
    "sae": _readme_sae,
    "steering": _readme_steering,
    "investigation": _readme_investigation,
}


def _generate_readme(bundle: HubArtifactBundle, repo_id: str) -> str:
    """Dispatch to the appropriate README generator for this bundle kind."""
    try:
        generator = _README_GENERATORS[bundle.kind]
    except KeyError as exc:
        raise ValueError(f"Unknown bundle kind: {bundle.kind!r}") from exc
    return generator(bundle.name, bundle.metadata, repo_id)


def _print_dry_run(bundle: HubArtifactBundle, repo_id: str, repo_url: str) -> None:
    """Print a dry-run summary to stdout using Rich."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    console.print(
        Panel(
            f"[bold green]DRY RUN[/bold green] — no files uploaded\n"
            f"[bold]Bundle:[/bold] {bundle.name}\n"
            f"[bold]Kind:[/bold]   {bundle.kind}\n"
            f"[bold]Target:[/bold] {repo_url}",
            title="[bold #176b87]mech publish[/bold #176b87]",
            border_style="cyan",
        )
    )

    table = Table(title="Files that would be uploaded", show_lines=True)
    table.add_column("Staged name", style="cyan")
    table.add_column("Source path", style="dim")
    table.add_column("Size", justify="right")

    # README (generated, no source path)
    readme_text = _generate_readme(bundle, repo_id)
    table.add_row(
        "README.md",
        "[dim](generated)[/dim]",
        f"{len(readme_text.encode()):,} B",
    )

    # bundle_metadata.json (generated)
    meta_text = json.dumps(bundle.metadata, indent=2, default=str)
    table.add_row(
        "bundle_metadata.json",
        "[dim](generated)[/dim]",
        f"{len(meta_text.encode()):,} B",
    )

    for src_path in bundle.local_paths:
        dest_name = _staged_name(bundle.kind, src_path)
        if src_path.exists():
            size_str = _format_size(src_path.stat().st_size)
        else:
            size_str = "missing"
            dest_name = f"[red]{dest_name}[/red]"
        table.add_row(dest_name, str(src_path), size_str)

    console.print(table)
    console.print(
        "\n[dim]To upload for real, run without [bold]--dry-run[/bold] "
        "(requires [bold]huggingface-cli login[/bold]).[/dim]"
    )
