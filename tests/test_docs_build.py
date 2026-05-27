"""Fast unit gate: validate mkdocs.yml parses without building the site."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_mkdocs_yml_parses() -> None:
    """mkdocs.yml exists and is valid YAML."""
    import yaml

    mkdocs_path = PROJECT_ROOT / "mkdocs.yml"
    assert mkdocs_path.exists(), "mkdocs.yml not found at repo root"
    with mkdocs_path.open() as fh:
        config = yaml.safe_load(fh)
    assert isinstance(config, dict), "mkdocs.yml did not parse as a mapping"
    assert "site_name" in config, "mkdocs.yml missing site_name"
    assert "nav" in config, "mkdocs.yml missing nav"
    assert "theme" in config, "mkdocs.yml missing theme"


def test_mkdocs_site_name() -> None:
    import yaml

    with (PROJECT_ROOT / "mkdocs.yml").open() as fh:
        config = yaml.safe_load(fh)
    assert config["site_name"] == "Mechanistic Interpretability Platform"


def test_mkdocs_theme_is_material() -> None:
    import yaml

    with (PROJECT_ROOT / "mkdocs.yml").open() as fh:
        config = yaml.safe_load(fh)
    assert config["theme"]["name"] == "material"


def test_nav_references_existing_docs() -> None:
    """Every file listed in nav must exist under docs/."""
    import yaml

    with (PROJECT_ROOT / "mkdocs.yml").open() as fh:
        config = yaml.safe_load(fh)

    docs_dir = PROJECT_ROOT / "docs"

    def collect_paths(nav_node: object) -> list[str]:
        paths: list[str] = []
        if isinstance(nav_node, str):
            paths.append(nav_node)
        elif isinstance(nav_node, dict):
            for v in nav_node.values():
                paths.extend(collect_paths(v))
        elif isinstance(nav_node, list):
            for item in nav_node:
                paths.extend(collect_paths(item))
        return paths

    nav_paths = collect_paths(config.get("nav", []))
    missing = [p for p in nav_paths if not (docs_dir / p).exists()]
    assert not missing, f"Nav references missing docs files: {missing}"


def test_docs_index_exists() -> None:
    assert (PROJECT_ROOT / "docs" / "index.md").exists()


def test_docs_getting_started_exists() -> None:
    assert (PROJECT_ROOT / "docs" / "getting-started.md").exists()


def test_docs_investigations_index_exists() -> None:
    assert (PROJECT_ROOT / "docs" / "investigations" / "index.md").exists()


def test_docs_publications_index_exists() -> None:
    assert (PROJECT_ROOT / "docs" / "publications" / "index.md").exists()


def test_docs_reference_families_exists() -> None:
    assert (PROJECT_ROOT / "docs" / "reference" / "families.md").exists()
