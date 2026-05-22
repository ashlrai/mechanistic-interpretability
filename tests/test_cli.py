from pathlib import Path

from typer.testing import CliRunner

from mech_interp import cli


def test_validate_command_accepts_default_experiments() -> None:
    result = CliRunner().invoke(cli.app, ["validate"])

    assert result.exit_code == 0
    assert "Validated 3 experiment spec" in result.output


def test_validate_command_fails_invalid_specs(tmp_path: Path) -> None:
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text(
        """
name: bad
family: nope
backend: transformerlens
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli.app, ["validate", "--directory", str(tmp_path)])

    assert result.exit_code == 1
    assert "Invalid experiment specs" in result.output
    assert "bad.yaml" in result.output
    assert "unsupported family 'nope'" in result.output
