"""Keeps docs/OPERATOR_MANUAL.md honest: every documented module and script must exist, and every
documented config file must exist. If a command is renamed, this test fails until the manual is updated."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs" / "OPERATOR_MANUAL.md"
TEXT = MANUAL.read_text(encoding="utf-8")


def _documented_modules() -> set[str]:
    return set(re.findall(r"python -m (algolab\.[\w.]+)", TEXT))


def _documented_scripts() -> set[str]:
    return set(re.findall(r"python (scripts/[\w./-]+\.py)", TEXT))


def _documented_configs() -> set[str]:
    found = set(re.findall(r"`(config/[\w./-]+\.json)`", TEXT))
    found |= set(re.findall(r"`(config/[\w./-]+/)`", TEXT))
    return found


def test_manual_documents_commands():
    assert _documented_modules(), "manual should document at least one python -m command"
    assert _documented_scripts(), "manual should document at least one script"


@pytest.mark.parametrize("module", sorted(_documented_modules()))
def test_documented_module_exists(module):
    assert importlib.util.find_spec(module) is not None, f"manual documents missing module {module}"


@pytest.mark.parametrize("script", sorted(_documented_scripts()))
def test_documented_script_exists(script):
    assert (ROOT / script).is_file(), f"manual documents missing script {script}"


@pytest.mark.parametrize("path", sorted(_documented_configs()))
def test_documented_config_exists(path):
    assert (ROOT / path).exists(), f"manual documents missing config {path}"


def test_manual_names_the_safety_defaults():
    for needle in ('"enabled": false', "DRY_RUN=true", "reset --confirm"):
        assert needle in TEXT, f"manual must state the safety default: {needle}"
