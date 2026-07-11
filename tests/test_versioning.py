"""Tests for the automated release version contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_release_version import SEMVER_RE, validate_release_tag
from src.version import __version__


ROOT = Path(__file__).resolve().parents[1]


def test_application_version_is_semver() -> None:
    assert SEMVER_RE.fullmatch(__version__)


def test_release_manifest_matches_application_version() -> None:
    manifest = json.loads((ROOT / ".release-please-manifest.json").read_text())
    assert manifest["."] == __version__


def test_release_tag_must_match_application_version() -> None:
    validate_release_tag(f"v{__version__}")

    with pytest.raises(ValueError, match="does not match"):
        validate_release_tag("v99.0.0")


def test_release_config_updates_application_version() -> None:
    config = json.loads((ROOT / "release-please-config.json").read_text())
    extra_files = config["packages"]["."]["extra-files"]
    assert {entry["path"] for entry in extra_files} == {"src/version.py"}
