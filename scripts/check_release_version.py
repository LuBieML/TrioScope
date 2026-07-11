#!/usr/bin/env python3
"""Validate that a release tag and TrioScope's source version agree."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.version import __version__


SEMVER_RE = re.compile(
    r"^(?:0|[1-9]\d*)\."
    r"(?:0|[1-9]\d*)\."
    r"(?:0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def validate_release_tag(tag: str, version: str = __version__) -> None:
    """Raise ``ValueError`` when *tag* is invalid or mismatches *version*."""
    if not SEMVER_RE.fullmatch(version):
        raise ValueError(f"src/version.py contains invalid SemVer: {version!r}")

    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ValueError(
            f"release tag {tag!r} does not match application version "
            f"{version!r}; expected {expected_tag!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Release tag, for example v1.2.3")
    args = parser.parse_args()

    try:
        validate_release_tag(args.tag)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Release tag {args.tag} matches TrioScope {__version__}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
