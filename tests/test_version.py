"""The version is defined once (``src/otaf/_version.py``); the files that repeat it must agree.

``CITATION.cff`` is parsed with regular expressions: it is flat YAML and the test
suite does not depend on a YAML parser.
"""

import importlib.metadata
import re
from pathlib import Path

import otaf

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")


def _citation_field(name):
    match = re.search(rf'^{name}:\s*"?([^"\n]+?)"?\s*$', (ROOT / "CITATION.cff").read_text(), re.MULTILINE)
    assert match, f"CITATION.cff has no '{name}' field"
    return match.group(1)


def _latest_changelog_release():
    """``(version, date)`` of the first released entry of CHANGELOG.md (after Unreleased)."""
    match = re.search(r"^## \[(\d[^\]]*)\] - (\S+)", (ROOT / "CHANGELOG.md").read_text(), re.MULTILINE)
    assert match, "CHANGELOG.md has no released entry"
    return match.group(1), match.group(2)


def test_version_is_semantic():
    assert SEMVER.match(otaf.__version__), otaf.__version__


def test_installed_metadata_matches_version():
    # pyproject.toml reads the version from otaf._version (setuptools dynamic attr).
    assert importlib.metadata.version("otaf") == otaf.__version__


def test_citation_matches_version():
    assert _citation_field("version") == otaf.__version__


def test_changelog_top_entry_matches_version_and_citation_date():
    version, date = _latest_changelog_release()
    assert version == otaf.__version__
    assert date == _citation_field("date-released")
