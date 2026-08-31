"""Tests for private-repo credentials in the VLM job bootstrap.

A PAT reaching a log line or a job's plaintext environment is a credential leak, and the job's own
metadata is world-readable to anyone who can inspect it. These pin both the injection and the
redaction, including the case that actually bites: ``git+https://host/o/r.git@ref`` already uses
``@`` for the ref, so credentials add a second one and a naive first-match split would parse the
token as the branch name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_vlm_job import _redact, _with_credentials  # noqa: E402

SOURCE = "git+https://github.com/prarabdhmisra/quantiphy.git@fix/prior-grounding-phrase"


class TestWithCredentials:
    def test_no_token_is_a_no_op(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        assert _with_credentials(SOURCE) == SOURCE

    def test_token_is_injected_after_the_scheme(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        assert _with_credentials(SOURCE) == (
            "git+https://x-access-token:ghp_secret@github.com/"
            "prarabdhmisra/quantiphy.git@fix/prior-grounding-phrase")

    def test_the_ref_still_parses_off_the_clean_url(self, monkeypatch):
        """The clean URL keeps exactly one ``@``, so the fallback clone's partition is safe."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        url = SOURCE.removeprefix("git+")
        clean, _, ref = url.partition("@")
        assert ref == "fix/prior-grounding-phrase"
        assert _with_credentials(clean).endswith("github.com/prarabdhmisra/quantiphy.git")

    def test_gh_token_is_accepted_too(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "ghp_other")
        assert "x-access-token:ghp_other@" in _with_credentials(SOURCE)

    def test_a_non_github_host_is_left_alone(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        other = "git+https://gitlab.com/o/r.git@main"
        assert _with_credentials(other) == other

    def test_existing_credentials_are_not_doubled(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        already = "git+https://user:pw@github.com/o/r.git@main"
        assert _with_credentials(already) == already

    def test_a_local_path_is_left_alone(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        assert _with_credentials("/srv/quantiphy") == "/srv/quantiphy"


class TestRedact:
    def test_the_token_is_replaced(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        assert _redact("failed on ghp_secret@github.com") == "failed on ***@github.com"

    def test_a_tokenised_command_is_scrubbed(self, monkeypatch):
        """The real leak: CalledProcessError stringifies the whole command list."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        message = f"Command '['uv', 'pip', 'install', '{_with_credentials(SOURCE)}']' returned 1"
        assert "ghp_secret" not in _redact(message)

    def test_no_token_set_is_a_no_op(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        assert _redact("nothing to hide") == "nothing to hide"
