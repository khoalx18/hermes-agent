"""Regression tests for the NUL-byte binary crash in cron/lifecycle_guard.py.

The incident: every agent command that invokes a real binary by absolute
path (e.g. ``C:/Python314/python.exe cos.py scan-discover``) crashed the
terminal with ``Failed to execute command: stat: embedded null character
in path`` (exit -1), blocking whole pipelines.

Root-cause chain in the guard:

  1. ``_iter_referenced_shell_scripts`` treats any executable containing
     ``/`` as a script path, so ``C:/Python314/python.exe`` is yielded as
     a script to scan.
  2. ``_read_referenced_script`` reads the binary as text and decodes it
     with ``errors="replace"`` — but the NUL byte is VALID UTF-8
     (U+0000), so ``\\x00`` survives the decode into the scanned string.
  3. ``_contains_unsafe_gateway_action`` recurses into that binary-shaped
     text; any token containing a NUL lands in ``Path(...).resolve()``,
     which raises ``ValueError: embedded null character in path`` — not
     an ``OSError``, so the guard's except clause does not catch it and
     the whole command fails.

These tests pin the required behaviour: binary files are skipped (not
decoded/recursed), absolute Windows binary paths pass through as safe
when no gateway action is present, and real shell scripts / ``-c``
payloads carrying an unsafe gateway action are still detected.

The source fix is tracked separately; these tests document the contract
and may fail until it lands.
"""

from pathlib import Path

import pytest

from cron.lifecycle_guard import (
    contains_gateway_lifecycle_command_or_referenced_script,
)


def test_absolute_windows_binary_path_is_safe():
    """A command invoking a real Windows binary must not raise and must be safe.

    ``C:/Python314/python.exe`` exists on the affected machines, so before
    the fix this command crashes with ``ValueError: embedded null character
    in path`` while scanning the binary as if it were a script. It must
    neither raise nor be treated as unsafe — there is no gateway action.
    """
    result = contains_gateway_lifecycle_command_or_referenced_script(
        "C:/Python314/python.exe cos.py scan-discover"
    )
    assert result is False


def test_nul_byte_referenced_script_is_skipped(tmp_path):
    """A referenced file containing a NUL byte is skipped, not scanned.

    The guard must not decode/recursively scan a file whose content is
    binary (contains ``\\x00``): no crash, and the command stays safe
    unless a real gateway action is present.
    """
    script = tmp_path / "evil.sh"
    script.write_bytes(b"#!/bin/sh\x00evil")

    # as_posix() is required on Windows: shlex treats "\\" as an escape, so a
    # native str(Path) (e.g. C:\\Windows\\Temp\\...) would be mangled and the
    # referenced file would silently not be scanned.
    result = contains_gateway_lifecycle_command_or_referenced_script(
        f"/bin/sh {script.as_posix()}"
    )
    assert result is False


def test_nul_byte_in_path_reference_does_not_crash(tmp_path):
    """A NUL byte inside a *path-shaped* token must not crash the guard.

    This is the crash path that bit production: after decoding binary
    content, a token like ``C:/evil\x00dir/evil.sh`` contains ``/`` so it
    is treated as a referenced script, and resolving a path with an
    embedded NUL raises ValueError. The guard must treat the binary
    content as opaque and skip it.
    """
    script = tmp_path / "binary.out"
    script.write_bytes(b"C:/evil\x00dir/evil.sh launch\n")

    result = contains_gateway_lifecycle_command_or_referenced_script(
        f"/bin/sh {script.as_posix()}"
    )
    assert result is False


def test_launchctl_submit_inline_command_still_detected():
    """An inline ``launchctl submit`` command is still unsafe."""
    result = contains_gateway_lifecycle_command_or_referenced_script(
        "launchctl submit -l ai.hermes.gateway-tmp -- /tmp/helper.sh"
    )
    assert result is True


def test_launchctl_submit_in_dash_c_payload_still_detected():
    """A ``-c`` payload carrying ``launchctl submit`` is still unsafe."""
    result = contains_gateway_lifecycle_command_or_referenced_script(
        'sh -c "launchctl submit -l ai.hermes.gateway-tmp -- helper.sh"'
    )
    assert result is True


def test_launchctl_submit_in_referenced_shell_script_still_detected(tmp_path):
    """A real ``.sh`` script containing ``launchctl submit`` is still unsafe.

    Guarding binaries must not weaken fail-closed detection: a genuine
    shell script that submits a launchd job is caught even when it is
    reached through a ``sh <script>`` reference.
    """
    script = tmp_path / "restart.sh"
    script.write_text("#!/bin/sh\nlaunchctl submit -l ai.hermes.gateway-tmp -- /tmp/helper.sh\n")

    result = contains_gateway_lifecycle_command_or_referenced_script(
        f"/bin/sh {script.as_posix()}"
    )
    assert result is True


def test_hermes_gateway_restart_still_detected():
    """The canonical ``hermes gateway restart`` foot-gun is still unsafe."""
    result = contains_gateway_lifecycle_command_or_referenced_script(
        "hermes gateway restart"
    )
    assert result is True


def test_binary_local_file_does_not_trigger_remote_read(tmp_path):
    """A binary local file must NOT trigger the read_remote_script fallback.

    This pins the production gateway path: ``tools/terminal_tool.py`` always
    passes a ``read_remote_script`` callback, and the callback re-reads the
    local file and decodes it with ``errors="replace"`` — so a NUL byte would
    survive and crash ``Path(...).resolve()`` again. Returning empty text
    (rather than ``None``) must keep the binary skip distinct from a local
    miss: no callback call, no raise, command stays safe.
    """
    script = tmp_path / "python.exe"
    script.write_bytes(b"MZ\x00\x90\x00C:/evil\x00dir/evil.sh launch\n")

    calls = []

    def production_equivalent_reader(path: str):
        # Mirrors _read_script_in_env in tools/terminal_tool.py: re-reads the
        # local file and decodes with errors="replace".
        calls.append(path)
        try:
            return Path(path).read_bytes().decode("utf-8", errors="replace")
        except Exception:
            return None

    result = contains_gateway_lifecycle_command_or_referenced_script(
        f"/bin/sh {script.as_posix()}",
        read_remote_script=production_equivalent_reader,
    )
    assert result is False
    assert calls == []


def test_remote_read_still_used_for_missing_local_file(tmp_path):
    """A genuinely missing local file still falls back to read_remote_script.

    The binary-skip fix must not swallow the remote-backend fallback: for a
    local miss (``None``), the callback is still consulted exactly as before.
    """
    missing = tmp_path / "missing.sh"
    calls = []

    def remote_reader(path: str):
        calls.append(path)
        return "launchctl submit -l ai.hermes.gateway-tmp -- /tmp/helper.sh"

    result = contains_gateway_lifecycle_command_or_referenced_script(
        f"/bin/sh {missing.as_posix()}",
        read_remote_script=remote_reader,
    )
    assert result is True
    assert len(calls) == 1
