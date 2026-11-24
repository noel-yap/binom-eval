"""Unit tests for `scripts/retry.sh` (run as a subprocess; no real sleeping)."""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "retry.sh"

# Records each requested pause instead of sleeping.
_SLEEP_STUB = '#!/usr/bin/env bash\necho "$1" >> "$SLEEP_LOG"\n'

# Fails FAILS times (with exit status STATUS), then succeeds. Logs every call's
# arguments, one line per call, so tests can count runs and check arguments.
_FLAKY_STUB = """#!/usr/bin/env bash
n=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > "$COUNT_FILE"
echo "$*" >> "$CALL_LOG"
[ "$n" -le "${FAILS:-0}" ] && exit "${STATUS:-1}"
exit 0
"""


def _executable(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class _Result:
    def __init__(self, proc: subprocess.CompletedProcess[str], tmp: Path) -> None:
        self.returncode = proc.returncode
        self.stderr = proc.stderr
        sleep_log = tmp / "sleeps"
        call_log = tmp / "calls"
        self.sleeps = (
            [int(x) for x in sleep_log.read_text().split()]
            if sleep_log.exists()
            else []
        )
        self.calls = call_log.read_text().splitlines() if call_log.exists() else []


@pytest.fixture
def run(tmp_path: Path):
    sleep = _executable(tmp_path / "fake_sleep", _SLEEP_STUB)
    flaky = _executable(tmp_path / "flaky.sh", _FLAKY_STUB)

    def _run(
        *args: str,
        fails: int = 0,
        status: int = 1,
        use_flaky: bool = True,
        command_args: tuple[str, ...] = (),
        env_extra: dict[str, str] | None = None,
        sleeper: Path | None = sleep,
    ) -> _Result:
        env = {
            **os.environ,
            "SLEEP_LOG": str(tmp_path / "sleeps"),
            "COUNT_FILE": str(tmp_path / "count"),
            "CALL_LOG": str(tmp_path / "calls"),
            "FAILS": str(fails),
            "STATUS": str(status),
            **(env_extra or {}),
        }
        env.pop("RETRY_SLEEP", None)
        if sleeper is not None:
            env["RETRY_SLEEP"] = str(sleeper)
        argv = ["bash", str(_SCRIPT), *args]
        if use_flaky:
            argv += [str(flaky), *command_args]
        proc = subprocess.run(argv, env=env, capture_output=True, text=True)
        return _Result(proc, tmp_path)

    return _run


class TestSucceeds:
    def test_runs_once_and_does_not_pause_when_command_succeeds(self, run) -> None:
        r = run("--")
        assert r.returncode == 0
        assert len(r.calls) == 1
        assert r.sleeps == []

    def test_retries_until_command_succeeds(self, run) -> None:
        r = run("--", fails=2)
        assert r.returncode == 0
        assert len(r.calls) == 3
        assert len(r.sleeps) == 2

    def test_succeeds_on_the_last_allowed_attempt(self, run) -> None:
        r = run("-n", "3", "--", fails=2)
        assert r.returncode == 0
        assert len(r.calls) == 3

    def test_runs_command_without_the_double_dash_separator(self, run) -> None:
        r = run()
        assert r.returncode == 0
        assert len(r.calls) == 1


class TestGivesUp:
    def test_stops_after_five_attempts_by_default(self, run) -> None:
        r = run("--", fails=99)
        assert r.returncode != 0
        assert len(r.calls) == 5
        assert len(r.sleeps) == 4
        assert "giving up after 5 attempts" in r.stderr

    def test_stops_after_the_requested_number_of_attempts(self, run) -> None:
        r = run("-n", "2", "--", fails=99)
        assert len(r.calls) == 2
        assert "giving up after 2 attempts" in r.stderr

    def test_does_not_retry_when_only_one_attempt_is_allowed(self, run) -> None:
        r = run("-n", "1", "--", fails=99)
        assert r.returncode != 0
        assert len(r.calls) == 1
        assert r.sleeps == []

    def test_exits_with_the_last_failure_status_of_the_command(self, run) -> None:
        r = run("-n", "2", "--", fails=99, status=7)
        assert r.returncode == 7


class TestBackoff:
    @staticmethod
    def _ceilings(stderr: str) -> list[int]:
        return [int(m) for m in re.findall(r"ceiling (\d+)s", stderr)]

    def test_ceiling_doubles_from_the_default_base(self, run) -> None:
        r = run("--", fails=99)
        assert self._ceilings(r.stderr) == [2, 4, 8, 16]

    def test_ceiling_starts_from_the_requested_base(self, run) -> None:
        r = run("-b", "3", "-n", "4", "--", fails=99)
        assert self._ceilings(r.stderr) == [3, 6, 12]

    def test_ceiling_stops_growing_at_the_cap(self, run) -> None:
        r = run("-b", "10", "-c", "15", "-n", "5", "--", fails=99)
        assert self._ceilings(r.stderr) == [10, 15, 15, 15]

    def test_ceiling_equal_to_the_cap_is_not_clamped_further(self, run) -> None:
        r = run("-b", "4", "-c", "8", "-n", "4", "--", fails=99)
        assert self._ceilings(r.stderr) == [4, 8, 8]

    def test_cap_below_the_base_limits_the_first_pause(self, run) -> None:
        r = run("-b", "8", "-c", "3", "-n", "2", "--", fails=99)
        assert self._ceilings(r.stderr) == [3]

    def test_every_pause_is_within_its_ceiling(self, run) -> None:
        r = run("-b", "50", "-c", "200", "-n", "5", "--", fails=99)
        ceilings = self._ceilings(r.stderr)
        assert len(r.sleeps) == len(ceilings) == 4
        assert all(0 <= s <= c for s, c in zip(r.sleeps, ceilings))

    def test_pauses_are_randomized_rather_than_fixed(self, run) -> None:
        r = run("-b", "100", "-c", "100", "-n", "30", "--", fails=99)
        assert len(r.sleeps) == 29
        assert len(set(r.sleeps)) > 1
        assert all(0 <= s <= 100 for s in r.sleeps)

    def test_pauses_use_the_sleep_command_when_no_override_is_set(
        self, run, tmp_path: Path
    ) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        _executable(bin_dir / "sleep", _SLEEP_STUB)
        r = run(
            "-n",
            "2",
            "--",
            fails=99,
            sleeper=None,
            env_extra={"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
        )
        assert len(r.sleeps) == 1


class TestArguments:
    def test_passes_the_command_arguments_through_on_every_attempt(self, run) -> None:
        r = run("-n", "2", "--", fails=1, command_args=("a b", "c"))
        assert r.returncode == 0
        assert r.calls == ["a b c", "a b c"]


class TestUsageErrors:
    def test_rejects_a_missing_command_after_the_separator(self, run) -> None:
        r = run("--", use_flaky=False)
        assert r.returncode == 2
        assert "no command given" in r.stderr

    def test_rejects_a_missing_command_without_the_separator(self, run) -> None:
        r = run(use_flaky=False)
        assert r.returncode == 2
        assert "no command given" in r.stderr

    def test_rejects_an_unknown_option(self, run) -> None:
        r = run("-x", "--")
        assert r.returncode == 2
        assert "usage:" in r.stderr
        assert r.calls == []
