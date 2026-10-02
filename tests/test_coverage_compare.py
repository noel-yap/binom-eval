"""Unit tests for `scripts/coverage_compare.py` (stdlib only)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "coverage_compare.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("coverage_compare", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cc = _load()


def _report(tmp_path: Path, name: str, percents: dict[str, float]) -> str:
    path = tmp_path / name
    data = {
        "files": {
            f: {"summary": {"percent_covered": p}} for f, p in percents.items()
        }
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


class TestPercents:
    def test_maps_each_file_to_its_percent_covered(self, tmp_path: Path) -> None:
        path = _report(tmp_path, "r.json", {"a.py": 90.5, "b.py": 0.0})
        assert cc.percents(path) == {"a.py": 90.5, "b.py": 0.0}


class TestMain:
    def test_lower_head_coverage_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = _report(tmp_path, "b.json", {"a.py": 90.0})
        head = _report(tmp_path, "h.json", {"a.py": 80.0})
        assert cc.main(base, head, ["a.py"]) == 1
        out = capsys.readouterr().out
        assert "FAIL  a.py: 90.00% -> 80.00%" in out
        assert "Coverage decreased vs. base in 1 file(s)." in out

    def test_equal_or_higher_head_coverage_passes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = _report(tmp_path, "b.json", {"a.py": 90.0, "c.py": 50.0})
        head = _report(tmp_path, "h.json", {"a.py": 90.0, "c.py": 60.0})
        assert cc.main(base, head, ["a.py", "c.py"]) == 0
        out = capsys.readouterr().out
        assert "ok    a.py: 90.00% -> 90.00%" in out
        assert "ok    c.py: 50.00% -> 60.00%" in out
        assert "No coverage regressions in changed files." in out

    def test_difference_below_rounding_precision_is_not_a_regression(
        self, tmp_path: Path
    ) -> None:
        base = _report(tmp_path, "b.json", {"a.py": 90.004})
        head = _report(tmp_path, "h.json", {"a.py": 89.996})
        assert cc.main(base, head, ["a.py"]) == 0

    def test_difference_at_rounding_precision_is_a_regression(
        self, tmp_path: Path
    ) -> None:
        base = _report(tmp_path, "b.json", {"a.py": 90.00})
        head = _report(tmp_path, "h.json", {"a.py": 89.99})
        assert cc.main(base, head, ["a.py"]) == 1

    def test_file_missing_from_base_is_skipped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = _report(tmp_path, "b.json", {})
        head = _report(tmp_path, "h.json", {"a.py": 10.0})
        assert cc.main(base, head, ["a.py"]) == 0
        assert "a.py" not in capsys.readouterr().out.split("\n\n")[0]

    def test_file_missing_from_head_is_skipped(self, tmp_path: Path) -> None:
        base = _report(tmp_path, "b.json", {"a.py": 90.0})
        head = _report(tmp_path, "h.json", {})
        assert cc.main(base, head, ["a.py"]) == 0

    def test_file_not_in_changed_list_is_ignored(self, tmp_path: Path) -> None:
        base = _report(tmp_path, "b.json", {"a.py": 90.0})
        head = _report(tmp_path, "h.json", {"a.py": 10.0})
        assert cc.main(base, head, []) == 0
        assert cc.main(base, head, ["a.py"]) == 1

    def test_only_regressed_files_are_counted_and_output_is_sorted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = _report(tmp_path, "b.json", {"z.py": 90.0, "a.py": 90.0, "m.py": 90.0})
        head = _report(tmp_path, "h.json", {"z.py": 50.0, "a.py": 95.0, "m.py": 40.0})
        assert cc.main(base, head, ["z.py", "a.py", "m.py"]) == 1
        out = capsys.readouterr().out
        assert out.index("a.py") < out.index("m.py") < out.index("z.py")
        assert "in 2 file(s)" in out
