#!/usr/bin/env python3
"""Fail if coverage of any file changed in a PR is lower than on the base.

Usage: coverage_compare.py BASE_COVERAGE.json HEAD_COVERAGE.json FILE...

The JSON files come from `coverage json`; FILEs are the paths changed in the
PR (e.g. from `git diff --name-only`). Files absent from either report (new,
deleted, or not measured) have no baseline and are skipped.
"""
import json
import sys


def percents(path: str) -> dict[str, float]:
    with open(path) as f:
        files = json.load(f)["files"]
    return {name: d["summary"]["percent_covered"] for name, d in files.items()}


def main(base_path: str, head_path: str, changed: list[str]) -> int:
    base, head = percents(base_path), percents(head_path)
    regressions = []
    for name in sorted(set(changed) & base.keys() & head.keys()):
        b, h = round(base[name], 2), round(head[name], 2)
        mark = "FAIL" if h < b else "ok"
        print(f"{mark:4}  {name}: {b:.2f}% -> {h:.2f}%")
        if h < b:
            regressions.append(name)
    if regressions:
        print(f"\nCoverage decreased vs. base in {len(regressions)} file(s).")
        return 1
    print("\nNo coverage regressions in changed files.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
