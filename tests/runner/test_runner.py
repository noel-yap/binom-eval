"""Unit tests for the `binom_eval.runner` package layer.

Covers the backend-agnostic pieces that live in the package root: env
scrubbing (`stripped_env`), the per-run workdir (`isolated_workdir`), the
pure model-probe parser (`_model_probe_rejected`), the `backend:model` spec
parser (`resolve_runner`), and the concurrent `run_eval_batch` driver. The
`ClaudeRunner` backend itself is tested in `test_claude_runner.py`;
`run_eval_batch` is exercised here against an injected fake `Runner`.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

import binom_eval
from binom_eval import (
    ClaudeRunner,
    CursorRunner,
    EvalRun,
    Runner,
    isolated_workdir,
    resolve_runner,
    stripped_env,
)
from binom_eval.runner import fake_home_env, resolve_skill_dirs


class _FakeRunner(Runner):
    """A `Runner` whose `run` delegates to an injected callable.

    Lets `run_eval_batch` be exercised against a backend that records or
    throttles calls without spawning any CLI; `version`/`preflight`/
    `validate_model` are unused by the batch driver and stubbed inert.
    """

    def __init__(self, run_fn: Any) -> None:
        self._run_fn = run_fn

    def version(self) -> str:
        return ""

    def preflight(self) -> str | None:
        return None

    def validate_model(self, model: str, timeout: int = 30) -> str | None:
        return None

    def run(
        self,
        prompt: str,
        repo_root: Path,
        skill_name: str,
        timeout: int = 300,
        *,
        isolate: bool = False,
        isolate_skill: tuple[Path, ...] | None = None,
        model: str,
    ) -> EvalRun:
        return self._run_fn(
            prompt, repo_root, skill_name, isolate=isolate, model=model
        )


class TestResolveRunner:
    def test_bare_model_without_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 'backend:model'"):
            resolve_runner("haiku")

    def test_missing_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 'backend:model'"):
            resolve_runner(None)

    def test_claude_prefix_strips_to_model(self) -> None:
        backend, model, runner = resolve_runner("claude:claude-opus-4-8")
        assert backend == "claude"
        assert model == "claude-opus-4-8"
        assert isinstance(runner, ClaudeRunner)

    def test_cursor_prefix_selects_cursor_backend(self) -> None:
        backend, model, runner = resolve_runner("cursor:sonnet-4.5")
        assert backend == "cursor"
        assert model == "sonnet-4.5"
        assert isinstance(runner, CursorRunner)

    def test_unknown_backend_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown eval backend 'gpt'"):
            resolve_runner("gpt:4o")

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValueError, match="empty model"):
            resolve_runner("claude:")


class TestStrippedEnv:
    def test_removes_nested_session_markers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "abc")
        monkeypatch.setenv("CLAUDE_CODE_CHILD_SESSION", "1")
        monkeypatch.setenv("OTHER", "v")
        env = stripped_env()
        assert "CLAUDECODE" not in env
        assert "CLAUDE_CODE_SESSION_ID" not in env
        assert "CLAUDE_CODE_CHILD_SESSION" not in env
        assert env.get("OTHER") == "v"


class TestFakeHomeEnv:
    def test_repoints_home_to_a_fresh_empty_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", "/real/home")
        with fake_home_env() as env:
            home = env["HOME"]
            assert home != "/real/home"
            assert env["USERPROFILE"] == home
            home_path = Path(home)
            assert home_path.is_dir()
            assert not any(home_path.iterdir())
        # The throwaway home is removed once the run ends.
        assert not Path(home).exists()

    def test_layers_on_stripped_env_credentials_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
        monkeypatch.setenv("CURSOR_API_KEY", "ck")
        with fake_home_env() as env:
            assert "CLAUDECODE" not in env
            assert env["ANTHROPIC_API_KEY"] == "ak"
            assert env["CURSOR_API_KEY"] == "ck"


class TestRunClaudeBatch:
    def test_runs_count_times_and_stamps_eval_id(self) -> None:
        def fake_run(
            prompt: str,
            repo_root: Path,
            skill_name: str,
            *,
            isolate: bool = False,
            isolate_skill: tuple[Path, ...] | None = None,
            model: str,
        ) -> EvalRun:
            return EvalRun(
                eval_id="", prompt=prompt, skill_invoked=True, assistant_text=""
            )

        runs = binom_eval.run_eval_batch(
            {"id": "e1", "prompt": "p", "prompt_input": "fixture body"},
            Path("."),
            "demo",
            count=3,
            model="m",
            runner=_FakeRunner(fake_run),
        )
        assert len(runs) == 3
        assert all(run.eval_id == "e1" for run in runs)
        assert all(run.prompt == "p" for run in runs)
        assert all(run.prompt_input == "fixture body" for run in runs)

    def test_isolate_skill_not_forwarded_when_none(self) -> None:
        # A custom `Runner` written before `isolate_skill` existed has no such
        # parameter; forwarding `isolate_skill=None` would raise TypeError.
        class LegacyRunner(_FakeRunner):
            def run(  # type: ignore[override]
                self,
                prompt: str,
                repo_root: Path,
                skill_name: str,
                timeout: int = 300,
                *,
                isolate: bool = False,
                model: str,
            ) -> EvalRun:
                return EvalRun(
                    eval_id="", prompt=prompt, skill_invoked=True, assistant_text=""
                )

        runs = binom_eval.run_eval_batch(
            {"id": "e1", "prompt": "p"},
            Path("."),
            "demo",
            count=2,
            model="m",
            runner=LegacyRunner(None),
        )
        assert len(runs) == 2

    def test_isolate_skill_forwarded_when_set(self, tmp_path: Path) -> None:
        seen: list[Any] = []

        class RecordingRunner(_FakeRunner):
            def run(self, *args: Any, **kwargs: Any) -> EvalRun:  # type: ignore[override]
                seen.append(kwargs.get("isolate_skill", "absent"))
                return EvalRun(
                    eval_id="", prompt="", skill_invoked=True, assistant_text=""
                )

        keep = (tmp_path / "skills/demo",)
        binom_eval.run_eval_batch(
            {"id": "e1", "prompt": "p"},
            tmp_path,
            "demo",
            count=2,
            isolate=True,
            isolate_skill=keep,
            model="m",
            runner=RecordingRunner(None),
        )
        binom_eval.run_eval_batch(
            {"id": "e1", "prompt": "p"},
            tmp_path,
            "demo",
            count=1,
            model="m",
            runner=RecordingRunner(None),
        )
        assert seen == [keep, keep, "absent"]

    def test_gate_caps_concurrent_runs(self) -> None:
        lock = threading.Lock()
        live = {"now": 0, "max": 0}

        def fake_run(
            prompt: str,
            repo_root: Path,
            skill_name: str,
            *,
            isolate: bool = False,
            isolate_skill: tuple[Path, ...] | None = None,
            model: str,
        ) -> EvalRun:
            with lock:
                live["now"] += 1
                live["max"] = max(live["max"], live["now"])
            # A brief overlap window so unthrottled runs would pile up; the
            # gate must still hold the peak at its count.
            time.sleep(0.02)
            with lock:
                live["now"] -= 1
            return EvalRun(
                eval_id="", prompt=prompt, skill_invoked=True, assistant_text=""
            )

        runs = binom_eval.run_eval_batch(
            {"id": "e1", "prompt": "p"},
            Path("."),
            "demo",
            count=6,
            gate=threading.Semaphore(2),
            model="m",
            runner=_FakeRunner(fake_run),
        )
        assert len(runs) == 6
        assert live["max"] <= 2


class TestIsolatedWorkdir:
    def test_without_isolation_yields_repo_root_unchanged(
        self, tmp_path: Path
    ) -> None:
        with isolated_workdir(tmp_path, isolate=False) as workdir:
            assert workdir == tmp_path

    def test_isolation_copies_tree_skips_ignored_and_cleans_up(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "keep.txt").write_text("payload", encoding="utf-8")
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "junk.pyc").write_text("nope", encoding="utf-8")

        with isolated_workdir(tmp_path, isolate=True) as workdir:
            assert workdir != tmp_path
            assert (workdir / "keep.txt").read_text(encoding="utf-8") == (
                "payload"
            )
            assert not (workdir / "__pycache__").exists()
            copied = workdir

        # The throwaway copy is removed once the run ends, and the original
        # tree is untouched.
        assert not copied.exists()
        assert (tmp_path / "keep.txt").exists()


def _resolve_form(
    root: Path, skill_name: str, path: str | None
) -> tuple[Path, ...]:
    """Resolve the flag form (`path` None) or the single-path form."""
    if path is None:
        return resolve_skill_dirs(root, skill_name, include_evaluated=True)
    return resolve_skill_dirs(root, skill_name, [path])


class TestResolveSkillDirs:
    """`resolve_skill_dirs` validates and resolves the skill to keep."""

    @staticmethod
    def _repo(root: Path, container: str) -> None:
        for name in ("demo", "other"):
            skill = root / container / name
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(name, encoding="utf-8")

    def test_by_name_finds_each_standard_root(self, tmp_path: Path) -> None:
        for container in (".claude/skills", ".cursor/skills", "skills"):
            self._repo(tmp_path, container)
        found = resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        root = tmp_path.resolve()
        assert found == (
            root / ".claude/skills/demo",
            root / ".cursor/skills/demo",
            root / "skills/demo",
        )

    def test_explicit_path_is_resolved(self, tmp_path: Path) -> None:
        self._repo(tmp_path, "pkg/skills")
        found = resolve_skill_dirs(tmp_path, "x", ["pkg/./skills/../skills/other"])
        assert found == (tmp_path.resolve() / "pkg/skills/other",)

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        self._repo(tmp_path, "skills")
        with pytest.raises(ValueError, match="absolute"):
            resolve_skill_dirs(tmp_path, "demo", [str(tmp_path / "skills/demo")])

    @pytest.mark.parametrize("path", ["..", "../x", "skills/../..", ""])
    def test_escaping_path_rejected(self, tmp_path: Path, path: str) -> None:
        repo = tmp_path / "repo"
        self._repo(repo, "skills")
        (tmp_path / "x").mkdir()
        with pytest.raises(ValueError, match="inside the repo root"):
            resolve_skill_dirs(repo, "demo", [path])

    def test_symlink_escaping_repo_rejected(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (repo / "link").symlink_to(outside)
        with pytest.raises(ValueError, match="inside the repo root"):
            resolve_skill_dirs(repo, "demo", ["link"])

    @pytest.mark.parametrize("path", [None, ".claude/skills/demo"])
    def test_symlinked_skill_keeps_its_unresolved_location(
        self, tmp_path: Path, path: str | None
    ) -> None:
        self._repo(tmp_path, "skills")
        (tmp_path / ".claude/skills").mkdir(parents=True)
        (tmp_path / ".claude/skills/demo").symlink_to("../../skills/demo")
        found = _resolve_form(tmp_path, "demo", path)
        assert tmp_path.resolve() / ".claude/skills/demo" in found

    @pytest.mark.parametrize("path", [None, ".claude/skills/demo"])
    def test_symlinked_skill_target_outside_repo_rejected(
        self, tmp_path: Path, path: str | None
    ) -> None:
        repo = tmp_path / "repo"
        (repo / ".claude/skills").mkdir(parents=True)
        (tmp_path / "outside").mkdir()
        (repo / ".claude/skills/demo").symlink_to(tmp_path / "outside")
        # SKILL.md so the container/name rule passes and the containment
        # check is what rejects the skill.
        (tmp_path / "outside/SKILL.md").write_text("s", encoding="utf-8")
        with pytest.raises(ValueError, match="inside the repo root"):
            _resolve_form(repo, "demo", path)

    @pytest.mark.parametrize(
        "path",
        [
            ".claude/skills/demo/sub",
            ".cursor/skills/demo/sub",
            "skills/demo/sub",
            ".claude/skills",
            "skills",
        ],
    )
    def test_path_inside_or_at_container_rejected(
        self, tmp_path: Path, path: str
    ) -> None:
        for container in (".claude/skills", ".cursor/skills", "skills"):
            self._repo(tmp_path, container)
            (tmp_path / container / "demo/sub").mkdir()
        with pytest.raises(ValueError, match="inside skill container"):
            resolve_skill_dirs(tmp_path, "demo", [path])

    def test_container_path_without_skill_md_rejected(self, tmp_path: Path) -> None:
        (tmp_path / ".claude/skills/foo").mkdir(parents=True)
        with pytest.raises(ValueError, match="containing SKILL.md"):
            resolve_skill_dirs(tmp_path, "foo", [".claude/skills/foo"])

    def test_container_path_with_skill_md_accepted(self, tmp_path: Path) -> None:
        self._repo(tmp_path, ".claude/skills")
        found = resolve_skill_dirs(tmp_path, "x", [".claude/skills/demo"])
        assert found == (tmp_path.resolve() / ".claude/skills/demo",)

    @pytest.mark.parametrize("path", ["examples/x/skill", "plugins/x/skills/foo"])
    def test_path_outside_containers_needs_no_skill_md(
        self, tmp_path: Path, path: str
    ) -> None:
        (tmp_path / path).mkdir(parents=True)
        assert resolve_skill_dirs(tmp_path, "x", [path]) == (
            tmp_path.resolve() / path,
        )

    def test_flag_form_ignores_stray_dir_without_skill_md(
        self, tmp_path: Path
    ) -> None:
        # A leftover empty `skills/demo/` must not fail the run while
        # `.claude/skills/demo/SKILL.md` holds the real skill.
        self._repo(tmp_path, ".claude/skills")
        (tmp_path / "skills/demo").mkdir(parents=True)
        (tmp_path / ".cursor/skills/demo").mkdir(parents=True)
        found = resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        assert found == (tmp_path.resolve() / ".claude/skills/demo",)

    def test_flag_form_none_with_skill_md_names_the_flag_and_lists_tried(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "skills/demo").mkdir(parents=True)
        with pytest.raises(ValueError) as excinfo:
            resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        message = str(excinfo.value)
        assert "--live-eval-isolate-skill:" in message
        assert "-path" not in message
        root = tmp_path.resolve()
        for base in (".claude/skills", ".cursor/skills", "skills"):
            assert str(root / base / "demo" / "SKILL.md") in message

    @pytest.mark.parametrize("path", [None, "nope/skill"])
    def test_missing_dir_raises(self, tmp_path: Path, path: str | None) -> None:
        self._repo(tmp_path, ".claude/skills")
        with pytest.raises(ValueError, match="skill directory not found"):
            _resolve_form(tmp_path, "absent", path)

    @staticmethod
    def _skills(root: Path, container: str, *names: str) -> None:
        for name in names:
            (root / container / name).mkdir(parents=True)
            (root / container / name / "SKILL.md").write_text(name, encoding="utf-8")

    def test_no_flag_and_no_paths_resolves_nothing(self, tmp_path: Path) -> None:
        self._skills(tmp_path, ".claude/skills", "demo")
        assert resolve_skill_dirs(tmp_path, "demo") == ()

    def test_several_paths_are_all_kept_in_order(self, tmp_path: Path) -> None:
        self._skills(tmp_path, ".claude/skills", "a", "b")
        self._skills(tmp_path, "pkg/skills", "c")
        found = resolve_skill_dirs(
            tmp_path, "x", ["pkg/skills/c", ".claude/skills/b", ".claude/skills/a"]
        )
        root = tmp_path.resolve()
        assert found == (
            root / "pkg/skills/c",
            root / ".claude/skills/b",
            root / ".claude/skills/a",
        )

    def test_flag_and_paths_are_additive(self, tmp_path: Path) -> None:
        self._skills(tmp_path, ".claude/skills", "demo", "helper")
        found = resolve_skill_dirs(
            tmp_path,
            "demo",
            [".claude/skills/helper"],
            include_evaluated=True,
        )
        root = tmp_path.resolve()
        assert found == (
            root / ".claude/skills/demo",
            root / ".claude/skills/helper",
        )

    def test_duplicates_are_removed_preserving_order(self, tmp_path: Path) -> None:
        self._skills(tmp_path, ".claude/skills", "demo", "helper")
        # The flag resolves `demo` to the very location a path also names
        # (spelled differently); a repeated path is collapsed too.
        found = resolve_skill_dirs(
            tmp_path,
            "demo",
            [
                ".claude/skills/helper",
                ".claude/./skills/demo",
                ".claude/skills/helper/",
            ],
            include_evaluated=True,
        )
        root = tmp_path.resolve()
        assert found == (
            root / ".claude/skills/demo",
            root / ".claude/skills/helper",
        )

    def test_one_bad_path_among_several_fails_the_call_naming_it(
        self, tmp_path: Path
    ) -> None:
        self._skills(tmp_path, ".claude/skills", "demo", "helper")
        with pytest.raises(ValueError, match="skills/typo"):
            resolve_skill_dirs(
                tmp_path,
                "demo",
                [".claude/skills/helper", ".claude/skills/typo"],
                include_evaluated=True,
            )

    def test_bad_flag_fails_even_when_paths_are_valid(self, tmp_path: Path) -> None:
        self._skills(tmp_path, ".claude/skills", "helper")
        with pytest.raises(ValueError, match="--live-eval-isolate-skill:"):
            resolve_skill_dirs(
                tmp_path,
                "absent",
                [".claude/skills/helper"],
                include_evaluated=True,
            )


class TestIsolatedWorkdirSkill:
    """`isolate_skill` copies only the evaluated skill from its container."""

    @staticmethod
    def _repo(root: Path, container: str) -> None:
        for name in ("demo", "other"):
            skill = root / container / name
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(name, encoding="utf-8")
        (root / "README.md").write_text("top", encoding="utf-8")

    def test_keeps_named_skill_and_drops_siblings(self, tmp_path: Path) -> None:
        self._repo(tmp_path, ".claude/skills")
        keep = resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert (workdir / ".claude/skills/demo/SKILL.md").exists()
            assert not (workdir / ".claude/skills/other").exists()
            assert (workdir / "README.md").exists()

    def test_finds_cursor_skill_root(self, tmp_path: Path) -> None:
        self._repo(tmp_path, ".cursor/skills")
        keep = resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert (workdir / ".cursor/skills/demo").is_dir()
            assert not (workdir / ".cursor/skills/other").exists()

    def test_explicit_path_in_standard_container_drops_its_siblings(
        self, tmp_path: Path
    ) -> None:
        self._repo(tmp_path, ".claude/skills")
        (tmp_path / ".claude/settings.json").write_text("{}", encoding="utf-8")
        keep = resolve_skill_dirs(tmp_path, "ignored", [".claude/skills/other"])
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert (workdir / ".claude/skills/other/SKILL.md").exists()
            assert not (workdir / ".claude/skills/demo").exists()
            assert (workdir / ".claude/settings.json").exists()

    def test_explicit_path_in_non_standard_parent_keeps_siblings(
        self, tmp_path: Path
    ) -> None:
        # `pkg/skills` is not the repo-root `skills` container, so it is not
        # narrowed: only standard containers are filtered.
        self._repo(tmp_path, "pkg/skills")
        keep = resolve_skill_dirs(tmp_path, "ignored", ["pkg/skills/other"])
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert (workdir / "pkg/skills/other/SKILL.md").exists()
            assert (workdir / "pkg/skills/demo/SKILL.md").exists()

    def test_skill_directly_under_repo_root_keeps_root_contents(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "my-skill").mkdir()
        (tmp_path / "my-skill/SKILL.md").write_text("s", encoding="utf-8")
        (tmp_path / "CLAUDE.md").write_text("c", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("p", encoding="utf-8")
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude/settings.json").write_text("{}", encoding="utf-8")
        (tmp_path / "other-dir").mkdir()
        keep = resolve_skill_dirs(tmp_path, "ignored", ["my-skill"])
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert (workdir / "my-skill/SKILL.md").exists()
            assert (workdir / "CLAUDE.md").exists()
            assert (workdir / "pyproject.toml").exists()
            assert (workdir / ".claude/settings.json").exists()
            assert (workdir / "other-dir").is_dir()

    def test_nested_non_standard_skill_keeps_siblings_but_drops_containers(
        self, tmp_path: Path
    ) -> None:
        self._repo(tmp_path, "plugins/x/skills")
        (tmp_path / "plugins/x/plugin.json").write_text("{}", encoding="utf-8")
        self._repo(tmp_path, ".claude/skills")
        self._repo(tmp_path, "skills")
        keep = resolve_skill_dirs(tmp_path, "ignored", ["plugins/x/skills/demo"])
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert (workdir / "plugins/x/skills/demo/SKILL.md").exists()
            assert (workdir / "plugins/x/skills/other/SKILL.md").exists()
            assert (workdir / "plugins/x/plugin.json").exists()
            for container in (".claude/skills", "skills"):
                assert list((workdir / container).iterdir()) == []

    def test_ignore_matches_dotdot_and_symlinked_root_spelling(
        self, tmp_path: Path
    ) -> None:
        real = tmp_path / "real"
        self._repo(real, ".claude/skills")
        (tmp_path / "sub").mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        keep = resolve_skill_dirs(real, "demo", include_evaluated=True)
        for spelling in (
            tmp_path / "sub" / ".." / "real",
            link,
            link / ".." / "link",
        ):
            with isolated_workdir(spelling, True, keep) as workdir:
                assert (workdir / ".claude/skills/demo").is_dir()
                assert not (workdir / ".claude/skills/other").exists()

    def test_symlinked_skill_dir_still_excludes_siblings(
        self, tmp_path: Path
    ) -> None:
        self._repo(tmp_path, "skills")
        (tmp_path / ".claude/skills").mkdir(parents=True)
        (tmp_path / ".claude/skills/demo").symlink_to("../../skills/demo")
        (tmp_path / ".claude/skills/other").symlink_to("../../skills/other")
        keep = resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        with isolated_workdir(tmp_path, True, keep) as workdir:
            # The kept symlink is dereferenced into a real directory; the
            # `skills/` container is a SKILL_ROOT too, so its non-kept
            # children (including the link's target) are no longer copied.
            assert (workdir / ".claude/skills/demo/SKILL.md").exists()
            assert not (workdir / ".claude/skills/other").is_symlink()
            assert not (workdir / ".claude/skills/other").exists()
            assert not (workdir / "skills/other").exists()

    def test_flag_form_excludes_siblings_in_other_containers(
        self, tmp_path: Path
    ) -> None:
        self._repo(tmp_path, ".claude/skills")
        self._repo(tmp_path, ".cursor/skills")
        self._repo(tmp_path, "skills")
        (tmp_path / "skills/demo").rename(tmp_path / "skills/gone")
        keep = resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        assert len(keep) == 2
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert (workdir / ".claude/skills/demo/SKILL.md").exists()
            assert (workdir / ".cursor/skills/demo/SKILL.md").exists()
            assert not (workdir / ".claude/skills/other").exists()
            assert not (workdir / ".cursor/skills/other").exists()
            assert not (workdir / "skills/gone").exists()
            assert not (workdir / "skills/other").exists()

    def test_flag_form_skill_only_in_claude_drops_other_containers(
        self, tmp_path: Path
    ) -> None:
        self._repo(tmp_path, ".claude/skills")
        # Siblings must look like skills (carry a SKILL.md) to be dropped;
        # bare directories are non-skill entries, which are now kept.
        for sibling in (".cursor/skills/sib", "skills/sib", ".cursor/skills/other"):
            (tmp_path / sibling).mkdir(parents=True)
            (tmp_path / sibling / "SKILL.md").write_text("s", encoding="utf-8")
        keep = resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert (workdir / ".claude/skills/demo/SKILL.md").exists()
            assert not (workdir / ".claude/skills/other").exists()
            assert not (workdir / ".cursor/skills/sib").exists()
            assert not (workdir / ".cursor/skills/other").exists()
            assert not (workdir / "skills/sib").exists()
            assert (workdir / "README.md").exists()

    def test_path_form_outside_roots_drops_only_standard_container_skills(
        self, tmp_path: Path
    ) -> None:
        self._repo(tmp_path, "examples/x")
        self._repo(tmp_path, ".claude/skills")
        self._repo(tmp_path, ".cursor/skills")
        self._repo(tmp_path, "skills")
        keep = resolve_skill_dirs(tmp_path, "ignored", ["examples/x/demo"])
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert (workdir / "examples/x/demo/SKILL.md").exists()
            # Non-standard parent: siblings are not excluded.
            assert (workdir / "examples/x/other/SKILL.md").exists()
            for container in (".claude/skills", ".cursor/skills", "skills"):
                assert (workdir / container).is_dir()
                assert list((workdir / container).iterdir()) == []

    @pytest.mark.parametrize("absolute", [False, True])
    def test_symlinked_skill_dir_is_copied_as_real_directory(
        self, tmp_path: Path, absolute: bool
    ) -> None:
        self._repo(tmp_path, "skills")
        (tmp_path / ".claude/skills").mkdir(parents=True)
        target = "../../skills/demo"
        if absolute:
            target = str((tmp_path / "skills/demo").resolve())
        (tmp_path / ".claude/skills/demo").symlink_to(target)
        (tmp_path / "skills/demo/__pycache__").mkdir()
        keep = resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        with isolated_workdir(tmp_path, True, keep) as workdir:
            copied = workdir / ".claude/skills/demo"
            assert copied.is_dir() and not copied.is_symlink()
            assert (copied / "SKILL.md").read_text(encoding="utf-8") == "demo"
            assert not (copied / "__pycache__").exists()
            (copied / "SKILL.md").write_text("changed", encoding="utf-8")
            (copied / "new.txt").write_text("x", encoding="utf-8")
        original = tmp_path / "skills/demo"
        assert (original / "SKILL.md").read_text(encoding="utf-8") == "demo"
        assert not (original / "new.txt").exists()

    @pytest.mark.parametrize("absolute", [False, True])
    @pytest.mark.parametrize("path", [None, ".claude/skills/demo"])
    def test_symlinked_container_is_copied_as_real_directory(
        self, tmp_path: Path, absolute: bool, path: str | None
    ) -> None:
        self._repo(tmp_path, "shared/skills")
        (tmp_path / ".claude").mkdir()
        target = "../shared/skills"
        if absolute:
            target = str((tmp_path / "shared/skills").resolve())
        (tmp_path / ".claude/skills").symlink_to(target)
        (tmp_path / "shared/skills/demo/__pycache__").mkdir()
        keep = _resolve_form(tmp_path, "demo", path)
        with isolated_workdir(tmp_path, True, keep) as workdir:
            container = workdir / ".claude/skills"
            assert container.is_dir() and not container.is_symlink()
            assert (container / "demo/SKILL.md").read_text(encoding="utf-8") == "demo"
            assert not (container / "demo/__pycache__").exists()
            assert not (container / "other").exists()
            (container / "demo/SKILL.md").write_text("changed", encoding="utf-8")
            (container / "new.txt").write_text("x", encoding="utf-8")
        original = tmp_path / "shared/skills"
        assert (original / "demo/SKILL.md").read_text(encoding="utf-8") == "demo"
        assert not (original / "new.txt").exists()

    def test_symlinked_container_and_skill_dir_both_symlinks(
        self, tmp_path: Path
    ) -> None:
        self._repo(tmp_path, "real")
        (tmp_path / "shared/skills").mkdir(parents=True)
        (tmp_path / "shared/skills/demo").symlink_to("../../real/demo")
        (tmp_path / "shared/skills/other").symlink_to("../../real/other")
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude/skills").symlink_to("../shared/skills")
        keep = resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        with isolated_workdir(tmp_path, True, keep) as workdir:
            container = workdir / ".claude/skills"
            assert container.is_dir() and not container.is_symlink()
            demo = container / "demo"
            assert demo.is_dir() and not demo.is_symlink()
            assert (demo / "SKILL.md").read_text(encoding="utf-8") == "demo"
            assert not (container / "other").exists()
            (demo / "SKILL.md").write_text("changed", encoding="utf-8")
        assert (tmp_path / "real/demo/SKILL.md").read_text(encoding="utf-8") == "demo"

    @staticmethod
    def _shared_repo(root: Path, absolute: bool) -> None:
        """One real `shared/skills`; `.claude/skills` and `.cursor/skills`
        are links to it (the latter via `.claude/skills`), the usual way
        Claude and Cursor share skill implementations."""
        TestIsolatedWorkdirSkill._repo(root, "shared/skills")
        (root / "shared/skills/_shared").mkdir()
        (root / "shared/skills/_shared/helper.md").write_text("h", encoding="utf-8")
        (root / ".claude").mkdir()
        (root / ".cursor").mkdir()
        claude, cursor = "../shared/skills", "../.claude/skills"
        if absolute:
            claude = str((root / "shared/skills").resolve())
            cursor = str(root.resolve() / ".claude/skills")
        (root / ".claude/skills").symlink_to(claude)
        (root / ".cursor/skills").symlink_to(cursor)

    @pytest.mark.parametrize("absolute", [False, True])
    @pytest.mark.parametrize("path", [None, ".claude/skills/demo"])
    def test_shared_target_container_leaks_no_sibling_anywhere(
        self, tmp_path: Path, absolute: bool, path: str | None
    ) -> None:
        # The link's target `shared/skills` is also a plain directory of the
        # tree that `copytree` walks, so it must be filtered too; otherwise
        # `other` stays visible at `shared/skills/other` (and a relative link
        # would resolve to it) even though the links themselves are clean.
        self._shared_repo(tmp_path, absolute)
        keep = _resolve_form(tmp_path, "demo", path)
        with isolated_workdir(tmp_path, True, keep) as workdir:
            for container in ("shared/skills", ".claude/skills", ".cursor/skills"):
                copied = workdir / container
                assert copied.is_dir()
                assert not (copied / "other").exists()
                assert (copied / "demo/SKILL.md").read_text(encoding="utf-8") == "demo"
                assert (copied / "_shared/helper.md").exists()
            for link in (".claude/skills", ".cursor/skills"):
                assert not (workdir / link).is_symlink()
            # Writes go to the copy: the links were turned into real
            # directories, not left pointing back into the original tree.
            (workdir / ".claude/skills/demo/SKILL.md").write_text(
                "changed", encoding="utf-8"
            )
            (workdir / ".cursor/skills/new.txt").write_text("x", encoding="utf-8")
        original = tmp_path / "shared/skills"
        assert (original / "demo/SKILL.md").read_text(encoding="utf-8") == "demo"
        assert (original / "other/SKILL.md").exists()
        assert not (original / "new.txt").exists()

    @pytest.mark.parametrize("absolute", [False, True])
    def test_container_under_symlinked_parent_leaks_no_sibling(
        self, tmp_path: Path, absolute: bool
    ) -> None:
        # `.claude -> ../shared/claude` makes `.claude/skills` a real
        # directory only through the link; the real location must be filtered
        # and the link replaced by a real, filtered directory.
        self._repo(tmp_path, "shared/claude/skills")
        target = "shared/claude"
        if absolute:
            target = str((tmp_path / "shared/claude").resolve())
        (tmp_path / ".claude").symlink_to(target)
        keep = resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert not (workdir / ".claude").is_symlink()
            for container in ("shared/claude/skills", ".claude/skills"):
                assert (workdir / container / "demo/SKILL.md").exists()
                assert not (workdir / container / "other").exists()
            (workdir / ".claude/skills/demo/SKILL.md").write_text(
                "changed", encoding="utf-8"
            )
        original = tmp_path / "shared/claude/skills"
        assert (original / "demo/SKILL.md").read_text(encoding="utf-8") == "demo"
        assert (original / "other/SKILL.md").exists()

    def test_shared_target_filter_applies_per_container_kept_names(
        self, tmp_path: Path
    ) -> None:
        # `.claude/skills` links to `shared/skills` (keeps `demo`) while
        # `skills/` is a separate real container where nothing is kept: the
        # kept name is per container, so `skills/demo` is dropped.
        self._shared_repo(tmp_path, False)
        self._repo(tmp_path, "skills")
        keep = resolve_skill_dirs(tmp_path, "demo", [".claude/skills/demo"])
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert (workdir / "shared/skills/demo").is_dir()
            assert not (workdir / "shared/skills/other").exists()
            assert not (workdir / "skills/demo").exists()
            assert not (workdir / "skills/other").exists()

    def test_dereference_tolerates_missing_copied_entry(
        self, tmp_path: Path
    ) -> None:
        from binom_eval.runner import _dereference_symlinks, _skill_ignore

        # The copy may lack an entry the original has (e.g. it was ignored or
        # removed meanwhile); that must not raise FileNotFoundError.
        repo = tmp_path / "repo"
        self._shared_repo(repo, False)
        dest = tmp_path / "dest"
        dest.mkdir()
        root = repo.resolve()
        keep = (root / ".claude/skills/demo",)
        _dereference_symlinks(keep, root, dest, _skill_ignore(keep, repo))
        assert list(dest.iterdir()) == []

    def test_dereference_raises_for_kept_skill_resolving_outside_root(
        self, tmp_path: Path
    ) -> None:
        from binom_eval.runner import _dereference_symlinks, _skill_ignore

        # Unreachable via validated input (`isolated_workdir` and
        # `resolve_skill_dirs` reject it first), but a direct call must get a
        # clear error rather than a silently deleted skill.
        repo = tmp_path / "repo"
        (repo / ".claude/skills").mkdir(parents=True)
        (tmp_path / "outside").mkdir()
        (repo / ".claude/skills/demo").symlink_to(tmp_path / "outside")
        dest = tmp_path / "dest"
        (dest / ".claude/skills").mkdir(parents=True)
        (dest / ".claude/skills/demo").symlink_to(tmp_path / "outside")
        root = repo.resolve()
        keep = (root / ".claude/skills/demo",)
        with pytest.raises(ValueError, match="isolate_skill must name"):
            _dereference_symlinks(keep, root, dest, _skill_ignore(keep, repo))

    @pytest.mark.parametrize("path", [None, ".claude/skills/demo"])
    def test_symlinked_container_outside_repo_rejected_up_front(
        self, tmp_path: Path, path: str | None
    ) -> None:
        repo = tmp_path / "repo"
        (repo / ".claude").mkdir(parents=True)
        self._repo(tmp_path / "outside", "skills")
        (repo / ".claude/skills").symlink_to(tmp_path / "outside/skills")
        with pytest.raises(ValueError, match="inside the repo root"):
            _resolve_form(repo, "demo", path)

    def test_symlinked_container_outside_repo_raises_when_called_directly(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        (repo / ".claude").mkdir(parents=True)
        self._repo(tmp_path / "outside", "skills")
        (repo / ".claude/skills").symlink_to(tmp_path / "outside/skills")
        with pytest.raises(ValueError, match="isolate_skill must name"):
            with isolated_workdir(repo, True, (repo / ".claude/skills/demo",)):
                pass

    @pytest.mark.parametrize(
        "path", [".claude/skills/demo/sub", ".claude/skills", "skills/demo/sub"]
    )
    def test_direct_call_inside_container_rejected(
        self, tmp_path: Path, path: str
    ) -> None:
        self._repo(tmp_path, ".claude/skills")
        self._repo(tmp_path, "skills")
        (tmp_path / ".claude/skills/demo/sub").mkdir()
        (tmp_path / "skills/demo/sub").mkdir()
        with pytest.raises(ValueError, match="inside skill container"):
            with isolated_workdir(tmp_path, True, (tmp_path.resolve() / path,)):
                pass

    def test_direct_call_container_dir_without_skill_md_rejected(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".claude/skills/foo").mkdir(parents=True)
        with pytest.raises(ValueError, match="containing SKILL.md"):
            with isolated_workdir(
                tmp_path, True, (tmp_path.resolve() / ".claude/skills/foo",)
            ):
                pass

    def test_unkept_symlinked_container_outside_repo_is_removed(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        self._repo(repo, ".claude/skills")
        (tmp_path / "outside/skills/stranger").mkdir(parents=True)
        (tmp_path / "outside/skills/stranger/SKILL.md").write_text("s", encoding="utf-8")
        (repo / "skills").symlink_to(tmp_path / "outside/skills")
        keep = resolve_skill_dirs(repo, "demo", include_evaluated=True)
        with isolated_workdir(repo, True, keep) as workdir:
            assert not (workdir / "skills").exists()
            assert not (workdir / "skills").is_symlink()

    @pytest.mark.parametrize("path", [None, ".claude/skills/demo"])
    def test_shared_non_skill_entries_kept_and_sibling_skills_dropped(
        self, tmp_path: Path, path: str | None
    ) -> None:
        self._repo(tmp_path, ".claude/skills")
        skills = tmp_path / ".claude/skills"
        (skills / "_shared").mkdir()
        (skills / "_shared/helper.md").write_text("h", encoding="utf-8")
        (skills / "README.md").write_text("r", encoding="utf-8")
        keep = _resolve_form(tmp_path, "demo", path)
        with isolated_workdir(tmp_path, True, keep) as workdir:
            copied = workdir / ".claude/skills"
            assert (copied / "demo/SKILL.md").exists()
            assert (copied / "_shared/helper.md").exists()
            assert (copied / "README.md").exists()
            assert not (copied / "other").exists()

    def test_shared_non_skill_entries_kept_in_symlinked_container(
        self, tmp_path: Path
    ) -> None:
        self._repo(tmp_path, "shared/skills")
        (tmp_path / "shared/skills/_shared").mkdir()
        (tmp_path / "shared/skills/README.md").write_text("r", encoding="utf-8")
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude/skills").symlink_to("../shared/skills")
        keep = resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        with isolated_workdir(tmp_path, True, keep) as workdir:
            copied = workdir / ".claude/skills"
            assert (copied / "_shared").is_dir()
            assert (copied / "README.md").exists()
            assert not (copied / "other").exists()

    def test_symlink_target_outside_repo_raises_when_called_directly(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        (repo / ".claude/skills").mkdir(parents=True)
        (tmp_path / "outside").mkdir()
        link = repo / ".claude/skills/demo"
        link.symlink_to(tmp_path / "outside")
        with pytest.raises(ValueError, match="isolate_skill must name"):
            with isolated_workdir(repo, True, (link,)):
                pass

    def test_still_applies_isolation_ignore(self, tmp_path: Path) -> None:
        self._repo(tmp_path, ".claude/skills")
        (tmp_path / ".claude/skills/demo/__pycache__").mkdir()
        keep = resolve_skill_dirs(tmp_path, "demo", include_evaluated=True)
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert not (workdir / ".claude/skills/demo/__pycache__").exists()

    @pytest.mark.parametrize("which", ["missing", "outside", "empty"])
    def test_bad_value_raises_when_called_directly(
        self, tmp_path: Path, which: str
    ) -> None:
        repo = tmp_path / "repo"
        self._repo(repo, ".claude/skills")
        (tmp_path / "outside").mkdir()
        bad = {
            "missing": (repo / "nope",),
            "outside": (tmp_path / "outside",),
            "empty": (),
        }[which]
        with pytest.raises(ValueError, match="isolate_skill must name"):
            with isolated_workdir(repo, True, bad):
                pass

    def test_absent_copies_whole_tree(self, tmp_path: Path) -> None:
        self._repo(tmp_path, ".claude/skills")
        with isolated_workdir(tmp_path, True, None) as workdir:
            assert (workdir / ".claude/skills/demo").is_dir()
            assert (workdir / ".claude/skills/other").is_dir()

    def test_not_isolated_ignores_skill_option(self, tmp_path: Path) -> None:
        keep = (tmp_path,)
        with isolated_workdir(tmp_path, False, keep) as workdir:
            assert workdir == tmp_path


class TestIsolatedWorkdirSeveralSkills:
    """Several kept skills: all kept names survive, every other skill goes."""

    @staticmethod
    def _skills(root: Path, container: str, *names: str) -> None:
        for name in names:
            (root / container / name).mkdir(parents=True)
            (root / container / name / "SKILL.md").write_text(name, encoding="utf-8")

    def test_same_container_keeps_both_and_drops_a_third(
        self, tmp_path: Path
    ) -> None:
        self._skills(tmp_path, ".claude/skills", "demo", "helper", "third")
        (tmp_path / ".claude/skills/README.md").write_text("r", encoding="utf-8")
        keep = resolve_skill_dirs(
            tmp_path, "demo", [".claude/skills/helper"], include_evaluated=True
        )
        with isolated_workdir(tmp_path, True, keep) as workdir:
            skills = workdir / ".claude/skills"
            assert (skills / "demo/SKILL.md").exists()
            assert (skills / "helper/SKILL.md").exists()
            assert not (skills / "third").exists()
            assert (skills / "README.md").exists()

    def test_two_paths_in_the_same_container(self, tmp_path: Path) -> None:
        self._skills(tmp_path, ".claude/skills", "a", "b", "c")
        keep = resolve_skill_dirs(
            tmp_path, "x", [".claude/skills/a", ".claude/skills/b"]
        )
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert sorted(p.name for p in (workdir / ".claude/skills").iterdir()) == [
                "a",
                "b",
            ]

    def test_paths_in_different_containers(self, tmp_path: Path) -> None:
        self._skills(tmp_path, ".claude/skills", "a", "x")
        self._skills(tmp_path, ".cursor/skills", "b", "y")
        self._skills(tmp_path, "skills", "z")
        keep = resolve_skill_dirs(
            tmp_path, "q", [".claude/skills/a", ".cursor/skills/b"]
        )
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert (workdir / ".claude/skills/a").is_dir()
            assert not (workdir / ".claude/skills/x").exists()
            assert (workdir / ".cursor/skills/b").is_dir()
            assert not (workdir / ".cursor/skills/y").exists()
            # Nothing is kept in `skills/`, so its skill is dropped too.
            assert not (workdir / "skills/z").exists()

    def test_standard_and_non_standard_paths_together(self, tmp_path: Path) -> None:
        self._skills(tmp_path, ".claude/skills", "a", "x")
        self._skills(tmp_path, "pkg/skills", "b", "sibling")
        keep = resolve_skill_dirs(
            tmp_path, "q", [".claude/skills/a", "pkg/skills/b"]
        )
        with isolated_workdir(tmp_path, True, keep) as workdir:
            assert (workdir / ".claude/skills/a").is_dir()
            assert not (workdir / ".claude/skills/x").exists()
            # A non-standard parent is not filtered, as with a single path.
            assert (workdir / "pkg/skills/b").is_dir()
            assert (workdir / "pkg/skills/sibling").is_dir()

    def test_symlinked_container_shared_by_two_kept_skills(
        self, tmp_path: Path
    ) -> None:
        self._skills(tmp_path, "shared/skills", "demo", "helper", "third")
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".cursor").mkdir()
        (tmp_path / ".claude/skills").symlink_to("../shared/skills")
        (tmp_path / ".cursor/skills").symlink_to("../.claude/skills")
        keep = resolve_skill_dirs(
            tmp_path, "demo", [".claude/skills/helper"], include_evaluated=True
        )
        with isolated_workdir(tmp_path, True, keep) as workdir:
            for container in ("shared/skills", ".claude/skills", ".cursor/skills"):
                copied = workdir / container
                assert (copied / "demo/SKILL.md").exists()
                assert (copied / "helper/SKILL.md").exists()
                assert not (copied / "third").exists()
            for link in (".claude/skills", ".cursor/skills"):
                assert not (workdir / link).is_symlink()
        assert (tmp_path / "shared/skills/third/SKILL.md").exists()

    def test_symlinked_skill_dirs_are_each_dereferenced(self, tmp_path: Path) -> None:
        self._skills(tmp_path, "impl", "demo", "helper")
        self._skills(tmp_path, ".claude/skills", "third")
        (tmp_path / ".claude/skills/demo").symlink_to("../../impl/demo")
        (tmp_path / ".claude/skills/helper").symlink_to("../../impl/helper")
        keep = resolve_skill_dirs(
            tmp_path, "demo", [".claude/skills/helper"], include_evaluated=True
        )
        with isolated_workdir(tmp_path, True, keep) as workdir:
            skills = workdir / ".claude/skills"
            for name in ("demo", "helper"):
                assert not (skills / name).is_symlink()
                assert (skills / name / "SKILL.md").read_text(
                    encoding="utf-8"
                ) == name
            assert not (skills / "third").exists()


class TestModelProbeRejected:
    """Unit tests for the pure probe-parser behind `validate_model`."""

    def test_format_model_error_appends_sorted_valid_models(self) -> None:
        from binom_eval.runner import _format_model_error

        msg = _format_model_error(
            "model not found: bad",
            ["claude-opus-4-8", "claude-haiku-4-5", "claude-opus-4-8"],
        )
        assert msg == (
            "model not found: bad; valid models: claude-haiku-4-5, claude-opus-4-8"
        )

    def test_format_model_error_leaves_message_unchanged_without_models(
        self,
    ) -> None:
        from binom_eval.runner import _format_model_error

        assert _format_model_error("model not found: bad", None) == (
            "model not found: bad"
        )

    # Trimmed stream-json from `claude -p --model <bad> --output-format
    # stream-json`: a synthetic assistant turn plus an is_error 404 result.
    _BAD = (
        '{"type":"assistant","message":{"model":"<synthetic>"},'
        '"error":"model_not_found"}\n'
        '{"type":"result","subtype":"success","is_error":true,'
        '"api_error_status":404,"result":"There\'s an issue with the '
        'selected model (nope). It may not exist."}\n'
    )
    _GOOD = (
        '{"type":"assistant","message":{"model":"claude-haiku-4-5"}}\n'
        '{"type":"result","subtype":"success","is_error":false,'
        '"result":"ok"}\n'
    )

    def test_rejects_unknown_model_with_cli_message(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        msg = _model_probe_rejected(self._BAD)
        assert msg is not None
        assert "may not exist" in msg

    def test_accepts_usable_model(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        assert _model_probe_rejected(self._GOOD) is None

    def test_ignores_blank_and_unparsable_lines(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        assert _model_probe_rejected("\n  \nnot json\n") is None

    # --- additional probe-parser scenarios (independent conditions) ---
    # An is_error result carrying HTTP 404 but no `model_not_found` marker.
    _BAD_404_ONLY = (
        '{"type":"assistant","message":{"model":"<synthetic>"}}\n'
        '{"type":"result","subtype":"success","is_error":true,'
        '"api_error_status":404,"result":"model unavailable"}\n'
    )
    # A `model_not_found` marker with no result line carrying a message.
    _BAD_NO_MESSAGE = (
        '{"type":"assistant","message":{"model":"<synthetic>"},'
        '"error":"model_not_found"}\n'
    )
    # Real "Not logged in" output: is_error true, but it is an auth failure
    # (api_error_status null, no model_not_found), so the model is not at fault.
    _AUTH_FAIL = (
        '{"type":"assistant","message":{"model":"<synthetic>"},'
        '"error":"authentication_failed"}\n'
        '{"type":"result","subtype":"success","is_error":true,'
        '"api_error_status":null,"result":"Not logged in"}\n'
    )
    # A 404 that appears on a non-result line must be ignored.
    _IS_ERROR_NOT_RESULT = (
        '{"type":"assistant","is_error":true,"api_error_status":404,'
        '"message":{"model":"x"}}\n'
        '{"type":"result","subtype":"success","is_error":false,"result":"ok"}\n'
    )

    def test_rejects_on_http_404_without_model_not_found_marker(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        assert _model_probe_rejected(self._BAD_404_ONLY) == "model unavailable"

    def test_rejects_with_default_message_when_no_result_text(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        assert _model_probe_rejected(self._BAD_NO_MESSAGE) == "model not found"

    def test_accepts_when_error_is_auth_not_model(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        # is_error is true, but it is an auth failure (no 404, no
        # model_not_found), so the probe must not blame the model.
        assert _model_probe_rejected(self._AUTH_FAIL) is None

    def test_ignores_http_404_on_a_non_result_event(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        assert _model_probe_rejected(self._IS_ERROR_NOT_RESULT) is None


class TestSkillHelpers:
    """Direct checks of each condition in the skill-isolation helpers."""

    @staticmethod
    def _skill(parent: Path, name: str) -> Path:
        skill = parent / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(name, encoding="utf-8")
        return skill

    def test_inside_true_for_existing_directory_under_root(
        self, tmp_path: Path
    ) -> None:
        from binom_eval.runner import _inside

        root = tmp_path.resolve()
        (root / "a").mkdir()
        assert _inside(root / "a", root)

    def test_inside_false_when_lexically_outside_root(self, tmp_path: Path) -> None:
        from binom_eval.runner import _inside

        root = (tmp_path / "repo").resolve()
        root.mkdir()
        (tmp_path / "sibling").mkdir()
        assert not _inside(tmp_path / "sibling", root)
        assert not _inside(root, root)

    def test_inside_false_when_not_an_existing_directory(
        self, tmp_path: Path
    ) -> None:
        from binom_eval.runner import _inside

        root = tmp_path.resolve()
        (root / "file").write_text("x", encoding="utf-8")
        assert not _inside(root / "missing", root)
        assert not _inside(root / "file", root)

    def test_inside_false_when_symlink_target_leaves_root(
        self, tmp_path: Path
    ) -> None:
        from binom_eval.runner import _inside

        root = (tmp_path / "repo").resolve()
        root.mkdir()
        (tmp_path / "outside").mkdir()
        (root / "in").mkdir()
        (root / "good").symlink_to(root / "in")
        (root / "bad").symlink_to(tmp_path / "outside")
        assert _inside(root / "good", root)
        assert not _inside(root / "bad", root)

    def test_sibling_skills_drops_only_unkept_skills(self, tmp_path: Path) -> None:
        from binom_eval.runner import _sibling_skills

        self._skill(tmp_path, "keep")
        self._skill(tmp_path, "drop")
        (tmp_path / "_shared").mkdir()
        (tmp_path / "README.md").write_text("r", encoding="utf-8")
        names = ["keep", "drop", "_shared", "README.md"]
        assert _sibling_skills(str(tmp_path), names, {"keep"}) == {"drop"}
        assert _sibling_skills(str(tmp_path), names, {"keep", "drop"}) == set()
        assert _sibling_skills(str(tmp_path), names, set()) == {"keep", "drop"}

    def test_skill_containers_track_kept_names_per_container(
        self, tmp_path: Path
    ) -> None:
        from binom_eval.runner import _skill_containers

        root = tmp_path.resolve()
        kept = self._skill(root / ".claude/skills", "demo")
        elsewhere = self._skill(root / "pkg", "x")
        both = _skill_containers((kept, elsewhere), root)
        assert both[root / ".claude/skills"] == {"demo"}
        assert both[root / "skills"] == set()
        assert both[root / ".cursor/skills"] == set()
        only_outside = _skill_containers((elsewhere,), root)
        assert only_outside[root / ".claude/skills"] == set()

    def test_skill_ignore_falls_back_to_resolved_root_spelling(
        self, tmp_path: Path
    ) -> None:
        from binom_eval.runner import _skill_ignore

        repo = tmp_path / "repo"
        keep = self._skill(repo / ".claude/skills", "demo")
        self._skill(repo / ".claude/skills", "other")
        link = tmp_path / "link"
        link.symlink_to(repo)
        root = repo.resolve()
        ignore = _skill_ignore((keep.resolve(),), link)
        container = root / ".claude/skills"
        # Walked as spelled (via the symlink): re-based lexically.
        assert ignore(str(link / ".claude/skills"), ["demo", "other"]) == {"other"}
        # Walked as the resolved root (a symlink target copy): same result.
        assert ignore(str(container), ["demo", "other"]) == {"other"}
        # A directory that is not a container drops nothing but the ignores.
        assert ignore(str(root / ".claude"), ["skills"]) == set()

    def test_resolve_dot_path_naming_the_root_itself_rejected(
        self, tmp_path: Path
    ) -> None:
        self._skill(tmp_path, "skills/demo")
        with pytest.raises(ValueError, match="inside the repo root"):
            resolve_skill_dirs(tmp_path, "demo", ["."])
        assert resolve_skill_dirs(tmp_path, "demo", ["skills/demo"])

    def test_resolve_symlink_pointing_at_the_root_rejected(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "loop").symlink_to(tmp_path)
        with pytest.raises(ValueError, match="not inside the repo root"):
            resolve_skill_dirs(tmp_path, "demo", ["loop"])

    def test_dereference_removes_unkept_symlink_to_a_file(
        self, tmp_path: Path
    ) -> None:
        from binom_eval.runner import _dereference_symlinks, _skill_ignore

        repo = tmp_path / "repo"
        keep = self._skill(repo / ".claude/skills", "demo")
        (repo / "skills").symlink_to(repo / ".claude/skills/demo/SKILL.md")
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "skills").symlink_to(repo / ".claude/skills/demo/SKILL.md")
        root = repo.resolve()
        kept = (root / keep.relative_to(repo),)
        _dereference_symlinks(kept, root, dest, _skill_ignore(kept, repo))
        assert not (dest / "skills").exists()
        assert not (dest / "skills").is_symlink()

    def test_dereference_leaves_real_directories_untouched(
        self, tmp_path: Path
    ) -> None:
        from binom_eval.runner import _dereference_symlinks, _skill_ignore

        repo = tmp_path / "repo"
        keep = self._skill(repo / ".claude/skills", "demo")
        dest = tmp_path / "dest"
        self._skill(dest / ".claude/skills", "demo")
        root = repo.resolve()
        kept = (root / keep.relative_to(repo),)
        _dereference_symlinks(kept, root, dest, _skill_ignore(kept, repo))
        assert (dest / ".claude/skills/demo/SKILL.md").is_file()
