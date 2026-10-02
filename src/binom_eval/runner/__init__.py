"""Invoking an agent CLI and parsing the result into `EvalRun`s.

The I/O layer of the harness: it is the only package that spawns
subprocesses and scrubs the environment. A `Runner` is a single live call;
`run_eval_batch` overlaps `count` independent calls for one eval to
measure the model's run-to-run variance. Evals are non-deterministic, so
nothing here caches — every trial is a fresh invocation.

Backends are pluggable: `Runner` is the backend-agnostic interface and
`ClaudeRunner` (in `claude_runner.py`) is the `claude -p` implementation.
The shared subprocess/env helpers (`stripped_env`, `fake_home_env`,
`isolated_workdir`, `_model_probe_rejected`) live here so every backend can
build on them. Live runs execute under a throwaway `HOME` (`fake_home_env`)
so no harness picks up the invoking user's settings -- user skill roots
(`~/.claude/skills`, `~/.cursor/skills`, ...), MCP config, or stored logins --
and grade only against the project; backends therefore authenticate from
environment credentials (`ANTHROPIC_API_KEY`, `CURSOR_API_KEY`) rather than a
stored session under the real home.

Concurrency is throttled by an optional shared `gate` (a `threading.Semaphore`
the caller threads through every run): trials within a batch, and whole evals
running in parallel above this layer, all acquire the one gate, so total live
calls never exceed its count regardless of suite size. Filesystem isolation is
optional too -- with `isolate=True` each run executes in a fresh throwaway copy
of `repo_root` (see `isolated_workdir`), so a skill that writes to the tree
cannot clobber a concurrent run.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from binom_eval.runner.retry import RetryableError, RetryPolicy
from binom_eval.stream_json import EvalRun, parse_stream_json, stream_error

DEFAULT_TIMEOUT_SECONDS = 300

# Back-off for one live trial. An API 500/overload or a CLI that dies
# mid-run surfaces as a nonzero exit or an `is_error` result event; grading
# such a trial as a behavioral failure would bias every posterior downward
# with noise unrelated to the skill, so backends retry the trial a few times
# (within the trial's own timeout budget) before marking the run errored.
# `_spawn_checked` raises every transient trial failure as `RetryableError`,
# which is exactly what `RetryPolicy` retries.
TRIAL_RETRY = RetryPolicy(
    max_attempts=3,
    base_delay_seconds=1.0,
    max_delay_seconds=8.0,
)

# Regenerable or heavy directories not copied into a per-run isolated
# workdir: caches are rebuilt on demand and dependency trees would dominate
# the per-run copy cost. `.git` is deliberately kept so skills that shell out
# to git still see a real repository.
ISOLATION_IGNORE = (
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
)

# Conventional skill containers under the repo root, searched (relative to
# it) by the `--live-eval-isolate-skill` form. Cursor also reads
# `.claude/skills/` and `skills/`, so all three are checked.
SKILL_ROOTS = (".claude/skills", ".cursor/skills", "skills")


# Markers Claude Code sets on every child process to signal a nested
# session. The CLI itself strips this exact trio when it needs a child to
# behave like a clean top-level invocation, so the eval runner mirrors it.
NESTED_SESSION_MARKERS = (
    "CLAUDECODE",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
)


def stripped_env() -> dict[str, str]:
    """A copy of the current environment with nested-session markers removed.

    The nested `claude -p` runs must not inherit the outer session's
    `CLAUDECODE`, `CLAUDE_CODE_SESSION_ID`, or `CLAUDE_CODE_CHILD_SESSION`
    markers, which would otherwise make the CLI behave as a nested session
    rather than a fresh top-level invocation.
    """
    return {
        k: v for k, v in os.environ.items() if k not in NESTED_SESSION_MARKERS
    }


@contextlib.contextmanager
def fake_home_env() -> Iterator[dict[str, str]]:
    """Yield a scrubbed env whose `HOME` points at a fresh empty directory.

    Layered on `stripped_env`, this repoints `HOME` (and `USERPROFILE` on
    Windows) at a throwaway temp directory for the duration of one run, so a
    spawned agent CLI cannot read the invoking user's home: no user skill
    roots (`~/.claude/skills`, `~/.cursor/skills`, ...), no per-user MCP or CLI
    config, and no stored login session. Evals therefore grade only against
    the project's own skills/agents, never whatever happens to be installed
    for the user. Because the stored session is hidden, every backend must
    authenticate from an environment credential (e.g. `ANTHROPIC_API_KEY`,
    `CURSOR_API_KEY`), which is preserved by `stripped_env`. The temp home is
    removed when the run ends.
    """
    env = stripped_env()
    with tempfile.TemporaryDirectory(prefix="binom-eval-home-", ignore_cleanup_errors=True) as home:
        env["HOME"] = home
        env["USERPROFILE"] = home
        yield env


def _model_probe_rejected(stdout: str) -> str | None:
    """Verdict for a model probe's stream-json `stdout`, with no I/O.

    Returns the CLI's human-readable error message when the run reports the
    model is unusable -- `error == "model_not_found"` on any event, or an
    `is_error` result carrying HTTP 404 -- and None otherwise. Kept pure so
    the parsing is unit-testable without spawning `claude`.
    """
    message: str | None = None
    rejected = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("error") == "model_not_found":
            rejected = True
        if event.get("type") == "result" and event.get("is_error"):
            if event.get("api_error_status") == 404:
                rejected = True
            message = event.get("result") or message
    return (message or "model not found") if rejected else None


def _run_error(
    returncode: int | None, stdout: str, stderr: str = ""
) -> str | None:
    """Why one completed CLI trial should not be graded, or None when clean.

    A nonzero exit means the CLI itself failed (the reason usually lands on
    stderr); a clean exit can still carry an errored stream (`is_error`
    result event, or no assistant events at all) -- see `stream_error`.
    """
    if returncode:
        lines = (stderr or stdout).strip().splitlines()
        tail = lines[-1] if lines else ""
        message = f"CLI exited with status {returncode}"
        return f"{message}: {tail}" if tail else message
    return stream_error(stdout)


def _errored_run(prompt: str, error: str) -> EvalRun:
    """An `EvalRun` marking a trial that produced no gradable result."""
    return EvalRun(
        eval_id="",
        prompt=prompt,
        skill_invoked=False,
        assistant_text="",
        tool_uses=[],
        model="",
        errored=True,
        error=error,
    )


def _spawn_checked(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    remaining: float,
    last_error: list[str],
) -> subprocess.CompletedProcess[str]:
    """Run one CLI trial attempt; raise `RetryableError` on any error signal.

    Converts the error outcomes -- `subprocess.TimeoutExpired`, nonzero exit,
    or an errored stream (see `_run_error`) -- into `RetryableError` so the
    `TRIAL_RETRY` loop in `_run_trial` retries them, recording the reason in
    `last_error` for the eventual errored run.
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired:
        last_error[0] = f"trial timed out after {remaining:.0f}s"
        raise RetryableError(last_error[0]) from None
    error = _run_error(proc.returncode, proc.stdout, proc.stderr)
    if error is not None:
        last_error[0] = error
        raise RetryableError(error)
    return proc


def _run_trial(
    prompt: str,
    timeout: int,
    attempt: Callable[[float, list[str]], EvalRun],
) -> EvalRun:
    """Drive one trial's attempts through `TRIAL_RETRY`.

    `attempt(remaining_seconds, last_error)` performs a single live call
    (typically via `_spawn_checked`, which records its failure reason in
    `last_error` before raising `RetryableError`). Returns the first
    successful attempt's run, or an errored `EvalRun` carrying the last
    failure reason once retries or the `timeout` budget are exhausted.
    """
    last_error = ["trial produced no result"]
    result = TRIAL_RETRY.execute(
        lambda remaining: attempt(remaining, last_error),
        timeout,
    )
    if result is None:
        return _errored_run(prompt, last_error[0])
    return result


def _format_model_error(
    message: str, valid_models: list[str] | None = None
) -> str:
    """Append a sorted valid-models suffix when the backend could enumerate them."""
    if not valid_models:
        return message
    known = ", ".join(sorted(set(valid_models)))
    return f"{message}; valid models: {known}"


def resolve_skill_dirs(
    repo_root: Path,
    skill_name: str,
    skill_paths: Sequence[str] = (),
    include_evaluated: bool = False,
) -> tuple[Path, ...]:
    """Resolve and validate the skill directories to keep when isolating.

    The result is the union of (a) with `include_evaluated`
    (`--live-eval-isolate-skill`), the conventional locations of `skill_name`
    under `repo_root` that hold a `SKILL.md` (a stray directory without one is
    ignored as long as another location has the skill; none at all is an
    error), and (b) every `skill_paths` entry
    (`--live-eval-isolate-skill-path`, repeatable), each relative to
    `repo_root` and inside it. The two are additive, so the evaluated skill
    plus helper skills it invokes can all be kept. Duplicates (the same
    normalized location, e.g. the flag's skill also given as a path) appear
    once, in first-seen order. With neither, the result is empty.

    Returned paths are lexically normalized locations under the resolved
    `repo_root`; a skill directory (or a container on the way to it) that is
    a symlink is deliberately left unresolved, because that is the location
    `copytree` walks and `isolated_workdir` later dereferences. Where it
    points must still resolve inside `repo_root`, checked here for the flag
    and path forms alike.

    Raises:
      ValueError: naming the offending path, when a `skill_paths` entry is
        absolute, lies outside `repo_root` (or is `repo_root` itself), when a
        skill directory or one of its containers is a symlink whose target
        resolves outside `repo_root`, when an entry lies at or under a
        standard container (`SKILL_ROOTS`) without being exactly
        `<container>/<name>` holding a `SKILL.md`, or when no matching skill
        directory exists (flag form: none holds a `SKILL.md`), so a typo never
        silently copies nothing (or everything). Any bad element fails the
        whole call.
    """
    root = repo_root.resolve()
    found: list[Path] = []
    if include_evaluated:
        candidates = [root / base / skill_name / "SKILL.md" for base in SKILL_ROOTS]
        evaluated = [path.parent for path in candidates if path.is_file()]
        if not evaluated:
            # A stray directory without a SKILL.md at one location must not
            # fail the run while another location holds the real skill, so
            # only candidates with a SKILL.md count; none at all is the error.
            tried = ", ".join(str(path) for path in candidates)
            raise ValueError(
                "--live-eval-isolate-skill: skill directory not found "
                f"(no SKILL.md at any of: {tried})"
            )
        found.extend(evaluated)
    for skill_path in skill_paths:
        if Path(skill_path).is_absolute():
            raise ValueError(
                "--live-eval-isolate-skill-path must be relative to the repo "
                f"root, got absolute path {skill_path!r}"
            )
        candidate = Path(os.path.normpath(root / skill_path))
        if candidate == root or root not in candidate.parents:
            raise ValueError(
                "--live-eval-isolate-skill-path must stay inside the repo "
                f"root {root}, got {skill_path!r}"
            )
        if not candidate.is_dir():
            raise ValueError(
                f"--live-eval-isolate-skill-path {skill_path!r}: skill "
                f"directory not found (tried {candidate})"
            )
        found.append(candidate)
    for path in found:
        _check_skill_location(path, root)
        target = path.resolve()
        if target == root or root not in target.parents:
            raise ValueError(
                f"skill directory {path} resolves to {target}, which is not "
                f"inside the repo root {root}"
            )
    return tuple(dict.fromkeys(found))


def _check_skill_location(path: Path, root: Path) -> None:
    """Reject a skill directory that lies at or under a standard container
    (`SKILL_ROOTS`) unless it is exactly `<container>/<name>` holding a
    `SKILL.md`. Judged lexically (a symlink is not resolved), so a path inside
    a skill, or a container itself, is rejected rather than resolved up (the
    sibling filter would otherwise drop every skill). Paths outside every
    container are unchecked. `path` is a normalized location under the
    resolved `root`.
    """
    parts = Path(os.path.normpath(path)).relative_to(root).parts
    for base in SKILL_ROOTS:
        container = Path(base).parts
        if parts[: len(container)] != container:
            continue
        if len(parts) != len(container) + 1 or not (path / "SKILL.md").is_file():
            raise ValueError(
                f"--live-eval-isolate-skill-path {'/'.join(parts)} is inside "
                f"skill container {base}; point it at the skill directory "
                f"itself ({base}/<name>) containing SKILL.md"
            )


def _is_skill(directory: str, name: str) -> bool:
    """Whether `directory/name` is a skill: a directory holding `SKILL.md`
    (followed through symlinks)."""
    return os.path.isfile(os.path.join(directory, name, "SKILL.md"))


def _sibling_skills(directory: str, names: list[str], kept: set[str]) -> set[str]:
    """The children of a skill container to drop: skills that are not kept.
    Non-skill entries (a README, a `_shared` directory, ...) stay."""
    return {name for name in names if name not in kept and _is_skill(directory, name)}


def _skill_containers(
    skill_dirs: tuple[Path, ...], root: Path
) -> dict[Path, set[str]]:
    """Map every location where sibling skills must be filtered to the names
    to keep there.

    The locations are the `SKILL_ROOTS` under the resolved `root` as spelled
    *and* where each resolves to inside `root`. A container that is a symlink
    (`.claude/skills -> ../shared/skills`, `.cursor/skills ->
    ../.claude/skills`) or sits under a symlinked parent (`.claude ->
    ../shared/claude`) shares its contents with its target, which `copytree`
    walks as an ordinary directory, so the target is filtered too. Containers
    that resolve to the same place share one set of kept names (the kept
    skill's name, per container).
    """
    lexical = [root / base for base in SKILL_ROOTS]

    def real(container: Path) -> Path:
        target = container.resolve()
        return target if root in target.parents else container

    kept: dict[Path, set[str]] = {real(container): set() for container in lexical}
    for skill_dir in skill_dirs:
        if skill_dir.parent in lexical:
            kept[real(skill_dir.parent)].add(skill_dir.name)
    return {**kept, **{container: kept[real(container)] for container in lexical}}


def _skill_ignore(
    skill_dirs: tuple[Path, ...], repo_root: Path
) -> Callable[[str, list[str]], set[str]]:
    """Build a `copytree` ignore callback excluding sibling skills.

    In a standard skill container, every child that looks like a skill (a
    directory containing `SKILL.md`) and is not a kept skill directory is
    skipped; non-skill entries (files, directories without `SKILL.md`) are
    kept. Containers are the `SKILL_ROOTS` under `repo_root` (whether or not
    they hold a kept skill, so sibling skills in the other conventional
    containers do not leak) plus the in-repo target of each symlinked
    container or container under a symlinked parent (see
    `_skill_containers`), so a shared real directory such as `shared/skills`
    is filtered where it is walked. A kept skill whose parent is not a
    standard container (e.g. an explicit path such as `my-skill` or
    `plugins/x/skills/foo`) leaves that parent unfiltered: it is copied
    normally, siblings included. `copytree` is walking `repo_root` as spelled
    (or, when copying a symlink target, the resolved root), so each visited
    directory is re-based onto the resolved root lexically. A symlink is never
    walked by `copytree`; see `_dereference_symlinks`. Composed with
    `ISOLATION_IGNORE`.
    """
    root = repo_root.resolve()
    keep = _skill_containers(skill_dirs, root)
    base = shutil.ignore_patterns(*ISOLATION_IGNORE)

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = set(base(directory, names))
        try:
            relative = Path(directory).relative_to(repo_root)
        except ValueError:
            relative = Path(directory).relative_to(root)
        kept = keep.get(Path(os.path.normpath(root / relative)))
        if kept is not None:
            skipped |= _sibling_skills(directory, names, kept)
        return skipped

    return ignore


@contextlib.contextmanager
def isolated_workdir(
    repo_root: Path,
    isolate: bool,
    isolate_skill: tuple[Path, ...] | None = None,
) -> Iterator[Path]:
    """Yield the working directory for a single run.

    When `isolate` is false, yields `repo_root` unchanged -- every run shares
    the one tree, which is safe only for skills that do not write to it. When
    true, copies `repo_root` into a fresh temporary directory (skipping the
    regenerable/heavy entries in `ISOLATION_IGNORE`) and yields that copy, so
    a skill that mutates the tree cannot clobber a concurrent run; the copy is
    removed when the run ends.

    Args:
      repo_root: The tree `claude -p` should run against.
      isolate: Whether to run in a throwaway copy rather than `repo_root`.
      isolate_skill: When not None, the skill directories to keep (as
        returned by `resolve_skill_dirs`); sibling skills (directories with a
        `SKILL.md`) in the standard containers (`SKILL_ROOTS`) are excluded
        from the copy, while non-skill entries there are kept. A skill outside
        those containers keeps its parent's other children. A symlinked skill
        directory, standard container, or parent of one is replaced in the
        copy by a real directory (filtered the same way); other symlinks are
        copied as links. The in-repo target of a symlinked container (e.g.
        `shared/skills` behind `.claude/skills`) is filtered too, wherever it
        is copied, so no sibling skill is visible at any location. A symlinked
        parent of a container is handled the same way (`.claude ->
        ../shared/claude`).

    Yields:
      The directory to use as the run's `cwd`.

    Raises:
      ValueError: when `isolate_skill` is empty or names a directory that
        does not exist inside `repo_root` (lexically, or via a symlinked skill
        directory or container whose target is outside it), or when a path
        lies at or under a standard container without being exactly
        `<container>/<name>` holding a `SKILL.md`.
    """
    if not isolate:
        yield repo_root
        return
    ignore = shutil.ignore_patterns(*ISOLATION_IGNORE)
    if isolate_skill is not None:
        root = repo_root.resolve()
        bad = [path for path in isolate_skill if not _inside(path, root)]
        if not isolate_skill or bad:
            raise ValueError(
                f"isolate_skill must name existing directories inside {root}, "
                f"got {[str(path) for path in isolate_skill]}"
            )
        for path in isolate_skill:
            _check_skill_location(Path(os.path.normpath(path)), root)
        kept = tuple(Path(os.path.normpath(path)) for path in isolate_skill)
        ignore = _skill_ignore(kept, repo_root)
    with tempfile.TemporaryDirectory(prefix="binom-eval-", ignore_cleanup_errors=True) as tmp:
        dest = Path(tmp) / repo_root.name
        shutil.copytree(repo_root, dest, symlinks=True, ignore=ignore)
        if isolate_skill is not None:
            _dereference_symlinks(kept, root, dest, ignore)
        yield dest


def _dereference_symlinks(
    skill_dirs: tuple[Path, ...],
    root: Path,
    dest: Path,
    ignore: Callable[[str, list[str]], set[str]],
) -> None:
    """Replace copied symlinks on the way to skills with real, filtered copies.

    `copytree(symlinks=True)` copies a symlink as a symlink and never walks
    it, so the ignore callback never filters it, and in the copy it dangles
    (relative link) or points back into the real repo (absolute link), so
    sibling skills would leak and writes would escape. Every symlink among the
    path components of each `SKILL_ROOTS` container and each kept skill
    directory (a symlinked parent such as `.claude`, the container, or the
    skill itself) is swapped, outermost first, for a copy of its resolved
    target made with `ignore`, so sibling skills are dropped from the copy
    exactly as in the main walk. The rest of the tree keeps its symlinks.
    A symlink on the way to a container with no kept skill whose target is
    outside `root` (or not a directory) is just removed: nothing of it can be
    copied, and it must not leave a link back out. On the way to a kept skill
    that cannot occur for validated input (`resolve_skill_dirs`); it raises
    rather than silently deleting the skill. `skill_dirs` are lexical
    locations under the resolved `root`.
    """
    required = {
        Path(*rel.parts[:i])
        for rel in (skill.relative_to(root) for skill in skill_dirs)
        for i in range(1, len(rel.parts) + 1)
    }
    prefixes = required | {
        Path(*Path(base).parts[:i])
        for base in SKILL_ROOTS
        for i in range(1, len(Path(base).parts) + 1)
    }
    for prefix in sorted(prefixes, key=lambda path: len(path.parts)):
        entry = dest / prefix
        if not entry.is_symlink():
            continue
        entry.unlink(missing_ok=True)
        target = (root / prefix).resolve()
        if target == root or root not in target.parents or not target.is_dir():
            if prefix in required:
                raise ValueError(
                    f"isolate_skill must name existing directories inside "
                    f"{root}, but {root / prefix} resolves to {target}"
                )
            continue
        shutil.copytree(target, entry, symlinks=True, ignore=ignore)


def _inside(path: Path, root: Path) -> bool:
    """Whether `path` is an existing directory under `root`, before and after
    following symlinks (`root` already resolved)."""
    lexical = Path(os.path.normpath(path))
    return (
        root in lexical.parents
        and path.is_dir()
        and root in path.resolve().parents
    )


class Runner(ABC):
    """A backend that can probe and invoke an agent CLI for one eval run.

    The harness depends only on this interface, so a suite can be graded
    against `claude -p`, `cursor`, or any other agent CLI by swapping in a
    different implementation. Concrete runners own the CLI specifics (binary
    name, flags, version probe); everything above this layer is backend-
    agnostic.
    """

    @abstractmethod
    def version(self) -> str:
        """Return the backend CLI version string, or '' when unavailable."""

    @abstractmethod
    def preflight(self) -> str | None:
        """Return why this backend cannot run live evals, or None when ready.

        Checked once before any trial so a missing CLI or absent credential
        fails the session fast with a clear message rather than surfacing as a
        run-time error on every trial. The message is backend-specific (e.g.
        which binary must be on PATH, which credential must be set).
        """

    @abstractmethod
    def validate_model(self, model: str, timeout: int = 30) -> str | None:
        """Confirm the backend can use `model`; return an error or None."""

    @abstractmethod
    def run(
        self,
        prompt: str,
        repo_root: Path,
        skill_name: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        *,
        isolate: bool = False,
        isolate_skill: tuple[Path, ...] | None = None,
        model: str,
    ) -> EvalRun:
        """Invoke the backend once and parse its output into an `EvalRun`."""


# Imported after `Runner` and the shared helpers are defined: the backend
# modules import them back from this package, so the names must already be
# bound when their module bodies execute.
from binom_eval.runner.claude_runner import ClaudeRunner  # noqa: E402
from binom_eval.runner.cursor_runner import CursorRunner  # noqa: E402

# The selectable backends, keyed by the prefix used in `--live-eval-model`
# (`backend:model`). The prefix is mandatory -- there is no default backend --
# so every live run names the harness it targets.
BACKENDS: dict[str, type[Runner]] = {
    "claude": ClaudeRunner,
    "cursor": CursorRunner,
}


def resolve_runner(spec: str | None) -> tuple[str, str, Runner]:
    """Parse a `--live-eval-model` spec into (backend, model, runner).

    The spec must be `backend:model` (e.g. `claude:haiku` or
    `cursor:sonnet-4.5`): the backend is always explicit so each run targets a
    single, named harness. The split is on the first colon only, so model
    names may themselves contain colons.

    Raises:
      ValueError: when the spec is missing, carries no `backend:` prefix, names
        an unknown backend, or has an empty model -- callers surface this as a
        clear command-line error.
    """
    known = ", ".join(sorted(BACKENDS))
    backend, sep, model = (spec or "").partition(":")
    if not sep:
        raise ValueError(
            "--live-eval-model must be 'backend:model' "
            f"(known backends: {known}); got {spec!r}"
        )
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown eval backend {backend!r} in --live-eval-model {spec!r}; "
            f"known backends: {known}"
        )
    if not model:
        raise ValueError(f"--live-eval-model {spec!r} has an empty model")
    return backend, model, BACKENDS[backend]()


def run_eval_batch(
    item: dict[str, Any],
    repo_root: Path,
    skill_name: str,
    count: int,
    *,
    gate: threading.Semaphore | None = None,
    isolate: bool = False,
    isolate_skill: tuple[Path, ...] | None = None,
    model: str,
    runner: Runner,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[EvalRun]:
    """Run one eval `count` times against `runner`, concurrently.

    Each run is an isolated `subprocess.run`, so threads share nothing
    mutable; concurrency just overlaps the model latency. Every trial is a
    fresh live call — repeated trials exist to measure the model's
    run-to-run variance.

    `gate`, when given, is a shared semaphore every trial acquires before
    spawning its subprocess; the same object passed across batches and across
    concurrently-running evals caps total live calls at its count. `isolate`
    is forwarded to the runner so each trial runs in its own throwaway copy of
    `repo_root` when set; `isolate_skill` narrows that copy to the evaluated
    skill (see `isolated_workdir`). `model` selects the specific model used for all
    trials in the batch. `runner` is the backend every trial runs against;
    it is backend-agnostic (`ClaudeRunner`, `CursorRunner`, ...).
    `timeout` sets the per-trial subprocess deadline in seconds; defaults to
    `DEFAULT_TIMEOUT_SECONDS`.
    """
    backend = runner
    eid = item["id"]
    prompt = item["prompt"]
    prompt_input = item.get("prompt_input", "")
    limit: contextlib.AbstractContextManager[Any] = (
        gate if gate is not None else contextlib.nullcontext()
    )

    # Forwarded only when set, so a `Runner` written before `isolate_skill`
    # existed (whose `run` does not accept it) keeps working.
    extra = {} if isolate_skill is None else {"isolate_skill": isolate_skill}

    def one(_: int) -> EvalRun:
        with limit:
            return backend.run(
                prompt,
                repo_root,
                skill_name,
                timeout,
                isolate=isolate,
                model=model,
                **extra,
            )

    with ThreadPoolExecutor(max_workers=count) as pool:
        runs = list(pool.map(one, range(count)))
    for run in runs:
        run.eval_id = eid
        run.prompt_input = prompt_input
    return runs


__all__ = [
    "BACKENDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "ISOLATION_IGNORE",
    "NESTED_SESSION_MARKERS",
    "SKILL_ROOTS",
    "TRIAL_RETRY",
    "ClaudeRunner",
    "CursorRunner",
    "Runner",
    "fake_home_env",
    "isolated_workdir",
    "resolve_runner",
    "resolve_skill_dirs",
    "run_eval_batch",
    "stripped_env",
]
