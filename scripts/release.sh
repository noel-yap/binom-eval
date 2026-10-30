#!/usr/bin/env bash
#
# Cut a release: infer/compute the next semantic version from the Conventional
# Commits since the last tag, create an annotated git tag, and push it.
#
# The package version is derived from the tag (hatch-vcs), so there is no
# manifest to bump. Pushing the tag starts the Release workflow's build and
# publish jobs (.github/workflows/release.yml).
#
# This one script is the only implementation of the bump logic. Two callers:
#   * a human, via `make release` / `make release-dry` (checks that the tree is
#     clean and on an in-sync `main` before tagging);
#   * the Release workflow's `tag` job, via `--ci`, on every push to `main`.
#
# Usage:
#   scripts/release.sh [--dry-run] [--ci] [major|minor|patch|<X.Y.Z>]
#
# With no level/version the bump is inferred from the commits since the last
# tag: a breaking change (a `type!:` subject or a `BREAKING CHANGE:` footer)
# -> major (minor while the current major is 0), any `feat` -> minor, else
# patch. `--dry-run` prints the computed version and notes without tagging.
#
# `--ci` is non-interactive and tags HEAD of the commit the workflow checked
# out. It skips the branch and remote-sync checks, and exits 0 without tagging
# when there are no commits since the last tag (HEAD is already released). On a
# tag it writes `tag=<name>` to $GITHUB_OUTPUT so later jobs know what to build.

set -euo pipefail

readonly MAIN_BRANCH="main"

err() {
  echo "release: $*" >&2
}

die() {
  err "$*"
  exit 1
}

# Echo the remote to push the tag to: `upstream` when it exists, else `origin`.
tag_remote() {
  if git remote get-url upstream >/dev/null 2>&1; then
    echo "upstream"
  else
    echo "origin"
  fi
}

# Echo the most recent version tag, or empty when there are none.
latest_tag() {
  git describe --tags --abbrev=0 2>/dev/null || true
}

# Echo the non-merge commit subjects since $1 (all history when $1 is empty),
# one per line.
commit_subjects() {
  local since="$1"
  if [[ -n "${since}" ]]; then
    git log --no-merges --pretty=format:'%s' "${since}..HEAD"
  else
    git log --no-merges --pretty=format:'%s'
  fi
}

# Echo the inferred bump level (major|minor|patch) from the commits since $1,
# given the current major version $2 (for the semver 0.x initial-dev rule).
infer_bump() {
  local since="$1"
  local current_major="$2"
  local range subjects bodies
  if [[ -n "${since}" ]]; then
    range="${since}..HEAD"
  else
    range="HEAD"
  fi
  subjects="$(commit_subjects "${since}")"
  bodies="$(git log --no-merges --pretty=format:'%B' "${range}")"
  if grep -qE '^[a-z]+(\([^)]*\))?!:' <<<"${subjects}" \
      || grep -qE '^BREAKING[ -]CHANGE:' <<<"${bodies}"; then
    if (( current_major == 0 )); then
      echo "minor"
    else
      echo "major"
    fi
    return
  fi
  if grep -qE '^feat(\([^)]*\))?:' <<<"${subjects}"; then
    echo "minor"
    return
  fi
  echo "patch"
}

# Echo version "$1" bumped by level "$2" (major|minor|patch).
bump_version() {
  local version="$1"
  local level="$2"
  local major minor patch
  IFS='.' read -r major minor patch <<<"${version}"
  case "${level}" in
    major) (( major += 1 )); minor=0; patch=0 ;;
    minor) (( minor += 1 )); patch=0 ;;
    patch) (( patch += 1 )) ;;
    *) die "unknown bump level: ${level}" ;;
  esac
  echo "${major}.${minor}.${patch}"
}

# Manual releases only: die unless HEAD is the in-sync tip of $MAIN_BRANCH.
check_main_in_sync() {
  local remote="$1"
  local branch remote_head
  branch="$(git branch --show-current)"
  [[ "${branch}" == "${MAIN_BRANCH}" ]] \
    || die "must release from '${MAIN_BRANCH}', on '${branch:-detached HEAD}'"
  [[ -z "$(git status --porcelain)" ]] \
    || die "working tree is not clean; commit or stash first"
  git fetch --quiet "${remote}" "${MAIN_BRANCH}" \
    || die "failed to fetch ${remote}/${MAIN_BRANCH}"
  remote_head="$(git rev-parse FETCH_HEAD)"
  [[ "$(git rev-parse HEAD)" == "${remote_head}" ]] \
    || die "local ${MAIN_BRANCH} is not in sync with ${remote}/${MAIN_BRANCH}"
}

main() {
  local dry_run=0
  local ci=0
  local requested=""
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --dry-run) dry_run=1 ;;
      --ci) ci=1 ;;
      major|minor|patch) requested="${arg}" ;;
      v[0-9]*.[0-9]*.[0-9]* | [0-9]*.[0-9]*.[0-9]*) requested="${arg}" ;;
      "") ;;
      *) die "unrecognized argument: ${arg}" ;;
    esac
  done

  if (( ci )); then
    [[ -n "${GITHUB_OUTPUT:-}" ]] || die "--ci requires GITHUB_OUTPUT"
  fi

  git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "not inside a git work tree"

  local remote
  remote="$(tag_remote)"
  if (( ! ci )); then
    check_main_in_sync "${remote}"
  else
    [[ -z "$(git status --porcelain)" ]] \
      || die "working tree is not clean"
  fi
  git fetch --quiet --tags "${remote}" || die "failed to fetch tags"

  local prev prefix current
  prev="$(latest_tag)"
  if [[ -n "${prev}" ]]; then
    if [[ "${prev}" == v* ]]; then prefix="v"; else prefix=""; fi
    current="${prev#v}"
  else
    prefix="v"
    current="0.0.0"
  fi

  local subjects
  subjects="$(commit_subjects "${prev}")"
  if [[ -z "${subjects}" ]]; then
    # HEAD is already released (the nearest tag is at HEAD) or the repo is empty.
    if (( ci )); then
      err "no commits since ${prev:-repo start}; nothing to release"
      return 0
    fi
    die "no commits since ${prev:-repo start}; nothing to release"
  fi

  local next
  if [[ -z "${requested}" ]]; then
    local level
    level="$(infer_bump "${prev}" "${current%%.*}")"
    err "inferred bump: ${level}"
    next="$(bump_version "${current}" "${level}")"
  elif [[ "${requested}" =~ ^(major|minor|patch)$ ]]; then
    next="$(bump_version "${current}" "${requested}")"
  else
    next="${requested#v}"
  fi

  local new_tag="${prefix}${next}"
  ! git rev-parse -q --verify "refs/tags/${new_tag}" >/dev/null \
    || die "tag ${new_tag} already exists"

  echo "Release ${prev:-<none>} -> ${new_tag}  (remote: ${remote})"
  echo "Commits since ${prev:-repo start}:"
  echo "  - ${subjects//$'\n'/$'\n'  - }"

  if (( dry_run )); then
    echo
    echo "(dry run; no tag created)"
    return
  fi

  local message notes
  notes="- ${subjects//$'\n'/$'\n'- }"
  printf -v message '%s\n\n%s\n' "${new_tag}" "${notes}"
  git tag -a "${new_tag}" -m "${message}"
  git push "${remote}" "${new_tag}"
  echo "Tagged ${new_tag} and pushed to ${remote}; the Release workflow will publish it."
  if (( ci )); then
    echo "tag=${new_tag}" >> "${GITHUB_OUTPUT}"
  fi
}

main "$@"
