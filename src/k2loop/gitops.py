from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

# never let these reach a commit / candidate diff (.k2loop/check.sh: the judge must not be editable by the coder)
PROTECTED = [".k2loop/check.sh"]
EXCLUDE_PATHS = [*PROTECTED, ".sdd", ".k2loop/runs", ".k2loop/plans", "node_modules", ".pio", ".venv", ":(glob)**/.aider*",
                 ":(glob)**/__pycache__/**", ".pytest_cache", ".ruff_cache", ".mypy_cache", ":(glob)**/*.pyc"]
IDENT = ["-c", "user.name=k2loop", "-c", "user.email=k2loop@localhost"]


def git(args: list[str], cwd: Path, check: bool = True, input_: str | None = None) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, input=input_)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()[:500]}")
    return p.stdout


def stage_all(d: Path) -> None:
    """Stage everything (respecting .gitignore), then un-stage paths that must never be committed.
    (Exclude-pathspecs on `git add` fail when the path is also gitignored, e.g. .sdd/.)"""
    git(["add", "-A"], d)
    git(["reset", "-q", "--", *EXCLUDE_PATHS], d, check=False)


def restore_protected(d: Path) -> bool:
    """Put the committed .k2loop/check.sh back in a candidate worktree before its checks run.
    Returns True when the candidate had changed it."""
    changed = bool(git(["status", "--porcelain", "--", *PROTECTED], d, check=False).strip())
    if changed:
        git(["checkout", "HEAD", "--", *PROTECTED], d, check=False)
        git(["clean", "-fq", "--", *PROTECTED], d, check=False)  # untracked in HEAD: drop the coder's copy
    return changed


def snapshot_diff(d: Path) -> str:
    stage_all(d)
    return git(["diff", "--cached", "--binary", "HEAD"], d)


def apply_diff(d: Path, diff: str) -> None:
    if not diff.strip():
        return
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as fh:
        fh.write(diff)
        patch = fh.name
    try:
        try:
            git(["apply", "--index", "--binary", "--whitespace=nowarn", patch], d)
        except RuntimeError:
            git(["apply", "--index", "--binary", "--3way", "--whitespace=nowarn", patch], d)
    finally:
        Path(patch).unlink(missing_ok=True)


def new_worktree(repo: Path, dest: Path, base_diff: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        remove_worktree(repo, dest)
    git(["worktree", "add", "--detach", str(dest), "HEAD"], repo)
    if base_diff:
        apply_diff(dest, base_diff)
    return dest


def remove_worktree(repo: Path, dest: Path) -> None:
    git(["worktree", "remove", "--force", str(dest)], repo, check=False)
    shutil.rmtree(dest, ignore_errors=True)
    git(["worktree", "prune"], repo, check=False)


def diff_lines(diff: str) -> int:
    return sum(1 for ln in diff.splitlines()
               if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---")))


def changed_files(diff: str) -> list[str]:
    out = []
    for ln in diff.splitlines():
        if ln.startswith("+++ b/"):
            out.append(ln[6:])
    return out


def commit_all(d: Path, msg: str) -> list[str]:
    stage_all(d)
    if not git(["diff", "--cached", "--name-only"], d).strip():
        return []
    git([*IDENT, "commit", "-q", "-m", msg], d) if not _has_identity(d) else git(["commit", "-q", "-m", msg], d)
    return [f for f in git(["diff", "--name-only", "HEAD~1", "HEAD"], d, check=False).splitlines() if f]


def _has_identity(d: Path) -> bool:
    return bool(git(["config", "user.email"], d, check=False).strip())


def commit_staged(d: Path, msg: str) -> list[str]:
    """Commit exactly what is in the index (e.g. after `apply_diff`, which uses --index).
    Never sweeps in untracked files that tools or the orchestrator dropped into the worktree."""
    names = [f for f in git(["diff", "--cached", "--name-only"], d).splitlines() if f]
    if not names:
        return []
    ident = [] if _has_identity(d) else IDENT
    git([*ident, "commit", "-q", "-m", msg], d)
    return names
