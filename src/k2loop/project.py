"""Project type detection and generation of .k2loop/check.sh (the single source of truth for 'tests pass')."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def detect(root: Path) -> dict:
    types: list[str] = []
    checks: list[tuple[str, str]] = []
    notes: list[str] = []

    has_py = any((root / f).exists() for f in ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"))
    py_tests = (root / "tests").is_dir() or any(root.glob("test_*.py")) or any(root.glob("**/tests/test_*.py"))
    if has_py or py_tests:
        types.append("python")
        if shutil.which("ruff"):
            checks.append(("lint", "ruff check ."))
        if py_tests:
            checks.append(("tests", "$PYTEST -x -q"))
        else:
            notes.append("Python: test bulunamadı (tests/ veya test_*.py). Sadece lint çalışacak.")

    pkg = root / "package.json"
    if pkg.exists():
        types.append("node")
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        scripts = data.get("scripts", {}) or {}
        deps = {**(data.get("dependencies") or {}), **(data.get("devDependencies") or {})}
        if (root / "tsconfig.json").exists() and "typescript" in deps:
            checks.append(("typecheck", "npx --no-install tsc --noEmit"))
        if "eslint" in deps:
            checks.append(("lint", "npx --no-install eslint ."))
        if "test" in scripts and "no test specified" not in scripts["test"]:
            checks.append(("tests", "npm test --silent"))
        else:
            notes.append("Node: package.json içinde 'test' script'i yok.")

    ini = root / "platformio.ini"
    if ini.exists():
        types.append("platformio")
        text = ini.read_text(errors="ignore", encoding="utf-8")
        checks.append(("build", "pio run"))
        if "[env:native]" in text:
            checks.append(("tests", "pio test -e native"))
        else:
            notes.append("PlatformIO: [env:native] yok; sadece derleme kontrol edilecek. "
                         "Donanım testleri otomatik yapılamaz.")

    mk = root / "Makefile"
    if mk.exists() and "\ntest:" in "\n" + mk.read_text(errors="ignore", encoding="utf-8") and not any(n == "tests" for n, _ in checks):
        types.append("make")
        checks.append(("tests", "make test"))

    if not types:
        types.append("unknown")
    if not checks:
        notes.append("Hiç kontrol komutu bulunamadı: çıktılar doğrulanmadan kabul edilir. "
                     ".k2loop/check.sh dosyasını elle düzenleyin.")
    return {"types": types, "checks": checks, "notes": notes}


CHECK_HEADER = r'''#!/usr/bin/env bash
# k2loop tarafından üretildi. Elle düzenleyebilirsiniz.
# Hem k2loop (aday seçimi) hem Bernstein (merge kapısı) "testler geçti mi" sorusunu buna sorar.
set -uo pipefail
ROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd || pwd)"
PY="python3"; [ -x "$ROOT/.venv/bin/python" ] && PY="$ROOT/.venv/bin/python"; [ -x ".venv/bin/python" ] && PY=".venv/bin/python"
# pytest: projenin venv'i > sistem python > uv (proje bağımlılıklarını geçici ortama kurar)
PYTEST="$PY -m pytest"
if ! "$PY" -c "import pytest" >/dev/null 2>&1 && command -v uv >/dev/null; then PYTEST="uv run --quiet --with pytest python -m pytest"; fi
# worktree'lerde bağımlılık klasörleri yok: ana repodakilere bağla (commit'e girmez, k2loop hariç tutar)
[ -f package.json ] && [ ! -e node_modules ] && [ -d "$ROOT/node_modules" ] && ln -s "$ROOT/node_modules" node_modules
[ -f package.json ] && [ ! -e node_modules ] && { npm ci --silent --no-audit --no-fund || npm install --silent --no-audit --no-fund; }
[ -f platformio.ini ] && [ ! -e .pio/libdeps ] && [ -d "$ROOT/.pio/libdeps" ] && mkdir -p .pio && ln -s "$ROOT/.pio/libdeps" .pio/libdeps
run() { echo "=== [$1] $2"; bash -c "$2"; rc=$?; if [ $rc -ne 0 ]; then echo "=== [$1] FAILED (exit $rc)"; exit $rc; fi; echo "=== [$1] OK"; }
'''


def render_check_sh(checks: list[tuple[str, str]]) -> str:
    body = [CHECK_HEADER]
    if not checks:
        body.append('echo "K2LOOP_NO_CHECKS"\nexit 0\n')
    for name, cmd in checks:
        body.append(f'run {name} "{cmd}"\n')
    body.append('echo "=== ALL CHECKS PASSED"\n')
    return "".join(body)
