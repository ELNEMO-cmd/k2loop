"""`k2loop` command line: init / plan / run / go / solve / doctor / status / laya-*."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .claude import version as claude_version
from .config import CONFIG_PATH, data_dir, load_config, write_laya_overrides
from .decisions import QUESTIONS, LayaClient, load_logs
from .jsonl import append_jsonl
from .llm import LLM
from .project import detect, render_check_sh

BERNSTEIN_ROLES = ["backend", "frontend", "qa", "docs", "devops", "data", "ml-engineer", "security",
                   "architect", "reviewer", "resolver", "ci-fixer", "manager"]


def bernstein_yaml(model: str, budget_min: int) -> str:
    t = max(2 * 3600, (budget_min + 45) * 60)
    roles = "\n".join(f"  {r}: {{cli: generic, model: {model}}}" for r in BERNSTEIN_ROLES)
    return f"""# k2loop tarafından üretildi: K2 Horizon (yerel) + Laya + Claude
# Her görev k2loop worker'ına (generic-cli) gider; Claude'u worker kendisi çağırır.
goal: "Execute the tasks of the given plan file. Do not invent new tasks."
cli: generic
max_agents: 1
internal_llm_provider: none
evolution_enabled: false
auto_decompose: false
merge_strategy: direct      # yerel merge (varsayılan 'pr' GitHub PR açmaya çalışır)
auto_merge: true
role_model_policy:
{roles}
quality_gates:
  enabled: true
  lint: false
  type_check: false
  tests: true
  test_command: "bash .k2loop/check.sh"
constraints:
  - "Work only inside the task scope."
  - "Checks: bash .k2loop/check.sh"
tuning:
  task:
    scope_timeout_s: {{small: {t}, medium: {t}, large: {t}}}
    xl_timeout_s: {t}
  orchestrator:
    max_agent_runtime_s: {t}
  agent:
    heartbeat_starting_timeout_s: 1800
"""


def _git(args: list[str], cwd: Path, check: bool = True) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise SystemExit(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


def _root() -> Path:
    p = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("Bu klasör bir git deposu değil. Önce: git init && git add -A && git commit -m ilk")
    return Path(p.stdout.strip())


# ---------------------------------------------------------------- init
def _exclude_locally(root: Path, before: set[str], new: set[str]) -> None:
    """Hide files created by `bernstein init` via .git/info/exclude (local only, never committed)."""
    if not new:
        return
    entries = set()
    for f in new:
        top = f.split("/", 1)[0]
        pre_existing = any(b == top or b.startswith(top + "/") for b in before) or (root / top).is_file() and "/" not in f
        entries.add(f if pre_existing or "/" not in f else top + "/")
    ex = root / ".git" / "info" / "exclude"
    ex.parent.mkdir(parents=True, exist_ok=True)
    cur = ex.read_text(encoding="utf-8").splitlines() if ex.exists() else []
    ex.write_text("\n".join(cur + ["# bernstein init (k2loop)"] + sorted(e for e in entries if e not in cur)) + "\n", encoding="utf-8")


def cmd_init(a: argparse.Namespace) -> int:
    cfg = load_config()
    root = Path.cwd()
    if not (root / ".git").exists():
        if not a.yes and input("Git deposu yok. 'git init' yapılsın mı? [E/h] ").strip().lower() in ("h", "n"):
            return 1
        _git(["init", "-q"], root)
    if not _git(["rev-parse", "--verify", "HEAD"], root, check=False).strip():
        _git(["add", "-A"], root)
        _git(["-c", "user.name=k2loop", "-c", "user.email=k2loop@localhost", "commit", "-q", "--allow-empty",
              "-m", "initial commit"], root)
    info = detect(root)
    (root / ".k2loop").mkdir(exist_ok=True)
    chk = root / ".k2loop" / "check.sh"
    if chk.exists() and not a.force:
        print(f"• {chk} zaten var, dokunulmadı (--force ile yeniden üret)")
    else:
        chk.write_text(render_check_sh(info["checks"]), encoding="utf-8")
        chk.chmod(0o755)
    if not (root / ".sdd").exists() and shutil.which("bernstein"):
        before = set(_git(["ls-files", "--others", "--exclude-standard"], root).splitlines())
        subprocess.run(["bernstein", "init"], cwd=root, stdin=subprocess.DEVNULL, timeout=300,
                       stdout=subprocess.DEVNULL)
        after = set(_git(["ls-files", "--others", "--exclude-standard"], root).splitlines())
        _exclude_locally(root, before, after - before - {"bernstein.yaml", ".gitignore"})
    by = root / "bernstein.yaml"
    if by.exists():
        by.rename(root / f"bernstein.yaml.bak-{time.strftime('%Y%m%d%H%M%S')}")
    budget = max(int(cfg["tot"]["task_time_budget_min"]), int(cfg["strong"]["task_time_budget_min"]))
    by.write_text(bernstein_yaml(cfg["llm"]["model"], budget), encoding="utf-8")
    gi = root / ".gitignore"
    lines = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    for pat in (".sdd/", ".k2loop/runs/", ".k2loop/plans/", "bernstein.yaml.bak-*", ".aider*",
                "__pycache__/", ".pytest_cache/"):
        if pat not in lines:
            lines.append(pat)
    gi.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _git(["add", ".k2loop/check.sh", "bernstein.yaml", ".gitignore"], root)
    if _git(["diff", "--cached", "--name-only"], root).strip():
        ident = [] if _git(["config", "user.email"], root, check=False).strip() else \
            ["-c", "user.name=k2loop", "-c", "user.email=k2loop@localhost"]
        _git([*ident, "commit", "-q", "-m", "k2loop init"], root)
    print(f"✓ Proje tipi: {', '.join(info['types'])}")
    for n, c in info["checks"]:
        print(f"  kontrol [{n}]: {c}")
    for n in info["notes"]:
        print(f"  ! {n}")
    print("✓ .k2loop/check.sh ve bernstein.yaml yazıldı ve commit'lendi.")
    print("Sıradaki adım:  k2loop go \"ne yapılacağını anlatan hedef\"")
    return 0


# ---------------------------------------------------------------- plan / run / go
def _latest_plan(root: Path) -> Path:
    plans = sorted((root / ".k2loop" / "plans").glob("*.yaml"))
    if not plans:
        raise SystemExit("Plan yok. Önce: k2loop plan \"hedef\"")
    return plans[-1]


def cmd_plan(a: argparse.Namespace) -> int:
    from .planner import make_plan, summarize
    root = _root()
    path = make_plan(load_config(), a.goal, root)
    print(summarize(path))
    print(f"\n✓ Plan: {path}")
    return 0


def _ensure_work_branch(root: Path, path: Path) -> str:
    """Bernstein refuses to merge into the default branch: work on k2loop/<plan> instead."""
    cur = _git(["rev-parse", "--abbrev-ref", "HEAD"], root).strip()
    if cur.startswith("k2loop/") or cur not in ("main", "master", "trunk", "develop"):
        return cur
    if _git(["status", "--porcelain", "--untracked-files=no"], root).strip():
        raise SystemExit("Commit'lenmemiş değişiklik var; önce commit/stash yapın.")
    br = "k2loop/" + path.stem[:60]
    exists = _git(["rev-parse", "--verify", "--quiet", br], root, check=False).strip()
    _git(["switch", "-q", br] if exists else ["switch", "-q", "-c", br], root)
    print(f"Çalışma dalı: {br}  (bitince inceleyip '{cur}' dalına merge edin)")
    return br


def _port_busy(port: int = 8052) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _common_dir(root: Path) -> str:
    c = Path(_git(["rev-parse", "--git-common-dir"], root).strip())
    return str(((root / c) if not c.is_absolute() else c).resolve())


def _wait_bernstein_settled(root: Path, timeout: float = 180) -> None:
    """Bernstein keeps merging/cleaning in the background after `run` returns."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        wts = [ln for ln in _git(["worktree", "list", "--porcelain"], root).splitlines()
               if ln.startswith("worktree ") and "/.sdd/" in ln]
        if not wts:
            return
        time.sleep(5)


def collect(root: Path, since: float = 0.0) -> int:
    """Bring k2loop commits that Bernstein failed to merge into the current branch (safe cherry-pick)."""
    from .jsonl import read_jsonl
    cfg = load_config()
    picked = 0
    for rec in read_jsonl(data_dir(cfg) / "commits.jsonl"):
        if rec.get("repo") != _common_dir(root) or rec.get("ts", 0) < since:
            continue
        sha = rec["sha"]
        if subprocess.run(["git", "cat-file", "-e", sha], cwd=root).returncode != 0:
            print(f"  ! {sha[:8]} bulunamadı (atlandı)")
            continue
        if subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=root).returncode == 0:
            continue
        subj = _git(["log", "-1", "--format=%s", sha], root).strip()
        if _git(["log", "--format=%s", "-200"], root).splitlines().count(subj) and \
                not _git(["diff", sha + "^", sha, "--stat"], root).strip():
            continue
        untracked = set(_git(["ls-files", "--others", "--exclude-standard"], root).splitlines())
        for f in _git(["diff-tree", "--no-commit-id", "--name-only", "-r", sha], root).splitlines():
            if f in untracked:  # remove only byte-identical untracked copies
                blob = subprocess.run(["git", "show", f"{sha}:{f}"], cwd=root, capture_output=True).stdout
                if (root / f).is_file() and (root / f).read_bytes() == blob:
                    (root / f).unlink()
        ident = [] if _git(["config", "user.email"], root, check=False).strip() else \
            ["-c", "user.name=k2loop", "-c", "user.email=k2loop@localhost"]
        r = subprocess.run(["git", *ident, "cherry-pick", "-x", sha], cwd=root, capture_output=True, text=True)
        if r.returncode == 0:
            picked += 1
            print(f"  ✓ alındı: {sha[:8]} {subj}")
        else:
            subprocess.run(["git", "cherry-pick", "--abort"], cwd=root, capture_output=True)
            print(f"  ! {sha[:8]} otomatik alınamadı; elle: git cherry-pick {sha}")
    return picked


def cmd_mode(a: argparse.Namespace) -> int:
    cfg = load_config()
    if not a.mode:
        print(f"Mod: {cfg['mode']}  ({'her şey yerel: ' + cfg['strong']['model'] + ' planlar/review eder' if cfg['mode'] == 'local' else 'Claude planlar, eskale eder, review eder'})")
        return 0
    from .config import write_overrides
    write_overrides({"mode": a.mode, "claude": {"enabled": a.mode == "hybrid"}})
    print(f"✓ Mod: {a.mode}")
    if a.mode == "local" and not strong_model_on_disk(cfg):
        print(f"  ! Güçlü model ({cfg['strong']['model']}) kurulu değil: k2loop strong")
    return 0


STRONG_MODELS = {  # kind -> (llama-swap model name, model folder, description)
    "moe": ("qwen3.6-35b-a3b", "models-moe", "Qwen3.6-35B-A3B MoE (3B aktif, hızlı)"),
    "dense": ("qwen3.8-27b", "models-strong", "Qwen3.8 27B yoğun (yavaş)"),
}


def strong_model_on_disk(cfg: dict) -> bool:
    folder = next((f for n, f, _ in STRONG_MODELS.values() if n == cfg["strong"]["model"]), "models-strong")
    d = Path(cfg["paths"]["data_dir"]) / folder
    return d.exists() and any(d.rglob("*.gguf"))


def cmd_strong(a: argparse.Namespace) -> int:
    from .config import write_overrides
    cfg = load_config()
    if not a.kind:
        cur = next((k for k, (n, _, _) in STRONG_MODELS.items() if n == cfg["strong"]["model"]), "?")
        for k, (n, folder, desc) in STRONG_MODELS.items():
            d = Path(cfg["paths"]["data_dir"]) / folder
            have = d.exists() and any(d.rglob("*.gguf"))
            print(f"{'→' if k == cur else ' '} {k:<6} {n:<18} {desc}  {'(diskte)' if have else '(indirilmemiş)'}")
        return 0
    name, folder, desc = STRONG_MODELS[a.kind]
    write_overrides({"strong": {"model": name}})
    print(f"✓ Güçlü model: {desc} ({name})")
    d = Path(cfg["paths"]["data_dir"]) / folder
    if not (d.exists() and any(d.rglob("*.gguf"))):
        print(f"  ! Model diskte yok ({d}). Kurmak için: ./install.sh --skip-root --strong {a.kind}")
    return 0


def cmd_collect(a: argparse.Namespace) -> int:
    n = collect(_root())
    print(f"{n} commit çalışma dalına alındı.")
    return 0


def _run_plan(path: Path) -> int:
    root = _root()
    if _port_busy():
        raise SystemExit("Bernstein zaten çalışıyor (port 8052). Önceki projede 'k2loop stop' (veya "
                         "'bernstein stop') çalıştırın; aynı anda tek proje çalışabilir.")
    br = _ensure_work_branch(root, path)
    env = dict(os.environ, BERNSTEIN_ADAPTER_ADMISSION_POLICY="off")
    print(f"Bernstein başlıyor: bernstein run {path}")
    print("İzlemek için başka terminalde:  bernstein live   (veya tarayıcı: bernstein gui serve)")
    t0 = time.time()
    rc = subprocess.call(["bernstein", "run", "--auto-approve", str(path)], cwd=root, env=env)
    print("Bernstein'ın arka plan merge/temizliği bekleniyor...")
    _wait_bernstein_settled(root)
    n = collect(root, since=t0)
    if n:
        print(f"{n} görev commit'i çalışma dalına alındı (Bernstein merge'ünün kaçırdıkları).")
    print(f"\nBitti (çıkış kodu {rc}). Sonuçlar '{br}' dalında:  git log --oneline {br}")
    return rc


def cmd_run(a: argparse.Namespace) -> int:
    root = _root()
    return _run_plan(Path(a.plan).resolve() if a.plan else _latest_plan(root))


def cmd_go(a: argparse.Namespace) -> int:
    from .planner import make_plan, summarize
    root = _root()
    if not (root / ".k2loop" / "check.sh").exists():
        raise SystemExit("Bu projede önce: k2loop init")
    path = make_plan(load_config(), a.goal, root)
    print(summarize(path))
    if not a.yes and input("\nBu plan çalıştırılsın mı? [E/h] ").strip().lower() in ("h", "n", "hayir", "no"):
        print(f"Plan kaydedildi, sonra çalıştırmak için: k2loop run {path}")
        return 0
    return _run_plan(path)


def cmd_solve(a: argparse.Namespace) -> int:
    """Run the ToT engine on a single task WITHOUT Bernstein (quick test)."""
    from .engine import Engine
    from .heartbeat import Heartbeat
    root = _root()
    if _git(["status", "--porcelain"], root).strip():
        raise SystemExit("Çalışma ağacında commit'lenmemiş değişiklik var. Önce commit/stash yapın (git status).")
    if not a.here:
        br = f"k2loop/solve-{time.strftime('%Y%m%d-%H%M%S')}"
        _git(["switch", "-c", br], root)
        print(f"Yeni dal: {br}")
    cfg = load_config()
    hb = Heartbeat([], echo=print)
    res = Engine(cfg, a.task, root, time.strftime("%Y%m%d-%H%M%S") + "-solve", hb=hb,
                 project_types=detect(root)["types"], title=a.task.splitlines()[0]).run()
    print(f"\nSonuç: {res.status}\n{res.summary}")
    if res.patch_path:
        print(f"Patch: {res.patch_path}")
    return 0 if res.status == "done" else 1


# ---------------------------------------------------------------- serve / ask
def cmd_serve(a: argparse.Namespace) -> int:
    from .gateway import serve
    serve(load_config())
    return 0


def cmd_ask(a: argparse.Namespace) -> int:
    """Send a goal to the running `k2loop serve` and print the live stream (same text as Open WebUI)."""
    import urllib.request
    cfg = load_config()
    ag = cfg["agent"]
    body = json.dumps({"model": ag.get("model_name", "k2agent"), "stream": True,
                       "messages": [{"role": "user", "content": a.goal}]}).encode()
    headers = {"Content-Type": "application/json"}
    if ag.get("api_key"):
        headers["Authorization"] = f"Bearer {ag['api_key']}"
    req = urllib.request.Request(f"http://127.0.0.1:{ag['port']}/v1/chat/completions", data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=None) as r:
            for raw in r:
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[5:])["choices"][0]["delta"]
                print(delta.get("content", ""), end="", flush=True)
    except KeyboardInterrupt:
        print("\n(izlemeyi bıraktın; hedef arka planda devam ediyor — /durum için: k2loop ask /durum)")
    except OSError as exc:
        raise SystemExit(f"k2loop serve çalışmıyor mu? ({exc})  →  systemctl --user status k2-agent")
    print()
    return 0


# ---------------------------------------------------------------- doctor / status
def _http_ok(url: str, timeout: float = 5, key: str = "") -> tuple[bool, str]:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 400, f"HTTP {r.status}"
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return False, "bağlantı reddedildi (servis kapalı)"
        return False, str(exc.reason)[:80]
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:80]


def cmd_doctor(a: argparse.Namespace) -> int:
    cfg = load_config()
    bad = 0

    def row(ok: bool, name: str, detail: str = "") -> None:
        nonlocal bad
        bad += 0 if ok else 1
        print(f"{'✓' if ok else '✗'} {name:<22} {detail}")

    print(f"k2loop {__version__}  config: {CONFIG_PATH} {'(var)' if CONFIG_PATH.exists() else '(yok, varsayılanlar)'}")
    tools = ["bernstein", "aider", "git", "generic-cli", "llama-swap"]
    if cfg["mode"] != "local":
        tools.append("claude")
    for tool in tools:
        row(bool(shutil.which(tool)), tool, shutil.which(tool) or "PATH'te yok")
    if cfg["mode"] != "local":
        cv = claude_version(cfg)
        row(bool(cv), "claude sürüm", cv or "kurulu değil")
    swap = cfg["models"]["swap_url"].rstrip("/")
    ok, detail = _http_ok(swap + "/running")
    row(ok, "llama-swap", f"{swap}/ui" if ok else detail + "  → systemctl --user status k2-swap")
    from .research import health as research_health
    rh = research_health(cfg)
    row(rh["searxng"], "SearXNG", cfg["research"]["searxng_url"] if rh["searxng"] else "kapalı → docker ps -a | grep k2-searxng")
    has_ldr_login = bool(cfg["research"].get("ldr_user"))
    row(rh["ldr"], "Local Deep Research", (cfg["research"]["ldr_url"] + ("" if has_ldr_login else
        "  ! secrets.env'de LDR_USER/LDR_PASSWORD yok (SearXNG'ye düşer)")) if rh["ldr"] else "kapalı → systemctl --user status k2-ldr")
    ok, detail = _http_ok(cfg["trace"]["endpoint"].rsplit("/v1/", 1)[0])
    row(ok or not cfg["trace"]["enabled"], "Phoenix (iz ağacı)", "http://127.0.0.1:6006" if ok else detail + "  → systemctl --user status k2-phoenix")
    ag = cfg["agent"]
    ok, detail = _http_ok(f"http://127.0.0.1:{ag['port']}/v1/models", key=ag.get("api_key") or "")
    row(ok, "k2agent (serve)", f"http://127.0.0.1:{ag['port']}/v1" if ok else detail + "  → systemctl --user status k2-agent")
    ok, detail = _http_ok("http://127.0.0.1:3000/health")
    row(ok, "Open WebUI", "http://127.0.0.1:3000" if ok else detail + "  → docker ps -a | grep k2-webui")
    if a.full:
        t = time.time()
        try:
            ans = LLM(cfg, "helper").chat([{"role": "user", "content": "Reply with the single word: pong"}],
                                          max_tokens=64, timeout=600)
            row("pong" in ans.lower(), "yardımcı model (CPU)", f"{time.time() - t:.1f}s → {ans[:40]!r}")
        except Exception as exc:  # noqa: BLE001
            row(False, "yardımcı model (CPU)", str(exc)[:120])
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        row(True, "GPU", out)
    else:
        row(False, "GPU", "nvidia-smi yok")
    ok, detail = LLM(cfg).health()
    if not ok and "[Errno 111] Connection refused" in detail:
        detail = "bağlantı reddedildi (servis kapalı)"
    row(ok, "K2 sunucusu", detail + (f" ctx={cfg['llm'].get('context')}" if ok else "  → journalctl --user -u k2-swap -n 50"))
    if ok and a.full:
        t = time.time()
        try:
            ans = LLM(cfg).chat([{"role": "user", "content": "Reply with the single word: pong"}], effort="low",
                                max_tokens=2048, timeout=600)
            row("pong" in ans.lower(), "K2 cevap testi", f"{time.time() - t:.1f}s → {ans[:40]!r}")
        except Exception as exc:  # noqa: BLE001
            row(False, "K2 cevap testi", str(exc))
    lc = LayaClient(cfg["laya"]["url"])
    ok, detail = lc.health()
    if not ok and "[Errno 111] Connection refused" in detail:
        detail = "bağlantı reddedildi (servis kapalı)"
    row(ok or not cfg["laya"]["enabled"], "Laya servisi", detail[:80] if ok else detail + "  → systemctl --user status k2-laya")
    if ok and a.full:
        ans = lc.predict({"log_tail": "ModuleNotFoundError: No module named 'requests'"}, QUESTIONS["test_classification"])
        row(bool(ans), "Laya tahmin testi", json.dumps(ans.get("category", {}).get("choice")) if ans else "hata")
    print("Laya modları:", cfg["laya"]["mode"])
    print(f"Çalışma modu: {cfg['mode']}")
    has_strong = strong_model_on_disk(cfg)
    sname = cfg["strong"]["model"]
    row(has_strong or cfg["mode"] != "local", "güçlü model", f"{sname} " + ("var" if has_strong else "yok (local mod için gerekli)"))
    if a.full and has_strong and cfg["mode"] == "local":
        from .swap import strong_model
        t = time.time()
        with strong_model(cfg) as ok:
            ans = LLM(cfg, "strong").chat([{"role": "user", "content": "Reply with the single word: pong"}],
                                          max_tokens=4096, timeout=900) if ok else ""
        row("pong" in ans.lower(), "güçlü model testi", f"{sname}: {time.time() - t:.0f}s (GPU takası dahil)")
    if a.full and cfg["claude"]["enabled"] and cfg["mode"] != "local":
        from .claude import run_claude
        res = run_claude(cfg, "Reply with the single word: pong", Path.cwd(), model="haiku", edit=False,
                         timeout=120, max_turns=1)
        row(res.ok, "Claude giriş testi", res.text[:40] if res.ok else (res.error[:120] or "başarısız → 'claude' ile giriş yapın"))
    print("\nHer şey ✓ ise: tarayıcıda http://127.0.0.1:3000 → model 'k2agent' → hedefini yaz"
          if not bad else f"\n{bad} sorun var.")
    return 0 if not bad else 1


def cmd_status(a: argparse.Namespace) -> int:
    return subprocess.call(["bernstein", "status"])


def cmd_stop(a: argparse.Namespace) -> int:
    return subprocess.call(["bernstein", "stop"], cwd=_root())


# ---------------------------------------------------------------- laya
def threshold_scan(rows: list[tuple[dict, str]], who: str) -> tuple[float, float, float | None, float]:
    """rows = (record, true label). Returns (rule_acc, pure_acc, best_threshold|None, best_acc) where the
    predictor `who` (laya|semif) decides when its confidence >= threshold and the rule decides otherwise."""
    n = len(rows)
    rule_acc = sum(r["rule"] == y for r, y in rows) / n
    pure = sum(r[who] == y for r, y in rows) / n
    best_t, best_acc = None, rule_acc
    for t in [x / 100 for x in range(50, 96, 5)]:
        acc = sum(((r[who] if (r.get(f"{who}_conf") or 0) >= t else r["rule"]) == y) for r, y in rows) / n
        if acc > best_acc + 0.02:
            best_t, best_acc = t, acc
    return rule_acc, pure, best_t, best_acc


def cmd_laya_eval(a: argparse.Namespace) -> int:
    from .config import write_overrides
    cfg = load_config()
    laya_upd: dict = {"mode": {}, "thresholds": {}}
    semif_thr: dict = {}
    for name in QUESTIONS:
        recs, outs = load_logs(cfg, name)
        labeled = [(r, outs[r["id"]]) for r in recs if r["id"] in outs]
        print(f"\n== {name}: {len(recs)} karar, {len(labeled)} etiketli")
        if len(labeled) < a.min_labels:
            print(f"   yetersiz veri (en az {a.min_labels} etiket gerekli) — gölge modda kalıyor")
            continue
        winner = None  # (gain, who, threshold, acc)
        for who, label in (("laya", "Laya"), ("semif", "SemIf")):
            rows = [(r, y) for r, y in labeled if r.get(who)]
            if len(rows) < a.min_labels:
                print(f"   {label}: yetersiz tahmin ({len(rows)})")
                continue
            rule_acc, pure, t, acc = threshold_scan(rows, who)
            print(f"   {label}: kural {rule_acc:.3f} · saf {label} {pure:.3f} · "
                  + (f"eşik {t:.2f} ile {acc:.3f}" if t is not None else "kuralı yenemiyor"))
            if t is not None and (winner is None or acc - rule_acc > winner[0]):
                winner = (acc - rule_acc, who, t, acc)
        if winner is None:
            print("   → kural kalsın (shadow)")
            continue
        _, who, t, acc = winner
        mode = "active" if who == "laya" else "semif"
        print(f"   → ÖNERİ: {mode}, eşik {t:.2f} (doğruluk {acc:.3f})")
        laya_upd["mode"][name] = mode
        if who == "laya":
            laya_upd["thresholds"][name] = t
        else:
            semif_thr[name] = t
    if a.apply and laya_upd["mode"]:
        write_laya_overrides(laya_upd)
        if semif_thr:
            write_overrides({"semif": {"thresholds": semif_thr}})
        print("\n✓ Uygulandı:", laya_upd, semif_thr or "")
    elif laya_upd["mode"]:
        print("\nUygulamak için: k2loop laya-eval --apply")
    return 0


def cmd_laya_export(a: argparse.Namespace) -> int:
    cfg = load_config()
    out = Path(a.out)
    n = 0
    out.unlink(missing_ok=True)
    for name, qs in QUESTIONS.items():
        recs, outs = load_logs(cfg, name)
        qid = next(iter(qs))
        for r in recs:
            if r["id"] in outs:
                append_jsonl(out, {"decision": name, "state": r["state"], "questions": qs,
                                   "label": {qid: outs[r["id"]]}})
                n += 1
    print(f"✓ {n} etiketli örnek → {out}\n  Laya'nın resmi fine-tune notebook'u ile kullanın (README'ye bakın).")
    return 0


def cmd_laya_mode(a: argparse.Namespace) -> int:
    names = list(QUESTIONS) if a.decision == "all" else [a.decision]
    write_laya_overrides({"mode": {n: a.mode for n in names}})
    print(f"✓ {', '.join(names)} → {a.mode}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="k2loop", description="K2 Horizon + Laya + Qwen: yerel ToT ajan döngüsü")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="projeyi hazırla (check.sh + bernstein.yaml)")
    p.add_argument("--force", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("serve", help="sürekli ajan döngüsü + Open WebUI için OpenAI uyumlu uç (systemd: k2-agent)")
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("ask", help="çalışan ajana hedef gönder ve canlı izle (Open WebUI'sız)")
    p.add_argument("goal")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("plan", help="3 alternatifli plan üret (yerel modda Qwen, hibritte Claude)")
    p.add_argument("goal")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("run", help="planı Bernstein ile çalıştır (varsayılan: son plan)")
    p.add_argument("plan", nargs="?")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("go", help="plan + onay + çalıştır")
    p.add_argument("goal")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(fn=cmd_go)
    p = sub.add_parser("solve", help="tek görevi Bernstein'sız dene (test için)")
    p.add_argument("task")
    p.add_argument("--here", action="store_true", help="yeni dal açmadan mevcut dalda çalış")
    p.set_defaults(fn=cmd_solve)
    p = sub.add_parser("doctor", help="kurulumu kontrol et")
    p.add_argument("--full", action="store_true", help="model/Claude cevap testleri de yap")
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("status", help="bernstein status")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("mode", help="hybrid (Claude) / local (tamamen yerel, güçlü yerel model) modu göster/değiştir")
    p.add_argument("mode", nargs="?", choices=["hybrid", "local"])
    p.set_defaults(fn=cmd_mode)
    p = sub.add_parser("strong", help="güçlü modeli göster/seç: moe (Qwen3.6-35B-A3B, hızlı) | dense (Qwen3.8 27B)")
    p.add_argument("kind", nargs="?", choices=list(STRONG_MODELS))
    p.set_defaults(fn=cmd_strong)
    p = sub.add_parser("collect", help="Bernstein'ın merge edemediği k2loop commit'lerini çalışma dalına al")
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("stop", help="bu projedeki Bernstein'ı durdur (arka planda çalışmaya devam eder)")
    p.set_defaults(fn=cmd_stop)
    p = sub.add_parser("laya-eval", help="gölge mod verisini değerlendir, eşik öner")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--min-labels", type=int, default=150)
    p.set_defaults(fn=cmd_laya_eval)
    p = sub.add_parser("laya-export", help="fine-tune için etiketli veri dışa aktar")
    p.add_argument("--out", default=str(Path.home() / "k2loop-laya-dataset.jsonl"))
    p.set_defaults(fn=cmd_laya_export)
    p = sub.add_parser("laya-mode", help="bir kararı kim versin: shadow (kural) / active (Laya) / semif")
    p.add_argument("decision", choices=[*QUESTIONS, "all"])
    p.add_argument("mode", choices=["shadow", "active", "semif"])
    p.set_defaults(fn=cmd_laya_mode)
    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
