"""Which local model really writes code that passes the checks? Measured, not assumed from the model size.

Every candidate the engine checks is logged to <data>/model_stats.jsonl (model name, project types, passed...).
`choose_coder` picks the coder from those numbers; `k2loop model-stats` shows them.
"""
from __future__ import annotations

import time
from pathlib import Path

from .config import data_dir
from .jsonl import append_jsonl, read_jsonl

CODERS = ("llm", "strong")   # config profiles that can write code through Aider


def _path(cfg: dict) -> Path:
    return data_dir(cfg) / "model_stats.jsonl"


def record(cfg: dict, profile: str, *, kind: str, types: list[str], usable: bool, passed: bool, failures: int,
           secs: float) -> None:
    try:
        append_jsonl(_path(cfg), {"ts": time.time(), "profile": profile, "model": cfg[profile]["model"],
                                  "kind": kind, "types": list(types), "usable": usable, "passed": passed,
                                  "failures": failures, "secs": round(secs, 1)})
    except (OSError, KeyError):
        pass


def summary(cfg: dict, types: list[str] | None = None) -> dict[str, dict]:
    """Per model name: n, usable, passed, rate (smoothed pass rate), mean_secs. `types` filters by project type."""
    want = set(types or [])
    out: dict[str, dict] = {}
    for r in read_jsonl(_path(cfg)):
        if want and not want & set(r.get("types") or []):
            continue
        s = out.setdefault(r.get("model", "?"), {"n": 0, "usable": 0, "passed": 0, "secs": 0.0})
        s["n"] += 1
        s["usable"] += bool(r.get("usable"))
        s["passed"] += bool(r.get("passed")) and bool(r.get("usable"))
        s["secs"] += float(r.get("secs") or 0)
    for s in out.values():
        s["rate"] = (s["passed"] + 1) / (s["n"] + 2)          # Laplace: few samples stay near 0.5
        s["mean_secs"] = s["secs"] / s["n"] if s["n"] else 0.0
    return out


def _available(cfg: dict) -> list[str]:
    return [p for p in CODERS if p == "llm" or cfg.get("mode") == "local"]


def _measured(cfg: dict, types: list[str]) -> dict[str, dict] | None:
    """Stats for every available coder with enough samples: this project type first, then all projects."""
    need = int(cfg["tot"].get("coder_min_samples", 20))
    profs = _available(cfg)
    for t in (types, None):
        s = summary(cfg, t)
        got = {p: s.get(cfg[p]["model"]) for p in profs}
        if all(v and v["n"] >= need for v in got.values()):
            return got
    return None


def choose_coder(cfg: dict, types: list[str], failed: str | None = None) -> tuple[str, str]:
    """(profile, reason). The model with the clearly better measured pass rate wins; without enough data, or when
    the difference is small, the model that did NOT just fail is tried (a different model makes different mistakes)."""
    profs = _available(cfg)
    if len(profs) == 1:
        return profs[0], "tek kod modeli var"
    m = _measured(cfg, types)
    other = next((p for p in profs if p != failed), profs[0])
    if m is None:
        if failed:
            return other, f"yeterli ölçüm yok; {cfg[failed]['model']} takıldı, farklı model deneniyor"
        return "llm", "yeterli ölçüm yok; varsayılan kod modeli"
    best = max(profs, key=lambda p: m[p]["rate"])
    worst = min(profs, key=lambda p: m[p]["rate"])
    gap = m[best]["rate"] - m[worst]["rate"]
    rates = ", ".join(f"{cfg[p]['model']} %{m[p]['rate'] * 100:.0f} (n={m[p]['n']})" for p in profs)
    if gap >= float(cfg["tot"].get("coder_margin", 0.05)) or not failed:
        return best, f"ölçüme göre en iyi: {rates}"
    return other, f"ölçümler yakın ({rates}); {cfg[failed]['model']} takıldı, farklı model deneniyor"


def format_table(cfg: dict) -> str:
    rows = []
    all_types = sorted({t for r in read_jsonl(_path(cfg)) for t in (r.get("types") or [])})
    for label, t in [("tüm projeler", None)] + [(x, [x]) for x in all_types]:
        s = summary(cfg, t)
        if not s:
            continue
        rows.append(f"\n{label}:")
        rows.append(f"  {'model':<22}{'aday':>6}{'değişiklik':>12}{'geçti':>8}{'geçme %':>9}{'ort. sn':>9}")
        for model, v in sorted(s.items(), key=lambda kv: -kv[1]["rate"]):
            rows.append(f"  {model:<22}{v['n']:>6}{v['usable']:>12}{v['passed']:>8}"
                        f"{v['passed'] / v['n'] * 100 if v['n'] else 0:>8.0f}%{v['mean_secs']:>9.0f}")
    return "\n".join(rows) if rows else "Henüz ölçüm yok (her aday çalıştıkça birikir)."
