"""OpenAI-compatible endpoint so Open WebUI can talk to the agent (model "k2agent").

A chat message becomes a goal in the agent's queue; the reply streams the agent's live events (research,
plan, ToT candidates, Laya decisions, test results...) as Markdown, like a Claude Code transcript. Closing the
browser does not stop the goal: the loop keeps working and the result is also in the event log.

Open WebUI's own background jobs (chat titles, tags, follow-up suggestions) are answered by the helper model
instead of becoming goals.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

from .agent import Agent
from .config import load_config
from .llm import LLM

HELP = """**k2agent komutları**
- Normal mesaj → yeni hedef. Türünü ajan anlar:
  - **araştırma**: "yerel LLM'ler için en iyi arama motorlarını karşılaştır"
  - **plan** (her konu): "3 ayda Almanca B1 için çalışma planı yap" → planlayıcılar tartışır, hakem seçer
  - **kod**: proje yolunu yaz: "`~/projem` içine CSV grafiği ekle, testleriyle"
- `/durum` → şu an ne yapıyor, sırada ne var
- `/iptal` → çalışan hedefi durdur (adım sınırında)
- `/izle` → çalışan hedefin akışını bu sohbete bağla
- `/yardim` → bu liste"""

STAGE_ICON = {"research": "🔎", "plan": "🗺️", "repo": "📁", "debate": "🗣️", "judge": "⚖️", "refine": "✍️",
              "verify": "🔍"}
DECISION_TR = {"test_classification": "hata türü", "branch_pruning": "dal budama", "retry_escalate": "sonraki adım"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content or "")


def is_webui_task(text: str) -> bool:
    return "### Task:" in text or "<chat_history>" in text


class Formatter:
    """Turns event-log entries of one goal into Markdown chunks."""

    def __init__(self) -> None:
        self.last_phase = ""
        self.step_started = False

    def __call__(self, ev: dict) -> str:
        k = ev.get("kind")
        f = getattr(self, "_" + str(k), None)
        try:
            return f(ev) if f else ""
        except Exception:  # noqa: BLE001
            return ""

    def _goal(self, ev: dict) -> str:
        st = ev.get("state")
        if st == "started":
            return f"🚀 **Başladım:** {ev.get('text', '')}\n\n"
        return ""

    def _intent(self, ev: dict) -> str:
        if ev.get("goal_kind") == "code":
            return f"🧭 Tür: **kod değişikliği** · proje `{ev.get('project') or '?'}`\n\n"
        if ev.get("goal_kind") == "plan":
            return "🧭 Tür: **plan** (araştırma → farklı açılardan planlar → tartışma → hakem)\n\n"
        return "🧭 Tür: **araştırma**\n\n"

    def _alt(self, ev: dict) -> str:
        if ev.get("state") == "started":
            return f"- ✏️ **{ev.get('name')}** planlayıcı yazıyor… ({ev.get('i')}/{ev.get('n')})\n"
        return (f"\n<details>\n<summary>📄 {ev.get('name')} planı</summary>\n\n{ev.get('text', '')}\n\n"
                "</details>\n\n")

    def _debate(self, ev: dict) -> str:
        return f"> 🗣️ **{ev.get('speaker')}** (tur {ev.get('round')}): {ev.get('text', '')}\n\n"

    def _judge(self, ev: dict) -> str:
        names, scores = ev.get("names") or [], ev.get("scores") or []
        table = " · ".join(f"{n}: **{s}**" for n, s in zip(names, scores)) if scores else "puan yok"
        return f"- ⚖️ Hakem: {table} → seçilen **{ev.get('best')}**. {ev.get('why', '')}\n"

    def _stage(self, ev: dict) -> str:
        return f"\n### {STAGE_ICON.get(ev.get('name'), '▶️')} {ev.get('msg', '')}\n\n"

    def _research(self, ev: dict) -> str:
        out = ""
        prog = ev.get("progress")
        msg = ev.get("msg") or ""
        if msg or prog is not None:
            out = f"- 🔎 {msg}" + (f" · %{prog}" if isinstance(prog, (int, float)) else "") + "\n"
        for it in ev.get("items") or []:
            out += f"  - [{it.get('title') or it.get('url')}]({it.get('url')})\n"
        if ev.get("url"):
            out += f"  - Ayrıntılı rapor: {ev['url']}\n"
        return out

    def _research_report(self, ev: dict) -> str:
        return f"\n<details>\n<summary>📚 Araştırma raporu</summary>\n\n{ev.get('text', '')}\n\n</details>\n\n"

    def _log(self, ev: dict) -> str:
        t = str(ev.get("text", ""))
        if t.startswith("[k2loop]") and "GPU" not in t:
            return ""  # engine internals: cand/phase events already show them
        return f"- {t.replace('[k2loop] ', '')}\n"

    def _plan(self, ev: dict) -> str:
        return f"\n<details open>\n<summary>🗺️ Seçilen plan</summary>\n\n```\n{ev.get('text', '')}\n```\n\n</details>\n\n"

    def _step(self, ev: dict) -> str:
        if ev.get("state") == "started":
            self.step_started = True
            self.last_phase = ""
            return f"\n#### ⚙️ Adım {ev.get('i')}/{ev.get('n')}: {ev.get('title')}\n\n"
        icon = {"done": "✅", "human": "🟡", "failed": "❌", "cancelled": "⏹"}.get(ev.get("state"), "•")
        return f"\n{icon} **Adım {ev.get('i')} sonucu:** {str(ev.get('summary', ''))[:400]}\n\n"

    def _round(self, ev: dict) -> str:
        return f"\n---\n\n## 🔁 Tur {ev.get('round')}/{ev.get('max')}: {ev.get('msg', '')}\n\n"

    def _verify(self, ev: dict) -> str:
        if ev.get("done"):
            return f"- 🎯 Hedef tamamlandı: {ev.get('why', '')}\n"
        s = f"- 🔁 Hedef henüz bitmedi: {ev.get('why', '')}\n"
        return s + "".join(f"  - eksik: {r}\n" for r in ev.get("remaining") or [])

    def _task_start(self, ev: dict) -> str:
        return "" if self.step_started else f"\n#### ⚙️ Görev: {ev.get('title')}\n\n"

    def _phase(self, ev: dict) -> str:
        key = f"{ev.get('stage')}|{ev.get('msg')}"
        if key == self.last_phase:
            return ""
        self.last_phase = key
        return f"- ⏱ `{ev.get('stage')}` {ev.get('msg')} · bütçenin %{ev.get('pct')}'i kullanıldı\n"

    def _cand_start(self, ev: dict) -> str:
        return f"- ⏺ **{ev.get('name')}** yazılıyor…\n"

    def _cand(self, ev: dict) -> str:
        ok = ev.get("passed")
        files = ", ".join(f"`{f}`" for f in (ev.get("files") or [])[:5]) or "değişiklik yok"
        s = (f"  - {'✅' if ok else '❌'} **{ev.get('name')}**: {ev.get('lines')} satır · {files} · test "
             f"{'GEÇTİ' if ok else 'KALDI (' + str(ev.get('failures')) + ')'} · {ev.get('secs')} sn\n")
        if not ok and ev.get("check_tail"):
            s += (f"\n<details>\n<summary>{ev.get('name')} test çıktısı</summary>\n\n```\n{ev['check_tail']}\n```\n"
                  "\n</details>\n\n")
        return s

    def _decision(self, ev: dict) -> str:
        return f"  - 🧠 {DECISION_TR.get(ev.get('name'), ev.get('name'))}: **{ev.get('used')}**\n"

    def _hypotheses(self, ev: dict) -> str:
        s = f"  - 💡 Hata ayıklama seviye {ev.get('level')}: hipotezler\n"
        for h in ev.get("items") or []:
            s += f"    - {'✔ denenecek' if h.get('kept') else '✖ budandı'} ({h.get('score')}) {h.get('text')}\n"
        return s

    def _review(self, ev: dict) -> str:
        s = f"  - 👀 Review oyları: {', '.join(map(str, ev.get('votes') or [])) or '—'}\n"
        for i in ev.get("issues") or []:
            s += f"    - {i}\n"
        return s

    def _task_end(self, ev: dict) -> str:
        return "" if self.step_started else f"\n**Sonuç:** {ev.get('status')} — {str(ev.get('summary'))[:400]}\n"

    def _final(self, ev: dict) -> str:
        return f"\n---\n\n{ev.get('text', '')}\n"


class Gateway:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.cfg = agent.cfg

    # -------------------------------------------------------------- answers
    def status_text(self) -> str:
        goals = self.agent.q.snapshot()
        run = [g for g in goals if g.status == "running"]
        queued = [g for g in goals if g.status == "queued"]
        done = [g for g in goals if g.status not in ("running", "queued")][-5:]
        s = "**Durum**\n\n"
        s += (f"- ⚙️ Çalışıyor: {run[0].text[:120]}\n" if run else "- 💤 Boşta, yeni hedef bekliyor\n")
        s += "".join(f"- ⏳ Sırada: {g.text[:120]}\n" for g in queued)
        if done:
            s += "\n**Son bitenler**\n" + "".join(f"- {g.status}: {g.text[:100]}\n" for g in reversed(done))
        return s

    def follow(self, gid: str, from_id: int, stop: threading.Event) -> Iterator[str]:
        """Markdown chunks for one goal until it ends ('' = keep-alive tick)."""
        fmt = Formatter()
        last_ping = time.time()
        for ev in self.agent.ev.follow(from_id, stop):
            if ev is None:
                if time.time() - last_ping > 10:
                    last_ping = time.time()
                    yield ""
                continue
            if ev.get("goal") != gid:
                continue
            chunk = fmt(ev)
            if chunk:
                last_ping = time.time()
                yield chunk
            if ev.get("kind") == "goal" and ev.get("state") in ("done", "failed", "cancelled"):
                return

    def handle_chat(self, body: dict) -> Iterator[str]:
        msgs = body.get("messages") or []
        text = _text(msgs[-1].get("content")) if msgs else ""
        cmd = text.strip().lower()
        if is_webui_task(text):
            norm = [{"role": m.get("role", "user"), "content": _text(m.get("content"))} for m in msgs]
            try:
                yield LLM(self.cfg, "helper").chat(norm, max_tokens=512, temperature=0.3)
            except Exception:  # noqa: BLE001
                yield ""
            return
        if cmd in ("/yardim", "/yardım", "/help") or not cmd:
            yield HELP
            return
        if cmd == "/durum":
            yield self.status_text()
            return
        if cmd == "/iptal":
            yield self.agent.q.cancel()
            return
        stop = threading.Event()
        try:
            if cmd == "/izle":
                cur = self.agent.current
                if not cur:
                    yield "Şu an çalışan bir hedef yok. " + self.status_text()
                    return
                yield f"👁 İzleniyor: **{cur.text[:200]}**\n\n"
                yield from self.follow(cur.id, max(0, cur.events_from - 1), stop)
                return
            start_id = self.agent.ev.last_id()
            g = self.agent.submit(text)
            pos = self.agent.q.position(g.id)
            if pos > 0:
                yield f"⏳ Sıraya aldım ({pos} hedef önde). Başlayınca burada canlı akacak; `/durum` ile bakabilirsin.\n\n"
            yield from self.follow(g.id, start_id, stop)
        finally:
            stop.set()


def make_handler(gw: Gateway):
    cfg = gw.cfg
    key = cfg["agent"].get("api_key") or ""
    model = cfg["agent"].get("model_name", "k2agent")

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _authorized(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0]
            if host not in ("127.0.0.1", "localhost", "::1", "[::1]"):
                return False  # DNS-rebinding guard
            if key and self.headers.get("Authorization", "") != f"Bearer {key}":
                return False
            return True

        def _json(self, code: int, obj: Any) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if not self._authorized():
                return self._json(401, {"error": "unauthorized"})
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._json(200, {"object": "list", "data": [
                    {"id": model, "object": "model", "created": 0, "owned_by": "k2loop"}]})
            if self.path.startswith("/status"):
                return self._json(200, {"current": gw.agent.current.text if gw.agent.current else None,
                                        "goals": [g.__dict__ for g in gw.agent.q.snapshot()[-20:]]})
            return self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if not self._authorized():
                return self._json(401, {"error": "unauthorized"})
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._json(415, {"error": "application/json required"})
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._json(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad json"})
            cid = "chatcmpl-" + uuid.uuid4().hex[:12]
            chunks = gw.handle_chat(body)
            if not body.get("stream"):
                text = "".join(chunks)
                return self._json(200, {"id": cid, "object": "chat.completion", "created": int(time.time()),
                                        "model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                                                     "message": {"role": "assistant", "content": text}}]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def send(delta: dict, finish: str | None = None) -> None:
                obj = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()

            try:
                send({"role": "assistant", "content": ""})
                for ch in chunks:
                    send({"content": ch})
                send({}, "stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser closed: the goal keeps running in the loop
            finally:
                chunks.close()
            self.close_connection = True

    return H


def serve(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    agent = Agent(cfg)
    threading.Thread(target=agent.run_forever, name="k2agent-loop", daemon=True).start()
    a = cfg["agent"]
    srv = ThreadingHTTPServer((a["host"], int(a["port"])), make_handler(Gateway(agent)))
    srv.daemon_threads = True
    print(f"k2agent: http://{a['host']}:{a['port']}/v1  (model: {a.get('model_name', 'k2agent')})", flush=True)
    try:
        srv.serve_forever()
    finally:
        agent.stopped.set()
