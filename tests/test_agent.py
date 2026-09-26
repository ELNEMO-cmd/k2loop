import copy
import json
import subprocess
import threading
import time

from k2loop.agent import Agent, GoalQueue
from k2loop.config import DEFAULTS
from k2loop.events import EventLog, set_context
from k2loop.gateway import Formatter, Gateway, is_webui_task


def _cfg(tmp_path):
    c = copy.deepcopy(DEFAULTS)
    c["paths"]["data_dir"] = str(tmp_path)
    c["trace"]["enabled"] = False
    return c


def test_event_log_ids_history_and_follow(tmp_path):
    ev = EventLog(tmp_path / "agent")
    a = ev.emit("log", text="one")
    set_context(goal="g-1")
    b = ev.emit("log", text="two")
    set_context()
    assert (a["id"], b["id"]) == (1, 2) and b["goal"] == "g-1" and "goal" not in a
    assert ev.last_id() == 2
    assert [e["text"] for e in ev.history(after=1)] == ["two"]
    stop = threading.Event()
    got = []

    def reader():
        for e in ev.follow(1, stop):
            if e:
                got.append(e["text"])
                if len(got) == 2:
                    stop.set()

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.2)
    ev.emit("log", text="ünicode üç")
    t.join(5)
    stop.set()
    assert got == ["two", "ünicode üç"]


def test_goal_queue_persist_and_cancel(tmp_path):
    q = GoalQueue(tmp_path / "goals.json")
    g1, g2 = q.add("birinci"), q.add("ikinci")
    assert q.position(g2.id) == 1
    assert q.next(0.1).id == g1.id
    assert "durduruluyor" in q.cancel() and q.cancel_requested
    assert "iptal" in q.cancel(g2.id)
    q.finish(g1, "cancelled", "x")
    q2 = GoalQueue(tmp_path / "goals.json")  # restart
    assert {g.id: g.status for g in q2.snapshot()} == {g1.id: "cancelled", g2.id: "cancelled"}
    g3 = q2.add("üçüncü")
    q2.next(0.1)
    assert GoalQueue(tmp_path / "goals.json").get(g3.id).status == "queued"  # running goal re-queued after restart


def test_intent_falls_back_to_heuristics(tmp_path, monkeypatch):
    repo = tmp_path / "proj"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    class Broken:
        def __init__(self, *a, **k):
            pass

        def ask_json(self, *a, **k):
            raise RuntimeError("model down")

    monkeypatch.setattr("k2loop.agent.LLM", Broken)
    ag = Agent(_cfg(tmp_path))
    it = ag.intent(f"{repo} içine login ekle")
    assert it["kind"] == "code" and it["project"] == str(repo)
    assert ag.intent("en iyi yerel arama motoru hangisi?")["kind"] == "research"
    assert ag.intent("yaz için bir antrenman planı yap")["kind"] == "plan"


def test_plan_goal_tot_debate_judge(tmp_path, monkeypatch):
    from k2loop.agent import Goal
    ag = Agent(_cfg(tmp_path))
    ag.cfg["agent"]["research_first"] = False
    ag.cfg["plan"]["branch_model"] = "llm"  # the test checks that branches and judge can use different models
    calls = []

    def fake_call(profile, prompt, **kw):
        calls.append((profile, prompt))
        if "Several candidate plans" in prompt:
            return '{"scores": [4, 9, 6], "best_index": 1, "why": "en gerçekçi", "borrow": ["haftalık kontrol"]}'
        if "FINAL plan" in prompt:
            assert "haftalık kontrol" in prompt
            return "## Özet\nson plan"
        if "debate round" in prompt:
            return "<<<MESAJ>>>\nsenin planında bütçe yok\n<<<PLAN>>>\n# revize\nadımlar"
        return "# plan\n1. adım"

    monkeypatch.setattr(ag, "_model_call", fake_call)
    g = Goal(id="g-t", text="3 ayda Almanca B1 planı", created=0)
    status, out = ag.plan_goal(g, {"topic": "German B1 in 3 months"})
    assert status == "done" and "son plan" in out
    kinds = [c[0] for c in calls]
    assert kinds == ["llm"] * 3 + ["llm"] * 3 + ["strong", "strong"]  # 3 branches, 3 debaters, judge, refine
    debate_prompts = [p for _, p in calls if "debate round" in p]
    assert "senin planında bütçe yok" in debate_prompts[2]  # later speakers hear earlier ones
    evs = [e for e in ag.ev.history() if e["kind"] in ("debate", "judge")]
    assert [e["kind"] for e in evs] == ["debate"] * 3 + ["judge"] and evs[-1]["best"] == "Kapsamlı"
    saved = list((tmp_path / "agent" / "plans").glob("*.md"))[0].read_text(encoding="utf-8")
    assert "## Tartışma" in saved and "bütçe yok" in saved


def test_turkish_slug():
    from k2loop.planner import slug
    assert slug("3 ayda Almanca B1 için çalışma planı") == "3-ayda-almanca-b1-icin-calisma-plani"
    assert slug("İŞÇİ", "x") == "isci" and slug("???", "goal") == "goal"


def test_formatter_renders_tot_events():
    f = Formatter()
    out = "".join(f(e) for e in [
        {"kind": "goal", "state": "started", "text": "hedef"},
        {"kind": "intent", "goal_kind": "code", "project": "/p"},
        {"kind": "step", "state": "started", "i": 1, "n": 2, "title": "add()"},
        {"kind": "cand", "name": "gen1", "passed": False, "failures": 2, "lines": 10, "files": ["a.py"],
         "secs": 3.2, "check_tail": "AssertionError"},
        {"kind": "hypotheses", "level": 1, "items": [{"text": "fix import", "score": 8, "kept": True},
                                                      {"text": "rewrite", "score": 2, "kept": False}]},
        {"kind": "decision", "name": "retry_escalate", "used": "retry_local"},
        {"kind": "log", "text": "[k2loop] gen1: internals"},
        {"kind": "debate", "round": 1, "speaker": "Pragmatik", "text": "bütçe eksik"},
        {"kind": "judge", "names": ["A", "B"], "scores": [3, 8], "best": "B", "why": "gerçekçi"},
        {"kind": "final", "text": "bitti"},
    ])
    assert "🗣️ **Pragmatik**" in out and "B: **8**" in out and "seçilen **B**" in out
    assert "Başladım" in out and "kod değişikliği" in out and "Adım 1/2" in out
    assert "❌ **gen1**" in out and "AssertionError" in out and "✔ denenecek" in out and "✖ budandı" in out
    assert "sonraki adım" in out and "internals" not in out and "bitti" in out


def test_gateway_commands_and_webui_tasks(tmp_path, monkeypatch):
    assert is_webui_task("### Task:\nGenerate a concise title\n<chat_history>...")
    assert not is_webui_task("projeme test ekle")

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

        def chat(self, msgs, **kw):
            return json.dumps({"title": "Başlık"}, ensure_ascii=False)

    monkeypatch.setattr("k2loop.gateway.LLM", FakeLLM)
    gw = Gateway(Agent(_cfg(tmp_path)))
    title = "".join(gw.handle_chat({"messages": [{"role": "user", "content": "### Task:\nGenerate a title"}]}))
    assert "Başlık" in title
    assert gw.agent.q.snapshot() == []  # background jobs never become goals
    assert "Boşta" in "".join(gw.handle_chat({"messages": [{"role": "user", "content": "/durum"}]}))
    assert "k2agent komutları" in "".join(gw.handle_chat({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "/yardim"}]}]}))


def test_goal_streams_until_final(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    ag = Agent(cfg)

    def fake_handle(g):
        set_context(goal=g.id)
        ag.emit("goal", state="started", text=g.text)
        ag.emit("stage", name="research", msg="araştırılıyor")
        ag.emit("final", text="rapor hazır", status="done")
        ag.emit("goal", state="done", text=g.text)
        ag.q.finish(g, "done", "rapor hazır")

    monkeypatch.setattr(ag, "handle", fake_handle)
    t = threading.Thread(target=ag.run_forever, daemon=True)
    t.start()
    text = "".join(Gateway(ag).handle_chat({"messages": [{"role": "user", "content": "yerel LLM araştır"}]}))
    ag.stopped.set()
    assert "araştırılıyor" in text and "rapor hazır" in text
