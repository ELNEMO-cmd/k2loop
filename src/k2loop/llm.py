"""OpenAI-compatible client for the local K2 Horizon server (llama.cpp or vLLM)."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from .config import effective_max_tokens

_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
_IFM_END = re.compile(r"</ifm\|[^>]*>")   # K2 Horizon's own reasoning markers, e.g. </ifm|think_faster>
_IFM_TAG = re.compile(r"<\/?ifm\|[^>]*>")


def strip_reasoning(text: str | None) -> str:
    """Remove inline reasoning when the server did not split it into reasoning_content."""
    if not text:
        return ""
    t = _THINK_RE.sub("", text)
    for tag in ("</think>", "</thinking>", "</reasoning>"):
        if tag in t:
            t = t.rsplit(tag, 1)[1]
    ends = list(_IFM_END.finditer(t))
    if ends:
        t = t[ends[-1].end():]
    return _IFM_TAG.sub("", t).strip()


def _first_json(s: str) -> Any:
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(s[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def extract_json(text: str | None) -> Any:
    """Best-effort JSON extraction from a model answer (fenced or bare)."""
    t = strip_reasoning(text)
    for cand in re.findall(r"```(?:json)?\s*(.*?)```", t, re.S) + [t]:
        obj = _first_json(cand)
        if obj is not None:
            return obj
    return None


class LLM:
    def __init__(self, cfg: dict[str, Any], profile: str = "llm"):
        self.cfg = cfg
        self.profile = profile
        c = cfg[profile]
        self.base = c["base_url"].rstrip("/")
        self.model = c["model"]
        self.key = c.get("api_key") or "none"

    def _post(self, path: str, body: dict, timeout: float) -> dict:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def chat(self, messages: list[dict], effort: str = "high", max_tokens: int | None = None,
             temperature: float | None = None, timeout: float | None = None) -> str:
        from .trace import set_attrs, span
        c = self.cfg[self.profile]
        ctk = {"reasoning_effort": effort} if self.profile == "llm" else dict(c.get("template_kwargs") or {})
        body = {
            "model": self.model, "messages": messages,
            "temperature": c["temperature"] if temperature is None else temperature,
            "top_p": c["top_p"], "max_tokens": max_tokens or effective_max_tokens(self.cfg, self.profile),
        }
        if ctk:
            body["chat_template_kwargs"] = ctk
        last: Exception | None = None
        with span(f"llm:{self.model}", "LLM", llm__model_name=self.model, llm__invocation_parameters=
                  {"effort": effort, "max_tokens": body["max_tokens"], "temperature": body["temperature"]},
                  input__value=messages[-1].get("content", "")) as sp:
            for attempt in range(3):
                try:
                    data = self._post("/chat/completions", body, timeout or float(c["request_timeout_s"]))
                    msg = data["choices"][0]["message"]
                    out = strip_reasoning(msg.get("content") or "")
                    usage = data.get("usage") or {}
                    set_attrs(sp, output__value=out, llm__token_count__prompt=usage.get("prompt_tokens"),
                              llm__token_count__completion=usage.get("completion_tokens"))
                    return out
                except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, json.JSONDecodeError) as exc:
                    last = exc
                    time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed ({self.model}): {last}")

    def ask_json(self, prompt: str, effort: str, system: str | None = None, **kw: Any) -> Any:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return extract_json(self.chat(msgs, effort=effort, **kw))

    def health(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(self.base + "/models", timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return True, f"models={[m.get('id') for m in data.get('data', [])]}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
