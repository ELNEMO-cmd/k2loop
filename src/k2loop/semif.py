"""SemIf-style decisions: read option probabilities from ONE forward pass of a small model, no text generation.

Technique from SemIf (https://github.com/theoleecj/semif, MIT): state + question + typed options go into the
prompt, the options are labelled A, B, C..., and the probabilities of those label tokens at the first answer
position are the decision scores. SemIf itself is a batch CLI that loads the model per call; here the same idea
runs on the helper model that llama-swap keeps loaded on the CPU (llama.cpp `logprobs`), so a decision costs one
short request. It is a second shadow predictor next to Laya; `k2loop laya-eval` compares rule / Laya / SemIf.
"""
from __future__ import annotations

import json
import math
import urllib.request
from typing import Any

LETTERS = "ABCDEFGHIJ"

SYSTEM = ("You are a decision function. Read the state, then answer the question by choosing the single best "
          "option. Reply with the option letter only.")


def build_messages(state: Any, question: str, options: dict[str, str]) -> tuple[list[dict], list[str]]:
    ids = list(options)[: len(LETTERS)]
    st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    lines = "\n".join(f"{LETTERS[i]}) {oid}: {options[oid]}" for i, oid in enumerate(ids))
    user = f"STATE:\n{st}\n\nQUESTION: {question}\n\nOPTIONS:\n{lines}\n\nAnswer with one letter."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], ids


def option_probs(top_logprobs: list[dict], ids: list[str]) -> dict[str, float]:
    """Map first-token top logprobs onto the option letters and renormalise over the options."""
    mass = {oid: 0.0 for oid in ids}
    for t in top_logprobs or []:
        tok = str(t.get("token", "")).strip().rstrip(")").strip()
        if len(tok) == 1 and tok.upper() in LETTERS[: len(ids)]:
            mass[ids[LETTERS.index(tok.upper())]] += math.exp(float(t.get("logprob", -100)))
    total = sum(mass.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in mass.items()}


class SemIfScorer:
    def __init__(self, cfg: dict, timeout: float = 60):
        c = cfg["semif"]
        prof = cfg[c.get("model_profile", "helper")]
        self.url = prof["base_url"].rstrip("/") + "/chat/completions"
        self.model = prof["model"]
        self.key = prof.get("api_key") or "none"
        self.timeout = float(c.get("timeout_s", timeout))

    def predict(self, state: Any, question: str, options: dict[str, str]) -> tuple[str | None, float | None, dict]:
        msgs, ids = build_messages(state, question, options)
        body = {"model": self.model, "messages": msgs, "max_tokens": 1, "temperature": 0,
                "logprobs": True, "top_logprobs": 20}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8", errors="ignore"))
            content = (data["choices"][0].get("logprobs") or {}).get("content") or []
            probs = option_probs(content[0].get("top_logprobs", []) if content else [], ids)
        except Exception:  # noqa: BLE001  (shadow predictor: never break the run)
            return None, None, {}
        if not probs:
            return None, None, {}
        best = max(probs, key=probs.get)
        return best, round(probs[best], 4), {k: round(v, 4) for k, v in probs.items()}
