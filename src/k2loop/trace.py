"""OpenTelemetry spans for Phoenix (http://127.0.0.1:6006): the ToT run becomes a browsable tree.

Uses OpenInference attribute names so Phoenix shows LLM / tool / chain spans properly.
Everything is a no-op when the opentelemetry packages or Phoenix are missing.
"""
from __future__ import annotations

import contextlib
import json
from typing import Any, Iterator

_tracer = None
_ready = False


def setup(cfg: dict) -> None:
    global _tracer, _ready
    if _ready:
        return
    _ready = True
    t = cfg.get("trace") or {}
    if not t.get("enabled", True):
        return
    try:
        from opentelemetry import trace as ot
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return
    provider = TracerProvider(resource=Resource.create({
        "service.name": "k2loop", "openinference.project.name": t.get("project", "k2loop")}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=t["endpoint"], timeout=5)))
    ot.set_tracer_provider(provider)
    import atexit
    atexit.register(provider.shutdown)  # short runs (`k2loop solve`) must flush their spans
    _tracer = ot.get_tracer("k2loop")


def _val(v: Any) -> Any:
    if isinstance(v, (str, bool, int, float)):
        return v if not isinstance(v, str) else v[:20_000]
    return json.dumps(v, ensure_ascii=False, default=str)[:20_000]


@contextlib.contextmanager
def span(name: str, kind: str = "CHAIN", parent: Any = None, **attrs: Any) -> Iterator[Any]:
    """kind: CHAIN | LLM | TOOL | AGENT | EVALUATOR | RETRIEVER. Yields the span (or None)."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name, context=parent) as sp:
        sp.set_attribute("openinference.span.kind", kind)
        for k, v in attrs.items():
            if v is not None:
                sp.set_attribute(k.replace("__", "."), _val(v))
        yield sp


def set_attrs(sp: Any, **attrs: Any) -> None:
    if sp is None:
        return
    for k, v in attrs.items():
        if v is not None:
            sp.set_attribute(k.replace("__", "."), _val(v))


def current_context() -> Any:
    """Capture the active context to parent spans created in another thread."""
    if _tracer is None:
        return None
    from opentelemetry import context
    return context.get_current()
