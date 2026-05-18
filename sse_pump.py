"""Daemon-thread + queue → SSE-frame helper.

Extracted from app_free._agent_stream so the orchestration can be
tested in isolation (no Neo4j, no LLM clients, no FastAPI required).

The helper runs a worker callable in a daemon thread, drains events
from a queue, and yields each as an SSE-formatted line. When the
consumer (e.g. FastAPI's StreamingResponse) disconnects, the
finally block signals the worker via a threading.Event and joins
with a short timeout. Workers with long-running steps (LLM calls,
large Cypher) should poll the event between steps to cooperate
with the teardown.

When called with ``dedup_text=True`` the pump also buffers all
``{"t": "text", "d": ...}`` events for the lifetime of the stream,
runs ``bullet_dedup.BulletDedup`` over the concatenated text right
before forwarding the terminating ``{"t": "done", ...}`` event, and
emits a single cleaned text frame in their place. This is the
verbatim-bullet suppression introduced after the 2026-05-17 baseline
measured a 31% verse-explanation duplicate rate on qwen3:14b.

The buffering trade-off: with ``dedup_text=True`` users no longer
see prose stream in as it arrives — they see it once at the end of
the agent loop. Tool/graph_update/verification events still pass
through immediately. The agent_stream caller opts in; legacy callers
(see ``tests/regression/test_sse_worker_leak.py``) keep the
forward-immediately default.
"""

from __future__ import annotations

import asyncio
import json
import queue as tqueue
import threading
import time
from collections.abc import AsyncIterator
from typing import Callable

from bullet_dedup import BulletDedup


async def pump_worker_into_sse(
    run_fn: Callable[[tqueue.SimpleQueue, threading.Event], None],
    *,
    join_timeout: float = 1.0,
    dedup_text: bool = False,
    keepalive_idle_sec: float = 10.0,
) -> AsyncIterator[str]:
    """Run `run_fn(queue, stop_event)` in a daemon thread; yield SSE frames.

    Encapsulates the daemon-thread + queue orchestration that powers
    streaming endpoints. The worker pushes dict events into the queue
    and a final None sentinel when done; this helper drains the queue
    and yields each event as an SSE-formatted line.

    On consumer disconnect the finally block sets the stop_event and
    joins the worker thread with a short timeout. Workers with long-
    running steps MUST poll stop_event between steps for the teardown
    to take effect; the daemon flag ensures the process can still exit
    even if a step blocks past the join timeout.

    With ``dedup_text=True`` text events are buffered until the
    ``done`` event arrives (or the stream ends), at which point
    ``bullet_dedup.BulletDedup`` strips verbatim verse-explanation
    duplicates and a single cleaned text frame is emitted before the
    ``done`` event is forwarded. A short summary line is printed to
    stdout for ops visibility; nothing about the suppression goes out
    over SSE.

    When the worker goes ``keepalive_idle_sec`` seconds without
    pushing any event into the queue, a SSE comment frame
    (``": keepalive\\n\\n"``) is yielded to keep the underlying TCP
    connection (and any HTTP proxies) from timing out during long
    LLM turns or the dedup buffering window. SSE clients ignore
    comment frames, so this is transparent to callers. The idle
    timer resets on any queue activity (including buffered text
    events) and after each keepalive is emitted.
    """
    q: tqueue.SimpleQueue = tqueue.SimpleQueue()
    stop_event = threading.Event()

    def target() -> None:
        try:
            run_fn(q, stop_event)
        finally:
            # Belt-and-braces sentinel — guarantees the consumer's
            # `if event is None: break` fires even if run_fn raised.
            q.put(None)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    dedup: BulletDedup | None = BulletDedup() if dedup_text else None
    text_buffer: list[str] = []
    text_flushed = False

    def _flush_text_frame() -> str | None:
        """Return a single SSE frame holding the dedup'd buffered text, or None."""
        nonlocal text_flushed
        if not dedup_text or text_flushed:
            return None
        text_flushed = True
        if not text_buffer:
            return None
        full = "".join(text_buffer)
        assert dedup is not None  # mypy; guarded by dedup_text
        cleaned, suppressed = dedup.filter_text(full)
        if suppressed:
            print(
                f"  [dedup] suppressed {len(suppressed)} verbatim bullets "
                f"in response (verses: {', '.join(suppressed)})"
            )
        return f"data: {json.dumps({'t': 'text', 'd': cleaned}, ensure_ascii=False)}\n\n"

    last_activity = time.monotonic()
    try:
        while True:
            try:
                event = q.get_nowait()
            except tqueue.Empty:
                if time.monotonic() - last_activity >= keepalive_idle_sec:
                    yield ": keepalive\n\n"
                    last_activity = time.monotonic()
                await asyncio.sleep(0.05)
                continue
            last_activity = time.monotonic()
            if event is None:
                # End of stream. If we were buffering text and never saw
                # a `done` event, flush the buffered text now so callers
                # that omit `done` (and tests that don't model it) still
                # see the prose.
                flushed = _flush_text_frame()
                if flushed is not None:
                    yield flushed
                break

            if dedup_text and isinstance(event, dict) and event.get("t") == "text":
                text_buffer.append(event.get("d", "") or "")
                continue

            if dedup_text and isinstance(event, dict) and event.get("t") == "done":
                flushed = _flush_text_frame()
                if flushed is not None:
                    yield flushed
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                continue

            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    finally:
        stop_event.set()
        thread.join(timeout=join_timeout)
