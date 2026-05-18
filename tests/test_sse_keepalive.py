"""
SSE keepalive frames during idle worker windows.

Context: ``sse_pump.pump_worker_into_sse`` buffers all text events when
called with ``dedup_text=True`` and only emits the dedup'd text frame
right before the terminating ``done`` event. On slow models (qwen3:14b
on a degraded server) the gap between the last visible event and the
final flush can exceed the client read-timeout — the client disconnects
mid-generation and the agent thread leaks until the join-timeout fires.

The fix: when the worker pushes no event into the queue for
``keepalive_idle_sec`` seconds, yield a SSE comment frame
(``": keepalive\\n\\n"``). Comment frames are ignored by SSE clients
but keep the underlying TCP connection (and intermediate HTTP proxies)
from timing out.

These tests use a scaled-down ``keepalive_idle_sec`` (50ms) so the
suite runs in <1s. Production default is 10s.
"""

from __future__ import annotations

import asyncio

import sse_pump


def _drain(gen) -> list[str]:
    """Synchronously drain `gen` until StopAsyncIteration, return frames."""
    frames: list[str] = []

    async def run() -> None:
        try:
            while True:
                frames.append(await gen.__anext__())
        except StopAsyncIteration:
            pass

    asyncio.run(run())
    return frames


def _keepalive_count(frames: list[str]) -> int:
    return sum(1 for f in frames if f.startswith(": keepalive"))


def test_no_idle_no_keepalive_with_dedup_off():
    """Regression guard: fast worker, dedup off → no keepalive frames."""

    def fast_worker(q, stop_event):
        for i in range(5):
            q.put({"t": "tick", "i": i})
        q.put(None)

    gen = sse_pump.pump_worker_into_sse(
        fast_worker,
        dedup_text=False,
        keepalive_idle_sec=0.05,
    )
    frames = _drain(gen)

    # 5 tick frames, no keepalives.
    data_frames = [f for f in frames if f.startswith("data: ")]
    assert len(data_frames) == 5, f"expected 5 data frames; got {frames!r}"
    assert _keepalive_count(frames) == 0, (
        f"unexpected keepalive frames on fast path: {frames!r}"
    )


def test_long_idle_emits_keepalive_frames_with_dedup_on():
    """Worker stalls during the dedup buffer window → keepalives keep the wire warm."""
    keepalive_sec = 0.05
    # Sleep ≥ 5 × keepalive_sec so we expect ≥ 2 keepalives even after the
    # ~15ms timer-resolution jitter on Windows asyncio. Production analog
    # is the 25s sleep producing keepalives at t=10s and t=20s; here we
    # scale that to ~250ms and 5×idle for headroom.
    idle_sec = keepalive_sec * 5

    def stalling_worker(q, stop_event):
        # Push one buffered text event so dedup mode actually buffers
        # *something* (this is the production scenario — text streams in,
        # gets buffered, then the worker stalls waiting on the next LLM
        # turn). With dedup_text=True the text frame is NOT yielded to
        # the client; the connection goes idle from the consumer's POV.
        q.put({"t": "text", "d": "starting…"})
        # Now stall. Use stop_event.wait so the test can still abort
        # cleanly if the generator is closed early.
        stop_event.wait(timeout=idle_sec)
        q.put({"t": "done"})
        q.put(None)

    gen = sse_pump.pump_worker_into_sse(
        stalling_worker,
        dedup_text=True,
        keepalive_idle_sec=keepalive_sec,
    )
    frames = _drain(gen)

    keepalives = _keepalive_count(frames)
    assert keepalives >= 2, (
        f"expected ≥2 keepalive frames during idle window; got {keepalives} "
        f"in {frames!r}"
    )
    # Every keepalive must be exactly the SSE comment frame.
    for f in frames:
        if f.startswith(":"):
            assert f == ": keepalive\n\n", f"malformed keepalive frame: {f!r}"


def test_steady_text_emits_no_keepalive_with_dedup_on():
    """Worker pushes text steadily inside the keepalive window → no keepalives.

    Even though dedup_text=True buffers the text frames, queue activity
    resets the idle timer. The connection only goes ``idle`` (from the
    pump's POV) when the worker stops pushing events at all.
    """
    keepalive_sec = 0.10
    tick_interval = keepalive_sec / 4  # well below the keepalive threshold

    def steady_worker(q, stop_event):
        for i in range(8):
            q.put({"t": "text", "d": f"chunk{i} "})
            if stop_event.wait(timeout=tick_interval):
                return
        q.put({"t": "done"})
        q.put(None)

    gen = sse_pump.pump_worker_into_sse(
        steady_worker,
        dedup_text=True,
        keepalive_idle_sec=keepalive_sec,
    )
    frames = _drain(gen)

    assert _keepalive_count(frames) == 0, (
        f"unexpected keepalive frames while worker was actively producing "
        f"events every {tick_interval}s: {frames!r}"
    )
