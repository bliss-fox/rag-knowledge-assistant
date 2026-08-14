"""Trace context for observability across pipeline stages.

Provides trace_id, trace_type (query/ingestion), per-stage timing,
finish() lifecycle, and to_dict() serialisation for JSON Lines output.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


@dataclass
class TraceContext:
    """Request-scoped trace context that records pipeline stages and timing.

    Attributes:
        trace_id: Unique identifier for this trace.
        trace_type: Either ``"query"`` or ``"ingestion"``.
        started_at: ISO-8601 timestamp when the trace was created.
        finished_at: ISO-8601 timestamp when ``finish()`` was called, or None.
        stages: Ordered list of recorded stage dicts.
        metadata: Arbitrary key/value pairs attached to the trace.
    """

    trace_type: Literal["query", "ingestion"] = "query"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = field(default=None)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Nanosecond performance counter avoids zero-duration traces on Windows
    # hosts whose floating-point monotonic clock has coarse effective
    # resolution for very short operations.
    _start_perf_ns: int = field(default_factory=time.perf_counter_ns, repr=False)
    _finish_perf_ns: Optional[int] = field(default=None, repr=False)
    _stage_timings: Dict[str, float] = field(default_factory=dict, repr=False)

    # ---- recording ---------------------------------------------------

    def record_stage(
        self,
        stage_name: str,
        data: Dict[str, Any],
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """Record data from a pipeline stage.

        Args:
            stage_name: Name of the stage (e.g. ``"dense_retrieval"``).
            data: Stage-specific payload (method, provider, details …).
            elapsed_ms: Pre-computed elapsed time in ms.  If *None* the
                caller should measure externally, or leave it to the
                ``stage_timer`` context-manager.
        """
        entry: Dict[str, Any] = {
            "stage": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = float(elapsed_ms)
            self._stage_timings[stage_name] = elapsed_ms
        self.stages.append(entry)

    # ---- lifecycle ----------------------------------------------------

    def finish(self) -> None:
        """Mark the trace as finished and record wall-clock end time."""
        self._finish_perf_ns = time.perf_counter_ns()
        self.finished_at = datetime.now(timezone.utc).isoformat()

    # ---- timing helpers -----------------------------------------------

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """Return elapsed time in milliseconds.

        Args:
            stage_name: If given, return the elapsed time recorded for
                that stage.  If *None*, return the total trace elapsed
                time (start → finish, or start → now if not yet
                finished).

        Returns:
            Elapsed milliseconds.

        Raises:
            KeyError: If *stage_name* was provided but not found.
        """
        if stage_name is not None:
            if stage_name not in self._stage_timings:
                raise KeyError(f"Stage '{stage_name}' has no recorded timing")
            return self._stage_timings[stage_name]

        end_ns = self._finish_perf_ns if self._finish_perf_ns is not None else time.perf_counter_ns()
        return max(end_ns - self._start_perf_ns, 1) / 1_000_000

    # ---- serialisation ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the trace to a plain dict suitable for ``json.dumps``.

        Returns:
            Dictionary with all trace data.
        """
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            # Preserve sub-centisecond precision.  Rounding here previously
            # converted valid short traces to 0.0 before SQLite aggregation.
            "total_elapsed_ms": self.elapsed_ms(),
            "stages": list(self.stages),
            "metadata": dict(self.metadata),
        }

    # ---- backwards-compat helper used in C5 / C6 -----------------------

    def get_stage_data(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve recorded data for a specific stage.

        Searches stages list (last-write-wins for duplicate names).

        Args:
            stage_name: Name of the stage to retrieve.

        Returns:
            The ``data`` dict of the matching stage, or *None*.
        """
        for entry in reversed(self.stages):
            if entry.get("stage") == stage_name:
                return entry.get("data")
        return None
