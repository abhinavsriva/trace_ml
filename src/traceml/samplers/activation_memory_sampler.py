import time
from collections import defaultdict, deque
from dataclasses import dataclass
from queue import Empty
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch

from .base_sampler import BaseSampler
from traceml.utils.patch import get_activation_queue


@dataclass
class _BatchStats:
    count: int
    sum_mb: float
    avg_mb: float
    max_mb: float
    min_nonzero_mb: Optional[float]


class ActivationMemorySampler(BaseSampler):
    """
    Drain-all activation-event sampler.

    Each call to `sample()`:
      - Drains the activation queue.
      - Aggregates per-device stats (avg/sum/max/min_nonzero) over those new events.
      - Returns a live snapshot dict.
      - If no new events arrive, returns the last snapshot.
      - If no event has ever arrived, returns a guidance note (hooks likely not attached).

    Also keeps:
      - A bounded, optional raw-event buffer (for logging).
      - Lightweight cumulative per-device stats across all drains since start.
    """

    def __init__(
        self,
        max_raw_events: int = 10_000,
        pressure_threshold: float = 0.9,
        store_raw: bool = True,
    ):
        super().__init__()
        self.pressure_threshold = float(pressure_threshold)
        self.store_raw = bool(store_raw)
        # raw event(each item: {"ts": float, "per_device_mb": {dev: mb}})
        self._raw_events: Deque[Dict[str, Any]] = deque(maxlen=int(max_raw_events))
        # Cumulative stats: dev -> (count_samples, sum_mb, max_mb)
        self._cumulative: Dict[str, Tuple[int, float, float]] = defaultdict(lambda: (0, 0.0, 0.0))
        # Last live snapshot + flag to know if we ever saw any data
        self._latest_snapshot: Dict[str, Any] = {}
        self._ever_seen: bool = False

    def _append_raw_event(self, ts: float, per_dev_mb: Dict[str, float]) -> None:
        """Push one raw event into the bounded buffer (if enabled)."""
        if not self.store_raw:
            return
        self._raw_events.append({"ts": float(ts), "per_device_mb": dict(per_dev_mb)})

    def _accumulate_cumulative(self, per_dev_mb: Dict[str, float]) -> None:
        """Update cumulative counters for each device."""
        for dev, mb in per_dev_mb.items():
            c_count, c_sum, c_max = self._cumulative[dev]
            mb_f = float(mb)
            self._cumulative[dev] = (c_count + 1, c_sum + mb_f, max(c_max, mb_f))

    @staticmethod
    def _compute_batch_stats(values_mb: List[float]) -> _BatchStats:
        """Compute summary stats for a non-empty list of MB values."""
        if not values_mb:
            return _BatchStats(0, 0.0, 0.0, 0.0, None)
        s = float(sum(values_mb))
        mx = float(max(values_mb))
        nz = [v for v in values_mb if v > 0.0]
        mnz = float(min(nz)) if nz else None
        return _BatchStats(count=len(values_mb), sum_mb=s, avg_mb=s / len(values_mb), max_mb=mx, min_nonzero_mb=mnz)

    def _pressure_flag(self, dev: str, batch_max_mb: float) -> Optional[bool]:
        """Returns True if batch max exceeds threshold of device capacity; None if unknown/not CUDA."""
        if not dev.startswith("cuda"):
            return None
        try:
            idx = int(dev.split(":", 1)[1])
        except Exception:
            return None
        try:
            props = torch.cuda.get_device_properties(idx)
            total_mb = props.total_memory / (1024 ** 2)
            return (batch_max_mb / total_mb) >= self.pressure_threshold
        except Exception:
            return None

    def _drain_queue(self) -> Tuple[int, Dict[str, List[float]]]:
        """
        Drain the activation queue completely and return:
          - number of events drained
          - mapping dev -> list of MB values in THIS drain cycle
        """
        q = get_activation_queue()
        drained_events = 0
        batch_per_dev: Dict[str, List[float]] = defaultdict(list)
        now = time.time()

        while True:
            try:
                ev = q.get_nowait()
            except Empty:
                break

            drained_events += 1
            per_dev = getattr(ev, "per_device_activation_mb", None)
            ts = getattr(ev, "timestamp", now)

            if not isinstance(per_dev, dict):
                continue

            self._append_raw_event(ts, per_dev)
            self._accumulate_cumulative(per_dev)

            for dev, mb in per_dev.items():
                batch_per_dev[dev].append(float(mb))

        return drained_events, batch_per_dev

    def _build_snapshot(self, drained_events: int, batch_per_dev: Dict[str, List[float]]) -> Dict[str, Any]:
        """Construct the live snapshot dict from this drain’s per-device values."""
        devices_out: Dict[str, Any] = {}
        overall_avg = 0.0
        n_devs = 0

        for dev, vals in batch_per_dev.items():
            stats = self._compute_batch_stats(vals)
            pressure = self._pressure_flag(dev, stats.max_mb)

            devices_out[dev] = {
                "count": stats.count,
                "sum_mb": round(stats.sum_mb, 4),
                "avg_mb": round(stats.avg_mb, 4),
                "max_mb": round(stats.max_mb, 4),
                "min_nonzero_mb": round(stats.min_nonzero_mb, 4) if stats.min_nonzero_mb is not None else None,
                "pressure_90pct": pressure,
            }
            overall_avg += stats.avg_mb
            n_devs += 1

        return {
            "timestamp": time.time(),
            "devices": devices_out,
            "overall_avg_mb": round(overall_avg / n_devs, 4) if n_devs else 0.0,
            "drained_events": drained_events,
            "stale": False,
        }

    def sample(self) -> Dict[str, Any]:
        """
        Drain the queue and compute a live snapshot.
        If nothing new arrived:
          - If we’ve seen data before: return last snapshot with `stale=True`.
          - If we’ve never seen data: return a guidance note about attaching hooks.
        """
        drained_events, batch_per_dev = self._drain_queue()

        if drained_events == 0:
            if self._ever_seen and self._latest_snapshot:
                snap = dict(self._latest_snapshot)
                snap["stale"] = True
                snap["drained_events"] = 0
                return snap
            else:
                self._latest_snapshot = {
                    "timestamp": time.time(),
                    "devices": {},
                    "overall_avg_mb": 0.0,
                    "drained_events": 0,
                    "stale": True,
                    "note": (
                        "No activation events received yet. "
                        "Attach hooks with @trace_model(..., trace_activations=True) "
                        "or trace_model_instance(model, trace_activations=True)."
                    ),
                }
                return self._latest_snapshot

        self._latest_snapshot = self._build_snapshot(drained_events, batch_per_dev)
        self._ever_seen = True
        return self._latest_snapshot

    def get_summary(self) -> Dict[str, Any]:
        """
        Summarize all drained data so far using cumulative counters.
        Returns a dict
        """
        per_dev_summary: Dict[str, Any] = {}
        for dev, (c_count, c_sum, c_max) in self._cumulative.items():
            avg = (c_sum / c_count) if c_count else 0.0
            per_dev_summary[dev] = {
                "cumulative_count": c_count,
                "cumulative_sum_mb": round(c_sum, 4),
                "cumulative_avg_mb": round(avg, 4),
                "cumulative_max_mb": round(c_max, 4),
            }

        return {
            "ever_seen": self._ever_seen,
            "per_device_cumulative": per_dev_summary,
            "raw_events_kept": len(self._raw_events),
            "last_snapshot": self._latest_snapshot,
        }
