import psutil
import os
import sys
from typing import List, Dict, Any
from .base_sampler import BaseSampler


class ProcessSampler(BaseSampler):
    """
    Sampler that tracks CPU and RAM usage of the current Python process
    (or a specified PID) over time using psutil.

    Useful for monitoring your training script and TraceML process itself.
    """

    def __init__(self, pid: int = None):
        super().__init__()

        # Monitor current process by default
        self.process = psutil.Process(pid or os.getpid())

        # Sampling history
        self.cpu_samples: List[float] = []
        self.ram_samples_mb: List[float] = []

        # Latest snapshot
        self._latest_snapshot: Dict[str, Any] = {
            "process_cpu_percent": 0.0,
            "process_ram_mb": 0.0,
        }

        # Prime CPU usage measurement
        try:
            self.process.cpu_percent(interval=None)
        except Exception as e:
            print(
                f"[TraceML] WARNING: process.cpu_percent() initial call failed: {e}",
                file=sys.stderr,
            )

    def sample(self) -> Dict[str, Any]:
        """
        Sample current CPU and RAM usage of the monitored process.
        Returns:
            Dict[str, Any]: Snapshot of CPU % and RAM MB used.
        """
        try:
            cpu_usage = self.process.cpu_percent(interval=None)
            ram_usage_mb = self.process.memory_info().rss / (1024**2)

            current_sample = {
                "process_cpu_percent": round(cpu_usage, 2),
                "process_ram_mb": round(ram_usage_mb, 2),
            }

            self.cpu_samples.append(cpu_usage)
            self.ram_samples_mb.append(ram_usage_mb)

            self._latest_snapshot = current_sample
            return current_sample

        except Exception as e:
            print(f"[TraceML] Process sampling error: {e}", file=sys.stderr)
            error_snapshot = {
                "process_cpu_percent": 0.0,
                "process_ram_mb": 0.0,
                "error": str(e),
            }
            self._latest_snapshot = error_snapshot
            return error_snapshot

    def get_live_snapshot(self) -> Dict[str, Any]:
        """
        Return the most recent CPU and RAM usage snapshot.
        """
        return self._latest_snapshot

    def get_summary(self) -> Dict[str, Any]:
        """
        Return average and peak CPU/RAM usage for the monitored process.
        """
        summary = {}
        try:
            if self.cpu_samples:
                summary["cpu_average_percent"] = round(
                    sum(self.cpu_samples) / len(self.cpu_samples), 2
                )
                summary["cpu_peak_percent"] = round(max(self.cpu_samples), 2)
            else:
                summary["cpu_average_percent"] = 0.0
                summary["cpu_peak_percent"] = 0.0

            if self.ram_samples_mb:
                summary["ram_average_mb"] = round(
                    sum(self.ram_samples_mb) / len(self.ram_samples_mb), 2
                )
                summary["ram_peak_mb"] = round(max(self.ram_samples_mb), 2)
            else:
                summary["ram_average_mb"] = 0.0
                summary["ram_peak_mb"] = 0.0

            summary["total_process_samples"] = len(self.cpu_samples)

        except Exception as e:
            print(f"[TraceML] Process summary error: {e}", file=sys.stderr)
            return {"error": str(e), "total_process_samples": 0}

        return summary
