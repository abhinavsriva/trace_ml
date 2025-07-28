import psutil
import sys
from typing import List, Optional, Dict, Any
from .base_sampler import BaseSampler

from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetCount,
    NVMLError,
)

class SystemSampler(BaseSampler):
    """
    Sampler that tracks CPU RAM and GPU usage over time using psutil.
    Collects usage percentages periodically and exposes live snapshots
    and statistical summaries.

    This sampler is safe to use in long-running processes and will
    continue operating even if individual sampling steps fail.
    """

    def __init__(self):
        super().__init__()

        # Initialize psutil.cpu_percent for non-blocking calls in `sample()`
        # This first call to psutil.cpu_percent returns 0.0,
        # subsequent calls return the CPU usage since the last call.
        try:
            psutil.cpu_percent(interval=None)
        except Exception as e:
            print(
                f"[TraceML] WARNING: psutil.cpu_percent initial call failed: {e}",
                file=sys.stderr,
            )

        self.cpu_samples: List[float] = []
        self.ram_percent_samples: List[float] = []
        self.ram_used_samples: List[float] = []
        self.ram_available_samples: List[float] = []
        self.ram_total_samples: List[float] = []

        self.gpu_util_samples: List[float] = []
        self.gpu_mem_used_samples: List[float] = []
        self.gpu_mem_total_samples: List[float] = []

        # GPU setup
        self.gpu_available = False
        self.gpu_count = 0
        try:
            nvmlInit()
            self.gpu_count = nvmlDeviceGetCount()
            self.gpu_available = True
        except NVMLError as e:
            print(f"[TraceML] WARNING: GPU not available: {e}", file=sys.stderr)

        self._latest_snapshot: Dict[str, Any] = {}


    def sample(self) -> Dict[str, Any]:
        """
        Poll current CPU and RAM usage and return it as a dictionary.
        This method is non-blocking.

        Returns:
            Dict[str, Any]: Includes "error" key if sampling fails.
        """
        try:
            # CPU usage (non-blocking after initial priming)
            cpu_usage = psutil.cpu_percent(interval=None)

            # RAM usage
            mem = psutil.virtual_memory()
            ram_percent_used = mem.percent
            ram_used = mem.used / (1024**2)
            ram_available = mem.available / (1024**2)
            ram_total = mem.total / (1024**2)

            # Store raw samples for summary calculation
            self.cpu_samples.append(cpu_usage)
            self.ram_percent_samples.append(ram_percent_used)
            self.ram_used_samples.append(ram_used)
            self.ram_available_samples.append(ram_available)
            self.ram_total_samples.append(ram_total)  # Store total as well

            current_sample = {
                "cpu_percent": round(cpu_usage, 2),
                "ram_percent_used": round(ram_percent_used, 2),
                "ram_used_mb": round(ram_used, 2),
                "ram_available_mb": round(ram_available, 2),
                "ram_total_mb": round(ram_total, 2),
            }

            if self.gpu_available:
                total_gpu_util = 0.0
                total_gpu_mem_used = 0.0
                total_gpu_mem_total = 0.0

                for i in range(self.gpu_count):
                    handle = nvmlDeviceGetHandleByIndex(i)
                    util = nvmlDeviceGetUtilizationRates(handle)
                    mem = nvmlDeviceGetMemoryInfo(handle)

                    total_gpu_util += util.gpu
                    total_gpu_mem_used += mem.used / (1024 ** 2)
                    total_gpu_mem_total += mem.total / (1024 ** 2)

                avg_gpu_util = total_gpu_util / self.gpu_count
                avg_gpu_mem_used = total_gpu_mem_used / self.gpu_count
                avg_gpu_mem_total = total_gpu_mem_total / self.gpu_count

                self.gpu_util_samples.append(avg_gpu_util)
                self.gpu_mem_used_samples.append(avg_gpu_mem_used)
                self.gpu_mem_total_samples.append(avg_gpu_mem_total)

                current_sample.update({
                    "gpu_util_percent": round(avg_gpu_util, 2),
                    "gpu_memory_used_mb": round(avg_gpu_mem_used, 2),
                    "gpu_memory_total_mb": round(avg_gpu_mem_total, 2),
                })

            self._latest_snapshot = current_sample
            return current_sample

        except Exception as e:
            print(f"[TraceML] System sampling error: {e}", file=sys.stderr)
            error_snapshot = {
                "cpu_percent": 0.0,
                "ram_percent_used": 0.0,
                "ram_used_mb": 0.0,
                "ram_available_mb": 0.0,
                "ram_total_mb": 0.0,
                "error": str(e),  # Error message (for debugging)
            }
            if self.gpu_available:
                error_snapshot.update(
                {
                    "gpu_util_percent": 0.0,
                    "gpu_memory_used_mb": 0.0,
                    "gpu_memory_total_mb": 0.0,
                }
            )
            self._latest_snapshot = error_snapshot
            return error_snapshot

    def get_live_snapshot(self) -> Dict[str, Any]:
        """
        Return the most recent CPU and RAM usage reading.

        Returns:
            Dict[str, Any]: Latest sampled values. Returns initialized default
                            if no samples have been successfully collected yet.
        """
        # Return the last updated snapshot
        return self._latest_snapshot

    def get_summary(self) -> Dict[str, Any]:
        """
        Compute and return summary statistics for CPU and RAM usage over the sampling period.
        """
        try:
            summary = {
                "total_system_samples": len(self.cpu_samples),
                "cpu_average_percent": round(
                    sum(self.cpu_samples) / len(self.cpu_samples),2) if self.cpu_samples else 0.0,
                "cpu_peak_percent": round(max(self.cpu_samples), 2) if self.cpu_samples else 0.0,
                "ram_average_percent_used": round(
                    sum(self.ram_percent_samples) / len(self.ram_percent_samples),2) if self.ram_percent_samples else 0.0,
                "ram_peak_percent_used": round(
                    max(self.ram_percent_samples), 2) if self.ram_percent_samples else 0.0,
                "ram_average_used": round(
                    sum(self.ram_used_samples) / len(self.ram_used_samples), 2) if self.ram_used_samples else 0.0,
                "ram_peak_used": round(
                    max(self.ram_used_samples), 2) if self.ram_used_samples else 0.0,
                "ram_average_available": round(
                    sum(self.ram_available_samples) / len(self.ram_available_samples),2) if self.ram_available_samples else 0.0,
                "ram_min_available": round(
                    min(self.ram_available_samples),2) if self.ram_available_samples else 0.0,
                "ram_total_mb": round(self.ram_total_samples[0], 2) if self.ram_total_samples else 0.0,
            }

            if self.gpu_available and self.gpu_util_samples:
                summary.update({
                    "gpu_average_util_percent": round(sum(self.gpu_util_samples) / len(self.gpu_util_samples), 2),
                    "gpu_peak_util_percent": round(max(self.gpu_util_samples), 2),
                    "gpu_average_memory_used": round(sum(self.gpu_mem_used_samples) / len(self.gpu_mem_used_samples),
                                                        2),
                    "gpu_peak_memory_used": round(max(self.gpu_mem_used_samples), 2),
                    "gpu_memory_total": round(self.gpu_mem_total_samples[0], 2),
                })
            else:
                summary.update({
                    "gpu_average_util_percent": 0.0,
                    "gpu_peak_util_percent": 0.0,
                    "gpu_average_memory_used": 0.0,
                    "gpu_peak_memory_used": 0.0,
                    "gpu_memory_total": 0.0,
                })
            return summary
        except Exception as e:
            print(f"[TraceML] System summary calculation error: {e}", file=sys.stderr)
            return {
                "error": str(e),
                "total_system_samples": 0,
            }


    def __del__(self):
        """ Safe shutdown"""
        if self.gpu_available:
            try:
                nvmlShutdown()
            except Exception:
                pass
