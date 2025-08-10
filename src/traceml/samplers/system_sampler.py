import psutil
import sys
from typing import List, Dict, Any
from .base_sampler import BaseSampler
import numpy as np

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
    Keeps per-GPU history internally, and exposes distilled live metrics
    (min/max/avg/imbalance) as well as a summary (global peak, lowest non-zero,
    average, variance).

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

        # Aggregated GPU history
        self.gpu_util_samples: List[float] = []
        self.gpu_mem_used_samples: List[float] = []
        self.gpu_mem_total_samples: List[float] = []

        # Per-GPU detailed history
        self.per_gpu_util_samples: Dict[int, List[float]] = {}
        self.per_gpu_mem_used_samples: Dict[int, List[float]] = {}
        self.per_gpu_mem_total: Dict[int, float] = {}  # constant per GPU

        # GPU setup
        self.gpu_available = False
        self.gpu_count = 0
        try:
            nvmlInit()
            self.gpu_count = nvmlDeviceGetCount()
            self.gpu_available = True
            for i in range(self.gpu_count):
                self.per_gpu_util_samples[i] = []
                self.per_gpu_mem_used_samples[i] = []
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
                gpu_utils = []
                gpu_mem_used = []
                gpu_mem_total = []

                for i in range(self.gpu_count):
                    try:
                        handle = nvmlDeviceGetHandleByIndex(i)
                        util = nvmlDeviceGetUtilizationRates(handle)
                        meminfo = nvmlDeviceGetMemoryInfo(handle)

                        util_pct = float(util.gpu)
                        used_mb = float(meminfo.used) / (1024 ** 2)
                        total_mb = float(meminfo.total) / (1024 ** 2)

                        gpu_utils.append(util_pct)
                        gpu_mem_used.append(used_mb)
                        gpu_mem_total.append(total_mb)

                        # update detailed history
                        self.per_gpu_util_samples[i].append(util_pct)
                        self.per_gpu_mem_used_samples[i].append(used_mb)
                        self.per_gpu_mem_total[i] = total_mb
                    except NVMLError as e:
                        print(f"[TraceML] NVML read failed for GPU {i}: {e}", file=sys.stderr)
                    except Exception as e:
                        print(f"[TraceML] Unexpected error reading GPU {i}: {e}", file=sys.stderr)

                # Convert to numpy arrays
                util_arr = np.array(gpu_utils, dtype=float)
                mem_used_arr = np.array(gpu_mem_used, dtype=float)
                mem_total_arr = np.array(gpu_mem_total, dtype=float)

                # Utilization
                avg_util = float(np.mean(util_arr)) if util_arr.size else 0.0
                max_util = float(np.max(util_arr)) if util_arr.size else 0.0
                nonzero_utils = util_arr[util_arr > 0]
                min_nonzero_util = float(np.min(nonzero_utils)) if nonzero_utils.size else 0.0
                imbalance_util = (max_util / min_nonzero_util) if min_nonzero_util > 0 else None

                # Memory distillation
                highest_mem = float(np.max(mem_used_arr)) if mem_used_arr.size else 0.0
                nonzero_mem = mem_used_arr[mem_used_arr > 0]
                lowest_nonzero_mem = float(np.min(nonzero_mem)) if nonzero_mem.size else 0.0

                # Count GPUs under high pressure (>90% of its own total)
                count_high_pressure = 0
                for used, total in zip(mem_used_arr, mem_total_arr):
                    if total > 0 and (used / total) > 0.9:
                        count_high_pressure += 1

                # Update aggregate history
                self.gpu_util_samples.append(avg_util)
                self.gpu_mem_used_samples.append(highest_mem)
                if mem_total_arr.size:
                    self.gpu_mem_total_samples.append(float(np.mean(mem_total_arr)))

                current_sample.update({
                    "gpu_total_count": self.gpu_count,
                    "gpu_util_avg_percent": round(avg_util, 2),
                    "gpu_util_min_nonzero_percent": round(min_nonzero_util, 2) if min_nonzero_util else 0.0,
                    "gpu_util_max_percent": round(max_util, 2),
                    "gpu_util_imbalance_ratio": round(imbalance_util, 2) if imbalance_util is not None else None,
                    "gpu_memory_highest_used_mb": round(highest_mem, 2),
                    "gpu_memory_lowest_nonzero_used_mb": round(lowest_nonzero_mem, 2) if lowest_nonzero_mem else 0.0,
                    "gpu_count_high_pressure": count_high_pressure,
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
                    "gpu_total_count": self.gpu_count,
                    "gpu_util_avg_percent": 0.0,
                    "gpu_util_min_nonzero_percent": 0.0,
                    "gpu_util_max_percent": 0.0,
                    "gpu_util_imbalance_ratio": None,
                    "gpu_memory_highest_used_mb": 0.0,
                    "gpu_memory_lowest_nonzero_used_mb": 0.0,
                    "gpu_count_high_pressure": 0,
                }
            )
            self._latest_snapshot = error_snapshot
            return error_snapshot

    def get_summary(self) -> Dict[str, Any]:
        try:
            summary: Dict[str, Any] = {
                "total_system_samples": len(self.cpu_samples),
                "cpu_average_percent": round(
                    float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0, 2
                ),
                "cpu_peak_percent": round(max(self.cpu_samples), 2) if self.cpu_samples else 0.0,
                "ram_average_percent_used": round(
                    float(np.mean(self.ram_percent_samples)) if self.ram_percent_samples else 0.0, 2
                ),
                "ram_peak_percent_used": round(
                    max(self.ram_percent_samples), 2
                ) if self.ram_percent_samples else 0.0,
                "ram_average_used": round(
                    float(np.mean(self.ram_used_samples)) if self.ram_used_samples else 0.0, 2
                ),
                "ram_peak_used": round(
                    max(self.ram_used_samples), 2
                ) if self.ram_used_samples else 0.0,
                "ram_average_available": round(
                    float(np.mean(self.ram_available_samples)) if self.ram_available_samples else 0.0, 2
                ),
                "ram_min_available": round(
                    min(self.ram_available_samples), 2
                ) if self.ram_available_samples else 0.0,
                "ram_total_mb": round(self.ram_total_samples[0], 2) if self.ram_total_samples else 0.0,
            }

            if self.gpu_available:
                # Flatten all per-GPU memory used history
                all_mem_usages = []
                nonzero_mem_usages = []
                for i in range(self.gpu_count):
                    usages = self.per_gpu_mem_used_samples.get(i, [])
                    all_mem_usages.extend(usages)
                    nonzero_mem_usages.extend([u for u in usages if u > 0])

                all_mem_arr = np.array(all_mem_usages, dtype=float) if all_mem_usages else np.array([], dtype=float)
                nonzero_mem_arr = np.array(nonzero_mem_usages, dtype=float) if nonzero_mem_usages else np.array([],
                                                                                                                dtype=float)

                global_peak = float(np.max(all_mem_arr)) if all_mem_arr.size else 0.0
                global_min_nonzero = float(np.min(nonzero_mem_arr)) if nonzero_mem_arr.size else 0.0
                avg_mem = float(np.mean(all_mem_arr)) if all_mem_arr.size else 0.0
                var_mem = float(np.var(all_mem_arr, ddof=1)) if all_mem_arr.size > 1 else 0.0

                # Aggregated util stats
                util_arr = np.array(self.gpu_util_samples, dtype=float) if self.gpu_util_samples else np.array([],
                                                                                                               dtype=float)
                average_gpu_util = float(np.mean(util_arr)) if util_arr.size else 0.0
                peak_gpu_util = float(np.max(util_arr)) if util_arr.size else 0.0

                summary.update({
                    "gpu_total_count": self.gpu_count,
                    "gpu_average_util_percent": round(average_gpu_util, 2),
                    "gpu_peak_util_percent": round(peak_gpu_util, 2),
                    "gpu_memory_global_peak_used_mb": round(global_peak, 2),
                    "gpu_memory_global_lowest_nonzero_used_mb": round(global_min_nonzero, 2),
                    "gpu_memory_average_used_mb": round(avg_mem, 2),
                    "gpu_memory_variance_mb": round(var_mem, 2),
                })
            else:
                summary.update({
                    "gpu_total_count": self.gpu_count,
                    "gpu_average_util_percent": 0.0,
                    "gpu_peak_util_percent": 0.0,
                    "gpu_memory_global_peak_used_mb": 0.0,
                    "gpu_memory_global_lowest_nonzero_used_mb": 0.0,
                    "gpu_memory_average_used_mb": 0.0,
                    "gpu_memory_variance_mb": 0.0,
                })

            return summary
        except Exception as e:
            print(f"[TraceML] System summary calculation error: {e}", file=sys.stderr)
            return {
                "error": str(e),
                "total_system_samples": 0,
            }

