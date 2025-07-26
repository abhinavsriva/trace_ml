import torch
import gc
import sys
from typing import Dict, Any, List, Tuple
from .base_sampler import BaseSampler
from src.traceml.decorator import get_model_queue


class LayerMemorySampler(BaseSampler):
    """
    A memory sampler that tracks parameter memory usage of PyTorch models at a per-layer level.
    This sampler operates in two modes:

    1. **Queue-based Sampling** (Preferred):
       If models are explicitly queued using the `@trace_model` decorator or the `trace__model()` function,
       this sampler will iterate over the queue and analyze each queued model without removing them.
       It ensures that each unique model (based on parameter signature) is only sampled once.

    2. **GC-based Fallback**:
       If the queue is empty (i.e., decorators are not used), the sampler will scan all live objects
       in memory using Python's garbage collector to find the largest `nn.Module` instance.
       This fallback helps default structure

    In both modes, a unique signature for each model is generated based on the shape of its parameters.
    This ensures deduplication and prevents redundant memory profiling.

    Memory usage is calculated per layer and aggregated to provide total memory footprint in megabytes (MB).
    Sampled snapshots are stored for historical tracking and summary reporting.
    """

    def __init__(self):
        """Initialize internal state for tracking model memory usage."""
        super().__init__()
        self.seen_signatures = set()  # set of previously seen model signatures
        self.memory_snapshots: List[Dict[str, Any]] = []
        self._latest_snapshot: Dict[str, Any] = {}
        self.total_samples = 0

    def _get_model_signature(self, model: torch.nn.Module) -> Tuple:
        """
        Generate a unique signature for the model.
        """
        try:
            return tuple((name, tuple(p.shape)) for name, p in model.named_parameters())
        except Exception:
            return ()

    def _sample_model(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Compute memory usage of a single model if it hasn't been sampled before."""
        try:
            signature = self._get_model_signature(model)
            if signature in self.seen_signatures:
                return None
            self.seen_signatures.add(signature)
            snapshot = {}
            total_memory_mb = 0.0
            for name, param in model.named_parameters():
                mem_mb = param.element_size() * param.nelement() / (1024 ** 2)
                snapshot[name] = round(mem_mb, 4)
                total_memory_mb += mem_mb

            snapshot_data = {
                "model_index": len(self.seen_signatures),
                "total_memory_mb": round(total_memory_mb, 4),
                "layer_memory_mb": snapshot,
                "model_signature": str(signature),
            }

            self._latest_snapshot = snapshot_data
            self.memory_snapshots.append(snapshot_data)
            self.total_samples += 1
            return snapshot_data

        except Exception as e:
            print(f"[TraceML] Error sampling model: {e}", file=sys.stderr)
            return None

    def _sample_from_gc(self) -> Dict[str, Any]:
        """
        Search all objects in memory for the largest nn.Module.
        """
        try:
            candidates = []
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, torch.nn.Module):
                        param_count = sum(p.numel() for p in obj.parameters())
                        candidates.append((param_count, id(obj), obj))
                except Exception:
                    continue

            if not candidates:
                return {
                    "model_index": -1,
                    "total_memory_mb": 0.0,
                    "layer_memory_mb": {},
                    "model_signature": "no_model_found",
                }

            candidates.sort(reverse=True)
            return self._sample_model(candidates[0][2]) or self._latest_snapshot

        except Exception as e:
            print(f"[TraceML] GC scan failed: {e}", file=sys.stderr)
            return {
                "model_index": -1,
                "total_memory_mb": 0.0,
                "layer_memory_mb": {},
                "model_signature": "gc_error",
                "error": str(e)
            }

    def sample(self) -> Dict[str, Any]:
        """
        Sample memory usage from models in the queue if present.
        Falls back to garbage-collected models otherwise.
        Returns the latest new snapshot or the last seen one.
        """
        queue = get_model_queue()

        if not queue.empty():
            for model in list(queue.queue):  # non-destructive iteration
                result = self._sample_model(model)
                if result:
                    return result  # return first new snapshot
            return self._latest_snapshot

        return self._sample_from_gc()

    def get_live_snapshot(self) -> Dict[str, Any]:
        """
        Return the latest sampled snapshot.
        """
        return self._latest_snapshot

    def get_summary(self) -> Dict[str, Any]:
        """
        Return summary statistics over all models seen.
        """
        total_models = len(self.seen_signatures)
        avg_total_memory = 0.0
        max_total_memory = 0.0

        if self.memory_snapshots:
            total_memory_values = [snap["total_memory_mb"] for snap in self.memory_snapshots]
            avg_total_memory = round(sum(total_memory_values) / len(total_memory_values), 4)
            max_total_memory = round(max(total_memory_values), 4)

        return {
            "total_models_seen": total_models,
            "total_samples_taken": self.total_samples,
            "average_model_memory": avg_total_memory,
            "peak_model_memory": max_total_memory,
            "last_model_snapshot": self._latest_snapshot,
        }
