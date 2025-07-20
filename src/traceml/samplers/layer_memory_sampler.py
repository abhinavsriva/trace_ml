import torch
import gc
import sys
from typing import Dict, Any, List, Tuple
from .base_sampler import BaseSampler


class LayerMemorySampler(BaseSampler):
    """
    Sampler that automatically tracks per-layer parameter memory usage of
    PyTorch models found in Python memory using garbage collection.
    """

    def __init__(self):
        super().__init__()
        # Set of previously seen model signatures
        self.seen_signatures = set()
        self.memory_snapshots: List[Dict[str, Any]] = []
        self._latest_snapshot: Dict[str, Any] = {}
        self.total_samples = 0

    def _find_largest_model(self):
        """
        Search all objects in memory for the largest nn.Module.
        """
        candidates = []
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.nn.Module):
                    param_count = sum(p.numel() for p in obj.parameters())
                    candidates.append((param_count, id(obj), obj))
            except Exception:
                continue

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][2]
        return None

    def _get_model_signature(self, model: torch.nn.Module) -> Tuple:
        """
        Generate a unique signature for the model.
        """
        try:
            return tuple((name, tuple(p.shape)) for name, p in model.named_parameters())
        except Exception:
            return ()

    def sample(self) -> Dict[str, Any]:
        """
        Scan for models, sample parameter memory usage, and store snapshot.
        Should return a structured dictionary to AVOID breaking downstream consumers.
        """
        try:
            model = self._find_largest_model()
            if model is None:
                empty_snapshot = {
                    "model_index": "none",
                    "total_memory_mb": 0.0,
                    "layer_memory_mb": {},
                    "model_signature": "no_model_found",
                    "info": "No model found in memory."
                }
                self._latest_snapshot = empty_snapshot
                return empty_snapshot

            signature = self._get_model_signature(model)
            if signature in self.seen_signatures:
                return self._latest_snapshot

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
            print(f"[TraceML] Layer memory sampling error: {e}", file=sys.stderr)
            error_snapshot = {
                "model_index": "error",
                "total_memory_mb": 0.0,
                "layer_memory_mb": {},
                "model_signature": "sampling_error",
                "error": str(e)
            }
            self._latest_snapshot = error_snapshot
            return error_snapshot

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
