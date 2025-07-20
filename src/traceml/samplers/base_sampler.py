from abc import ABC, abstractmethod
from typing import Any

class BaseSampler(ABC):
    """
    Abstract base class for samplers that monitor runtime metrics,
    such as CPU usage, tensor allocations, or custom events.

    Samplers may be stateful and are typically polled periodically.
    """

    @abstractmethod
    def sample(self) -> None:
        """
        Collect the latest data point (e.g., current CPU usage, validate tensor refs).
        Called regularly by the tracker loop.
        """
        pass

    @abstractmethod
    def get_live_snapshot(self) -> Any:
        """
        Return the latest snapshot in a format suitable for logging or visualization.

        Examples:
        - CPUSampler: {"cpu_percent": 72.3}
        - TensorSampler: {"total_memory_mb": 120.5, "tensors": [...]}
        """
        pass