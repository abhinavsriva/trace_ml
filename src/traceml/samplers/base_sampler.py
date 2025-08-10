from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseSampler(ABC):
    """
    Abstract base class for samplers that monitor runtime metrics,
    such as CPU usage, tensor allocations, or custom events.

    Samplers may be stateful and are typically polled periodically.
    """

    def __init__(self):
        # Optional: Initialize common sampler-level properties or perform global setup
        pass

    @abstractmethod
    def sample(self) -> Dict[str, Any]:
        """
        Collect the latest data point(s) and return them as a dictionary.
        This method should be non-blocking. It is called regularly by the tracker loop.

        Returns:
            Dict[str, Any]: A dictionary containing the sampled metrics.
                            Should return a dict with 'error' key if sampling fails.
        """
        pass

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """
        Compute and return summary statistics for the collected metrics over the sampling period.

        Returns:
            Dict[str, Any]: Summary statistics. Should return an empty or error
                            dict if no data or calculation fails.
        """
        pass
