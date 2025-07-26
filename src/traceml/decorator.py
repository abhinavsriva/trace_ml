import functools
import sys
import threading
from typing import Callable
import torch.nn as nn
from queue import Queue

# Shared queue used by decorator and sampler
model_queue: Queue = Queue()

def trace_model(
    sample_layer_memory: bool = True,
    trace_gradients: bool = False,
    trace_activations: bool = False,
) -> Callable:
    """
    Decorator to trace a model when it's defined.
    Queues the model for sampling.

    Args:
        sample_layer_memory (bool): Track parameter memory usage (default: True).
        trace_gradients (bool): Track gradient memory usage (planned).
        trace_activations (bool): Track activation memory usage (planned).
    """

    def decorator(cls):
        if not isinstance(cls, type) or not issubclass(cls, nn.Module):
            raise TypeError("@trace_model can only be applied to nn.Module subclasses.")

        original_init = cls.__init__
        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            # First call the original __init__
            original_init(self, *args, **kwargs)
            try:
                if sample_layer_memory:
                    model_queue.put(self)
                # TODO: register gradient/activation hooks here
            except Exception as e:
                print(f"[TraceML] Failed to trace model: {e}", file=sys.stderr)

        cls.__init__ = wrapped_init
        return cls
    return decorator

def trace_model_instance(model: nn.Module):
    """
    Manually trace a PyTorch model instance (useful for functional or sequential models).

    Args:
        model (nn.Module): The model instance to trace.
    """
    try:
        if isinstance(model, nn.Module):
            model_queue.put(model)
        else:
            raise TypeError("trace_model_instance expects an nn.Module.")
    except Exception as e:
        print(f"[TraceML] Failed to trace model instance: {e}", file=sys.stderr)


def get_model_queue():
    """Return the shared model queue."""
    return model_queue
