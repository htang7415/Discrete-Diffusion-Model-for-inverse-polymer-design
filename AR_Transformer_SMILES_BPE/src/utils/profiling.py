"""Profiling utilities for performance analysis."""

import time
from contextlib import contextmanager
from typing import Dict, Optional, Callable, Any
from functools import wraps
import torch


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all timers."""
        self.timings: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    @contextmanager
    def time(self, name: str):
        """Context manager for timing a block of code.

        Args:
            name: Name of the timing measurement.

        Example:
            timer = Timer()
            with timer.time("forward_pass"):
                output = model(input)
        """
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.timings[name] = self.timings.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics.

        Returns:
            Dictionary with total time, count, and average for each measurement.
        """
        stats = {}
        for name in self.timings:
            total = self.timings[name]
            count = self.counts[name]
            stats[name] = {
                "total_seconds": round(total, 4),
                "count": count,
                "avg_ms": round((total / count) * 1000, 4) if count > 0 else 0.0
            }
        return stats

    def print_stats(self, title: str = "Profiling Results"):
        """Print timing statistics.

        Args:
            title: Title for the output.
        """
        stats = self.get_stats()
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")

        # Sort by total time
        sorted_names = sorted(stats.keys(), key=lambda x: stats[x]["total_seconds"], reverse=True)

        for name in sorted_names:
            s = stats[name]
            print(f"{name:30s} | Total: {s['total_seconds']:8.3f}s | "
                  f"Count: {s['count']:6d} | Avg: {s['avg_ms']:8.3f}ms")

        print(f"{'='*50}\n")


def profile_function(timer: Timer, name: Optional[str] = None):
    """Decorator for profiling a function.

    Args:
        timer: Timer instance to use.
        name: Optional name for the measurement (defaults to function name).

    Example:
        timer = Timer()

        @profile_function(timer)
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        measurement_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with timer.time(measurement_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


class GPUProfiler:
    """GPU-specific profiling utilities."""

    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current GPU memory statistics.

        Returns:
            Dictionary with memory stats in GB.
        """
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
        }

    @staticmethod
    def reset_peak_stats():
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def synchronize():
        """Synchronize CUDA for accurate timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def profile_cuda(self, name: str, timer: Timer):
        """Profile a CUDA operation with proper synchronization.

        Args:
            name: Name of the operation.
            timer: Timer instance.

        Example:
            profiler = GPUProfiler()
            with profiler.profile_cuda("forward", timer):
                output = model(input)
        """
        self.synchronize()
        with timer.time(name):
            yield
        self.synchronize()


class SamplingProfiler:
    """Profiler for the sampling/generation pipeline."""

    def __init__(self):
        self.timer = Timer()
        self.gpu = GPUProfiler()

    def reset(self):
        """Reset profiler."""
        self.timer.reset()
        self.gpu.reset_peak_stats()

    @contextmanager
    def profile_step(self, name: str):
        """Profile a sampling step.

        Args:
            name: Name of the step.
        """
        self.gpu.synchronize()
        with self.timer.time(name):
            yield
        self.gpu.synchronize()

    def get_results(self) -> Dict:
        """Get profiling results.

        Returns:
            Dictionary with timing and memory stats.
        """
        return {
            "timing": self.timer.get_stats(),
            "memory": self.gpu.get_memory_stats()
        }

    def print_results(self):
        """Print profiling results."""
        self.timer.print_stats("Sampling Profiling Results")
        mem = self.gpu.get_memory_stats()
        if mem:
            print(f"GPU Memory: {mem['allocated_gb']:.2f}GB allocated, "
                  f"{mem['max_allocated_gb']:.2f}GB peak")


class TrainingProfiler:
    """Profiler for the training pipeline."""

    def __init__(self):
        self.timer = Timer()
        self.gpu = GPUProfiler()
        self.step_times = []

    def reset(self):
        """Reset profiler."""
        self.timer.reset()
        self.gpu.reset_peak_stats()
        self.step_times = []

    @contextmanager
    def profile_step(self, name: str):
        """Profile a training step.

        Args:
            name: Name of the step.
        """
        self.gpu.synchronize()
        start = time.perf_counter()
        with self.timer.time(name):
            yield
        self.gpu.synchronize()
        self.step_times.append(time.perf_counter() - start)

    def get_throughput(self, batch_size: int) -> float:
        """Get training throughput in samples/second.

        Args:
            batch_size: Training batch size.

        Returns:
            Samples processed per second.
        """
        if not self.step_times:
            return 0.0
        avg_step_time = sum(self.step_times) / len(self.step_times)
        return batch_size / avg_step_time if avg_step_time > 0 else 0.0

    def get_results(self) -> Dict:
        """Get profiling results.

        Returns:
            Dictionary with timing and memory stats.
        """
        return {
            "timing": self.timer.get_stats(),
            "memory": self.gpu.get_memory_stats(),
            "step_times": self.step_times
        }

    def print_results(self, batch_size: int = 256):
        """Print profiling results.

        Args:
            batch_size: Training batch size for throughput calculation.
        """
        self.timer.print_stats("Training Profiling Results")
        mem = self.gpu.get_memory_stats()
        if mem:
            print(f"GPU Memory: {mem['allocated_gb']:.2f}GB allocated, "
                  f"{mem['max_allocated_gb']:.2f}GB peak")
        print(f"Throughput: {self.get_throughput(batch_size):.1f} samples/sec")
