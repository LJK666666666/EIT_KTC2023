"""
Physics backend abstraction for test-time optimization.
"""
from abc import ABC, abstractmethod
from typing import Dict


class PhysicsBackend(ABC):
    """Abstract physics backend."""

    @abstractmethod
    def prepare(self, measurements, sigma_init) -> None:
        """Prepare backend state for one sample."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, sigma):
        """Predict measurements from sigma."""
        raise NotImplementedError


def create_physics_backend(name: str, **kwargs) -> PhysicsBackend:
    """Factory for physics backend."""
    backend_name = name.lower()
    if backend_name == "linearized_ktc":
        from .physics_linearized_ktc import LinearizedKTCBackend
        return LinearizedKTCBackend(**kwargs)
    raise ValueError(f"Unknown physics backend: {name}")

