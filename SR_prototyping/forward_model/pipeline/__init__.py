"""
Simulation pipeline module.

Provides high-level functions to simulate complete frames and sequences
through the optical chain.
"""

from .simulate_frame import FrameSimulator, simulate_frame, DebugOutput
from .simulate_sequence import SequenceSimulator, simulate_sequence, ShiftPattern

__all__ = [
    "FrameSimulator",
    "simulate_frame",
    "DebugOutput",
    "SequenceSimulator",
    "simulate_sequence",
    "ShiftPattern",
]
