"""Minimal mask input for the retained RGB-D lifting code."""
from dataclasses import dataclass
import numpy as np


@dataclass
class MaskInstance:
    mask: np.ndarray
