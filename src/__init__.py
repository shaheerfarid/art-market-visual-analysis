"""
Art Market Visual Analysis System
==================================

A comprehensive computer vision pipeline for analyzing artwork patterns,
color theory, and curatorial sequencing in auction catalogues.

Modules:
--------
- utils: Image loading and preprocessing utilities
- color_analysis: Color extraction and quantization
- spatial_analysis: Geometry, symmetry, and edge detection
- semantic_analysis: CLIP-based content classification
- sequence_analysis: Temporal pattern and transition detection
"""

__version__ = "1.0.0"
__author__ = "Muhammad Shaheer Bin Farid"

from . import utils
from . import color_analysis
from . import spatial_analysis
from . import semantic_analysis
from . import sequence_analysis

__all__ = [
    "utils",
    "color_analysis",
    "spatial_analysis",
    "semantic_analysis",
    "sequence_analysis"
]
