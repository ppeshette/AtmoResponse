"""Published algorithm examples used by AtmoResponse."""

from . import sam, wildfire_sam
from .adapters import as_algorithm
from .agriculture import canopy_chlorophyll_rsi, vegetation_water_indices
from .cdom import cdom_absorption
from .cyanobacteria import cyanobacteria_index
from .mineral import aloh_2200_depth

__all__ = [
    "aloh_2200_depth",
    "as_algorithm",
    "canopy_chlorophyll_rsi",
    "cdom_absorption",
    "cyanobacteria_index",
    "sam",
    "vegetation_water_indices",
    "wildfire_sam",
]
