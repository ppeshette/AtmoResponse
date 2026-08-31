"""Published algorithm examples used by AtmoResponse."""

from . import agriculture, endmembers, sam, water, wildfire_sam
from .adapters import as_algorithm
from .agriculture import canopy_chlorophyll_rsi, canopy_present, vegetation_water_indices
from .cdom import cdom_absorption
from .cyanobacteria import cyanobacteria_index
from .mineral import aloh_2200_depth
from .water import mndwi, water_candidate

__all__ = [
    "agriculture",
    "aloh_2200_depth",
    "as_algorithm",
    "canopy_chlorophyll_rsi",
    "canopy_present",
    "cdom_absorption",
    "cyanobacteria_index",
    "endmembers",
    "mndwi",
    "sam",
    "vegetation_water_indices",
    "water",
    "water_candidate",
    "wildfire_sam",
]
