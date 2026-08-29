"""Public package boundary for AtmoResponse."""

from ._version import __version__
from .aod import AodEstimate, AodQuery, AodSource, resolve_aod
from .cache import CacheConfig
from .catalog import SceneAssets, SceneQuery, SceneRecord, build_index, get_scene_assets, search_scenes
from .data import CachedSceneFiles, DownloadResult, cache_scene_files, download_file
from .lut import CorrectionCoefficients, reflectance_from_radiance, radiance_from_reflectance
from .sensitivity import Algorithm, LabeledScore, SensitivityResult, evaluate_sensitivity

__all__ = [
    "__version__",
    "Algorithm",
    "AodEstimate",
    "AodQuery",
    "AodSource",
    "CacheConfig",
    "CachedSceneFiles",
    "CorrectionCoefficients",
    "DownloadResult",
    "LabeledScore",
    "SceneAssets",
    "SceneQuery",
    "SceneRecord",
    "SensitivityResult",
    "build_index",
    "cache_scene_files",
    "download_file",
    "evaluate_sensitivity",
    "get_scene_assets",
    "radiance_from_reflectance",
    "reflectance_from_radiance",
    "resolve_aod",
    "search_scenes",
]
