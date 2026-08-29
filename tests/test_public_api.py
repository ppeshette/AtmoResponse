import atmoresponse


def test_public_api_exports_scaffolded_boundary():
    expected = {
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
    }

    assert expected.issubset(set(atmoresponse.__all__))


def test_package_has_version():
    assert atmoresponse.__version__
