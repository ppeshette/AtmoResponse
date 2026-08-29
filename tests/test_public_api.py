import atmoresponse


def test_package_root_exports_stable_data_front_door():
    assert atmoresponse.__all__ == [
        "__version__",
        "CacheConfig",
        "CachedSceneFiles",
        "SceneAssets",
        "SceneQuery",
        "SceneRecord",
        "cache_scene_files",
        "get_scene_assets",
        "search_scenes",
    ]


def test_package_has_version():
    assert atmoresponse.__version__


def test_domain_apis_stay_under_modules():
    from atmoresponse import aod, catalog, data, extract, lut, sensitivity

    assert callable(catalog.search_scenes)
    assert callable(data.cache_scene_files)
    assert callable(extract.radiance_at)
    assert callable(lut.reflectance_from_radiance)
    assert callable(aod.resolve_aod)
    assert callable(sensitivity.evaluate_sensitivity)


def test_low_level_helpers_are_not_root_exports():
    assert "build_index" not in atmoresponse.__all__
    assert "download_file" not in atmoresponse.__all__
    assert "radiance_at" not in atmoresponse.__all__
    assert "reflectance_from_radiance" not in atmoresponse.__all__
    assert "resolve_aod" not in atmoresponse.__all__
    assert "evaluate_sensitivity" not in atmoresponse.__all__
