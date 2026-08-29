import atmoresponse


def test_package_root_exports_stable_data_front_door():
    assert atmoresponse.__all__ == [
        "__version__",
        "CacheConfig",
        "LocalSceneFiles",
        "SceneAssets",
        "SceneQuery",
        "SceneRecord",
    ]


def test_package_has_version():
    assert atmoresponse.__version__


def test_domain_apis_stay_under_modules():
    from atmoresponse import (
        aeronet,
        aod,
        catalog,
        data,
        downloads,
        geo,
        lut,
        sensitivity,
        tanager_catalog,
        tanager_data,
        tanager_hdf5,
    )

    assert callable(aod.gather_aod)
    assert callable(aod.best_aod)
    assert callable(aod.summarize_aod)
    assert callable(aeronet.from_aeronet)
    assert atmoresponse.SceneRecord is catalog.SceneRecord
    assert callable(data.cache_scene_files)
    assert callable(downloads.download_file)
    assert callable(geo.haversine_km)
    assert callable(lut.reflectance_from_radiance)
    assert callable(aod.resolve_aod)
    assert callable(sensitivity.evaluate_sensitivity)
    assert callable(tanager_catalog.search_scenes)
    assert callable(tanager_data.cache_scene_files)
    assert callable(tanager_hdf5.radiance_at)
    assert callable(tanager_hdf5.shipped_aod_summary)


def test_low_level_helpers_are_not_root_exports():
    assert "build_index" not in atmoresponse.__all__
    assert "cache_scene_files" not in atmoresponse.__all__
    assert "download_file" not in atmoresponse.__all__
    assert "from_aeronet" not in atmoresponse.__all__
    assert "gather_aod" not in atmoresponse.__all__
    assert "get_scene_assets" not in atmoresponse.__all__
    assert "haversine_km" not in atmoresponse.__all__
    assert "radiance_at" not in atmoresponse.__all__
    assert "search_scenes" not in atmoresponse.__all__
    assert "shipped_aod_summary" not in atmoresponse.__all__
    assert "reflectance_from_radiance" not in atmoresponse.__all__
    assert "resolve_aod" not in atmoresponse.__all__
    assert "evaluate_sensitivity" not in atmoresponse.__all__
