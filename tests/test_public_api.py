import atmoresponse


def test_package_root_exports_stable_data_front_door():
    assert atmoresponse.__all__ == [
        "__version__",
        "CacheConfig",
        "HyperspectralCube",
        "LandCoverSample",
        "LocalSceneFiles",
        "SceneAssets",
        "SceneQuery",
        "SceneRecord",
        "SurfaceClassification",
    ]


def test_package_has_version():
    assert atmoresponse.__version__


def test_domain_apis_stay_under_modules():
    from atmoresponse import (
        aeronet,
        aod,
        catalog,
        cube,
        data,
        downloads,
        geo,
        lut,
        plotting,
        sensitivity,
        surface_classes,
        tanager_catalog,
        tanager_data,
        tanager_ortho,
    )

    assert callable(aod.gather_aod)
    assert callable(aod.best_aod)
    assert callable(aod.summarize_aod)
    assert callable(aeronet.from_aeronet)
    assert atmoresponse.HyperspectralCube is cube.HyperspectralCube
    assert atmoresponse.LandCoverSample is surface_classes.LandCoverSample
    assert atmoresponse.SceneRecord is catalog.SceneRecord
    assert atmoresponse.SurfaceClassification is surface_classes.SurfaceClassification
    assert callable(data.cache_scene_files)
    assert callable(downloads.download_file)
    assert callable(geo.haversine_km)
    assert callable(lut.reflectance_from_radiance)
    assert callable(aod.resolve_aod)
    assert callable(sensitivity.run_tanager)
    assert callable(sensitivity.run_emit)
    assert callable(sensitivity.evaluate)
    assert callable(plotting.sensitivity_figure)
    assert callable(surface_classes.classify_scene_surface)
    assert callable(tanager_catalog.search_scenes)
    assert callable(tanager_data.cache_scene_files)
    assert callable(tanager_data.fetch_scene)
    assert callable(tanager_ortho.radiance_at)
    assert callable(tanager_ortho.shipped_aod_summary)

    from atmoresponse import recipes

    assert callable(recipes.as_algorithm)


def test_low_level_helpers_are_not_root_exports():
    assert "build_index" not in atmoresponse.__all__
    assert "cache_scene_files" not in atmoresponse.__all__
    assert "classify_scene_surface" not in atmoresponse.__all__
    assert "download_file" not in atmoresponse.__all__
    assert "from_aeronet" not in atmoresponse.__all__
    assert "gather_aod" not in atmoresponse.__all__
    assert "get_scene_assets" not in atmoresponse.__all__
    assert "haversine_km" not in atmoresponse.__all__
    assert "radiance_at" not in atmoresponse.__all__
    assert "radiance_cube" not in atmoresponse.__all__
    assert "reflectance_cube" not in atmoresponse.__all__
    assert "search_scenes" not in atmoresponse.__all__
    assert "shipped_aod_summary" not in atmoresponse.__all__
    assert "reflectance_from_radiance" not in atmoresponse.__all__
    assert "resolve_aod" not in atmoresponse.__all__
    assert "evaluate_sensitivity" not in atmoresponse.__all__


def test_recipes_exports_sam_modules_not_individual_sam_helpers():
    from atmoresponse import recipes

    assert recipes.sam.__name__ == "atmoresponse.recipes.sam"
    assert recipes.wildfire_sam.__name__ == "atmoresponse.recipes.wildfire_sam"
    assert "sam" in recipes.__all__
    assert "wildfire_sam" in recipes.__all__
    assert "sam_angles" not in recipes.__all__
    assert "labeled_sam_score" not in recipes.__all__
