import pytest

from qal.data import (
    get_concentration_target_preset,
    get_target_preset,
    list_product_families,
    load_concentration_target_presets,
    load_product_family_presets,
)


def test_packaged_concentration_presets_are_available():
    presets = load_concentration_target_presets()

    assert set(presets) == {
        "icg",
        "icg_mini",
        "q700",
        "q700_mini",
        "q800",
        "q800_mini",
        "o38",
        "o38_mini",
        "mbl",
    }
    assert presets["icg"]["well_ids"][0] == "1000 nM"
    assert presets["q700"]["well_ids"][0] == "1000 nM"
    assert presets["q800"]["well_ids"][0] == "600 nM"
    assert all(preset["well_ids"][-1] == "Control" for preset in presets.values())


def test_all_product_family_presets_are_packaged():
    expected_families = {
        "concentration",
        "depth",
        "resolution",
        "uniformity_distortion",
        "radiometric_emitter",
        "depth_resolution",
        "reference_sets",
    }

    assert set(list_product_families()) == expected_families
    for family in expected_families:
        presets = load_product_family_presets(family)
        assert presets
        assert all(preset["family"] for preset in presets.values())


def test_depth_family_inherits_shared_well_ids():
    presets = load_product_family_presets("rds")

    assert set(presets) == {
        "icg",
        "icg_mini",
        "q700",
        "q700_mini",
        "q800",
        "q800_mini",
        "o38",
        "o38_mini",
        "mbl",
    }
    assert presets["icg"]["old_sku"] == "RDS-ICG-ST03-QUEL03"
    assert presets["q800"]["old_sku"] == "RDS-80Q-LU05-QUEL01"
    assert presets["q800"]["well_ids"][-1] == "Control"


def test_catalog_skus_are_published_for_current_products():
    concentration = load_product_family_presets("concentration")
    resolution = load_product_family_presets("resolution")
    ret = load_product_family_presets("ret")

    assert concentration["q800"]["old_sku"] == "RCS-80Q-LU04-QUEL01"
    assert concentration["q800"]["current_sku"] == "S_RCS_STD_Q800-01_LUNG-04_rA"
    assert resolution["q800"]["old_sku"] == "RRT-80Q-LU04-QUEL01"
    assert ret["icg"]["old_sku"] == "RRL-825-PW01-QUEL01"
    assert ret["q700"]["old_sku"] == "RRL-720-PW01-QUEL01"
    assert ret["q700"]["peak_wavelength_nm"] == 720
    assert ret["q800"]["peak_wavelength_nm"] == 805
    assert ret["icg"]["peak_wavelength_nm"] == 825
    assert "central_wavelength_range_nm" not in ret["icg"]

    depth_resolution = load_product_family_presets("depth_resolution")
    assert depth_resolution["icg_v_wedge"]["old_sku"] is None
    assert depth_resolution["icg_v_wedge"]["sku_status"] == "not_assigned"


def test_concentration_preset_aliases_are_supported():
    assert get_concentration_target_preset("Q800-01")["tag"] == "q800"
    assert get_concentration_target_preset("ICG-equivalent")["tag"] == "icg"


def test_custom_concentration_target_uses_supplied_ids():
    ids = [f"Well {index}" for index in range(1, 9)] + ["Control"]

    preset = get_concentration_target_preset(
        "custom",
        custom_well_ids=ids,
    )

    assert preset["tag"] == "custom"
    assert preset["well_ids"] == ids


@pytest.mark.parametrize(
    "ids",
    [
        ["A", "Control"],
        [*[f"Well {index}" for index in range(1, 9)], "Blank"],
    ],
)
def test_custom_concentration_target_requires_nine_wells_and_control(ids):
    with pytest.raises(ValueError):
        get_concentration_target_preset("custom", custom_well_ids=ids)


def test_ret_wavelength_tag_resolves_catalog_or_custom():
    catalog = get_target_preset("ret", "720")
    assert catalog["tag"] == "q700"
    assert catalog["peak_wavelength_nm"] == 720
    assert catalog["old_sku"] == "RRL-720-PW01-QUEL01"

    q800_peak = get_target_preset("ret", "805")
    assert q800_peak["tag"] == "q800"
    assert q800_peak["peak_wavelength_nm"] == 805

    custom = get_target_preset("ret", "630")
    assert custom["tag"] == "custom_630"
    assert custom["peak_wavelength_nm"] == 630
    assert custom["serial_template"] == "630-PW01-RRLxxx"
    assert custom["old_sku"] is None
    assert custom["current_sku"] is None
    assert "fluorophore_code" not in custom
    assert "fluorophore_label" not in custom
    assert custom["well_ids"] == ["RET"]

