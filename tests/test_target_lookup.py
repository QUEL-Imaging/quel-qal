import pytest

from qal.data import get_target_preset
from qal.util import find_target_presets, lookup_target_preset, target_data_labels


@pytest.mark.parametrize(
    ("query", "family_key", "tag", "sku"),
    [
        ("RCS-ICG-ST01-QUEL03", "concentration", "icg", "RCS-ICG-ST01-QUEL03"),
        ("RDS-80Q-LU05-QUEL01", "depth", "q800", "RDS-80Q-LU05-QUEL01"),
        ("RCS-S-IC1-S1-A-113", "concentration", "icg", "RCS-ICG-ST01-QUEL03"),
        ("V3-ICG-ST01-RCS007", "concentration", "icg", "RCS-ICG-ST01-QUEL03"),
        ("V1-80Q-LU05-RDS007", "depth", "q800", "RDS-80Q-LU05-QUEL01"),
        ("RCS-ICG-ST", "concentration", "icg", "RCS-ICG-ST01-QUEL03"),
        ("RRT-70Q", "resolution", "q700", "RRT-70Q-ST01-QUEL01"),
        ("RCS-M-IC1-S1-A-104", "concentration", "icg_mini", "RCS-ICG-SM01-QUEL01"),
        ("V3-ICG-SM02-RDS007", "depth", "icg_mini", "RDS-ICG-SM03-QUEL01"),
        ("S_RCS_STD_ICG-01_STND-01_rA", "concentration", "icg", "RCS-ICG-ST01-QUEL03"),
        ("RCS-O38-LU02-QUEL01", "concentration", "o38", "RCS-O38-LU02-QUEL01"),
        ("V3-ICG-ST01-RDS000", "depth", "icg", "RDS-ICG-ST03-QUEL03"),
        ("V2-O38-LM03-RDS006", "depth", "o38_mini", None),
        ("V3-MBL-R01-RCS01", "concentration", "mbl", None),
        ("v2_R3_C07", "concentration", "icg", "RCS-ICG-ST01-QUEL03"),
        ("RRL-825-PW01-QUEL01", "radiometric_emitter", "icg", "RRL-825-PW01-QUEL01"),
    ],
)
def test_lookup_target_preset_resolves_serials_and_fragments(
    query,
    family_key,
    tag,
    sku,
):
    preset = lookup_target_preset(query)

    assert preset["family_key"] == family_key
    assert preset["tag"] == tag
    assert preset.get("old_sku") == sku


def test_lookup_target_preset_returns_concentration_well_ids():
    preset = lookup_target_preset("RCS-ICG-ST01-QUEL03")
    labels = target_data_labels(preset)

    assert labels["well_ids"][0] == "1000 nM"
    assert labels["well_ids"][-1] == "Control"
    assert labels["fluorophore_label"] == "ICG-eq"
    assert labels["qal_graph_type"] == "concentration"


def test_lookup_q800_concentration_uses_600_nm_series():
    preset = lookup_target_preset("RCS-80Q-LU")

    assert preset["tag"] == "q800"
    assert preset["well_ids"][0] == "600 nM"


def test_partial_family_fluorophore_query_is_ambiguous():
    with pytest.raises(ValueError, match="Ambiguous"):
        lookup_target_preset("RCS-ICG")

    matches = find_target_presets("RCS-ICG")
    assert {preset["tag"] for preset in matches} == {"icg", "icg_mini"}


def test_lookup_can_be_limited_to_one_family():
    preset = lookup_target_preset("80Q", family="resolution")

    assert preset["family_key"] == "resolution"
    assert preset["tag"] == "q800"

    tagged = lookup_target_preset("icg", family="concentration")
    assert tagged["old_sku"] == "RCS-ICG-ST01-QUEL03"
    assert tagged["current_sku"] == "S_RCS_STD_ICG-01_STND-01_rA"


def test_unknown_serial_raises():
    with pytest.raises(ValueError, match="No product preset matches"):
        lookup_target_preset("XYZ-NOPE-0001")


def test_legacy_o38_keeps_1000_nm_well_series():
    preset = lookup_target_preset("V1-O38-LU02-RCS001")

    assert preset["tag"] == "o38"
    assert preset["status"] == "superseded"
    assert preset["replaced_by"] == "q800"
    assert preset["well_ids"][0] == "1000 nM"


def test_older_icg_depth_serial_is_backwards_compatible():
    preset = lookup_target_preset("V3-ICG-ST01-RDS000")

    assert preset["old_sku"] == "RDS-ICG-ST03-QUEL03"
    assert preset["backwards_compatible"] is True
    assert "ST01-RDS" in preset["notes"]
    preset = get_target_preset("rds", "q700_mini")

    assert preset["old_sku"] == "RDS-70Q-SM03-QUEL01"
    assert preset["well_ids"][-1] == "Control"


def test_catalog_ret_serial_keeps_explicit_peak_wavelength():
    q700 = lookup_target_preset("720-PW01-RRL003")
    assert q700["tag"] == "q700"
    assert q700["peak_wavelength_nm"] == 720
    assert q700["current_sku"] == "S_RET_PWR_720_rA"

    q800 = lookup_target_preset("800-PW01-RRL001")
    assert q800["tag"] == "q800"
    assert q800["peak_wavelength_nm"] == 805
    assert q800["old_sku"] == "RRL-800-PW01-QUEL01"


def test_custom_ret_serial_infers_wavelength_without_catalog_skus():
    preset = lookup_target_preset("630-PW01-RRL001")

    assert preset["family_key"] == "radiometric_emitter"
    assert preset["tag"] == "custom_630"
    assert preset["peak_wavelength_nm"] == 630
    assert preset["form_code"] == "PW01"
    assert preset["serial_template"] == "630-PW01-RRLxxx"
    assert preset["old_sku"] is None
    assert preset["current_sku"] is None
    assert "fluorophore_code" not in preset
    assert "fluorophore_label" not in preset
    assert preset["well_ids"] == ["RET"]
    labels = target_data_labels(preset)
    assert labels["peak_wavelength_nm"] == 630


def test_custom_ret_can_be_looked_up_by_wavelength_when_family_is_ret():
    preset = lookup_target_preset("630", family="ret")

    assert preset["tag"] == "custom_630"
    assert preset["peak_wavelength_nm"] == 630


def test_bare_wavelength_without_ret_family_does_not_synthesize():
    with pytest.raises(ValueError, match="No product preset matches"):
        lookup_target_preset("630")


def test_unknown_ret_shaped_query_outside_optical_range_still_fails():
    with pytest.raises(ValueError, match="No product preset matches"):
        lookup_target_preset("100-PW01-RRL001")
