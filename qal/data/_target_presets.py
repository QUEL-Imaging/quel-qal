"""Validated product presets for QUEL target families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional

import yaml


_FAMILY_RESOURCES = {
    "concentration": "product_presets/concentration.yaml",
    "depth": "product_presets/depth.yaml",
    "resolution": "product_presets/resolution.yaml",
    "uniformity_distortion": "product_presets/uniformity_distortion.yaml",
    "radiometric_emitter": "product_presets/radiometric_emitter.yaml",
    "depth_resolution": "product_presets/depth_resolution.yaml",
    "reference_sets": "product_presets/reference_sets.yaml",
}
_FAMILY_ALIASES = {
    "rcs": "concentration",
    "rds": "depth",
    "rrt": "resolution",
    "rud": "uniformity_distortion",
    "ret": "radiometric_emitter",
    "rrl": "radiometric_emitter",
    "rkt": "reference_sets",
    "dr": "depth_resolution",
}
_FAMILY_PRODUCT_CODES = {
    "concentration": "RCS",
    "depth": "RDS",
    "resolution": "RRT",
    "uniformity_distortion": "RUD",
    "radiometric_emitter": "RET",
    "depth_resolution": "DR",
    "reference_sets": "RKT",
}
_ALIASES = {
    "icg-equivalent": "icg",
    "q800-01": "q800",
    "q700-01": "q700",
    "80q": "q800",
    "70q": "q700",
    "o38": "o38",
    "otl38": "o38",
    "otl-38": "o38",
    "mbl": "mbl",
    "methylene-blue": "mbl",
    "methylene blue": "mbl",
}


def list_product_families() -> tuple[str, ...]:
    """Return product-family tags with packaged YAML definitions."""
    return tuple(_FAMILY_RESOURCES)


def normalize_product_family(family: str) -> str:
    """Return the canonical family key for a name or product code."""
    normalized_family = str(family).strip().casefold()
    normalized_family = _FAMILY_ALIASES.get(
        normalized_family,
        normalized_family,
    )
    if normalized_family not in _FAMILY_RESOURCES:
        options = ", ".join(_FAMILY_RESOURCES)
        raise ValueError(
            f"Unknown product family {family!r}; choose from {options}"
        )
    return normalized_family


def load_product_family_presets(
    family: str,
    yaml_path: Optional[str | Path] = None,
) -> dict[str, dict[str, Any]]:
    """Load one product family's presets and merge shared family metadata."""
    normalized_family = normalize_product_family(family)

    source = (
        Path(yaml_path)
        if yaml_path is not None
        else files("qal.data").joinpath(
            _FAMILY_RESOURCES[normalized_family]
        )
    )
    with source.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)

    if not isinstance(document, Mapping):
        raise ValueError(f"{normalized_family} YAML must contain a mapping")
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported {normalized_family} schema version")
    raw_presets = document.get("presets")
    if not isinstance(raw_presets, Mapping) or not raw_presets:
        raise ValueError(f"{normalized_family} YAML has no presets")

    shared = {
        key: value
        for key, value in document.items()
        if key not in {"schema_version", "presets"}
    }
    product_code = shared.get(
        "product_code",
        _FAMILY_PRODUCT_CODES[normalized_family],
    )
    presets = {}
    for raw_tag, raw_preset in raw_presets.items():
        tag = str(raw_tag).strip().casefold()
        if not isinstance(raw_preset, Mapping):
            raise ValueError(f"Preset {tag!r} must contain a mapping")
        presets[tag] = {
            **shared,
            **dict(raw_preset),
            "tag": tag,
            "family_key": normalized_family,
            "product_code": raw_preset.get("product_code", product_code),
        }
    return presets


def iter_product_presets(
    yaml_paths: Optional[Mapping[str, str | Path]] = None,
) -> list[dict[str, Any]]:
    """Load every packaged product preset as a flat list of copies."""
    presets = []
    for family in _FAMILY_RESOURCES:
        yaml_path = None if yaml_paths is None else yaml_paths.get(family)
        for preset in load_product_family_presets(
            family,
            yaml_path=yaml_path,
        ).values():
            presets.append(dict(preset))
    return presets


def get_target_preset(
    family: str,
    tag: str,
    *,
    custom_well_ids: Optional[Sequence[str]] = None,
    yaml_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Return one preset selected by family and tag."""
    normalized_family = normalize_product_family(family)
    if normalized_family == "concentration":
        return get_concentration_target_preset(
            tag,
            custom_well_ids=custom_well_ids,
            yaml_path=yaml_path,
        )
    if custom_well_ids is not None:
        raise ValueError("custom_well_ids is only supported for concentration")

    presets = load_product_family_presets(
        normalized_family,
        yaml_path=yaml_path,
    )
    normalized_tag = str(tag).strip().casefold()
    normalized_tag = _ALIASES.get(normalized_tag, normalized_tag)
    if normalized_tag in presets:
        return dict(presets[normalized_tag])
    if normalized_family == "radiometric_emitter":
        custom_ret = ret_preset_from_tag(normalized_tag, yaml_path=yaml_path)
        if custom_ret is not None:
            return custom_ret
        options = ", ".join([*sorted(presets), "<wavelength_nm>"])
        raise ValueError(
            f"Unknown {normalized_family} target tag {tag!r}; "
            f"choose from {options}"
        )
    options = ", ".join(sorted(presets))
    raise ValueError(
        f"Unknown {normalized_family} target tag {tag!r}; "
        f"choose from {options}"
    )


def _validate_well_ids(well_ids: Sequence[str], name: str) -> list[str]:
    ids = [str(well_id).strip() for well_id in well_ids]
    if len(ids) != 9:
        raise ValueError(f"{name} must define exactly nine well IDs")
    if any(not well_id for well_id in ids):
        raise ValueError(f"{name} contains an empty well ID")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate well IDs")
    if ids[-1].casefold() != "control":
        raise ValueError(f"{name} must place Control last")
    return ids


def load_concentration_target_presets(
    yaml_path: Optional[str | Path] = None,
) -> dict[str, dict[str, Any]]:
    """Load and validate concentration-target presets from YAML."""
    presets = load_product_family_presets(
        "concentration",
        yaml_path=yaml_path,
    )
    for tag, preset in presets.items():
        if "well_ids" not in preset:
            raise ValueError(f"Preset {tag!r} is missing well_ids")
        preset["well_ids"] = _validate_well_ids(
            preset["well_ids"],
            f"Preset {tag!r}",
        )
    return presets


def get_concentration_target_preset(
    tag: str,
    *,
    custom_well_ids: Optional[Sequence[str]] = None,
    yaml_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Return one preset selected by tag, or a validated custom definition."""
    normalized_tag = str(tag).strip().casefold()
    normalized_tag = _ALIASES.get(normalized_tag, normalized_tag)

    if normalized_tag == "custom":
        if custom_well_ids is None:
            raise ValueError(
                "custom_well_ids is required when target tag is 'custom'"
            )
        return {
            "tag": "custom",
            "display_name": "Custom concentration target",
            "well_ids": _validate_well_ids(
                custom_well_ids,
                "Custom concentration target",
            ),
        }

    presets = load_concentration_target_presets(yaml_path)
    if normalized_tag not in presets:
        options = ", ".join([*sorted(presets), "custom"])
        raise ValueError(
            f"Unknown concentration target tag {tag!r}; choose from {options}"
        )
    return dict(presets[normalized_tag])


_RET_WAVELENGTH_MIN_NM = 400
_RET_WAVELENGTH_MAX_NM = 1100
_RET_DEFAULT_FORM_CODE = "PW01"
_RET_SHARED_KEYS = (
    "family",
    "product_code",
    "qal_workflow",
    "source",
    "adjustable_radiance_range_uW_cm2_sr",
    "calibration_file_per_unit",
    "well_ids",
    "family_key",
)


def _validate_ret_wavelength_nm(wavelength_nm: int) -> int:
    value = int(wavelength_nm)
    if not _RET_WAVELENGTH_MIN_NM <= value <= _RET_WAVELENGTH_MAX_NM:
        raise ValueError(
            f"RET wavelength {value} nm is outside "
            f"{_RET_WAVELENGTH_MIN_NM}-{_RET_WAVELENGTH_MAX_NM} nm"
        )
    return value


def _ret_serial_wavelength_nm(preset: Mapping[str, Any]) -> Optional[int]:
    token = str(preset.get("serial_template") or "").split("-", 1)[0]
    if token.isdigit():
        return int(token)
    return None


def build_custom_ret_preset(
    wavelength_nm: int,
    *,
    form_code: str = _RET_DEFAULT_FORM_CODE,
    yaml_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Return a synthesized RET preset for a non-catalog wavelength."""
    wavelength_nm = _validate_ret_wavelength_nm(wavelength_nm)
    normalized_form = str(form_code or _RET_DEFAULT_FORM_CODE).strip().upper()
    if not normalized_form:
        normalized_form = _RET_DEFAULT_FORM_CODE

    catalog = load_product_family_presets(
        "radiometric_emitter",
        yaml_path=yaml_path,
    )
    template = next(iter(catalog.values()))
    preset = {
        key: template[key]
        for key in _RET_SHARED_KEYS
        if key in template
    }
    preset.update(
        {
            "tag": f"custom_{wavelength_nm}",
            "display_name": (
                f"Custom radiometric emitter target ({wavelength_nm} nm)"
            ),
            "form_code": normalized_form,
            "serial_template": f"{wavelength_nm}-{normalized_form}-RRLxxx",
            "peak_wavelength_nm": wavelength_nm,
            "old_sku": None,
            "current_sku": None,
            "sku_status": "not_in_catalog",
            "status": "custom",
        }
    )
    return preset


def resolve_ret_preset(
    wavelength_nm: int,
    *,
    form_code: str = _RET_DEFAULT_FORM_CODE,
    yaml_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Return a catalog RET preset, or a custom one if the wavelength is new."""
    wavelength_nm = _validate_ret_wavelength_nm(wavelength_nm)
    catalog = load_product_family_presets(
        "radiometric_emitter",
        yaml_path=yaml_path,
    )
    for preset in catalog.values():
        if _ret_serial_wavelength_nm(preset) == wavelength_nm:
            return dict(preset)
    for preset in catalog.values():
        if preset.get("peak_wavelength_nm") == wavelength_nm:
            return dict(preset)
    return build_custom_ret_preset(
        wavelength_nm,
        form_code=form_code,
        yaml_path=yaml_path,
    )


def ret_preset_from_tag(
    tag: str,
    *,
    yaml_path: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    """Return a RET preset for a wavelength tag such as ``630`` or ``custom_630``."""
    text = str(tag).strip().casefold()
    if text.startswith("custom_"):
        text = text.removeprefix("custom_")
    if text.endswith("nm"):
        text = text[:-2].rstrip("-_ ")
    if not text.isdigit():
        return None
    try:
        return resolve_ret_preset(int(text), yaml_path=yaml_path)
    except ValueError:
        return None

