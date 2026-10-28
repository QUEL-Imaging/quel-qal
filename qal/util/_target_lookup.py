"""Select QUEL product presets from catalog SKUs or unit serials."""

from __future__ import annotations

import re
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from qal.data._target_presets import (
    iter_product_presets,
    list_product_families,
    normalize_product_family,
    resolve_ret_preset,
)

_FAMILY_CODES = {
    "RCS": "concentration",
    "CONCENTRATION": "concentration",
    "RDS": "depth",
    "DEPTH": "depth",
    "RRT": "resolution",
    "RESOLUTION": "resolution",
    "RUD": "uniformity_distortion",
    "RET": "radiometric_emitter",
    "RRL": "radiometric_emitter",
    "RADIOMETRIC": "radiometric_emitter",
    "RKT": "reference_sets",
    "KIT": "reference_sets",
    "DR": "depth_resolution",
}
_FLUOROPHORE_CODES = {
    "ICG": "ICG",
    "ICG-EQUIVALENT": "ICG",
    "ICG-EQ": "ICG",
    "IC1": "ICG",
    "80Q": "80Q",
    "Q800": "80Q",
    "Q800-01": "80Q",
    "Q81": "80Q",
    "70Q": "70Q",
    "Q700": "70Q",
    "Q700-01": "70Q",
    "Q71": "70Q",
    "O38": "O38",
    "038": "O38",
    "OTL38": "O38",
    "OTL-38": "O38",
    "MBL": "MBL",
    "MB": "MBL",
    "METHYLENE": "MBL",
    "METHYLENEBLUE": "MBL",
    "S800": "S800",
    "S8": "S800",
}
_SIZE_CODES = {
    "ST": "standard",
    "STD": "standard",
    "STANDARD": "standard",
    "LU": "standard",
    "SM": "mini",
    "MIN": "mini",
    "MINI": "mini",
    "LM": "mini",
}
_IDENTIFIER_FIELDS = (
    "old_sku",
    "current_sku",
    "serial_template",
    "legacy_serial_template",
)
_IDENTIFIER_LIST_FIELDS = (
    "alternate_skus",
    "superseded_skus",
    "identifiers",
    "legacy_serial_templates",
)
_LABEL_FIELDS = (
    "well_ids",
    "fluorophore_label",
    "fluorophore_code",
    "display_name",
    "old_sku",
    "current_sku",
    "qal_graph_type",
    "chart",
    "groups",
    "channel_nm",
    "peak_wavelength_nm",
    "well_diameter_mm",
    "status",
    "replaced_by",
    "notes",
)
_UNIT_FAMILY_PREFIXES = ("RCS", "RDS", "RRT", "RRS", "RRL", "RUD", "RET")
_TRAILING_FAMILY_CODES = {
    "RCS": "concentration",
    "RDS": "depth",
    "RRT": "resolution",
    "RRS": "resolution",
}
_VERSION_PREFIXES = {"V1", "V2", "V3", "V3PROTO"}
_RET_SERIAL_PATTERN = re.compile(
    r"^(?P<nm>\d{3,4})-(?P<form>[A-Z]{2}\d{2})-RRL(?:XXX|\d+)?$"
)
_RET_OLD_SKU_PATTERN = re.compile(
    r"^RRL-(?P<nm>\d{3,4})-(?P<form>[A-Z]{2}\d{2})(?:-QUEL\d+)?$"
)
_RET_CURRENT_SKU_PATTERN = re.compile(
    r"^S-RET-(?:PWR|PLG)-(?P<nm>\d{3,4})(?:-RA)?$"
)


def _normalize(value: str) -> str:
    text = str(value).strip().upper().replace("_", "-")
    compact: list[str] = []
    previous_dash = False
    for char in text:
        if char.isalnum():
            compact.append(char)
            previous_dash = False
        elif not previous_dash:
            compact.append("-")
            previous_dash = True
    return "".join(compact).strip("-")


def _strip_unit(text: str) -> str:
    if not text:
        return text
    parts = text.split("-")
    last = parts[-1]
    if last in {"XXX", "XX", "X", "PXX"}:
        return "-".join(parts[:-1])
    if last.isdigit() and len(last) >= 2:
        return "-".join(parts[:-1])
    if len(last) >= 4 and last[0] == "P" and last[1:].isdigit():
        return "-".join(parts[:-1])
    for prefix in _UNIT_FAMILY_PREFIXES:
        if not last.startswith(prefix):
            continue
        rest = last[len(prefix):]
        if rest.isdigit() or rest in {"XXX", "XX", "X"}:
            parts[-1] = prefix
            return "-".join(parts)
    return text


def _stems_from(value: str) -> set[str]:
    text = _normalize(value)
    if not text:
        return set()
    return {stem for stem in {text, _strip_unit(text)} if stem}


def _catalog_sku(preset: Mapping[str, Any]) -> Any:
    return preset.get("old_sku")


def _parse_ret_query(
    query: str,
    *,
    family_key: Optional[str] = None,
) -> Optional[tuple[int, str]]:
    if family_key is not None and family_key != "radiometric_emitter":
        return None
    text = _normalize(query)
    for pattern in (
        _RET_SERIAL_PATTERN,
        _RET_OLD_SKU_PATTERN,
        _RET_CURRENT_SKU_PATTERN,
    ):
        match = pattern.fullmatch(text)
        if match is None:
            continue
        form = match.groupdict().get("form") or "PW01"
        return int(match.group("nm")), form
    if (
        family_key == "radiometric_emitter"
        and text.isdigit()
        and len(text) in {3, 4}
    ):
        return int(text), "PW01"
    return None


def _iter_identifier_values(preset: Mapping[str, Any]):
    for field in _IDENTIFIER_FIELDS:
        value = preset.get(field)
        if isinstance(value, (list, tuple)):
            yield from (str(item) for item in value if item)
        elif value:
            yield str(value)
    for field in _IDENTIFIER_LIST_FIELDS:
        yield from (str(item) for item in preset.get(field) or [])


def _preset_stems(preset: Mapping[str, Any]) -> set[str]:
    stems: set[str] = set()
    for value in _iter_identifier_values(preset):
        stems.update(_stems_from(value))
    return stems


def _split_trailing_family(token: str) -> Optional[str]:
    for code, family in _TRAILING_FAMILY_CODES.items():
        if token.startswith(code):
            rest = token[len(code):]
            if rest == "" or rest.isdigit() or rest in {"XXX", "XX", "X"}:
                return family
    return None


def _parse_shorthand_constraints(parts: list[str]) -> dict[str, str]:
    if not parts or (
        parts[0] not in _VERSION_PREFIXES and parts[0] != "PR"
    ):
        return {}
    last = parts[-1]
    family = None
    if last == "CU":
        family = "depth"
    elif last.startswith("C"):
        family = "concentration"
    elif last.startswith("D"):
        family = "depth"
    elif last == "RNA" or last.startswith("R"):
        family = "resolution"
    if family is None:
        return {}
    fluorophore = "MBL" if any(
        part in {"MBL", "Q1"} or (part.startswith("Q") and part[1:].isdigit())
        for part in parts
    ) else "ICG"
    return {
        "family_key": family,
        "fluorophore_code": fluorophore,
        "size": "standard",
    }


def _parse_constraints(query: str) -> dict[str, str]:
    parts = [part for part in _normalize(query).split("-") if part]
    constraints: dict[str, str] = {}
    if not parts:
        return constraints

    trailing_family = False
    family = _FAMILY_CODES.get(parts[0])
    if family is not None:
        constraints["family_key"] = family
        parts = parts[1:]
        if len(parts) >= 2 and parts[0] in {"S", "M"}:
            constraints["size"] = "standard" if parts[0] == "S" else "mini"
            fluorophore = _FLUOROPHORE_CODES.get(parts[1])
            if fluorophore is not None:
                constraints["fluorophore_code"] = fluorophore
            return constraints
    else:
        trailing = _split_trailing_family(parts[-1])
        if trailing is not None:
            constraints["family_key"] = trailing
            trailing_family = True
            parts = parts[:-1]
        if parts and parts[0] in _VERSION_PREFIXES:
            parts = parts[1:]

    if parts:
        fluorophore = _FLUOROPHORE_CODES.get(parts[0])
        if fluorophore is not None:
            constraints["fluorophore_code"] = fluorophore
            parts = parts[1:]
    if trailing_family and "fluorophore_code" not in constraints:
        constraints["fluorophore_code"] = "ICG"

    if parts:
        token = parts[0]
        size = _SIZE_CODES.get(token) or _SIZE_CODES.get(token[:2])
        if size is not None:
            constraints["size"] = size
        if not trailing_family and token not in _FLUOROPHORE_CODES:
            constraints["form_prefix"] = token

    if "family_key" not in constraints:
        shorthand = _parse_shorthand_constraints(
            [part for part in _normalize(query).split("-") if part]
        )
        if shorthand:
            return shorthand
    return constraints


def _constraint_match(
    preset: Mapping[str, Any],
    constraints: Mapping[str, str],
) -> bool:
    if not constraints:
        return False
    if (
        "family_key" in constraints
        and preset.get("family_key") != constraints["family_key"]
    ):
        return False
    if (
        "fluorophore_code" in constraints
        and str(preset.get("fluorophore_code") or "").upper()
        != constraints["fluorophore_code"]
    ):
        return False
    if "size" in constraints and preset.get("size") != constraints["size"]:
        return False
    form_prefix = constraints.get("form_prefix")
    if form_prefix:
        form_code = _normalize(str(preset.get("form_code") or ""))
        sku_tokens = _normalize(str(_catalog_sku(preset) or "")).split("-")
        form_token = sku_tokens[2] if len(sku_tokens) > 2 else ""
        if not (
            form_code.startswith(form_prefix)
            or form_token.startswith(form_prefix)
        ):
            return False
    return True


def _exact_match(query: str, preset: Mapping[str, Any]) -> bool:
    normalized = _normalize(query)
    for value in _iter_identifier_values(preset):
        if normalized == _normalize(value):
            return True
    return str(preset.get("tag") or "").casefold() == query.strip().casefold()


def _serial_match(query: str, preset: Mapping[str, Any]) -> bool:
    preset_stems = _preset_stems(preset)
    if _stems_from(query) & preset_stems:
        return True
    stripped_query = _strip_unit(_normalize(query))
    if len(stripped_query) < 3:
        return False
    return any(
        stripped_query.startswith(stem) and len(stem) >= 6
        for stem in preset_stems
    )


def _prefix_match(query: str, preset: Mapping[str, Any]) -> bool:
    stripped_query = _strip_unit(_normalize(query))
    if len(stripped_query) < 3:
        return False
    return any(
        stem.startswith(stripped_query) for stem in _preset_stems(preset)
    )


def _attribute_match(query: str, preset: Mapping[str, Any]) -> bool:
    return _constraint_match(preset, _parse_constraints(query))


_MATCH_STAGES = (
    _exact_match,
    _serial_match,
    _prefix_match,
    _attribute_match,
)


def _format_match(preset: Mapping[str, Any]) -> str:
    sku = _catalog_sku(preset) or preset.get("current_sku") or preset.get("tag")
    return f"{preset.get('family_key')}/{preset.get('tag')} ({sku})"


def _matched_presets(
    query: str,
    *,
    family: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return presets from the strongest matching stage.

    Stages, strongest first:
    1. Exact identifier (catalog SKU, current SKU, listed identifier, or tag)
    2. Serial template (unit number stripped)
    3. Identifier prefix
    4. Attribute filter (family, fluorophore, size, form)
    5. Custom RET serial or wavelength (non-catalog units)
    """
    text = str(query).strip()
    if not text:
        raise ValueError(
            "query must be a non-empty serial number or SKU fragment"
        )

    family_key = (
        normalize_product_family(family) if family is not None else None
    )
    candidates = [
        preset
        for preset in iter_product_presets()
        if family_key is None or preset.get("family_key") == family_key
    ]
    for matcher in _MATCH_STAGES:
        matches = [
            dict(preset) for preset in candidates if matcher(text, preset)
        ]
        if matches:
            matches.sort(
                key=lambda preset: (
                    str(preset.get("family_key") or ""),
                    str(preset.get("tag") or ""),
                )
            )
            return matches
    parsed_ret = _parse_ret_query(text, family_key=family_key)
    if parsed_ret is None:
        return []
    wavelength_nm, form_code = parsed_ret
    try:
        return [resolve_ret_preset(wavelength_nm, form_code=form_code)]
    except ValueError:
        return []


def find_target_presets(
    query: str,
    *,
    family: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return product presets that match a SKU, serial, or fragment."""
    return _matched_presets(query, family=family)


def lookup_target_preset(
    query: str,
    *,
    family: Optional[str] = None,
) -> dict[str, Any]:
    """Return the unique product preset matching a SKU, serial, or fragment."""
    matches = _matched_presets(query, family=family)
    if not matches:
        hint = f" in family {family!r}" if family is not None else ""
        raise ValueError(f"No product preset matches {query!r}{hint}")
    if len(matches) != 1:
        options = ", ".join(_format_match(preset) for preset in matches)
        raise ValueError(
            f"Ambiguous target query {query!r}; matches: {options}"
        )
    return dict(matches[0])


def lookup_concentration_preset(query: str) -> dict[str, Any]:
    """Return the concentration preset matching a SKU, serial, or fragment."""
    preset = lookup_target_preset(query, family="concentration")
    if not preset.get("well_ids"):
        raise ValueError(
            f"Concentration preset {preset.get('tag')!r} is missing well_ids"
        )
    return preset


def target_data_labels(preset: Mapping[str, Any]) -> dict[str, Any]:
    """Return well IDs and other labels used to annotate analysis output."""
    return {
        field: preset[field]
        for field in _LABEL_FIELDS
        if field in preset and preset[field] is not None
    }


def _print_preset(preset: Mapping[str, Any]) -> None:
    labels = target_data_labels(preset)
    print(preset.get("display_name") or preset.get("tag"))
    print(
        f"  family: {preset.get('family_key')} ({preset.get('product_code')})"
    )
    print(f"  tag: {preset.get('tag')}")
    for field in ("old_sku", "current_sku"):
        value = preset.get(field)
        if value:
            print(f"  {field}: {value}")
    well_ids = labels.get("well_ids")
    if isinstance(well_ids, Sequence) and not isinstance(well_ids, str):
        print("  well_ids:")
        for well_id in well_ids:
            print(f"    - {well_id}")
    for field in (
        "fluorophore_label",
        "qal_graph_type",
        "chart",
        "groups",
        "channel_nm",
        "peak_wavelength_nm",
        "well_diameter_mm",
        "status",
        "replaced_by",
        "notes",
    ):
        if field in labels:
            print(f"  {field}: {labels[field]}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = ArgumentParser(
        description=(
            "Look up QUEL product presets from a catalog SKU, unit serial, "
            "or unique fragment."
        )
    )
    parser.add_argument(
        "query",
        help="SKU or serial such as RCS-ICG-ST01-QUEL03 or RDS-S-Q81-D2-A-007",
    )
    parser.add_argument(
        "--family",
        choices=list_product_families(),
        help="Limit matches to one product family",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print every match instead of requiring a unique preset",
    )
    args = parser.parse_args(argv)

    if args.all:
        matches = find_target_presets(args.query, family=args.family)
        if not matches:
            raise SystemExit(f"No product preset matches {args.query!r}")
        for preset in matches:
            _print_preset(preset)
            print()
        return

    _print_preset(lookup_target_preset(args.query, family=args.family))


if __name__ == "__main__":
    main()
