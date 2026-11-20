"""Render one clinical event as a text fragment, e.g. "Heart rate: 88 bpm"."""

import re

import numpy as np

from clin_jepa.data.helpers.schema import CONCEPT_SOURCE
from clin_jepa.data.helpers.source import CONFIG_ROOT
from clin_jepa.utils import load_config


def _display_names() -> dict[tuple[str, str], str]:
    """Display names of the WeavEHR concepts, as configs/data/features.yaml lists them."""
    features = load_config(CONFIG_ROOT / "features.yaml")
    rules = features["observations"] + features["actions"]
    return {(CONCEPT_SOURCE, rule.get("name", rule["concept"])): rule["display_name"] for rule in rules}


DISPLAY_NAMES: dict[tuple[str, str], str] = _display_names()


_RATE_PATTERN = re.compile(r"rate:([\d.]+)\s*([^|]*)")


def get_display_name(source_table: str, variable_name: str) -> str:
    """Look up human-readable display name for a variable."""
    name = DISPLAY_NAMES.get((source_table, variable_name))
    if name is not None:
        return name
    return variable_name.replace("_", " ").strip()


def format_numeric(value: int | float) -> str:
    """Integers as integers, other values with two decimals and no trailing zeros."""
    if value == int(value):
        return str(int(value))
    formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return formatted


def parse_raw_text_value(
    variable_name: str,
    numeric_value: float | None,
    text_value: str,
    unit: str,
    source_table: str,
) -> str:
    """Parse structured text_value from raw tables into natural language."""
    name = get_display_name(source_table, variable_name)
    parts = [name]

    has_numeric = numeric_value is not None and not (
        isinstance(numeric_value, float) and np.isnan(numeric_value)
    )

    if source_table == "inputevents":
        if has_numeric:
            parts.append(f": {format_numeric(numeric_value)}")
            if unit:
                parts.append(f" {unit}")

        if text_value:
            rate_match = _RATE_PATTERN.search(text_value)
            if rate_match:
                try:
                    rate_val = format_numeric(float(rate_match.group(1).strip()))
                except ValueError:
                    rate_val = rate_match.group(1).strip()
                rate_unit = rate_match.group(2).strip()
                parts.append(f", {rate_val}")
                if rate_unit:
                    parts.append(f" {rate_unit}")

            if "|" in text_value:
                category = text_value.split("|", 1)[1].strip()
                if category:
                    parts.append(f" {category}")

    elif source_table == "prescriptions":
        route = ""
        strength = ""
        if text_value:
            if "route:" in text_value:
                after_route = text_value.split("route:", 1)[1]
                if "|" in after_route:
                    route = after_route.split("|", 1)[0].strip()
                    strength = after_route.split("|", 1)[1].strip()
                else:
                    route = after_route.strip()
            elif "|" in text_value:
                strength = text_value.split("|", 1)[1].strip()
            else:
                strength = text_value.strip()

        if has_numeric:
            parts.append(f": {format_numeric(numeric_value)}")
            if unit:
                parts.append(f" {unit}")

        if strength:
            parts.append(f" {strength}")
        if route:
            parts.append(f" {route}")

    elif source_table == "emar":
        if text_value and text_value not in ("", "nan"):
            parts.append(f": {text_value}")

    return "".join(parts)


def format_event(
    source_table: str,
    variable_name: str,
    numeric_value: float | None,
    text_value: str | None,
    unit: str,
) -> str | None:
    """Convert a single event into a text fragment."""
    has_numeric = numeric_value is not None and not (
        isinstance(numeric_value, float) and np.isnan(numeric_value)
    )
    has_text = (
        text_value is not None
        and str(text_value) not in ("", "nan", "None")
    )

    if not has_numeric and not has_text:
        return None

    if source_table in ("inputevents", "prescriptions", "emar"):
        return parse_raw_text_value(
            variable_name, numeric_value,
            str(text_value) if has_text else "",
            unit, source_table,
        )

    name = get_display_name(source_table, variable_name)

    if has_numeric and has_text:
        text_str = str(text_value)
        if unit:
            return f"{name}: {format_numeric(numeric_value)} {unit} ({text_str})"
        return f"{name}: {format_numeric(numeric_value)} ({text_str})"

    if has_numeric:
        if unit:
            return f"{name}: {format_numeric(numeric_value)} {unit}"
        return f"{name}: {format_numeric(numeric_value)}"

    return f"{name}: {text_value}"


