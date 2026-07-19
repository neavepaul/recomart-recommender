"""Robust parsing for RetailRocket item-property CSV files."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

EXPECTED_HEADER = ["timestamp", "itemid", "property", "value"]


def read_item_properties(path: Path) -> Iterator[dict[str, str]]:
    """Read item properties without losing commas from the value column.

    Proper CSV quoting is handled by ``csv.reader``. If a producer emits an
    unquoted comma in the final value field, all fields from position four
    onward are joined back together because ``value`` is the last logical
    column in this dataset.
    """
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty item-property CSV: {path}") from None
        if header != EXPECTED_HEADER:
            raise ValueError(
                f"Unexpected header in {path}: {header!r}; expected {EXPECTED_HEADER!r}"
            )
        for line_number, fields in enumerate(reader, 2):
            if len(fields) < 4:
                raise ValueError(
                    f"Malformed item-property row in {path} at line {line_number}: "
                    f"expected at least 4 fields, found {len(fields)}"
                )
            yield {
                "timestamp": fields[0],
                "itemid": fields[1],
                "property": fields[2],
                "value": ",".join(fields[3:]),
            }

