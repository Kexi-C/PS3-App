#!/usr/bin/env python3
"""Python 3.10+; standard library only. No pip installation is needed.

Usage:
    python acv_rank.py acv_case_01.xlsx
    python acv_rank.py acv_test_case.xlsx --output acv_predictions.csv
    python acv_rank.py acv_case_01.xlsx --sheet Sheet1

The first worksheet is used unless --sheet is provided. Headers must be in
its first row. Train length and car IDs are detected from those headers.
This is the compact-schema algorithm, not a case-04 adapter.
"""

import argparse
import csv
import posixpath
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

PARAMETERS = (
    "Indoor Average Temperature",
    "ACV Control Temperature (Cooling)",
    "ACV Running Mode",
    "ACV Information Valid",
)


# ---------------------------------------------------------------------------
# XLSX input adapter. Numeric cells are read without relying on display format.
# Supports ordinary Excel numeric, shared-string and inline-string cells.
# Formula cells require cached values: this reader does not evaluate formulas.
# ---------------------------------------------------------------------------
def read_xlsx_rows(
    path: Path, sheet_name: str | None = None
) -> Iterator[tuple[int, list[object]]]:
    """Yield (Excel row number, cell values), preserving empty cell positions."""
    if path.suffix.lower() != ".xlsx":
        raise ValueError("Input must be an .xlsx file.")

    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("{*}sheets/{*}sheet")
        selected = next(
            (s for s in sheets if sheet_name is None or s.get("name") == sheet_name),
            None,
        )
        if selected is None:
            raise ValueError(f"Worksheet not found: {sheet_name!r}")

        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        rel_id = next(
            value for key, value in selected.attrib.items()
            if key.rsplit("}", 1)[-1] == "id"
        )

        def member_path(target: str) -> str:
            return posixpath.normpath(
                target.lstrip("/") if target.startswith("/")
                else posixpath.join("xl", target)
            )

        sheet_rel = next(r for r in relationships if r.get("Id") == rel_id)
        if sheet_rel.get("TargetMode") == "External":
            raise ValueError("External worksheets are not supported.")
        worksheet_path = member_path(sheet_rel.attrib["Target"])

        shared_strings = []
        for rel in relationships:
            if rel.get("Type", "").endswith("/sharedStrings"):
                strings_root = ET.fromstring(
                    archive.read(member_path(rel.attrib["Target"]))
                )
                shared_strings = [
                    "".join(node.text or "" for node in item.findall(".//{*}t"))
                    for item in strings_root.findall("{*}si")
                ]
                break

        with archive.open(worksheet_path) as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag.rsplit("}", 1)[-1] != "row":
                    continue

                row = []
                for cell in element.findall("{*}c"):
                    letters = re.match(r"[A-Z]+", cell.attrib["r"])
                    if letters is None:
                        raise ValueError("Invalid Excel cell reference.")
                    column = 0
                    for letter in letters.group():
                        column = column * 26 + ord(letter) - ord("A") + 1
                    column -= 1
                    while len(row) <= column:
                        row.append(None)

                    kind = cell.get("t")
                    value = cell.find("{*}v")
                    if kind == "inlineStr":
                        row[column] = "".join(
                            n.text or "" for n in cell.findall("{*}is//{*}t")
                        )
                    elif value is not None and value.text is not None:
                        if kind == "s":
                            row[column] = shared_strings[int(value.text)]
                        elif kind == "b":
                            row[column] = value.text == "1"
                        else:
                            row[column] = value.text

                yield int(element.attrib["r"]), row
                element.clear()


# ---------------------------------------------------------------------------
# Algorithm: independent Valid + Automatic Cooling filter for each car,
# sum positive doubled-temperature errors, divide by that car's valid count.
# ---------------------------------------------------------------------------
def temperature_times_two(value: object) -> int:
    """25 -> 50; 25.5 -> 51. Accept numeric cells and numeric text.
    Rejects missing/non-numeric values and values off the 0.5-degree grid.
    """
    try:
        doubled = Decimal(str(value)) * 2
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid temperature: {value!r}") from exc

    if not doubled.is_finite() or doubled != doubled.to_integral_value():
        raise ValueError(f"Expected an integer or half-degree temperature: {value!r}")
    return int(doubled)


@dataclass
class Result:
    totals: list[int]        # Sum of positive errors in doubled units.
    valid_counts: list[int]  # Includes eligible observations with zero error.
    scores: list[float]      # doubled scale
    ranking: list[int]       # Actual car numbers, most suspicious first.
    car_ids: list[str]       # Exact header IDs, in the same order as score arrays.

    @property
    def train_length(self) -> int:
        return len(self.car_ids)

    @property
    def ranked_car_ids(self) -> list[str]:
        """Return ranked IDs with the exact formatting used in the headers."""
        original_ids = {int(car_id): car_id for car_id in self.car_ids}
        return [original_ids[car] for car in self.ranking]


def score_rows(rows: Iterable[tuple[int, list[object]]]) -> Result:
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        raise ValueError("Worksheet is empty.")
    _, headers = first

    # Discover every distinct car ID from the selected worksheet's headers.
    # Count IDs instead of using the largest ID, because IDs may have gaps.
    car_columns: dict[str, dict[str, int]] = {}
    for column, header in enumerate(headers):
        match = re.fullmatch(r"Car (\d+) - (.+)", str(header))
        if not match:
            continue
        car_id, parameter = match.groups()
        columns = car_columns.setdefault(car_id, {})
        if parameter in PARAMETERS:
            if parameter in columns:
                raise ValueError(f"Duplicate header: {header}")
            columns[parameter] = column

    if not car_columns:
        raise ValueError(
            "No car headers found. Expected headers such as "
            "'Car 01 - Indoor Average Temperature' in the first row."
        )

    # Numeric ordering matches the original Java car order, even when the
    # worksheet columns are reordered or use nonconsecutive car numbers.
    car_ids = sorted(car_columns, key=int)
    if len({int(car_id) for car_id in car_ids}) != len(car_ids):
        raise ValueError(
            "Ambiguous car IDs: the same car number has multiple formats "
            "(for example, '1' and '01'). Use one format per car."
        )
    train_length = len(car_ids)

    missing = [
        f"Car {car_id} - {parameter}"
        for car_id in car_ids
        for parameter in PARAMETERS
        if parameter not in car_columns[car_id]
    ]
    if missing:
        raise ValueError("Missing required headers:\n  " + "\n  ".join(missing))

    # Equivalent to Java's index[4][trainLength], sized from the headers.
    index = [
        [car_columns[car_id][parameter] for car_id in car_ids]
        for parameter in PARAMETERS
    ]
    total = [0] * train_length
    valcnt = [0] * train_length
    width = max(max(columns) for columns in index) + 1

    for row_number, row in rows:
        # Empty trailing Excel cells may be absent from the stored row.
        row = list(row) + [None] * max(0, width - len(row))
        for car in range(train_length):
            # Exact, case-sensitive string comparisons, as in the Java code.
            if (
                row[index[3][car]] == "Valid"
                and row[index[2][car]] == "Automatic Cooling"
            ):
                try:
                    indoor = temperature_times_two(row[index[0][car]])
                    target = temperature_times_two(row[index[1][car]])
                except ValueError as exc:
                    raise ValueError(
                        f"Row {row_number}, Car {car_ids[car]}: {exc}"
                    ) from exc

                total[car] += max(0, indoor - target)
                valcnt[car] += 1  # Count zero-error observations too.

    empty = [car_ids[car] for car, count in enumerate(valcnt) if count == 0]
    if empty:
        raise ValueError("No eligible observations for car(s): " + ", ".join(empty))

    scores = [total[car] / valcnt[car] for car in range(train_length)]

    order = sorted(range(train_length), key=lambda car: scores[car])
    order.reverse()
    return Result(
        totals=total,
        valid_counts=valcnt,
        scores=scores,
        ranking=[int(car_ids[car]) for car in order],
        car_ids=car_ids,
    )


def rank_workbook(path: Path, sheet_name: str | None = None) -> Result:
    rows = read_xlsx_rows(path, sheet_name)
    try:
        return score_rows(rows)
    finally:
        rows.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xlsx_file", type=Path, help="Input .xlsx workbook")
    parser.add_argument("--sheet", help="Worksheet name; default: first worksheet")
    parser.add_argument("--output", type=Path, help="Optional prediction CSV path")
    args = parser.parse_args()

    try:
        result = rank_workbook(args.xlsx_file, args.sheet)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["file_id", "ranked_cars"])
                writer.writerow([
                    args.xlsx_file.name,
                    "|".join(result.ranked_car_ids),
                ])
    except (OSError, ValueError, BadZipFile, ET.ParseError, KeyError, StopIteration) as exc:
        parser.exit(1, f"Error: {exc}\n")


    print(f"\nRank positions for dataset {args.xlsx_file}:")
    print(" ".join(map(str, result.ranking)))


if __name__ == "__main__":
    main()
