"""JUMLA-QSL-22 manifest construction for the Saudi-to-Qatari pilot.

The dataset workbook is parsed without Excel or ``openpyxl`` so the audit can
run on a fresh Windows machine.  This module never copies raw videos: its
manifest stores absolute paths to the locally downloaded front-view ``rec0``
files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


@dataclass(frozen=True)
class JumlaRow:
    code: int
    arabic: str

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(token for token in self.arabic.split() if token)


def _cell_column(reference: str) -> str:
    return "".join(character for character in reference if character.isalpha())


def _shared_strings(archive: ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.itertext()) for node in root.findall(f"{NS}si")]


def read_workbook(path: Path) -> list[JumlaRow]:
    """Read JUMLA codes and Arabic intent text from its first worksheet."""
    with ZipFile(path) as archive:
        strings = _shared_strings(archive)
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[JumlaRow] = []
    for index, row in enumerate(sheet.findall(f".//{NS}row")):
        if index == 0:
            continue
        values: dict[str, str] = {}
        for cell in row.findall(f"{NS}c"):
            value = cell.findtext(f"{NS}v", default="")
            if cell.get("t") == "s" and value:
                value = strings[int(value)]
            values[_cell_column(cell.get("r", ""))] = value.strip()
        if not values.get("A") or not values.get("C"):
            continue
        rows.append(JumlaRow(code=int(float(values["A"])), arabic=values["C"]))
    if not rows:
        raise ValueError(f"No JUMLA rows were read from {path}")
    return rows


def folder_name(participant: str, code: int) -> str:
    # DataPort uses padded folder names for AS/MA but unpadded names for AT.
    suffix = f"{code:03d}" if participant in {"AS", "MA"} else str(code)
    return f"f_{participant}{suffix}"


def _records(rows: Iterable[JumlaRow], data_root: Path) -> list[dict[str, str]]:
    rows = list(rows)
    singleton_tokens = {row.arabic for row in rows if len(row.tokens) == 1}
    isolated = [row for row in rows if len(row.tokens) == 1]
    covered = [row for row in rows if row.tokens and all(token in singleton_tokens for token in row.tokens)]
    output: list[dict[str, str]] = []
    for row in isolated:
        clip = data_root / "Participant_AS" / folder_name("AS", row.code) / "rec0.svo"
        output.append(
            {
                "clip_id": f"AS_{row.code:03d}", "participant": "AS", "role": "proxy_isolated_lexicon",
                "code": str(row.code), "arabic": row.arabic, "tokens": " ".join(row.tokens),
                "video_path": str(clip), "available": str(clip.exists()).lower(),
            }
        )
    for participant in ("AT", "MA"):
        for row in covered:
            clip = data_root / f"Participant_{participant}" / folder_name(participant, row.code) / "rec0.svo"
            output.append(
                {
                    "clip_id": f"{participant}_{row.code:03d}", "participant": participant, "role": "heldout_continuous_test",
                    "code": str(row.code), "arabic": row.arabic, "tokens": " ".join(row.tokens),
                    "video_path": str(clip), "available": str(clip.exists()).lower(),
                }
            )
    return output


def build_manifest(workbook: Path, data_root: Path, output_csv: Path) -> dict[str, int]:
    records = _records(read_workbook(workbook), data_root)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["clip_id", "participant", "role", "code", "arabic", "tokens", "video_path", "available"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    counts = {
        "isolated_total": sum(item["role"] == "proxy_isolated_lexicon" for item in records),
        "isolated_available": sum(item["role"] == "proxy_isolated_lexicon" and item["available"] == "true" for item in records),
        "heldout_total": sum(item["role"] == "heldout_continuous_test" for item in records),
        "heldout_available": sum(item["role"] == "heldout_continuous_test" and item["available"] == "true" for item in records),
    }
    return counts
