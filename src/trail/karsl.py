"""KArSL label handling and conservative Arabic lexical matching.

The KArSL release publishes its 502 label names in an ``.xlsx`` workbook.
This module reads the workbook with the Python standard library so the data
pipeline does not depend on a desktop Excel installation.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")


def normalize_arabic(text: str) -> str:
    """Normalize only spelling variants, never semantic variants.

    This deliberately keeps token boundaries and does *not* split slash
    alternatives in KArSL labels.  Candidates therefore remain auditable.
    """

    text = _DIACRITICS.sub("", text.strip())
    text = text.replace("ـ", "")
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}))
    return " ".join(text.split())


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.itertext()) for node in root.findall("x:si", _NS)]


def load_labels(path: Path) -> dict[int, str]:
    """Return ``{sign_id: Arabic label}`` from KARSL-502_Labels.xlsx."""

    with zipfile.ZipFile(path) as archive:
        strings = _shared_strings(archive)
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    labels: dict[int, str] = {}
    for row in root.findall(".//x:sheetData/x:row", _NS):
        values: dict[str, str] = {}
        for cell in row.findall("x:c", _NS):
            ref = cell.attrib.get("r", "")
            column = re.match(r"[A-Z]+", ref)
            value = cell.findtext("x:v", default="", namespaces=_NS)
            if not column:
                continue
            if cell.attrib.get("t") == "s" and value:
                value = strings[int(value)]
            values[column.group()] = value
        try:
            sign_id = int(values.get("A", ""))
        except ValueError:
            continue
        label = values.get("B", "").strip()
        if label:
            labels[sign_id] = label
    if len(labels) != 502:
        raise ValueError(f"Expected 502 KArSL labels, found {len(labels)}")
    return labels


def exact_normalized_index(labels: dict[int, str]) -> dict[str, int]:
    """Index unambiguous labels only; ambiguous normalized labels are dropped."""

    candidates: dict[str, list[int]] = {}
    for sign_id, label in labels.items():
        candidates.setdefault(normalize_arabic(label), []).append(sign_id)
    return {key: ids[0] for key, ids in candidates.items() if len(ids) == 1}
