"""Parse 3DZSignDB SiGML into TRAIL's symbolic phonological fields.

3DZSignDB stores expert-reviewed Algerian Sign Language lexical entries as
HamNoSys-derived SiGML.  This creates a transparent, auditable source
inventory for the proposal's h (handshape), l (location), mu (movement), and
o (orientation) fields.  It does *not* create labels for real KArSL/IshaRah
video: those require a separately validated visual alignment/calibration step.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as etree
from collections import Counter
from pathlib import Path


HANDSHAPE = {
    "hamfist", "hamflathand", "hamcee12", "hamceeall", "hamceeopen", "hamfinger2", "hamfinger23",
    "hamfinger2345", "hamfinger23spread", "hamindexfinger", "hammiddlefinger", "hampinch12",
    "hampinch12open", "hampinchall", "hampinky", "hamringfinger", "hamthumb", "hamthumbball",
}
LOCATION = {
    "hambetween", "hamcheek", "hamchest", "hamchin", "hamear", "hamearlobe", "hamelbowinside",
    "hamforehead", "hamhead", "hamheadtop", "hamlips", "hamneck", "hamnose", "hamnostrils",
    "hamshoulders", "hamshouldertop", "hamstomach", "hamunderchin", "hamupperarm",
}
ORIENTATION_PREFIXES = ("hampalm", "hamhandback", "hamorirelative", "hamfingernail", "hamfingerpad", "hamfingerside")
MOVEMENT_PREFIXES = ("hammove", "hamarc", "hamcircle", "hamclock", "hamellipse", "hamwavy", "hamswinging", "hamtwisting", "hambrushing", "hamcross", "hamalternatingmotion", "hamrepeat")


def _tags(path: Path) -> list[str]:
    root = etree.parse(path).getroot()
    return [node.tag.rsplit("}", 1)[-1] for node in root.iter()]


def _first(tags: list[str], values: set[str], fallback: str) -> str:
    return next((tag for tag in tags if tag in values), fallback)


def _joined(tags: list[str], predicate, fallback: str) -> str:
    found = sorted({tag for tag in tags if predicate(tag)})
    return "+".join(found) if found else fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Extracted 3DZSignDB root containing data/sigml.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.root / "data" / "sigml"
    paths = sorted(folder.glob("*.sigml"))
    if not paths:
        raise SystemExit(f"No SiGML files found under {folder}")
    rows = []
    for path in paths:
        tags = _tags(path)
        rows.append({
            "sign_id": path.stem,
            "sigml_path": str(path),
            "handshape_h": _first(tags, HANDSHAPE, "unknown"),
            "location_l": _first(tags, LOCATION, "neutral_or_unspecified"),
            "movement_mu": _joined(tags, lambda tag: tag.startswith(MOVEMENT_PREFIXES), "hold_or_unspecified"),
            "orientation_o": _joined(tags, lambda tag: tag.startswith(ORIENTATION_PREFIXES), "unspecified"),
            "all_manual_tags": "+".join(tag for tag in tags if tag.startswith("ham") and "nonmanual" not in tag),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    audit = {
        "entries": len(rows),
        "handshape_inventory": dict(Counter(row["handshape_h"] for row in rows)),
        "location_inventory": dict(Counter(row["location_l"] for row in rows)),
        "warning": "This is a notation-derived source lexicon. It is not a visual KArSL/IshaRah/Egyptian handshape-labelled dataset.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Parsed {len(rows)} SiGML entries into {args.output}")


if __name__ == "__main__":
    main()
