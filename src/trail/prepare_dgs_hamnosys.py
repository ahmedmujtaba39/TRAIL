"""Export real-human Public DGS Corpus sign segments with their HamNoSys labels.

The public DGS iLex export contains three linked pieces of information:
``type`` entries carrying the corpus' HamNoSys citation form, ``token``
entries linking a time-aligned occurrence to a type, and lexical tier tags
linking that token to a recording interval.  This tool makes that join
explicit and auditable.  It deliberately retains the raw HamNoSys sequence:
mapping notation symbols to TRAIL's compact ``h`` inventory is a separate,
versioned step rather than an undocumented heuristic.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as etree
from collections import Counter
from pathlib import Path


def codepoints(value: str) -> str:
    """A portable representation for PUA HamNoSys symbols in CSV/JSON."""
    return " ".join(f"U+{ord(symbol):04X}" for symbol in value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ilex", type=Path, required=True, help="A Public DGS Corpus .ilex transcript export.")
    parser.add_argument("--output", type=Path, required=True, help="CSV of time-aligned lexical segments.")
    args = parser.parse_args()

    root = etree.parse(args.ilex).getroot()
    types = {
        item.attrib["id"]: item.attrib
        for item in root.findall("type")
        if item.attrib.get("hamnosys")
    }
    tokens = {
        item.attrib["id"]: item.attrib["type"]
        for item in root.findall("token")
        if item.attrib.get("type") in types
    }
    lexical_tiers = {
        item.attrib["id"]
        for item in root.findall("tier")
        if item.attrib.get("name", "").startswith("Lexem/Geb")
    }

    rows: list[dict[str, str]] = []
    for tag in root.findall("tag"):
        if tag.attrib.get("tier") not in lexical_tiers:
            continue
        type_id = tokens.get(tag.attrib.get("token_dom", ""))
        if type_id is None:
            continue
        lexical_type = types[type_id]
        hamnosys = lexical_type["hamnosys"]
        rows.append({
            "segment_id": tag.attrib["id"],
            "tier_id": tag.attrib["tier"],
            "token_id": tag.attrib["token_dom"],
            "type_id": type_id,
            "sign_name": lexical_type.get("name", ""),
            "english_name": lexical_type.get("english", ""),
            "timecode_start": tag.attrib["timecode_start"],
            "timecode_end": tag.attrib["timecode_end"],
            "hamnosys": hamnosys,
            "hamnosys_codepoints": codepoints(hamnosys),
        })
    if not rows:
        raise SystemExit("No lexical segments with HamNoSys-linked types were found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "source": str(args.ilex),
        "segments": len(rows),
        "unique_types": len({row["type_id"] for row in rows}),
        "unique_hamnosys_forms": len({row["hamnosys"] for row in rows}),
        "top_types": Counter(row["sign_name"] for row in rows).most_common(20),
        "label_contract": "Raw corpus HamNoSys strings are retained. They are source notation, not yet TRAIL h classes.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(rows)} DGS segments covering {audit['unique_types']} HamNoSys-labelled types.")


if __name__ == "__main__":
    main()
