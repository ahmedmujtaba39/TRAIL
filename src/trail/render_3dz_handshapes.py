"""Render labelled avatar frames from 3DZSignDB SiGML for h-token pretraining.

The output is explicitly synthetic Algerian-Sign-Language data.  It may be
used to pretrain a HamNoSys-derived handshape recogniser, but every downstream
report must evaluate its transfer on held-out real KArSL before using it in
TRAIL.  No synthetic score is treated as evidence of real-language accuracy.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from urllib.parse import quote


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV from trail-prepare-3dz-phonology.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8765", help="Local server rooted at 3dzsigndb/web-simulator.")
    parser.add_argument("--frames-per-sign", type=int, default=3)
    parser.add_argument("--wait-seconds", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    try:
        from selenium import webdriver
        from selenium.webdriver.edge.options import Options
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install selenium to render 3DZ avatar examples.") from error
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8", newline="")))
    if args.limit is not None: rows = rows[:args.limit]
    if not rows: raise SystemExit("The phonology manifest is empty.")
    options = Options(); options.add_argument("--headless=new"); options.add_argument("--window-size=1024,768"); options.add_argument("--use-angle=swiftshader")
    driver = webdriver.Edge(options=options)
    output_rows = []
    try:
        driver.get(args.server.rstrip("/") + "/index.html")
        time.sleep(2.0)
        canvas = driver.find_element("css selector", "canvas.canvasAv")
        for index, row in enumerate(rows, 1):
            name = Path(row["sigml_path"]).name
            url = args.server.rstrip("/") + "/sigml/" + quote(name)
            driver.execute_script("CWASA.playSiGMLURL(arguments[0]);", url)
            for frame in range(args.frames_per_sign):
                time.sleep(args.wait_seconds)
                destination = args.output / row["handshape_h"] / f"{row['sign_id']}_{frame:02d}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                canvas.screenshot(str(destination))
                output_rows.append({"image_path": str(destination), "sign_id": row["sign_id"], "handshape_h": row["handshape_h"], "source": "3dz_avatar"})
            print(f"[{index}/{len(rows)}] {row['sign_id']}: {row['handshape_h']}")
    finally:
        driver.quit()
    manifest = args.output / "render_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "sign_id", "handshape_h", "source"])
        writer.writeheader(); writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} synthetic frames and {manifest}")


if __name__ == "__main__":
    main()
