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
    manifest = args.output / "render_manifest.csv"
    if manifest.exists():
        output_rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    else:
        # A previous interrupted renderer may have completed PNGs before it
        # could write its manifest. Recover those frames without rerendering.
        output_rows = [
            {"image_path": str(path), "sign_id": path.stem.rsplit("_", 1)[0], "handshape_h": path.parent.name, "source": "3dz_avatar"}
            for path in args.output.glob("*/*.png")
        ]
    completed = {(row["sign_id"], row["image_path"]) for row in output_rows}
    new_manifest = not manifest.exists()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_handle = manifest.open("a", encoding="utf-8", newline="")
    manifest_writer = csv.DictWriter(manifest_handle, fieldnames=["image_path", "sign_id", "handshape_h", "source"])
    if new_manifest:
        manifest_writer.writeheader()
        manifest_writer.writerows(output_rows); manifest_handle.flush()
    driver = webdriver.Edge(options=options)
    try:
        driver.get(args.server.rstrip("/") + "/index.html")
        time.sleep(2.0)
        canvas = driver.find_element("css selector", "canvas.canvasAv")
        for index, row in enumerate(rows, 1):
            destinations = [args.output / row["handshape_h"] / f"{row['sign_id']}_{frame:02d}.png" for frame in range(args.frames_per_sign)]
            if all((row["sign_id"], str(destination)) in completed and destination.exists() for destination in destinations):
                print(f"[{index}/{len(rows)}] class={row['handshape_h']}: already saved")
                continue
            name = Path(row["sigml_path"]).name
            url = args.server.rstrip("/") + "/sigml/" + quote(name)
            driver.execute_script("CWASA.playSiGMLURL(arguments[0]);", url)
            for frame in range(args.frames_per_sign):
                time.sleep(args.wait_seconds)
                destination = destinations[frame]
                record = {"image_path": str(destination), "sign_id": row["sign_id"], "handshape_h": row["handshape_h"], "source": "3dz_avatar"}
                if (row["sign_id"], str(destination)) in completed and destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                canvas.screenshot(str(destination))
                output_rows.append(record); manifest_writer.writerow(record); manifest_handle.flush()
            # Avoid printing Arabic file names to a Windows cp1252 console.
            print(f"[{index}/{len(rows)}] class={row['handshape_h']}")
    finally:
        driver.quit()
        manifest_handle.close()
    print(f"Wrote {len(output_rows)} synthetic frames and {manifest}")


if __name__ == "__main__":
    main()
