"""Prepare executable snippet files from the benchmark CSV.

Writes files to `data/snippets/{snippet_id}.(py|cpp)` and creates
`data/snippets/manifest.csv` with metadata needed for measured runs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

import feature_engineering


def main(csv_path: str = "eco_logic_synthetic_benchmark.csv"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    out_dir = Path("data") / "snippets"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["snippet_id", "path", "language", "input_scale_N", "hardware_tdp", "hardware_cores", "target_energy_joules"])
        for _, row in df.iterrows():
            sid = row.get("snippet_id") or ""
            code = row.get("source_code") or row.get("source") or row.get("code") or ""
            if not code or not isinstance(code, str):
                continue
            lang = feature_engineering.detect_language(code)
            ext = "py" if lang == "Python" else "cpp"
            fname = f"{sid}.{ext}"
            fpath = out_dir / fname
            with fpath.open("w", encoding="utf-8") as fh:
                fh.write(code)

            writer.writerow([sid, str(fpath), row.get("input_scale_N", ""), row.get("hardware_tdp", ""), row.get("hardware_cores", ""), row.get("target_energy_joules", "")])

    print(f"Wrote snippets to {out_dir} and manifest to {manifest_path}")


if __name__ == "__main__":
    main()
