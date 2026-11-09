# 03_run_model_inference.py
"""
Urban Place Perception - Human-Centered GeoAI
Pipeline Step 3 - Florence-2 VLM Inference
===========================================
Runs the Florence-2 Vision-Language Model on each sampled street-level image,
producing per-image perception scores with human-readable evidence and
reasoning sentences for full interpretability.

The script respects a ``--debug`` flag (or the ``DEBUG`` constant in ``config``)
to allow deterministic fallback scores when the model cannot be loaded.
If ``--debug`` is *off* and model loading fails, the script aborts *without*
writing or overwriting ``results.csv`` and prints the required error message.

Output: outputs/vlm/results.csv

Run with:  python 03_run_model_inference.py [--debug]
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR, DEBUG as CONFIG_DEBUG
from utils.inference import infer_image

parser = argparse.ArgumentParser(description='Run model inference')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with deterministic fallback')
args = parser.parse_args()

debug = args.debug or CONFIG_DEBUG


# Output CSV column names - include explanation column
INDICATORS = ["safety", "walkability", "greenery", "visual_quality", "explanation"]


def main(debug: bool):
    sample_path = Path(DATA_DIR) / "sample_metadata.csv"
    out_path = Path(OUTPUT_DIR) / "results.csv"

    df = pd.read_csv(sample_path)
    results = []
    try:
        for _, row in df.iterrows():
            img_path = Path(row["filepath"])
            # Pass debug flag to inference utility
            scores = infer_image(img_path, debug=debug)
            result_row = {
                "image_id": row["image_id"],
                "filepath": row["filepath"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
            }
            result_row.update(scores)
            results.append(result_row)
    except RuntimeError as e:
        # Abort without writing CSV when not in debug mode
        if not debug:
            print("Model inference failed. No final results.csv was written.")
            if out_path.exists():
                out_path.unlink()
            sys.exit(1)
        else:
            # In debug mode re‑raise to surface unexpected errors
            raise

    # Write results only if loop completed successfully
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"[OK] Inference completed -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference for urban place perception.")
    parser.add_argument("--debug", action="store_true", help="Enable deterministic fallback scores if model fails.")
    args = parser.parse_args()
    # Command‑line flag overrides config constant
    use_debug = args.debug or CONFIG_DEBUG
    main(debug=use_debug)
