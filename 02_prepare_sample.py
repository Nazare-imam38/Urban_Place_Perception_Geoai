# 02_prepare_sample.py
"""
Urban Place Perception - Human-Centered GeoAI
Pipeline Step 2 - Prepare Reproducible Image Sample
====================================================
Selects a reproducible random sample of street-level images for VLM inference.

Steps:
1. Reads data/metadata.csv (produced by Step 1).
2. Filters down to SAMPLE_SIZE rows (default 50) using the SEED constant
   to ensure full reproducibility.
3. Writes data/sample_metadata.csv - used by all downstream pipeline steps.

Run with:  python 02_prepare_sample.py
"""

import pandas as pd
from config import DATA_DIR, SAMPLE_SIZE, SEED

def main():
    meta_path = f"{DATA_DIR}/metadata.csv"
    out_path = f"{DATA_DIR}/sample_metadata.csv"
    df = pd.read_csv(meta_path)
    # reproducible random choice without replacement
    sampled = df.sample(n=SAMPLE_SIZE, random_state=SEED, replace=False)
    sampled.to_csv(out_path, index=False)
    print(f"[OK] Sampled {len(sampled)} rows -> {out_path}")

if __name__ == "__main__":
    main()
