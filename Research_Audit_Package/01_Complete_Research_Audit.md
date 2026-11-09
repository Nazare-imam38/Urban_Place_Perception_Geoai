# 01 — Complete Research Audit
## Urban Place Perception — Human-Centered GeoAI

**Audit conducted by:** Kiro AI Research Auditor  
**Date:** November 2026  
**Subject:** Nazar E Imam  
**Repository root:** `urban-place-perception-geoai/`

---

## 1. Project Identity and Objectives

| Property | Value |
|---|---|
| Project title | Urban Place Perception — Human-Centered GeoAI |
| Version | 1.0 |
| Scientific domain | GeoAI, Urban Analytics, Explainable AI, Computer Vision |
| Primary objective | Infer explainable multi-dimensional urban place perception scores from street-level imagery using a Vision-Language Model |
| Target application | Urban planning, walkability analysis, smart city design, equitable neighbourhood evaluation |
| Programming language | Python 3.13 |
| Repository type | Standalone script-based pipeline (no Git repository detected) |

The project develops a six-stage automated pipeline that: acquires geotagged street-level images, runs them through the Florence-2 Vision-Language Model using a structured chain-of-thought prompt, parses the model's natural-language output into typed numerical fields, aggregates the scores spatially, and presents results in an interactive GeoAI map with per-location explainability popups.

---

## 2. Research Background

Urban perception research attempts to quantify how human beings experience urban environments — whether a street feels safe, walkable, green, or visually appealing. Historically, this required expensive crowdsourced surveys such as MIT Place Pulse. CNN-based classifiers trained on such surveys produce scores without explaining which visual features drove the score.

This project explores an alternative: using a zero-shot Vision-Language Model (VLM) that can simultaneously identify visual scene content and produce structured, human-readable justifications for each perception score. This directly addresses the interpretability gap in prior automated approaches.

The README explicitly identifies three gaps this project addresses:
1. **Opacity** — CNN classifiers produce scores without human-interpretable justification
2. **Survey dependence** — crowdsourced datasets require expensive annotation at scale
3. **Spatial generalizability** — models trained on one city fail to generalize

The project addresses gap (1) via chain-of-thought visual prompting and gap (2) via zero-shot VLM inference requiring no labelled perception data.

---

## 3. Research Questions (as stated in README)

1. Can a Vision-Language Model identify the urban visual characteristics that influence human perception of safety, walkability, greenery, and visual quality?
2. Does structured textual reasoning produced by a VLM provide interpretable, location-specific evidence that aligns with established urban design principles?
3. What spatial patterns emerge in VLM-generated perception scores across NYC neighbourhoods, and do they reflect known urban inequality?

**Stated hypothesis:** A prompted Vision-Language Model can produce perception scores accompanied by explicit visual evidence that are spatially coherent and align with qualitative urban design principles — without requiring human-annotated training data.

---

## 4. Datasets

### Dataset 1 — Kaggle Google Street View (ACTUALLY USED)

| Property | Value |
|---|---|
| Name | Google Street View — Kaggle |
| Kaggle slug | paulchambaz/google-street-view |
| Content | ~10,000 numbered PNG images (0.png–9999.png) + coords.csv |
| Actual geographic coverage | **GLOBAL** (verified from metadata.csv: coordinates span Brazil, Argentina, Australia, Russia, Africa, Southeast Asia, Western Europe) |
| Documented geographic coverage | README states NYC (−74.2591°W to −73.7004°W, 40.4774°N to 40.9176°N) — this is INCORRECT |
| Format | PNG, CSV |
| Sample used | 50 images (seed=42) |
| CRS | WGS 84 (EPSG:4326) |
| Preprocessing | Random sampling → copy to data/images/ → metadata.csv |

**Critical discrepancy:** The README, REPORT.md, and config.py all state the study area is New York City. The actual coordinates in metadata.csv are globally distributed and contain no NYC coordinates. The Folium interactive map is centred on NYC coordinates despite none of the 50 images being from NYC. This is a fundamental mismatch between documentation and actual data.

### Dataset 2 — Mapillary API (ATTEMPTED, FAILED)

| Property | Value |
|---|---|
| API | Mapillary Graph API |
| Target area | NYC (defined in MAPILLARY_BBOX) |
| Status | Script implemented (01_download_mapillary.py) but encountered persistent HTTP 500 errors per README; abandoned in favour of Kaggle fallback |

### Dataset 3 — image_metadata.csv (PRESENT, UNUSED)

| Property | Value |
|---|---|
| Content | 5 Lahore, Pakistan locations (Gulberg, DHA, Old Lahore, Model Town, Liberty Market) |
| Status | Not consumed by any pipeline script; leftover from earlier development phase |

### Dataset 4 — Florence-2-base (PRETRAINED MODEL)

| Property | Value |
|---|---|
| Source | microsoft/Florence-2-base on HuggingFace Hub |
| Architecture | Encoder-decoder transformer with image-text grounding |
| Use mode | Zero-shot, no fine-tuning |
| Load status | **NOT SUCCESSFULLY LOADED** in any run producing results in this repository |

---

## 5. Complete Methodology

### 5.1 Numbered Research Workflow

1. **Data acquisition** — Download Kaggle GSV dataset via Kaggle CLI; extract zip; parse coords.csv to map image filenames to coordinates; randomly sample 50 images (seed=42); copy to data/images/; write metadata.csv
2. **Sample preparation** — Re-read metadata.csv; apply reproducible 50-image sample (seed=42); write sample_metadata.csv
3. **VLM inference** — Load Florence-2-base; for each image: open as RGB PIL Image; tokenize with structured prompt; generate up to 512 new tokens with beam search (n=3); decode output; parse via regex into typed fields (description, urban elements, per-dimension score + evidence + reasoning, overall, confidence)
4. **Results output** — Assemble 22-field dictionary per image; write to outputs/vlm/results.csv
5. **Visualization** — Read results.csv; generate 6 charts: average scores bar, radar top-5, violin+box, feature frequency, Pearson correlation heatmap, top urban elements
6. **Interactive map** — Build Folium map; add CircleMarker per image; populate popup with base64 thumbnail + explainability chain HTML; add optional grid choropleth; save map.html
7. **Spatial aggregation** — Filter results to non-(0,0) coordinates; project to EPSG:3857; create 500m rectangular grid; spatial join; compute mean scores per cell; write GeoJSON + CSV

### 5.2 Methodology Flowchart

```
Street-level Images (Kaggle GSV / Mapillary)
              │
              ▼
 [01] Data acquisition: download, extract, sample 50 (seed=42), metadata.csv
              │
              ▼
 [02] Sample preparation: reproducible re-sampling, sample_metadata.csv
              │
              ▼
 [03] Florence-2 VLM inference per image
       ├─ Structured chain-of-thought prompt
       ├─ Generate: Description, Urban_elements, Safety, Walkability, Greenery, Visual_quality, Overall
       ├─ Regex parse → typed fields
       └─ Write results.csv (22 fields/image)
              │
              ▼
 [04] Visualization: 6 publication-quality charts (dark mode)
              │
              ▼
 [05] Interactive GeoAI map: Folium + per-location explainability popups
              │
              ▼
 [06] Spatial aggregation: 500m grid, mean scores → GeoJSON + CSV
```

### 5.3 Core Inference Logic (utils/inference.py)

The inference utility implements three key design decisions:

**A. Evidence-before-score prompting:**  
The prompt structure forces the model to list visual evidence BEFORE assigning a numeric score. This is a deliberate chain-of-thought technique that grounds the score in identified visual features, improving interpretability and reducing hallucination.

**B. Regex-based structured parsing:**  
Six compiled regex patterns extract fields from the structured output. Each perception dimension requires: `<dimension>: <float> Evidence: <text> Reasoning: <text>`. If parsing fails for any field, the value is set to None rather than fabricating a score.

**C. Deterministic debug fallback:**  
When the model cannot load, scores are derived from `hashlib.sha256(image_path.encode())` seeded into `random.Random`, producing uniform samples in [0.4, 0.9]. This ensures the full pipeline can be tested without a GPU.

---

## 6. Experiments and Results

### 6.1 Experiment: VLM Inference on 50 Images

| Property | Value |
|---|---|
| Images | 50 globally distributed street-view images from Kaggle GSV |
| Model | Florence-2-base — NOT LOADED (debug fallback used) |
| Run 1 output | outputs/florence2/results.csv — 50 rows, schema: 9 columns (no evidence/reasoning) |
| Run 2 output | outputs/vlm/results.csv — 50 rows, schema: 22 columns (full explainability) |
| Evidence for model failure | Both CSVs contain "Model unavailable; using deterministic fallback scores" |

**Score summary from outputs/florence2/results.csv (selected rows):**

| image_id | safety | walkability | greenery | visual_quality |
|---|---|---|---|---|
| img_002 | 0.909 | 0.272 | 0.678 | 0.206 |
| img_004 | 0.855 | 0.775 | 0.406 | 0.620 |
| img_010 | 0.893 | 0.627 | 0.931 | 0.182 |
| img_020 | 0.043 | 0.726 | 0.945 | 0.723 |
| img_022 | 0.727 | 0.949 | 0.604 | 0.331 |
| img_023 | 0.027 | 0.188 | 0.210 | 0.368 |

Note: These values are hash-derived deterministic pseudorandom scores. They do not represent real VLM reasoning.

### 6.2 Experiment: Chart Generation

**Status: VERIFIED COMPLETE**  
6 chart files present in outputs/vlm/charts/:
- average_scores.png, radar_top5.png, violin_boxplots.png, feature_frequency.png, correlation_heatmap.png, top_urban_elements.png

### 6.3 Experiment: Interactive Map

**Status: VERIFIED COMPLETE**  
map.html exists in outputs/vlm/ and outputs/florence2/. The vlm version includes full explainability popups.

### 6.4 Quantitative Performance

**No verified quantitative performance evaluation was found.**

The REPORT.md states: "a strong positive relationship between greenery and visual_quality (r≈0.68), while safety correlates modestly with walkability (r≈0.42)." These values cannot be verified as legitimate findings because all scores are hash-derived pseudorandom values.

---

## 7. Output Files Inventory

| File | Status | Description |
|---|---|---|
| outputs/vlm/results.csv | EXISTS | 50-row inference results (debug fallback) |
| outputs/florence2/results.csv | EXISTS | Earlier 50-row run (simpler schema) |
| outputs/vlm/map.html | EXISTS | Interactive GeoAI explainability map |
| outputs/florence2/map.html | EXISTS | Earlier map (simpler popups) |
| outputs/vlm/charts/average_scores.png | EXISTS | Bar chart of mean scores |
| outputs/vlm/charts/radar_top5.png | EXISTS | Radar chart top-5 images |
| outputs/vlm/charts/violin_boxplots.png | EXISTS | Score distributions |
| outputs/vlm/charts/feature_frequency.png | EXISTS | Urban element frequency |
| outputs/vlm/charts/correlation_heatmap.png | EXISTS | Pearson correlation matrix |
| outputs/vlm/charts/top_urban_elements.png | EXISTS | Top-10 elements horizontal bar |
| outputs/florence2/charts/average_scores.png | EXISTS | Earlier run bar chart |
| outputs/florence2/charts/radar_top5.png | EXISTS | Earlier run radar chart |
| data/metadata.csv | EXISTS | 50-image metadata with global coordinates |
| data/sample_metadata.csv | EXISTS | 50-image sample subset |
| data/image_metadata.csv | EXISTS | 5-image Lahore reference (unused) |

---

## 8. Reproducibility Assessment

**Reproducible with conditions:**
- Kaggle API token required for image acquisition
- HuggingFace model download (~2GB) required for real VLM inference
- GPU strongly recommended (Florence-2-base runs on CPU but slowly)
- seed=42 ensures identical 50-image sample on each run
- Without GPU/model: fully reproducible in debug mode (hash-derived scores)

---

## 9. Research Maturity

**Classification: Exploratory Prototype**

The project is a well-architected pipeline prototype with genuine methodological novelty in its prompt design and explainability framework. The core experiment (VLM inference on real images) has not been successfully executed. Advancing to "Completed Applied Research" requires: (1) successful Florence-2 inference with GPU, (2) use of geographically accurate imagery for the stated study area, and (3) validation against human perception data.
