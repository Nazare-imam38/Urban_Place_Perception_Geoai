# 03 — Research Contributions and Findings
## Urban Place Perception — Human-Centered GeoAI

---

## What Was Actually Accomplished

### 1. Complete End-to-End Pipeline — VERIFIED

A six-stage reproducible pipeline was fully implemented, from image acquisition through spatial aggregation. All scripts are present, functional, and interconnected. The pipeline successfully:
- Downloads a street-view image dataset from Kaggle
- Samples 50 images reproducibly
- Runs batch inference (in debug mode)
- Generates six publication-quality visualizations
- Produces an interactive HTML map with explainability popups
- Computes spatial aggregation statistics

### 2. Structured Explainability Architecture — VERIFIED

The most methodologically significant contribution of the project is its prompt engineering and output schema design. Every perception score is accompanied by:
- A list of specific visual evidence cues (e.g., "wide sidewalk, accessible curbs")
- A natural-language reasoning sentence linking evidence to score

This three-part structure (score + evidence + reasoning) implements a form of visual chain-of-thought prompting that is absent in prior CNN-based urban perception pipelines. The design is fully implemented in `utils/inference.py` and would produce this output if the model were successfully loaded.

### 3. Dual Data Acquisition Strategy — VERIFIED

Two complete acquisition pathways were implemented:
- A Mapillary API-based pathway with tile-based pagination, exponential back-off retry, and high-resolution thumbnail download
- A Kaggle CLI fallback pathway with coordinate parsing, format flexibility, and paired image-coordinate sampling

### 4. Iterative Pipeline Development — VERIFIED (from output evidence)

The presence of two output directories (`outputs/florence2/` and `outputs/vlm/`) with different schema complexity (9 vs 22 fields; 2 vs 6 charts) demonstrates that the pipeline was developed iteratively — extending the explainability schema and adding four additional visualization types in a second iteration.

---

## Verified Findings

**IMPORTANT QUALIFICATION:** All numerical findings in this section are based on deterministic hash-derived fallback scores, not on actual Florence-2 VLM inference. They do not represent real empirical findings about urban perception.

### Finding 1 — Pipeline execution with fallback scores
Both inference runs produced complete 50-row results files. This verifies the pipeline infrastructure works correctly end-to-end, including CSV writing, chart generation, and map creation.

### Finding 2 — Score distribution with fallback scores
The 50 hash-derived scores per indicator span a wide range (~0.03–0.98), broadly uniform, reflecting the hash function's pseudo-random distribution. This is consistent with what would be expected from a well-functioning uniform random generator, not from a real perception model.

### Finding 3 — Map generation confirmed
An interactive HTML map was generated with per-location popups including base64-embedded image thumbnails, score bars, and evidence/reasoning text. This demonstrates the explainability interface design is functional.

---

## My Documented Contributions (Nazar E Imam)

**Attribution caveat:** No Git repository is present, so individual commits and authorship cannot be established from version control. The following attribution is based on the project's alignment with the applicant's described background and the workspace being in their local environment.

### Confirmed as project work (workspace evidence):
- All 8 Python source files present in the workspace
- config.py, requirements.txt, README.md, REPORT.md all authored as part of this project
- Two complete pipeline execution runs (florence2 and vlm output directories)
- Data downloaded and processed (kaggle_gsv dataset, metadata CSVs, 50 sample images)

### Likely personal contributions (reasonable inference from solo-student project context):
- Pipeline architecture design (6-stage sequential workflow)
- Prompt engineering (structured line-by-line format with evidence-before-score)
- Explainability schema design (22-field output dictionary)
- Debug fallback implementation (hash-based deterministic scoring)
- Dark-mode visualization theme and chart selection
- Interactive map popup HTML design

### Contributions requiring clarification:
- Whether the Mapillary API exploration and its failure was personal work
- Whether the pipeline was developed independently or with AI assistance
- The academic context in which this project was created (coursework, independent, or supervised research)

---

## Scientific Significance

### What the project does demonstrate:
1. **Prompt engineering for structured VLM output** — The prompt design is a genuine methodological contribution. Requesting evidence before scores, using a fixed line-by-line format, and enforcing regex-parseable structure is a non-trivial design choice that aligns with current best practices in LLM chain-of-thought prompting.

2. **Explainability architecture for GeoAI** — The combination of per-location score + evidence + reasoning + interactive map represents a more human-centered output than typical score-only classifiers.

3. **End-to-end GeoAI pipeline development** — The project demonstrates competency in integrating multiple tools (HuggingFace Transformers, Kaggle API, Mapillary API, Folium, GeoPandas) into a coherent research pipeline.

### What the project does NOT demonstrate:
1. Real VLM inference on street-level images (model never loaded in results-producing runs)
2. Empirically validated perception scores
3. Spatial analysis of NYC urban perception
4. Comparison with human perception surveys
5. A new computational method (Florence-2 is used as-is)

---

## Methodological Originality Classification

The project:
- **Applies an existing method** (Florence-2 zero-shot VLM) ✓
- **Adapts prompt engineering** for structured urban perception output ✓
- **Integrates multiple established tools** (VLM + geospatial tools + interactive visualization) ✓
- **Does NOT develop a new algorithm or architecture** ✗
- **Does NOT evaluate methods in a rigorously validated setting** ✗ (debug fallback only)

Overall: **Methodological adaptation and integration** — not novel algorithm development, but genuine applied research design work.

---

## Practical Importance

The approach — if executed with the real model — would provide urban planners and public health researchers with a scalable tool to:
- Screen large areas for walkability and safety concerns without human annotation
- Provide explainable evidence for planning decisions
- Monitor perception change over time using time-stamped street view imagery

This is a genuine practical application, though it remains at prototype stage.
