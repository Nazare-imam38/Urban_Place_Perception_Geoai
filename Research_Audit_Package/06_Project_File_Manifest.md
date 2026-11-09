# 06 — Project File Manifest
## Urban Place Perception — Human-Centered GeoAI

All paths are relative to the repository root `urban-place-perception-geoai/`.

---

## Source Code Files

| File | Significance |
|---|---|
| `config.py` | Central configuration: project title, all path constants, model name (Florence-2-base), SAMPLE_SIZE=50, SEED=42, Mapillary credentials, Kaggle slug, urban element vocabulary, pipeline stage descriptions |
| `utils/inference.py` | **Core inference module** — Florence-2 model loading singleton, structured chain-of-thought prompt, 6 regex parsers, `infer_image()` function, debug fallback, `image_to_base64()`, `_compute_confidence()` |
| `01_download_mapillary.py` | Mapillary API download: tile-based pagination, exponential back-off retry, high-resolution thumbnail download, metadata CSV writing |
| `01_download_kaggle_dataset.py` | Kaggle dataset download: CLI invocation, ZIP extraction, coords.csv parsing (header-flexible), image–coordinate pairing, 50-image sampling, metadata.csv writing |
| `02_prepare_sample.py` | Reproducible sample selection: reads metadata.csv, applies seed=42 sample, writes sample_metadata.csv |
| `03_run_model_inference.py` | Batch inference controller: reads sample_metadata.csv, calls `infer_image()` per row, handles RuntimeError abort on model failure, writes results.csv |
| `04_generate_visualizations.py` | Six chart generators: bar chart, radar chart, violin+box plots, feature frequency bar, Pearson correlation heatmap, top urban elements horizontal bar — all in dark mode |
| `05_create_interactive_map.py` | Folium map builder: `_build_popup()` HTML generator, safety-score color encoding, grid choropleth overlay, LayerControl, `map.html` output |
| `06_spatial_summary.py` | Spatial aggregation: GeoDataFrame creation, EPSG:3857 projection, 500m rectangular grid, spatial join, mean score aggregation, GeoJSON + CSV output, geopandas fallback |

---

## Research Documentation

| File | Significance |
|---|---|
| `README.md` | Complete project overview: motivation, research gap, research questions, hypothesis, contributions, dataset table, methodology flowchart, results section, interactive map description, limitations, future work, how-to-run instructions, references |
| `REPORT.md` | Short GIScience research report: abstract, introduction, related work, methodology, dataset, experiments & results (with correlation claims), discussion, limitations, future work, references |
| `requirements.txt` | Pinned dependency list: torch==2.3.0, transformers==4.44.0, folium==0.16.0, pandas==2.2.2, numpy==1.26.4, matplotlib==3.9.0, seaborn, scikit-learn, huggingface-hub, requests, kaggle |

---

## Data Files

| File/Directory | Significance |
|---|---|
| `data/kaggle_gsv/dataset/` | ~10,000 numbered PNG images (0.png–9999.png) + coords.csv — the actual data used for inference |
| `data/images/` | 50 sampled PNG images copied from kaggle_gsv/dataset/ — actual inference input |
| `data/metadata.csv` | 50-row CSV: image_id, filepath, latitude, longitude — coordinates are GLOBAL (not NYC) |
| `data/sample_metadata.csv` | 50-row reproducible sample from metadata.csv used by inference script |
| `data/image_metadata.csv` | 5-row Lahore reference dataset — NOT used by any pipeline script |
| `data/google-street-view.zip` | Original Kaggle download archive |

---

## Output Files

| File | Significance |
|---|---|
| `outputs/vlm/results.csv` | **Primary result** — 50 rows, 22 columns, full explainability schema including evidence and reasoning fields; all scores are debug fallback values |
| `outputs/florence2/results.csv` | Earlier result — 50 rows, 9 columns (simpler schema without evidence/reasoning); also debug fallback |
| `outputs/vlm/map.html` | **Interactive GeoAI map** — Folium map with full explainability popups; base64 thumbnails, score bars, evidence badges, reasoning text |
| `outputs/florence2/map.html` | Earlier map with simpler popups |
| `outputs/vlm/charts/average_scores.png` | Bar chart of mean scores per indicator |
| `outputs/vlm/charts/radar_top5.png` | Radar chart of top-5 images by overall score |
| `outputs/vlm/charts/violin_boxplots.png` | Score distributions per indicator (violin + box overlay) |
| `outputs/vlm/charts/feature_frequency.png` | Frequency of detected urban elements across 50 images |
| `outputs/vlm/charts/correlation_heatmap.png` | Pearson correlation matrix of four perception indicators |
| `outputs/vlm/charts/top_urban_elements.png` | Horizontal bar chart of top-10 most common elements |
| `outputs/florence2/charts/average_scores.png` | Earlier average scores chart |
| `outputs/florence2/charts/radar_top5.png` | Earlier radar chart |

---

## Configuration / Metadata Files

| File | Significance |
|---|---|
| `02_prepare_sample.py.metadata.json` | Kiro IDE metadata for prepare_sample script |
| `03_run_model_inference.py.metadata.json` | Kiro IDE metadata for inference script |
| `04_generate_visualizations.py.metadata.json` | Kiro IDE metadata for visualization script |
| `05_create_interactive_map.py.metadata.json` | Kiro IDE metadata for map script |
| `config.py.metadata.json` | Kiro IDE metadata for config |
| `requirements.txt.metadata.json` | Kiro IDE metadata for requirements |
| `pyrefly.toml` | Python type checking configuration |
| `.vscode/settings.json` | VS Code workspace settings |

---

## Absent Files (Architecturally Expected but Not Present)

| Expected file | Reason expected | Reason absent |
|---|---|---|
| `.git/` | Version control | This is not a Git repository |
| `outputs/vlm/spatial_summary.geojson` | Stage 6 spatial aggregation output | Not generated — all coordinates are far outside any reasonable grid |
| Fine-tuned model weights | If model was adapted | Not applicable — zero-shot only |
| Validation dataset | For score comparison | Not present — no human survey data integrated |
| Environment file (.env) | API keys | Mapillary token and Kaggle credentials stored directly in config.py |

---

## Dependency Summary

```
Deep learning:     torch==2.3.0, torchvision==0.18.0
NLP/VLM:           transformers==4.44.0, huggingface-hub==0.25.0
GIS/Spatial:       geopandas (optional, not in requirements.txt — fallback provided), folium==0.16.0
Data:              pandas==2.2.2, numpy==1.26.4
Visualization:     matplotlib==3.9.0, seaborn (version pinned via scikit-learn)
Image:             pillow==10.4.0
ML utilities:      scikit-learn==1.5.1
HTTP:              requests==2.32.3
Progress:          tqdm==4.66.5
Data acquisition:  kaggle>=1.6.0
```

Note: `geopandas` and `shapely` are not in requirements.txt but are imported in `06_spatial_summary.py` with a graceful fallback if unavailable.
