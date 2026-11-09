# 08 — Research Visuals
## Urban Place Perception — Human-Centered GeoAI

**Important note:** All charts were generated from deterministic hash-derived debug fallback scores, not from actual Florence-2 VLM inference. They demonstrate the visualization pipeline's capability but do not represent real empirical findings.

---

| File | Description | Source path |
|---|---|---|
| `average_scores.png` | Bar chart of mean perception scores per indicator (50 images, debug fallback scores) | `outputs/vlm/charts/average_scores.png` |
| `correlation_heatmap.png` | Pearson correlation matrix of four indicators — NOTE: based on pseudorandom scores, not real VLM output | `outputs/vlm/charts/correlation_heatmap.png` |
| `violin_boxplots.png` | Score distribution per indicator (violin + box overlay, dark mode) | `outputs/vlm/charts/violin_boxplots.png` |
| `feature_frequency.png` | Frequency of detected urban visual elements — NOTE: debug mode returns fixed element list per image | `outputs/vlm/charts/feature_frequency.png` |

---

## Additional outputs in original project (not copied here due to size)

| File | Location |
|---|---|
| `map.html` (interactive GeoAI map) | `outputs/vlm/map.html` |
| `radar_top5.png` | `outputs/vlm/charts/radar_top5.png` |
| `top_urban_elements.png` | `outputs/vlm/charts/top_urban_elements.png` |
