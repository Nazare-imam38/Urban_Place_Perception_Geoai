# 05 — Admissions Research Profile
## Nazar E Imam — Urban Place Perception GeoAI Project

---

## A. Three-Sentence Research Summary

I investigated whether Vision-Language Models can replace expensive human surveys for automated urban place perception analysis, developing a six-stage GeoAI pipeline that uses Florence-2 to generate structured, explainable scores for safety, walkability, greenery, and visual quality from geotagged street-level imagery.

The pipeline implements a structured chain-of-thought prompting strategy that forces the model to identify visual evidence before assigning numeric scores, producing a per-location explainability chain consisting of a scene description, detected urban elements, four scored dimensions with evidence and reasoning, and an interactive Folium map.

The experiment revealed that deploying Florence-2-base on CPU without a dedicated GPU made real inference infeasible at this scale, and that the global-coverage Kaggle dataset used does not match the NYC bounding box specified in the research design — two critical limitations that clearly identify the next research steps: GPU-based inference with geographically correct data and validation against human perception surveys.

---

## B. Detailed Research Summary (~350 words)

Urban perception — how safe, walkable, green, or visually appealing a street feels to pedestrians — is fundamental to equitable urban planning and smart city design. Traditional methods rely on crowdsourced human surveys, which are expensive to conduct at scale and inherently limited to the locations where surveys were administered. Recent CNN-based approaches automate this measurement but produce opaque scores without explaining which visual features drove the prediction.

My project explored a different approach: using Florence-2, a compact Vision-Language Model, to simultaneously identify visual scene content and produce structured natural-language justifications for each perception score. The core methodological contribution is a chain-of-thought prompt that requires the model to list visual evidence — sidewalk conditions, vegetation, lighting, building quality — before assigning a numeric score in the range 0–1 for each of four perception dimensions. This evidence-first structure produces a fully interpretable output chain for every image location.

The pipeline processes 50 geotagged street-level images through six stages: data acquisition from Mapillary or Kaggle (a Mapillary API fallback was implemented after persistent server errors), reproducible sampling, batch VLM inference, publication-quality visualization (six chart types), an interactive Folium map with per-location explainability popups, and grid-based spatial aggregation at 500-metre resolution. The interactive map, designed as an "Explainable GeoAI interface", renders each location marker with the scene description, detected urban elements as badges, and per-dimension score bars with corresponding evidence and reasoning text.

Executing the project revealed two critical limitations. First, Florence-2-base requires a CUDA-capable GPU for practical inference; CPU-only execution in the development environment made loading the model infeasible, so all results were produced using a deterministic debug fallback. Second, the Kaggle street-view dataset used as a fallback turned out to contain globally distributed images rather than the NYC-specific imagery the study design intended. Together, these limitations define a clear research agenda: the pipeline architecture and prompt engineering are complete, but the core hypothesis — that structured VLM prompting produces spatially coherent, human-aligned perception scores — remains to be tested against real model output and validated against human perception benchmarks such as Place Pulse 2.0.

---

## C. Academic CV Bullet Points

- **Developed a six-stage Human-Centered GeoAI pipeline** applying Microsoft Florence-2 Vision-Language Model to urban place perception analysis; implemented structured chain-of-thought prompting to generate evidence-grounded perception scores (safety, walkability, greenery, visual quality) with per-location reasoning sentences for explainable spatial inference.

- **Engineered a reproducible data acquisition and inference framework** integrating the Mapillary REST API (tile-based pagination, exponential back-off), Kaggle CLI, HuggingFace Transformers, GeoPandas 500m grid spatial aggregation, and Folium interactive GeoAI maps; identified and documented a critical geographic data discrepancy between documented NYC study area and actual global dataset coverage.

- **Demonstrated end-to-end applied GeoAI pipeline development** including dual data source handling, debug fallback mechanism for GPU-free reproducibility, six publication-quality dark-mode visualizations, and interactive HTML map with embedded base64 thumbnails and structured explainability popups — identifying GPU-based model execution and human perception ground truth validation as the key next research steps.

---

## D. Master's Letter of Motivation Research Paragraph (~115 words)

During my undergraduate studies in Geoinformatics Engineering at NUST, I developed a GeoAI pipeline that uses a Vision-Language Model — Microsoft's Florence-2 — to infer urban place perception from street-level imagery. Rather than training a classifier on existing survey data, I designed a structured prompt that asks the model to identify visual evidence before assigning scores for safety, walkability, greenery, and visual quality, producing an interpretable reasoning chain for every image location. Testing the pipeline on a global street-view dataset exposed a fundamental gap: scores derived from a model that could not run on available hardware, and imagery that did not cover the intended study area, neither confirmed nor refuted my central hypothesis. That unresolved question — whether structured VLM prompting genuinely produces human-aligned perception scores — is exactly what I want to investigate rigorously at Master's level.

---

## E. Faculty Outreach Paragraph (~90 words)

My recent GeoAI project investigates whether Vision-Language Models can replace crowdsourced urban perception surveys by generating explainable, evidence-grounded scores from street-level imagery. I developed a structured chain-of-thought prompt that forces the model to identify specific visual cues before assigning numeric scores, producing an interpretable output chain rather than an opaque prediction. The experiment revealed that the hypothesis — spatially coherent VLM perception scores without labelled training data — remains untested because model execution required GPU resources unavailable during development. I am interested in continuing this investigation rigorously, and your work on [specific faculty research topic] connects directly to the spatial coherence and validation questions that my prototype raised.

---

## F. Master's Research Direction

**Research question:** Do structured chain-of-thought prompts improve the agreement between Vision-Language Model urban perception scores and human survey ratings compared to unstructured prompting, and which prompt design choices most strongly influence score validity?

**Investigation approach:**
- Collect 500–1,000 geotagged images from cities covered by Place Pulse 2.0
- Run Florence-2 (base and large) with three prompt variants: score-only, evidence-first, and full chain-of-thought
- Compute Spearman ρ and RMSE between VLM scores and human ratings per dimension
- Conduct ablation study isolating evidence ordering, output format structure, and urban element vocabulary

This is well-scoped for a two-year research Master's: public datasets, no new model training required, clear quantitative evaluation.

---

## G. PhD Research Direction

**Research question:** Can a spatially aware urban perception model — pre-trained on street-level imagery and spatial context features — transfer across cities with different urban morphologies and cultural contexts more effectively than per-image zero-shot VLM inference?

**Extension from Master's:** The Master's establishes whether VLM prompting produces valid scores in a single-city controlled setting. The PhD extends this by: (1) systematically characterizing cross-city transfer and cultural bias in VLM perception reasoning, (2) developing a spatial context augmentation approach that incorporates OpenStreetMap features and adjacent image context during inference, and (3) building a multi-city benchmark for geographically coherent urban perception assessment.

This connects to the emerging literature on geospatial foundation models and spatial representation learning, with applications to equitable urban analytics.

---

## H. Research Skills Demonstrated by This Project

| Skill | Evidence |
|---|---|
| Vision-Language Models (HuggingFace) | Florence-2-base integration with AutoProcessor + AutoModelForCausalLM |
| Prompt engineering | Structured chain-of-thought format; evidence-before-score ordering; regex parsing |
| Python pipeline development | Six-stage script pipeline with config management and CLI flags |
| Geospatial data handling | Geopandas, Folium, EPSG:4326→3857 projection, spatial join, grid generation |
| API integration | Mapillary Graph API (pagination, retry), Kaggle CLI |
| Data engineering | Reproducible sampling, metadata CSV management, fallback mechanisms |
| Scientific visualization | 6 publication-quality charts (matplotlib, seaborn, dark mode) |
| Spatial analysis | 500m grid aggregation, GeoJSON export |
| Research documentation | README with research questions, hypothesis, methodology, limitations, future work |
| Critical self-assessment | Identifying model failure, geographic data discrepancy, and validation gap |

---

## I. Potential Faculty Research Directions

### 1. Human-Centered GeoAI / Explainable Spatial AI
- **Relevant to:** Professors working on explainable AI for geospatial applications, VLM-based scene understanding, urban analytics
- **Connection:** The project's core contribution is an explainability architecture (score + evidence + reasoning) that aligns with human-centered GeoAI principles
- **Example alignment area:** Yuhao Kang (UChicago), Gengchen Mai (UT Austin), Xiaojiang Li (Temple), Zhenghong Li (UTK)

### 2. Urban Analytics and Spatial Data Science
- **Relevant to:** Professors studying walkability, safety perception, urban inequality, neighborhood change
- **Connection:** The pipeline produces per-location walkability and safety scores with spatial aggregation, directly useful for urban planning research
- **Example alignment area:** Programs at MIT Media Lab, Penn Urban Spatial Analytics, UCL CASA

### 3. Computer Vision and Remote Sensing (ground-level imagery)
- **Relevant to:** Professors applying CV to street-level imagery, urban scene understanding, place recognition
- **Connection:** The prompt engineering and evaluation framework could support research on VLM evaluation in geographic contexts

### 4. GeoAI and Foundation Models
- **Relevant to:** Professors developing spatially aware language models or geospatial foundation models
- **Connection:** The project reveals limitations of per-image VLM inference (no spatial context) and motivates spatially grounded extensions

---

## J. Interview Questions and Answers

**Q1. What motivated you to use a VLM rather than a CNN for urban perception scoring?**

CNNs trained on Place Pulse produce single numbers without explanation. A VLM can simultaneously describe what it sees and justify the score, making the output auditable. For planning applications where decisions affect people's lives, knowing that a street scored 0.3 on safety "because of poor lighting, no sidewalks, and high traffic density" is more useful than just the number 0.3.

**Q2. Why did you use chain-of-thought prompting with evidence before score?**

If you ask the model for a score directly, it may confabulate a number without grounding. By requiring evidence first, you force the model to commit to a visual observation before the score is generated. This mirrors how human annotators approach such tasks — they identify features first, then rate. Whether this actually reduces hallucination in Florence-2 is an open empirical question I would want to test properly.

**Q3. Your results are all debug fallback scores. What exactly went wrong with the model?**

Florence-2-base requires loading roughly 450M parameters. On CPU-only hardware, the model can be loaded but inference is extremely slow (several minutes per image). Given 50 images, this becomes impractical. More likely, the model either failed to load due to a memory or dependency issue, or the environment triggered the try/except fallback silently. The debug mode was used to complete the pipeline, which was the right decision for testing the infrastructure, but it means the core hypothesis was not tested.

**Q4. You documented this as an NYC study but used global imagery. How did this happen?**

The original design planned to use Mapillary, which would have provided genuine NYC imagery within the bounding box. When Mapillary failed, the Kaggle Google Street View dataset was used as a fallback. That dataset is not NYC-specific — its coordinates span the globe. I should have verified the geographic extent of the fallback dataset before treating it as equivalent to the original data. This was a significant oversight that I would address by running the pipeline with actual NYC-bounded Mapillary or Google Street View imagery.

**Q5. What would you need to convert this prototype into a publishable paper?**

Four things: (1) GPU access to run Florence-2 on real images — even 50 well-chosen NYC images would be a start. (2) Replace the Kaggle dataset with geographically correct NYC imagery. (3) Obtain Place Pulse 2.0 human ratings for a subset of the same locations and compute alignment metrics. (4) Conduct a prompt ablation study comparing the evidence-first format against alternatives. With those four additions, there would be a defensible empirical contribution.

**Q6. How does your project relate to spatial foundation models?**

Spatial foundation models like GeoGPT or GeoLLM aim to pre-train language models with geographic awareness — embedding spatial relationships, place semantics, and map reasoning. My project essentially asks a vision-language foundation model (Florence-2) to do spatial analysis without any spatial pre-training. That's a stress test: can a geographically naive VLM reason coherently about urban space? The spatial coherence limitations I would expect to find motivate the connection to spatial foundation models — which could provide the geographic context that Florence-2 currently lacks.

**Q7. What is the difference between your approach and the Place Pulse classifier approach?**

Place Pulse trains a CNN on pairwise human ratings, learning to predict preference directly from image features. My approach uses zero-shot prompting: no labelled perception data is needed, and the model generalizes through natural language. The trade-off is validation — Place Pulse classifiers are directly calibrated to human judgments, while VLM scores need external validation. My approach is more scalable (no annotation required) but currently unvalidated.

**Q8. Why did you choose 500m as the grid cell size for spatial aggregation?**

500m is a common urban analysis unit — roughly one standard city block in North American grids, and the scale at which pedestrian environment conditions are reasonably homogeneous. It's also used in urban planning metrics (e.g., 500m walkability catchments). However, I would be interested in testing sensitivity to cell size — whether 200m or 1km cells change the spatial patterns, and whether spatial autocorrelation is maximized at a particular scale.

**Q9. What would you do differently if you started this project over?**

Three things: First, verify the geographic content of any fallback dataset before using it. Second, establish GPU access before designing the experiment around model inference. Third, start with a smaller validated sample — even 10 images with confirmed human ratings — and verify model alignment before scaling to 50.

**Q10. What does this project tell you about the challenges of deploying AI in geographic contexts?**

Geographic AI faces challenges that standard computer vision does not: spatial validity (does the model's output make geographic sense?), cultural transferability (does the model reason the same way about Nairobi streets as New York streets?), and data-location consistency (is the imagery actually from where you think it is?). This project exposed all three: I couldn't verify spatial coherence without real model output, I couldn't test transferability without multi-city data, and I had a silent data-location mismatch that invalidated the geographic framing. These are exactly the kinds of challenges that distinguish GeoAI as a field from general-purpose AI application.
