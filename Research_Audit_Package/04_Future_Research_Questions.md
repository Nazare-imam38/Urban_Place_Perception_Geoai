# 04 — Future Research Questions
## Urban Place Perception — Human-Centered GeoAI

The following three research questions emerge directly from the project's actual findings and limitations. Each is grounded in specific evidence from the audit.

---

## Research Question 1

**To what extent do zero-shot Vision-Language Model perception scores align with human crowdsourced urban perception ratings, and what aspects of the prompt structure most strongly influence score validity?**

### Motivating finding or limitation
The project's most critical limitation is that the Florence-2 model was never successfully executed on real images — all 50 result rows contain deterministic hash-derived placeholder scores. The hypothesis that structured VLM prompting produces spatially coherent, human-aligned perception scores therefore remains entirely untested. The prompt design (evidence before score, fixed line-by-line structure) is methodologically interesting but empirically unvalidated.

### Why this matters
Urban planning decisions based on AI-generated walkability or safety scores require confidence that those scores correspond to how real people perceive the environment. Without validation against human perception, the tool provides no reliable signal. This question directly addresses the gap between the project's methodological design and its current evidential status.

### Proposed approach
1. Successfully run Florence-2-base (or a larger variant such as Florence-2-large) on 500–1000 geotagged street-view images from a city where Place Pulse 2.0 human ratings exist
2. Compute Spearman rank correlation and Pearson r between VLM scores and human ratings per dimension
3. Conduct ablation study: compare baseline prompt (score only) vs. evidence-first prompt vs. full chain-of-thought prompt
4. Evaluate whether requesting evidence before score improves alignment (testing the core methodological claim)

### Possible datasets
- MIT Place Pulse 2.0 (Boston, New York, Chicago, London — 110,988 images with pairwise human ratings)
- Mapillary NYC imagery (the originally intended dataset)
- Google Street View Static API (city of choice)

### Testable hypothesis
VLM scores generated using an evidence-first structured prompt will correlate more strongly with human Place Pulse ratings (Spearman ρ > 0.40) than scores generated using a score-only prompt (ρ < 0.25), for at least three of the four perception dimensions.

### Evaluation strategy
- Primary metric: Spearman ρ between VLM scores and human ratings per dimension
- Secondary: Prompt ablation comparison across three prompt variants
- Tertiary: Qualitative analysis of evidence–score consistency

### Master's thesis feasibility
YES — This is well-scoped for a one-year research-based Master's thesis. The data are publicly available, the methodology is clear, and the comparison requires no new model training.

### PhD extension
A PhD could extend this by investigating fine-tuning the VLM on city-specific perception data, testing cross-city transfer, and developing uncertainty quantification methods to identify images where VLM confidence diverges from human agreement.

---

## Research Question 2

**How does urban place perception inferred by Vision-Language Models vary across cities with different urban morphologies, and does the model's reasoning reflect culturally specific visual cues or transfer generic global patterns?**

### Motivating finding or limitation
The Kaggle dataset that was actually used contains globally distributed images (South America, Africa, Eastern Europe, Southeast Asia, Australia) — not New York City as documented. This accidental use of a non-NYC dataset reveals a genuine research opportunity: the project's pipeline could, in principle, process images from any location. The key unresolved question is whether VLM perception reasoning transfers meaningfully across radically different urban contexts.

### Why this matters
Most urban perception AI research is conducted on Western, English-speaking cities. If VLMs encode culturally biased notions of "safety" or "walkability" (e.g., associating walkability with wide sidewalks common in North American cities, but penalizing narrow but functional pedestrian alleys common in Asian cities), then the models could reproduce spatial inequalities when deployed for planning in non-Western contexts.

### Proposed approach
1. Collect matched samples of 200–500 images from three cities with distinct urban morphologies: an American grid city (New York), a Southeast Asian mixed-development city (Singapore or Colombo), and an African informal-settlement city (Nairobi or Lagos)
2. Run Florence-2 (or GPT-4V) on all images using the same structured prompt
3. Compare score distributions and detected urban element frequencies across cities
4. Analyse reasoning sentences qualitatively for culturally specific language patterns

### Possible datasets
- Mapillary API (global coverage)
- Google Street View Static API
- KartaView / OpenStreetCam
- Existing urban perception datasets: Treepedia, Morphocode datasets

### Testable hypothesis
Florence-2 will produce significantly lower walkability scores for Nairobi street images than for New York images of equivalent pedestrian infrastructure, with reasoning sentences citing the absence of features (paved sidewalks, crosswalks) that are less common in informal urban contexts but do not reflect poor walkability for local residents.

### Evaluation strategy
- Quantitative: ANOVA/Kruskal-Wallis test of score distributions across cities
- Qualitative: Content analysis of reasoning sentences for culturally loaded language
- Ground truth: Local resident surveys where feasible

### Master's thesis feasibility
YES — Three-city comparative study is achievable with public imagery APIs. Requires minimal data infrastructure.

### PhD extension
A PhD could develop a cross-cultural urban perception benchmark and investigate fine-tuning strategies that reduce cultural bias, connecting to broader questions of AI fairness in spatial decision support systems.

---

## Research Question 3

**Can a spatially aware foundation model improve the geographic coherence and spatial consistency of VLM-generated urban perception scores compared to per-image processing without spatial context?**

### Motivating finding or limitation
The current pipeline processes each of the 50 images independently. There is no mechanism to enforce spatial consistency — two adjacent images on the same block could receive very different scores if their visual content differs slightly. Furthermore, the spatial aggregation step (06_spatial_summary.py) only averages scores after the fact; it does not provide spatial context during inference. Research Question 3 explores whether incorporating spatial information at inference time (street context, neighbourhood features, adjacent images) would improve score coherence.

### Why this matters
Urban perception is inherently spatial — a street's perceived safety is influenced not just by what is immediately visible but by the surrounding neighbourhood context. Ignoring this during inference may cause the scores to be spatially noisy. This is particularly relevant for spatial applications like walkability routing or urban inequality mapping, where smooth spatial gradients are expected.

### Proposed approach
1. Extend the pipeline to incorporate OpenStreetMap semantic features (street type, land use, POI density) as additional context in the prompt
2. Alternatively, explore retrieval-augmented generation: for each query image, retrieve the top-k most similar labelled images from Place Pulse as few-shot examples
3. Compare spatial autocorrelation (Moran's I) of scores from the three approaches: (a) current per-image zero-shot, (b) OSM-augmented prompt, (c) few-shot retrieval
4. Use a test dataset of 1000+ images with spatial coverage sufficient to compute local Moran's I

### Possible datasets
- OpenStreetMap (global, free)
- Place Pulse 2.0 as few-shot pool
- Mapillary or Google Street View for imagery

### Testable hypothesis
Perception scores generated with OSM-augmented prompts will exhibit higher spatial autocorrelation (Moran's I) than zero-shot scores, indicating that incorporating neighbourhood context produces more geographically coherent outputs.

### Evaluation strategy
- Moran's I for global spatial autocorrelation
- Local Indicators of Spatial Association (LISA) to identify hotspots
- Qualitative map inspection
- Optional: holdout image prediction from neighbourhood average

### Master's thesis feasibility
YES — Using public APIs and existing tools (PySAL, OSMnx), this is achievable in one year. The key novel contribution is the prompt augmentation with spatial features.

### PhD extension
A PhD could develop a full GeoAI foundation model that jointly encodes street-level imagery and spatial context embeddings, connecting to the emerging literature on spatial foundation models (GeoGPT, GeoLLM, Space2Vec) and providing a rigorous benchmark for spatially consistent urban perception inference.
