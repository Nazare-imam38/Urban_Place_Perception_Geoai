# 07 — Questions for the Project Owner
## Urban Place Perception — Human-Centered GeoAI

The following questions require your personal input. They cannot be answered from the repository alone.

---

## Section A — Authorship and Individual Contribution

**A1.** Did you write all pipeline scripts yourself, or were some generated with AI assistance (e.g., GitHub Copilot, ChatGPT, Kiro)?  
*Why this matters: Graduate applications require accurate attribution. If AI tools assisted, you should describe your role as directing and verifying rather than writing every line.*

**A2.** The project has no Git history. Was the project developed in a different repository or environment before being moved to this workspace? If there is a Git history elsewhere, it could provide timestamps and commit messages that support authorship claims.

**A3.** Is there a supervisor, collaborator, or team member whose contribution should be acknowledged? Or was this entirely independent work?

**A4.** Was this project created for a specific course, competition, research lab, or as an independent portfolio project?

---

## Section B — Execution Status

**B1.** Was Florence-2 ever successfully executed on real images (not debug fallback) during this project?  
- If yes: Where are those results? (They are not in the current repository.)
- If no: What prevented it — hardware constraints, model loading errors, time constraints?

**B2.** The Mapillary API returned persistent HTTP 500 errors. Was this a documented API issue at the time of your work, or do you believe it was a configuration problem? Did you try alternative Mapillary API endpoints or contact their support?

**B3.** Were any results from this project presented or submitted anywhere? (Course assignment, lab meeting, conference poster, competition entry, GitHub public repository?)

---

## Section C — Geographic Data Discrepancy

**C1.** The documentation consistently states the study area is New York City. The actual metadata.csv contains globally distributed coordinates (South America, Africa, Southeast Asia, Australia, Eastern Europe). Were you aware of this discrepancy?

**C2.** The Kaggle dataset (paulchambaz/google-street-view) is documented as Google Street View imagery. Are you aware of what cities or regions it actually covers? The coords.csv in the extracted dataset — did you examine the coordinate range before using it?

**C3.** Would you describe the NYC framing as intentional (a study design you plan to execute with proper NYC data) or an error in the documentation?

---

## Section D — Publication and Academic Context

**D1.** Is this project published, submitted for publication, under review, or intended for submission? If submitted, where?

**D2.** Is there a related manuscript, poster, or technical report associated with this project that is not in this repository?

**D3.** Does the REPORT.md represent a submitted assignment or is it internal documentation?

---

## Section E — Technical Details

**E1.** Why are there two separate output directories (`outputs/florence2/` and `outputs/vlm/`)? Were these two separate pipeline runs at different development stages, or were they run simultaneously with different configurations?

**E2.** The earlier run (`outputs/florence2/`) has only 9 columns in results.csv (no evidence/reasoning), while the later run (`outputs/vlm/`) has 22 columns. Did you redesign the output schema between runs? What prompted the extension to the full explainability schema?

**E3.** The `data/image_metadata.csv` file contains Lahore, Pakistan neighbourhood data (Gulberg, DHA, Old Lahore, Model Town, Liberty Market). Is this from a different project or an earlier version of this project? Was it originally intended to use Lahore imagery?

**E4.** Were there any results from running the pipeline with real Florence-2 output on any subset of images, even if small (1–5 images)? If so, what did the output look like?

---

## Section F — Research Development Plans

**F1.** Do you plan to continue developing this project — specifically, to run Florence-2 with GPU access and validate against human perception data?

**F2.** If you had a GPU available now, which city and dataset would you use for the actual experiment?

**F3.** Have you explored using the OpenAI GPT-4V API or other hosted multimodal models as an alternative to running Florence-2 locally? These could produce real inference results without local GPU requirements.

---

## Section G — Claims Requiring Your Confirmation

The following claims appear in the documentation but cannot be independently verified:

**G1.** REPORT.md states "greenery correlates strongly with visual quality (r≈0.68)" — this needs to be flagged as based on hash-derived scores, not real VLM output. In your applications, do not present these correlations as empirical research findings. Are you comfortable characterizing these as pipeline test results rather than research conclusions?

**G2.** README.md states "our experiments on a curated NYC sample demonstrate the pipeline's capability" — these are debug results, not real model inference. In your applications, how would you characterize the experimental evidence you have produced?

**G3.** The README lists as a contribution "Zero-shot urban perception: No labelled perception data is required; the model generalises through natural-language prompting." This is a correct description of the design intent but has not been empirically demonstrated. Are you comfortable distinguishing "designed to work this way" from "demonstrated to work this way" in your application materials?

---

## Notes for Application Preparation

Based on this audit, the strongest and most defensible claims for your graduate applications are:

1. You designed and implemented a complete six-stage GeoAI pipeline with genuine methodological thought around explainability
2. You developed a structured chain-of-thought prompting architecture for urban perception
3. You identified and documented significant research limitations (model execution failure, geographic data mismatch, validation gap)
4. You demonstrated competency across multiple technical domains: HuggingFace Transformers, geospatial Python, REST APIs, interactive mapping, and publication-quality visualization

The weakest claims — which should be qualified carefully — are:
- Any claim about empirical results or numerical findings (all scores are debug fallback values)
- The NYC geographic framing (the data is not from NYC)
- The "demonstrated capability" language in the abstract and report
