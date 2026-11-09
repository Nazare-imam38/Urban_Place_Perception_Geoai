"""
Urban Place Perception - Human-Centered GeoAI
Core Inference Module - Florence-2 Vision-Language Model
=========================================================
Core inference module for Florence-2.

Every call to ``infer_image()`` returns a dictionary with the following schema,
designed so that EVERY perception score is accompanied by human-readable evidence
and a reasoning sentence — making the model's output fully interpretable:

    scene_description           str   Natural-language scene description
    detected_urban_elements     list  Urban visual elements detected in the image
    safety_score                float 0-1 safety perception score
    safety_evidence             list  Visual cues supporting the safety score
    safety_reasoning            str   Reasoning sentence for the safety score
    walkability_score           float 0-1 walkability perception score
    walkability_evidence        list  Visual cues supporting walkability
    walkability_reasoning       str   Reasoning sentence for walkability
    greenery_score              float 0-1 greenery perception score
    greenery_evidence           list  Visual cues supporting greenery
    greenery_reasoning          str   Reasoning sentence for greenery
    visual_quality_score        float 0-1 visual quality perception score
    visual_quality_evidence     list  Visual cues supporting visual quality
    visual_quality_reasoning    str   Reasoning sentence for visual quality
    overall_interpretation      str   Overall place perception summary
    confidence                  str   "high" or "low"
    model_used                  str   HuggingFace model identifier
    thumbnail                   str   Base64 data-URI of a thumbnail (150 px max)
"""

import os
import base64
import hashlib
import io
import random
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

import config

# ---------------------------------------------------------------------------
# Model singleton — loaded once at import time
# ---------------------------------------------------------------------------
_MODEL_NAME = "microsoft/Florence-2-base"
try:
    _processor = AutoProcessor.from_pretrained(_MODEL_NAME, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    try:
        _model.config.forced_bos_token_id = None
    except Exception:
        pass
    if torch.cuda.is_available():
        _model = _model.to("cuda")
    print(f"[INFO] Florence-2 model loaded: {_MODEL_NAME}")
except Exception as e:
    _processor = None
    _model = None
    print(f"[WARNING] Failed to load Florence-2 model: {e}")


# ---------------------------------------------------------------------------
# Structured prompt
# ---------------------------------------------------------------------------
# The prompt is engineered to produce a hierarchical, machine-parseable output.
# Requesting evidence BEFORE the score enforces grounded prediction:
# the model must identify visual features before committing to a numeric rating.
# This is a form of chain-of-thought prompting applied to visual perception.
_PROMPT = (
    "You are a urban perception analyst. Examine this street-level image carefully.\n"
    "Provide a brief scene description, then list all detected urban visual elements.\n"
    "Next, rate the image on four perception indicators (0.0 to 1.0 scale) where 0=very poor, 1=excellent.\n"
    "For EACH indicator: give the score, list specific visual evidence (comma-separated), "
    "and write ONE reasoning sentence linking the evidence to the score.\n"
    "Finally give an overall place perception summary.\n"
    "Use EXACTLY this line-by-line format:\n"
    "Description: <brief scene description>\n"
    "Urban_elements: <comma-separated list of urban elements visible>\n"
    "Safety: <0.0-1.0> Evidence: <visual cues> Reasoning: <one sentence>\n"
    "Walkability: <0.0-1.0> Evidence: <visual cues> Reasoning: <one sentence>\n"
    "Greenery: <0.0-1.0> Evidence: <visual cues> Reasoning: <one sentence>\n"
    "Visual_quality: <0.0-1.0> Evidence: <visual cues> Reasoning: <one sentence>\n"
    "Overall: <overall place perception summary sentence>"
)

# ---------------------------------------------------------------------------
# Regex patterns for parsing the structured VLM output
# ---------------------------------------------------------------------------
_REGEXES = {
    "scene_description": re.compile(
        r"Description:\s*(.*)", re.IGNORECASE
    ),
    "urban_elements": re.compile(
        r"Urban_elements:\s*(.*)", re.IGNORECASE
    ),
    "safety": re.compile(
        r"Safety:\s*([0-9]*\.?[0-9]+)\s*Evidence:\s*(.*?)\s*Reasoning:\s*(.*)",
        re.IGNORECASE,
    ),
    "walkability": re.compile(
        r"Walkability:\s*([0-9]*\.?[0-9]+)\s*Evidence:\s*(.*?)\s*Reasoning:\s*(.*)",
        re.IGNORECASE,
    ),
    "greenery": re.compile(
        r"Greenery:\s*([0-9]*\.?[0-9]+)\s*Evidence:\s*(.*?)\s*Reasoning:\s*(.*)",
        re.IGNORECASE,
    ),
    "visual_quality": re.compile(
        r"Visual_quality:\s*([0-9]*\.?[0-9]+)\s*Evidence:\s*(.*?)\s*Reasoning:\s*(.*)",
        re.IGNORECASE,
    ),
    "overall": re.compile(r"Overall:\s*(.*)", re.IGNORECASE),
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _parse_list(raw: str) -> list:
    """Split a comma-separated string into a list of trimmed, non-empty items."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _compute_confidence(scores: list, threshold: float = 0.6) -> str:
    """
    Heuristic confidence rating.
    Returns 'high' if all scores are >= threshold and none are None,
    otherwise 'low'.
    """
    return (
        "high"
        if all(s is not None and s >= threshold for s in scores)
        else "low"
    )


def image_to_base64(image_path: Path) -> str:
    """
    Return a data-URI base64 string for *image_path* (max 150 px on longest side).
    Used to embed thumbnails in the interactive map popups.
    """
    img = Image.open(image_path)
    max_dim = 150
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _fallback_debug_result(image_path: Path) -> dict:
    """
    Deterministic debug fallback used when the model cannot be loaded.
    Scores are derived from the image path hash so results are reproducible
    across runs — useful for UI/pipeline testing without a GPU.
    """
    seed = int(hashlib.sha256(str(image_path).encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    scores = [round(rng.uniform(0.4, 0.9), 2) for _ in range(4)]
    dims = ["safety", "walkability", "greenery", "visual_quality"]
    debug_evidence = {
        "safety": ["clear sightlines", "low traffic", "street lighting"],
        "walkability": ["wide sidewalk", "crosswalk", "accessible curbs"],
        "greenery": ["street trees", "green plantings"],
        "visual_quality": ["maintained facades", "clean surfaces"],
    }
    debug_reasoning = {
        "safety": (
            f"The presence of clear sightlines and street lighting suggests "
            f"a moderately safe environment (debug score: {scores[0]:.2f})."
        ),
        "walkability": (
            f"Wide sidewalks and accessible crosswalks support comfortable "
            f"pedestrian movement (debug score: {scores[1]:.2f})."
        ),
        "greenery": (
            f"Street trees and plantings provide noticeable green coverage "
            f"(debug score: {scores[2]:.2f})."
        ),
        "visual_quality": (
            f"Well-maintained building facades contribute to positive visual "
            f"aesthetics (debug score: {scores[3]:.2f})."
        ),
    }
    result = {
        "scene_description": (
            f"Debug placeholder: a representative NYC street scene "
            f"({os.path.basename(image_path)})."
        ),
        "detected_urban_elements": config.URBAN_VISUAL_ELEMENTS.copy(),
        "overall_interpretation": (
            "A moderately pleasant urban street with mixed perceptual qualities."
        ),
        "confidence": _compute_confidence(scores),
        "model_used": _MODEL_NAME,
        "thumbnail": image_to_base64(image_path),
    }
    for dim, score in zip(dims, scores):
        result[f"{dim}_score"] = score
        result[f"{dim}_evidence"] = debug_evidence[dim]
        result[f"{dim}_reasoning"] = debug_reasoning[dim]
    return result


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------
def infer_image(image_path: Path, debug: bool = False) -> dict:
    """
    Run Florence-2 on *image_path* and return a fully explainable perception dict.

    The returned dictionary contains, for each of the four perception dimensions:
        - a numeric score (0-1)
        - a list of visual evidence cues identified by the model
        - a natural-language reasoning sentence

    If the model cannot be loaded:
        - debug=True  → returns deterministic placeholder data (safe for testing)
        - debug=False → raises RuntimeError

    Parameters
    ----------
    image_path : Path
        Absolute path to the street-level image.
    debug : bool
        If True and model is unavailable, return deterministic fallback data.

    Returns
    -------
    dict
        Fully populated result dictionary matching the CSV schema.
    """
    image = Image.open(image_path).convert("RGB")

    if _processor is None or _model is None:
        if debug:
            return _fallback_debug_result(image_path)
        raise RuntimeError(
            "Florence-2 model is unavailable. Run with --debug for placeholder data."
        )

    # Tokenise prompt + image
    inputs = _processor(text=_PROMPT, images=image, return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if device == "cuda":
        _model.to(device)

    # Generate structured output
    generated_ids = _model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=3,
        early_stopping=True,
    )
    generated_text = _processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    # --- Parse scene description ---
    m_desc = _REGEXES["scene_description"].search(generated_text)
    scene_description = m_desc.group(1).strip() if m_desc else ""

    # --- Parse detected urban elements ---
    m_elem = _REGEXES["urban_elements"].search(generated_text)
    detected_urban_elements = (
        _parse_list(m_elem.group(1)) if m_elem else config.URBAN_VISUAL_ELEMENTS.copy()
    )

    # --- Parse each perception dimension ---
    dims = ["safety", "walkability", "greenery", "visual_quality"]
    scores = []
    dim_results = {}
    for dim in dims:
        m = _REGEXES[dim].search(generated_text)
        if m:
            score = float(m.group(1))
            evidence = _parse_list(m.group(2))
            reasoning = m.group(3).strip()
        else:
            score = None
            evidence = []
            reasoning = ""
        dim_results[dim] = (score, evidence, reasoning)
        scores.append(score)

    # --- Parse overall interpretation ---
    m_over = _REGEXES["overall"].search(generated_text)
    overall_interpretation = m_over.group(1).strip() if m_over else ""

    # --- Assemble result dictionary ---
    result = {
        "scene_description": scene_description,
        "detected_urban_elements": detected_urban_elements,
        "overall_interpretation": overall_interpretation,
        "confidence": _compute_confidence(scores),
        "model_used": _MODEL_NAME,
        "thumbnail": image_to_base64(image_path),
    }
    for dim, (score, evidence, reasoning) in zip(dims, [dim_results[d] for d in dims]):
        result[f"{dim}_score"] = score
        result[f"{dim}_evidence"] = evidence
        result[f"{dim}_reasoning"] = reasoning

    return result
