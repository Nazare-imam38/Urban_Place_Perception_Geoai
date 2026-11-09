# config.py
"""
Urban Place Perception - Human-Centered GeoAI
Project Configuration
=====================
Centralised configuration constants for the full pipeline.

All path constants are relative to the repository root and will be created
automatically if they do not already exist.
"""
import os

# ---------------------------------------------------------------------------
# Project identity
# ---------------------------------------------------------------------------
PROJECT_TITLE = "Urban Place Perception - Human‑Centered GeoAI"
PROJECT_VERSION = "1.0"
PROJECT_DESCRIPTION = "A Human‑Centered GeoAI pipeline that infers urban place perception using Vision‑Language Models and provides explainable, interactive visualizations."
PIPELINE_DESCRIPTION = "Download street‑level images → Run Florence‑2 VLM inference → Parse scene description, urban elements and perception scores → Aggregate spatially → Visualize via interactive map and publication‑quality charts."


# Base directories (will be created if missing)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # repository root
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "vlm")
# Backward-compatible alias for visible features
VISIBLE_FEATURES = ["trees", "sidewalk", "crosswalk", "cars", "street lights", "commercial buildings"]  # legacy name

# Study parameters
SAMPLE_SIZE = 50  # number of images to process
SEED = 42

# Mapillary API configuration (kept for reference)
MAPILLARY_ACCESS_TOKEN = "MLY|36486651850982627|e5b936a5a6e766c3ab398bc18aaeb39f"
# NYC bounding box: west, south, east, north (lon, lat)
MAPILLARY_BBOX = "-74.2591,40.4774,-73.7004,40.9176"

# Kaggle dataset slug (fallback source)
KAGGLE_DATASET_SLUG = "paulchambaz/google-street-view"

# Model specification (HuggingFace identifier)
MODEL_NAME = "microsoft/Florence-2-base"
FLORENCE_MODEL = MODEL_NAME

# Fallback dataset URL (publicly available NYC StreetLearn sample)
DATASET_URL = (
    "https://raw.githubusercontent.com/google-research/streetlearn/main/data/nyc/nyc.zip"
)

# Urban visual elements for inference (order matters for CSV schema)
URBAN_VISUAL_ELEMENTS = [
    "trees",
    "sidewalk",
    "crosswalk",
    "cars",
    "street lights",
    "commercial buildings",
]

# ---------------------------------------------------------------------------
# Pipeline stages (used for logging / documentation)
# ---------------------------------------------------------------------------
PIPELINE_STAGES = [
    "Street-Level Image",
    "Vision-Language Model (Florence-2)",
    "Scene Understanding",
    "Urban Visual Elements Detection",
    "Visual Reasoning per Perception Dimension",
    "Perception Indicators (Safety · Walkability · Greenery · Visual Quality)",
    "Interactive GeoAI Map",
    "Spatial Aggregation",
]

# Debug mode flag (can be overridden via CLI arguments)
GRID_SIZE_METERS = 500  # grid cell size (metres) for spatial summary
DEBUG = False

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
