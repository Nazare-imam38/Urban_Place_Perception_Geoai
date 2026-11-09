# 05_create_interactive_map.py
"""
Urban Place Perception - Human-Centered GeoAI
Pipeline Step 5 - Explainable GeoAI Interactive Map
====================================================
Generates an interactive Folium map — an "Explainable GeoAI Interface" —
where every location marker opens a richly structured popup containing:

    • Street-level image thumbnail
    • Scene description (VLM-generated)
    • Detected urban visual elements
    • Per-dimension perception score + visual evidence + reasoning sentence
    • Overall place perception summary
    • Confidence rating

The marker colour encodes the safety score (green = high, red = low).
Optional grid-level choropleth overlay is added from spatial_summary.geojson.

Output: outputs/vlm/map.html

Run with:  python 05_create_interactive_map.py
"""

import ast
import pandas as pd
import folium
from folium import IFrame
from pathlib import Path
from config import OUTPUT_DIR
from utils.inference import image_to_base64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _score_to_color(score: float) -> str:
    """Linear red→green interpolation for a score in [0, 1]."""
    score = max(0.0, min(1.0, score))
    r = int((1 - score) * 200)
    g = int(score * 180)
    return f"#{r:02x}{g:02x}30"


def _score_bar(score: float) -> str:
    """Return an HTML score bar with a numeric label."""
    if score is None:
        return "<span style='color:#888'>N/A</span>"
    pct = int(score * 100)
    # colour gradient: red → amber → green
    if pct >= 70:
        colour = "#4caf50"
    elif pct >= 45:
        colour = "#ff9800"
    else:
        colour = "#f44336"
    return (
        f"<div style='display:flex;align-items:center;gap:6px;margin:2px 0'>"
        f"  <div style='flex:1;background:#333;border-radius:4px;height:8px;'>"
        f"    <div style='width:{pct}%;background:{colour};border-radius:4px;height:8px;'></div>"
        f"  </div>"
        f"  <span style='font-weight:bold;color:{colour};min-width:36px;font-size:12px'>{score:.2f}</span>"
        f"</div>"
    )


def _parse_list_field(raw) -> list:
    """Parse a list field that may be stored as a string representation."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                return ast.literal_eval(raw)
            except Exception:
                pass
        return [item.strip() for item in raw.split(",") if item.strip()]
    return []


def _build_popup(row: pd.Series) -> str:
    """
    Build an HTML popup card that exposes the full explainability chain:
    Scene Description → Urban Elements → per-dimension score + evidence + reasoning
    → Overall Summary → Confidence.
    """
    # Thumbnail
    try:
        thumb = image_to_base64(Path(row["filepath"]))
        thumb_html = (
            f"<img src='{thumb}' "
            f"style='width:100%;border-radius:6px;margin-bottom:8px;'>"
        )
    except Exception:
        thumb_html = ""

    # Urban elements as badges
    elements = _parse_list_field(row.get("detected_urban_elements", []))
    if not elements:
        elements = _parse_list_field(row.get("urban_visual_elements", []))
    badge_html = " ".join(
        f"<span style='background:#1e3a5f;color:#90caf9;border-radius:10px;"
        f"padding:1px 7px;font-size:10px;white-space:nowrap'>{e}</span>"
        for e in elements
    ) if elements else "<span style='color:#888'>—</span>"

    # Perception dimensions
    dims = [
        ("🔐 Safety",       "safety"),
        ("🚶 Walkability",  "walkability"),
        ("🌿 Greenery",     "greenery"),
        ("👁 Visual Quality", "visual_quality"),
    ]
    dims_html = ""
    for label, key in dims:
        score = row.get(f"{key}_score", None)
        evidence = _parse_list_field(row.get(f"{key}_evidence", []))
        reasoning = row.get(f"{key}_reasoning", "")
        evidence_str = (
            "<span style='color:#90caf9'>" +
            " · ".join(f"<em>{e}</em>" for e in evidence) +
            "</span>"
        ) if evidence else "<span style='color:#666'>—</span>"

        dims_html += (
            f"<div style='margin:8px 0;padding:8px;background:#1a2332;"
            f"border-radius:6px;border-left:3px solid #1565c0'>"
            f"  <div style='font-weight:600;color:#e3f2fd;font-size:12px;margin-bottom:4px'>"
            f"    {label}</div>"
            f"  {_score_bar(score)}"
            f"  <div style='font-size:10px;margin-top:4px;color:#b0bec5'>"
            f"    <strong style='color:#78909c'>Evidence:</strong> {evidence_str}</div>"
            f"  <div style='font-size:10px;margin-top:3px;color:#b0bec5'>"
            f"    <strong style='color:#78909c'>Reasoning:</strong> {reasoning}</div>"
            f"</div>"
        )

    # Overall summary
    overall = row.get("overall_interpretation", "")
    confidence = row.get("confidence", "")
    conf_color = "#4caf50" if confidence == "high" else "#ff9800"

    popup_html = f"""
<div style='font-family:Inter,Segoe UI,sans-serif;background:#0d1b2a;
            color:#cfd8dc;border-radius:10px;padding:12px;width:300px;
            max-height:520px;overflow-y:auto;box-shadow:0 4px 20px rgba(0,0,0,0.5)'>

  {thumb_html}

  <div style='font-size:10px;color:#78909c;margin-bottom:4px'>
    📍 {row.get("image_id","")}</div>

  <div style='font-size:11px;color:#90a4ae;margin-bottom:8px;
              border-left:2px solid #37474f;padding-left:8px;font-style:italic'>
    {row.get("scene_description","—")}</div>

  <div style='font-size:10px;font-weight:600;color:#78909c;
              text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px'>
    🏙 Urban Elements</div>
  <div style='margin-bottom:10px;line-height:1.8'>{badge_html}</div>

  <div style='font-size:10px;font-weight:600;color:#78909c;
              text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px'>
    📊 Perception Indicators</div>
  {dims_html}

  <div style='margin-top:10px;padding:8px;background:#1a2332;border-radius:6px'>
    <div style='font-size:10px;font-weight:600;color:#78909c;
                text-transform:uppercase;margin-bottom:4px'>💬 Overall Summary</div>
    <div style='font-size:10px;color:#b0bec5'>{overall}</div>
  </div>

  <div style='margin-top:8px;font-size:10px;color:#78909c'>
    Confidence: <strong style='color:{conf_color}'>{confidence.upper()}</strong>
  </div>

</div>
"""
    return popup_html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results_path = Path(OUTPUT_DIR) / "results.csv"
    df = pd.read_csv(results_path)

    # Base map — CartoDB Dark Matter for a modern GeoAI aesthetic
    m = folium.Map(
        location=[40.73, -73.99],
        zoom_start=11,
        tiles="CartoDB dark_matter",
    )

    # Optional grid choropleth overlay
    spatial_geojson_path = Path(OUTPUT_DIR) / "spatial_summary.geojson"
    if spatial_geojson_path.is_file():
        def _make_style(metric):
            def style_fn(feature):
                s = feature["properties"].get(metric) or 0
                return {
                    "fillColor": _score_to_color(s),
                    "color": "#1565c0",
                    "weight": 0.4,
                    "fillOpacity": 0.35,
                }
            return style_fn

        for metric, label in [
            ("safety_score", "Safety Grid"),
            ("walkability_score", "Walkability Grid"),
            ("greenery_score", "Greenery Grid"),
            ("visual_quality_score", "Visual Quality Grid"),
        ]:
            folium.GeoJson(
                spatial_geojson_path,
                name=label,
                style_function=_make_style(metric),
                tooltip=folium.GeoJsonTooltip(
                    fields=[metric],
                    aliases=[label.replace(" Grid", "")],
                    style="font-family:Inter,sans-serif;font-size:11px;"
                ),
                show=False,  # collapsed by default; user can enable
            ).add_to(m)

    # Point markers
    locations = []
    for _, row in df.iterrows():
        lat = row.get("latitude", 0.0) or 0.0
        lon = row.get("longitude", 0.0) or 0.0
        if lat == 0.0 and lon == 0.0:
            continue
        locations.append([lat, lon])

        safety = row.get("safety_score", 0.5) or 0.5
        popup_html = _build_popup(row)
        iframe = IFrame(popup_html, width=324, height=540)
        popup = folium.Popup(iframe, max_width=340)

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color=_score_to_color(safety),
            fill=True,
            fill_color=_score_to_color(safety),
            fill_opacity=0.85,
            weight=1.5,
            popup=popup,
            tooltip=(
                f"Safety: {safety:.2f} | "
                f"Walk: {row.get('walkability_score',0):.2f} | "
                f"Green: {row.get('greenery_score',0):.2f}"
            ),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    if locations:
        m.fit_bounds(locations)

    out_path = Path(OUTPUT_DIR) / "map.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"[OK] Explainable GeoAI map saved → {out_path}")


if __name__ == "__main__":
    main()
