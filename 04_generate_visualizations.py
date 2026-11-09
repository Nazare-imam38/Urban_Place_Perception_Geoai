# 04_generate_visualizations.py
"""
Urban Place Perception - Human-Centered GeoAI
Pipeline Step 4 - Publication-Quality Visualizations
=====================================================
Generates comprehensive, publication-ready charts from VLM inference results.

Charts produced (saved as PNG under outputs/vlm/charts/ in dark-mode style):
  • Average perception scores bar chart
  • Radar chart of top-5 highest-scoring images
  • Violin & box plots per perception indicator (score distribution)
  • Frequency bar chart of detected urban visual elements
  • Pearson correlation matrix heatmap across perception indicators
  • Top urban visual elements aggregated across the full dataset

Run with:  python 04_generate_visualizations.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pathlib import Path
from config import OUTPUT_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INDICATORS = ["safety_score", "walkability_score", "greenery_score", "visual_quality_score"]

# Dark‑mode style settings
plt.style.use('dark_background')
DARK_BG = "#212121"
ACCENT = "#90caf9"
BAR_COLOR = "#4a90e2"

# ---------------------------------------------------------------------------
# Basic bar chart of average scores (existing)
# ---------------------------------------------------------------------------
def _save_bar_chart(df: pd.DataFrame, out_path: Path):
    avg_scores = df[INDICATORS].mean()
    plt.figure(figsize=(6, 4), facecolor=DARK_BG)
    avg_scores.plot(kind="bar", color=BAR_COLOR)
    plt.ylim(0, 1)
    plt.title("Average Perception Scores (50 images)", color=ACCENT)
    plt.ylabel("Score (0-1)", color=ACCENT)
    plt.xticks(rotation=0, color=ACCENT)
    plt.yticks(color=ACCENT)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor=DARK_BG)
    plt.close()
    print(f"[OK] Bar chart saved -> {out_path}")

# ---------------------------------------------------------------------------
# Radar chart of top‑5 images (existing)
# ---------------------------------------------------------------------------
def _save_radar_chart(df: pd.DataFrame, out_path: Path):
    df["overall"] = df[INDICATORS].mean(axis=1)
    top5 = df.nlargest(5, "overall")
    categories = INDICATORS
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(6, 6), facecolor=DARK_BG)
    ax = plt.subplot(111, polar=True)
    for _, row in top5.iterrows():
        values = row[INDICATORS].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row["image_id"])
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=ACCENT)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.set_title("Radar Chart - Top 5 Images (overall score)", color=ACCENT)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor=DARK_BG)
    plt.close()
    print(f"[OK] Radar chart saved -> {out_path}")

# ---------------------------------------------------------------------------
# Violin + Box plots per indicator
# ---------------------------------------------------------------------------
def _save_violin_boxplots(df: pd.DataFrame, out_path: Path):
    """Create side-by-side violin and box plots for the four perception indicators."""
    plt.figure(figsize=(10, 6), facecolor=DARK_BG)
    data = [df[ind].dropna() for ind in INDICATORS]
    # Violin plot
    parts = plt.violinplot(data, positions=range(1, len(INDICATORS) + 1), showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:  # type: ignore[index]
        pc.set_facecolor(BAR_COLOR)
        pc.set_alpha(0.6)
    # Box plot overlay
    plt.boxplot(data, positions=range(1, len(INDICATORS) + 1), widths=0.1, patch_artist=True,
                boxprops=dict(facecolor=ACCENT, color=ACCENT),
                medianprops=dict(color='white'),
                whiskerprops=dict(color=ACCENT),
                capprops=dict(color=ACCENT),
                flierprops=dict(color=ACCENT, markeredgecolor=ACCENT))
    plt.xticks(range(1, len(INDICATORS) + 1), [ind.replace('_score', '').capitalize() for ind in INDICATORS], color=ACCENT)
    plt.yticks(np.linspace(0, 1, 6), color=ACCENT)
    plt.ylim(0, 1)
    plt.title('Score Distribution per Indicator', color=ACCENT)
    plt.xlabel('Indicator', color=ACCENT)
    plt.ylabel('Score (0-1)', color=ACCENT)
    plt.grid(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor=DARK_BG)
    plt.close()
    print(f"[OK] Violin/Box plot saved -> {out_path}")

# ---------------------------------------------------------------------------
# Frequency of detected urban visual elements
# ---------------------------------------------------------------------------
def _parse_elements(series: pd.Series) -> list:
    """Parse a pandas Series where each entry may be a list or a string representation of a list.
    Returns a flat list of all elements."""
    elements = []
    for val in series.dropna():
        if isinstance(val, list):
            elements.extend(val)
        elif isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    elements.extend(parsed)
                else:
                    elements.append(str(parsed))
            except Exception:
                elements.extend([e.strip() for e in val.split(',') if e.strip()])
        else:
            elements.append(str(val))
    return elements

def _save_feature_frequency(df: pd.DataFrame, out_path: Path, top_n: int = 15):
    """Bar chart of the most frequent urban visual elements detected across the dataset."""
    elements = _parse_elements(df["urban_visual_elements"])
    freq = pd.Series(elements).value_counts().head(top_n)
    plt.figure(figsize=(8, 5), facecolor=DARK_BG)
    freq.sort_values().plot(kind="barh", color=ACCENT)
    plt.title('Top Urban Visual Elements Frequency', color=ACCENT)
    plt.xlabel('Count', color=ACCENT)
    plt.ylabel('Element', color=ACCENT)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor=DARK_BG)
    plt.close()
    print(f"[OK] Feature frequency chart saved -> {out_path}")

# ---------------------------------------------------------------------------
# Correlation matrix heatmap
# ---------------------------------------------------------------------------
def _save_correlation_heatmap(df: pd.DataFrame, out_path: Path):
    """Heatmap of Pearson correlations between the four perception indicators."""
    corr = df[INDICATORS].corr()
    plt.figure(figsize=(6, 5), facecolor=DARK_BG)
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Correlation'},
                annot_kws={'color': 'white'},
                fmt='.2f')
    plt.title('Correlation Matrix of Perception Indicators', color=ACCENT)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor=DARK_BG)
    plt.close()
    print(f"[OK] Correlation heatmap saved -> {out_path}")

# ---------------------------------------------------------------------------
# Top urban elements overall (horizontal bar)
# ---------------------------------------------------------------------------
def _save_top_urban_elements(df: pd.DataFrame, out_path: Path, top_n: int = 10):
    """Horizontal bar chart of the most common urban elements overall (aggregated)."""
    elements = _parse_elements(df["urban_visual_elements"])
    freq = pd.Series(elements).value_counts().head(top_n)
    plt.figure(figsize=(8, 5), facecolor=DARK_BG)
    freq.sort_values().plot(kind="barh", color=BAR_COLOR)
    plt.title('Top Urban Elements Across Dataset', color=ACCENT)
    plt.xlabel('Count', color=ACCENT)
    plt.ylabel('Element', color=ACCENT)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor=DARK_BG)
    plt.close()
    print(f"[OK] Top urban elements chart saved -> {out_path}")

# ---------------------------------------------------------------------------
# Main driver - invoke all visualizations
# ---------------------------------------------------------------------------
def main():
    results_path = Path(OUTPUT_DIR) / "results.csv"
    df = pd.read_csv(results_path)
    charts_dir = Path(OUTPUT_DIR) / "charts"
    # Existing visualizations
    _save_bar_chart(df, charts_dir / "average_scores.png")
    _save_radar_chart(df, charts_dir / "radar_top5.png")
    # New publication‑quality visualizations
    _save_violin_boxplots(df, charts_dir / "violin_boxplots.png")
    _save_feature_frequency(df, charts_dir / "feature_frequency.png")
    _save_correlation_heatmap(df, charts_dir / "correlation_heatmap.png")
    _save_top_urban_elements(df, charts_dir / "top_urban_elements.png")

if __name__ == "__main__":
    main()
