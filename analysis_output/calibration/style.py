"""Styling utilities for journal-compliant figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:  # seaborn is optional at runtime
    sns = None

# Figure geometry in inches
FIG_WIDTH_1COL = 3.4
FIG_WIDTH_2COL = 7.0
GOLDEN_RATIO = 0.618  # pleasing aspect ratio

# Base font sizes (approximate journal guidance)
FONT_SIZE_BASE = 6
FONT_SIZE_TITLE = 6
FONT_SIZE_LEGEND = 5.0

# Fonts — Helvetica is sans-serif
FONT_FAMILY_SANS = "Helvetica"  # or "Helvetica Neue"
# Provide sensible fallbacks in case Helvetica isn't available
FONT_SANS_FALLBACKS = [FONT_FAMILY_SANS, "Arial", "DejaVu Sans"]

# Colorblind-safe palette (Okabe-Ito)
COLOR_PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#F0E442", "#56B4E9", "#E69F00", "#999999",
]

# Marker styling for scatter overlays (matched with merged calibration plots)
MARKER_SIZE = 42
MARKER_EDGE_WIDTH = 0.55
MARKER_FACE_COLOR = None  # default to series color
MARKER_EDGE_COLOR = "#1a1a1a"
MARKER_ALPHA = 0.9

# Standard plot bounds for coverage metrics
COVERAGE_Y_RANGE = (0.5, 1.0)


def apply_publication_style(figsize_width: float = FIG_WIDTH_1COL,
                            aspect_ratio: float = GOLDEN_RATIO,
                            dpi: int = 300) -> None:
    """Configure Matplotlib rcParams for publication-quality figures."""
    mpl.rcParams.update(
        {
            "figure.figsize": (figsize_width, figsize_width * aspect_ratio),
            "figure.dpi": dpi,
            "savefig.dpi": dpi,

            # ✅ use sans-serif family for Helvetica
            "font.family": "sans-serif",
            "font.sans-serif": FONT_SANS_FALLBACKS,

            # Helvetica lacks math glyphs — STIX covers math well
            "mathtext.fontset": "stix",

            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_BASE,
            "axes.labelweight": "semibold",
            "axes.titleweight": "semibold",
            "legend.fontsize": FONT_SIZE_LEGEND,
            "legend.frameon": False,
            "xtick.labelsize": FONT_SIZE_BASE,
            "ytick.labelsize": FONT_SIZE_BASE,
            "axes.prop_cycle": plt.cycler(color=COLOR_PALETTE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.2,
        }
    )

    if sns is not None:
        sns.set_theme(style="whitegrid")
        sns.set_palette(COLOR_PALETTE)
        sns.set_context(
            "notebook",
            rc={
                # ✅ mirror the sans-serif setting in seaborn
                "font.family": "sans-serif",
                "font.sans-serif": ",".join(FONT_SANS_FALLBACKS),
                "axes.titlesize": FONT_SIZE_TITLE,
                "axes.labelsize": FONT_SIZE_BASE,
                "legend.fontsize": FONT_SIZE_LEGEND,
                "xtick.labelsize": FONT_SIZE_BASE,
                "ytick.labelsize": FONT_SIZE_BASE,
            },
        )


def new_figure(width: float = FIG_WIDTH_1COL,
               aspect_ratio: float = GOLDEN_RATIO,
               dpi: int = 300) -> tuple[plt.Figure, plt.Axes]:
    apply_publication_style(width, aspect_ratio, dpi)
    fig, ax = plt.subplots()
    return fig, ax


def styled_subplots(width: float = FIG_WIDTH_1COL,
                    height: float | None = None,
                    aspect_ratio: float = GOLDEN_RATIO,
                    dpi: int = 300,
                    **kwargs):
    """Create subplots with the shared publication style applied."""
    if height is None:
        height = width * aspect_ratio
    else:
        aspect_ratio = height / width if width else aspect_ratio
    apply_publication_style(width, aspect_ratio, dpi)
    kwargs.setdefault("figsize", (width, height))
    return plt.subplots(**kwargs)
