"""Styling utilities for journal-compliant figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Figure geometry in inches
FIG_WIDTH_1COL = 3.4
FIG_WIDTH_2COL = 7.0
GOLDEN_RATIO = 0.618  # pleasing aspect ratio

# Base font sizes (approximate journal guidance)
FONT_SIZE_BASE = 8.5
FONT_SIZE_TITLE = 9.5
FONT_SIZE_LEGEND = 8.0
FONT_FAMILY_SERIF = "Times New Roman"
FONT_FAMILY_SANS = "Arial"

# Colorblind-safe palette (Okabe-Ito)
COLOR_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion (orange)
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#999999",  # grey
]


def apply_publication_style(figsize_width: float = FIG_WIDTH_1COL,
                            aspect_ratio: float = GOLDEN_RATIO,
                            dpi: int = 300) -> None:
    """Configure Matplotlib rcParams for publication-quality figures."""
    mpl.rcParams.update(
        {
            "figure.figsize": (figsize_width, figsize_width * aspect_ratio),
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.family": "serif",
            "font.serif": [FONT_FAMILY_SERIF],
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


def new_figure(width: float = FIG_WIDTH_1COL,
               aspect_ratio: float = GOLDEN_RATIO,
               dpi: int = 300) -> plt.Figure:
    apply_publication_style(width, aspect_ratio, dpi)
    fig, ax = plt.subplots()
    return fig
