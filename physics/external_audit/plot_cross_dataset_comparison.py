"""Publication-quality figures: PRC2025 vs DASHlink external pilot.

Compares absolute MAE levels and qualitative inductive-bias effects
(energy features, fuel-flow target, ML vs physics) across datasets.

Important: absolute MAE is *not* directly comparable across datasets
(different label construction, aircraft mix, and interval scales). These
figures support qualitative cross-dataset narrative for the paper.

PRC2025 values are taken from project tables (flight-level split):
  - OpenAP-only: figures/table_model_comparison.csv
  - Direct / Energy hybrids: figures/table_significance_v3_e6.csv
  - Flow E+W: figures/table_fuel_flow.csv (best XGB)

DASHlink values: audit_results/dashlink_pilot/ (15-flight pilot).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "cross_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data (kg MAE unless noted)
# ---------------------------------------------------------------------------
# PRC2025 — Level-1 flight holdout (project tables; approximate where noted)
PRC = {
    "direct_baseline": 86.31,   # OpenAP Hybrid (base + physics), V3 E6 baseline
    "direct_energy": 84.48,     # Energy Hybrid
    "flow_energy": 80.06,       # Fuel flow (E+W) XGB — best flow MAE
    "physics_only": 655.03,     # OpenAP-only
    # ΔMAE = new − baseline  (negative ⇒ improvement)
    "energy_delta": -1.82,      # Energy Hybrid vs OpenAP Hybrid
    "energy_ci": (-2.92, -0.67),
    "flow_delta": -3.70,        # Flow E+W (80.06) − Direct E+W (83.76)
    "flow_ci": None,            # not a single pooled CI in tables; leave open
    "ml_best": 80.06,           # best ML on PRC for this comparison
}

# DASHlink Project 85 pilot — audit_results/dashlink_pilot
DASH = {
    "direct_baseline": 25.54,   # Direct · base + physics
    "direct_energy": 20.70,     # Direct · base + energy + physics
    "flow_energy": 18.06,       # Flow · base + energy + physics
    "physics_only": 140.05,     # OpenAP-only
    "energy_delta": -4.85,      # Base+Energy vs Base
    "energy_ci": (-6.87, -2.88),
    "flow_delta": -2.64,        # Flow+Energy vs Direct+Energy
    "flow_ci": (-4.63, -0.75),
    "ml_best": 18.06,
}

# Consistent brand colors
COLOR_PRC = "#2E5A88"       # deep steel blue
COLOR_DASH = "#C45C26"      # burnt orange
PALETTE = [COLOR_PRC, COLOR_DASH]
DATASET_LABELS = ["PRC2025", "DASHlink pilot"]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _annotate_bars(ax, bars, fmt="{:.1f}", offset=0.02, fontsize=8.5):
    """Place value labels above (or below for negative) bars."""
    ymax = ax.get_ylim()[1]
    ymin = ax.get_ylim()[0]
    span = ymax - ymin
    for bar in bars:
        h = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        if h >= 0:
            y = h + offset * span
            va = "bottom"
        else:
            y = h - offset * span
            va = "top"
        ax.text(x, y, fmt.format(h), ha="center", va=va, fontsize=fontsize, fontweight="medium")


def fig1_model_performance_comparison(path: Path) -> None:
    """Fig 1 — Grouped bars: MAE by model type for PRC2025 vs DASHlink.

    Communicates absolute error *levels* differ by dataset scale while ranking
    (physics worst, Flow best among ML) is shared.
    """
    models = [
        "Physics-only\n(OpenAP)",
        "Direct\nBaseline",
        "Direct\n+ Energy",
        "Flow\n+ Energy",
    ]
    prc_vals = [
        PRC["physics_only"],
        PRC["direct_baseline"],
        PRC["direct_energy"],
        PRC["flow_energy"],
    ]
    dash_vals = [
        DASH["physics_only"],
        DASH["direct_baseline"],
        DASH["direct_energy"],
        DASH["flow_energy"],
    ]

    x = np.arange(len(models))
    width = 0.36

    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # --- Panel A: all models including physics (log-scale friendly split) ---
    # Use two panels: left = physics + ML (broken scale via twin approach:
    # actually use log y for left full comparison is clearest for 655 vs 25)
    ax = axes[0]
    b1 = ax.bar(
        x - width / 2, prc_vals, width, label="PRC2025", color=COLOR_PRC, edgecolor="white", linewidth=0.6
    )
    b2 = ax.bar(
        x + width / 2, dash_vals, width, label="DASHlink pilot", color=COLOR_DASH, edgecolor="white", linewidth=0.6
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("MAE [kg]  (log scale)")
    ax.set_title("A. Model MAE by dataset (log scale)")
    ax.legend(frameon=True, loc="upper right")
    ax.set_ylim(10, 1200)
    # Custom annotations (log scale)
    for bars, vals in ((b1, prc_vals), (b2, dash_vals)):
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v * 1.12,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # --- Panel B: ML-only linear scale (readable ranking) ---
    ax2 = axes[1]
    ml_models = models[1:]
    x2 = np.arange(len(ml_models))
    b3 = ax2.bar(
        x2 - width / 2,
        prc_vals[1:],
        width,
        label="PRC2025",
        color=COLOR_PRC,
        edgecolor="white",
        linewidth=0.6,
    )
    b4 = ax2.bar(
        x2 + width / 2,
        dash_vals[1:],
        width,
        label="DASHlink pilot",
        color=COLOR_DASH,
        edgecolor="white",
        linewidth=0.6,
    )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(ml_models)
    ax2.set_ylabel("MAE [kg]")
    ax2.set_title("B. ML models only (linear scale)")
    ax2.legend(frameon=True, loc="upper right")
    _annotate_bars(ax2, b3, fmt="{:.1f}")
    _annotate_bars(ax2, b4, fmt="{:.1f}")
    ax2.set_ylim(0, max(prc_vals[1:]) * 1.25)

    fig.suptitle(
        "Model Performance Comparison: PRC2025 vs DASHlink",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Note: absolute MAE scales are not comparable across datasets (different labels & fleets).\n"
        "PRC = flight-level EUROCONTROL; DASHlink = 15-flight Project 85 pilot (integrated fuel flow).",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def fig2_energy_feature_benefit(path: Path) -> None:
    """Fig 2 — ΔMAE when adding energy-state features (negative = improvement).

    Shows both datasets improve with energy features; DASHlink pilot effect is
    larger in absolute kg on its own scale, with CI excluding zero.
    """
    labels = DATASET_LABELS
    deltas = [PRC["energy_delta"], DASH["energy_delta"]]
    # Asymmetric error bars from bootstrap 95% CIs: shape (2, n) = lower/upper
    yerr_arr = np.array(
        [
            [
                PRC["energy_delta"] - PRC["energy_ci"][0],
                DASH["energy_delta"] - DASH["energy_ci"][0],
            ],
            [
                PRC["energy_ci"][1] - PRC["energy_delta"],
                DASH["energy_ci"][1] - DASH["energy_delta"],
            ],
        ]
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        deltas,
        width=0.55,
        color=PALETTE,
        edgecolor="white",
        linewidth=0.8,
        yerr=yerr_arr,
        capsize=6,
        error_kw={"elinewidth": 1.4, "capthick": 1.4, "ecolor": "#333333"},
    )
    ax.axhline(0, color="black", lw=1.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ΔMAE [kg]\n(Base+Energy − Base; negative = better)")
    ax.set_title("Effect of Adding Energy Features")
    _annotate_bars(ax, bars, fmt="{:.2f}", offset=0.04)

    # Significance callouts (placed to avoid bar / footer collisions)
    ax.text(
        0.0,
        PRC["energy_ci"][0] - 0.35,
        "95% CI excludes 0",
        ha="center",
        va="top",
        fontsize=8,
        color=COLOR_PRC,
    )
    ax.text(
        1.0,
        DASH["energy_ci"][0] - 0.35,
        "95% CI excludes 0",
        ha="center",
        va="top",
        fontsize=8,
        color=COLOR_DASH,
    )

    ax.set_ylim(min(d["energy_ci"][0] for d in (PRC, DASH)) - 1.2, 0.8)
    fig.text(
        0.5,
        -0.02,
        "PRC: Energy Hybrid vs OpenAP Hybrid  |  DASHlink: Base+Energy vs Base (pilot)",
        ha="center",
        fontsize=8,
        style="italic",
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def fig3_fuel_flow_advantage(path: Path) -> None:
    """Fig 3 — How much Flow+Energy beats matched Direct (ΔMAE).

    Negative ΔMAE means Fuel-Flow target recovers lower interval fuel error.
    """
    labels = DATASET_LABELS
    deltas = [PRC["flow_delta"], DASH["flow_delta"]]

    # DASHlink has CI; PRC uses point estimate only (draw as bar without CI)
    yerr_lo = [0.0, DASH["flow_delta"] - DASH["flow_ci"][0]]
    yerr_hi = [0.0, DASH["flow_ci"][1] - DASH["flow_delta"]]
    yerr = np.array([yerr_lo, yerr_hi])

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        deltas,
        width=0.55,
        color=PALETTE,
        edgecolor="white",
        linewidth=0.8,
        yerr=yerr,
        capsize=6,
        error_kw={"elinewidth": 1.4, "capthick": 1.4, "ecolor": "#333333"},
    )
    ax.axhline(0, color="black", lw=1.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ΔMAE [kg]\n(Flow+Energy − Direct+Energy; negative = Flow better)")
    ax.set_title("Fuel-Flow Target Advantage over Direct Regression")
    _annotate_bars(ax, bars, fmt="{:.2f}", offset=0.05)

    ax.text(
        0,
        PRC["flow_delta"] - 0.35,
        "point est. (E+W)",
        ha="center",
        va="top",
        fontsize=8,
        color=COLOR_PRC,
    )
    ax.text(
        1,
        DASH["flow_ci"][0] - 0.25,
        "95% CI excludes 0",
        ha="center",
        va="top",
        fontsize=8,
        color=COLOR_DASH,
    )

    lo = min(PRC["flow_delta"], DASH["flow_ci"][0] if DASH["flow_ci"] else DASH["flow_delta"])
    ax.set_ylim(lo - 1.0, 0.6)
    fig.text(
        0.5,
        -0.02,
        "PRC: best Flow E+W vs Direct E+W  |  DASHlink pilot: Flow+Energy vs matched Direct+Energy",
        ha="center",
        fontsize=8,
        style="italic",
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def fig4_physics_vs_ml(path: Path) -> None:
    """Fig 4 — ML improvement over OpenAP physics baseline.

    Side-by-side: absolute physics MAE, best ML MAE, and relative reduction %.
    """
    labels = DATASET_LABELS
    physics = [PRC["physics_only"], DASH["physics_only"]]
    ml = [PRC["ml_best"], DASH["ml_best"]]
    reduction_pct = [
        100.0 * (p - m) / p for p, m in zip(physics, ml)
    ]
    abs_improve = [p - m for p, m in zip(physics, ml)]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # Left: physics vs best ML (grouped bars, log for PRC physics)
    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.36
    b1 = ax.bar(
        x - width / 2,
        physics,
        width,
        label="Physics-only (OpenAP)",
        color="#6B7280",
        edgecolor="white",
    )
    b2 = ax.bar(
        x + width / 2,
        ml,
        width,
        label="Best ML (Flow+Energy)",
        color=[COLOR_PRC, COLOR_DASH],
        edgecolor="white",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE [kg]  (log scale)")
    ax.set_title("A. Physics baseline vs best ML")
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    for bars, vals in ((b1, physics), (b2, ml)):
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v * 1.15,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    ax.set_ylim(10, 1200)

    # Right: relative improvement %
    ax2 = axes[1]
    bars = ax2.bar(
        x,
        reduction_pct,
        width=0.5,
        color=PALETTE,
        edgecolor="white",
        linewidth=0.8,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("MAE reduction vs OpenAP [%]")
    ax2.set_title("B. Relative ML improvement over physics")
    ax2.set_ylim(0, 100)
    for bar, pct, imp in zip(bars, reduction_pct, abs_improve):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{pct:.1f}%\n(−{imp:.0f} kg)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="medium",
        )

    fig.suptitle(
        "Physics vs ML Improvement Across Datasets",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.03,
        "Both datasets: hybrid ML cuts OpenAP error by >85%. Absolute kg reductions differ by domain scale.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    fig1_model_performance_comparison(OUT_DIR / "fig_cross_dataset_model_mae.png")
    fig2_energy_feature_benefit(OUT_DIR / "fig_cross_dataset_energy_benefit.png")
    fig3_fuel_flow_advantage(OUT_DIR / "fig_cross_dataset_flow_advantage.png")
    fig4_physics_vs_ml(OUT_DIR / "fig_cross_dataset_physics_vs_ml.png")

    # Also copy key outputs next to audit pilot for convenience
    audit_fig = ROOT / "audit_results" / "dashlink_pilot" / "figures"
    audit_fig.mkdir(parents=True, exist_ok=True)
    import shutil

    for name in (
        "fig_cross_dataset_model_mae.png",
        "fig_cross_dataset_energy_benefit.png",
        "fig_cross_dataset_flow_advantage.png",
        "fig_cross_dataset_physics_vs_ml.png",
    ):
        src = OUT_DIR / name
        if src.exists():
            shutil.copy2(src, audit_fig / name)
    print(f"\nAll figures in: {OUT_DIR}")
    print(f"Copies in:      {audit_fig}")


if __name__ == "__main__":
    main()
