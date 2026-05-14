"""Paper-figure rendering for the SEMANTICS 2026 submission.

Reads a scores CSV produced by `mirror analyze` and writes three PDF
figures plus a summary stats CSV under `<output_dir>/`. Each figure
defends one of the load-bearing claims:

  Figure 1 - ablation effects (syntax x few_shot).
  Figure 2 - DSL comparison (dcpl vs odrl), best-ablation slice.
  Figure 3 - structural-vs-semantic agreement, per DSL.

All numeric values plotted are also dumped to `summary_stats.csv` so
paper prose can quote exact numbers without re-deriving them.
"""

import logging
import os
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from src.analyze import STRUCTURAL_FIELDS

METRICS: tuple[str, ...] = (
    *(f"structural_{f}" for f in STRUCTURAL_FIELDS),
    "semantic",
)

METRIC_LABELS: dict[str, str] = {
    "structural_matching": "matching",
    "structural_alignment": "alignment",
    "structural_type_consistency": "type consistency",
    "structural_content_fidelity": "content fidelity",
    "semantic": "semantic",
}

# Okabe-Ito colorblind-safe palette.
PALETTE: dict[str, str] = {
    "few_shot_off": "#0072B2",
    "few_shot_on": "#D55E00",
    "dcpl": "#009E73",
    "odrl": "#CC79A7",
    "ab_00": "#999999",
    "ab_10": "#D55E00",
    "ab_01": "#0072B2",
    "ab_11": "#009E73",
}


def _paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _col(df: Any, name: str) -> pd.Series:
    """Single-column accessor narrowed to `Series`.

    Pandas' `__getitem__` overloads return `Series | DataFrame` even on a
    plain DataFrame because duplicate labels can fan out; we never have
    duplicate columns. The `Any` parameter type also accepts the result of
    a boolean-mask slice (which pyright can't narrow to DataFrame on its
    own), so the same helper covers both `df[col]` and `df[mask][col]`.
    """
    return cast(pd.Series, df[name])


def _filter(df: Any, mask: Any) -> pd.DataFrame:
    """Boolean-mask slice narrowed to `DataFrame`."""
    return cast(pd.DataFrame, df[mask])


def _to_nullable_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype("boolean")
    return series.map(
        {"True": True, "False": False, True: True, False: False}
    ).astype("boolean")


def _load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("ablation_syntax", "ablation_few_shot", "sym1_success", "sym2_success"):
        df[col] = _to_nullable_bool(_col(df, col))
    for col in METRICS:
        df[col] = pd.to_numeric(_col(df, col), errors="coerce")
    return df


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return `(mean, lo, hi)`; NaNs in `values` are dropped first."""
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return (float("nan"), float("nan"), float("nan"))
    m = float(clean.mean())
    if len(clean) == 1:
        return (m, m, m)
    rng = np.random.default_rng(seed=seed)
    draws = rng.choice(clean, size=(n_boot, len(clean)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])
    return (m, float(lo), float(hi))


def _successful(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to cells with sym1 and sym2 both True."""
    mask = (
        _col(df, "sym1_success").fillna(False)
        & _col(df, "sym2_success").fillna(False)
    )
    return _filter(df, mask.astype(bool))


# --------------------------------------------------------------------------- #
# Figure 1 - Ablation effects                                                 #
# --------------------------------------------------------------------------- #

def _figure_ablation(df: pd.DataFrame) -> tuple[Figure, pd.DataFrame]:
    """Faceted by model.

    Top strip (one panel per model): sym1 ∧ sym2 success rate per
    ablation cell, hue = DSL.
    Lower grid (`n_models` rows × `n_metrics` cols): interaction plot
    per (model, metric) — x = syntax, two lines for few_shot.
    """
    models = sorted(_col(df, "model").dropna().unique())
    n_models = len(models)
    dsls = sorted(_col(df, "dsl").dropna().unique())
    ablations: list[tuple[bool, bool]] = [
        (False, False), (True, False), (False, True), (True, True),
    ]
    ablation_labels = ["syn=0,fs=0", "syn=1,fs=0", "syn=0,fs=1", "syn=1,fs=1"]

    success = (
        _col(df, "sym1_success").fillna(False)
        & _col(df, "sym2_success").fillna(False)
    )
    df = df.assign(_success=success.astype(bool))
    succ_df = _successful(df)

    fig = plt.figure(figsize=(13.5, 3.0 + 2.4 * n_models))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.55 * n_models], hspace=0.4,
    )
    succ_gs = outer[0].subgridspec(1, n_models, wspace=0.35)
    inter_gs = outer[1].subgridspec(
        n_models, len(METRICS), hspace=0.6, wspace=0.4,
    )

    succ_rows: list[dict[str, Any]] = []
    n_groups = len(ablations)
    bar_width = 0.8 / max(len(dsls), 1)
    xs = np.arange(n_groups)
    for m_idx, model in enumerate(models):
        ax = fig.add_subplot(succ_gs[0, m_idx])
        model_df = _filter(df, _col(df, "model") == model)
        for k, dsl in enumerate(dsls):
            rates: list[float] = []
            for (s, f) in ablations:
                mask = (
                    (_col(model_df, "dsl") == dsl)
                    & (_col(model_df, "ablation_syntax").fillna(False) == s)
                    & (_col(model_df, "ablation_few_shot").fillna(False) == f)
                )
                n = int(mask.sum())
                ok = int(_col(_filter(model_df, mask), "_success").sum())
                rate = ok / n if n > 0 else float("nan")
                rates.append(rate)
                succ_rows.append({
                    "figure": "fig1_success",
                    "model": model,
                    "dsl": dsl,
                    "ablation_syntax": s,
                    "ablation_few_shot": f,
                    "n_cells": n,
                    "n_success": ok,
                    "rate": rate,
                })
            offset = (k - (len(dsls) - 1) / 2) * bar_width
            ax.bar(
                xs + offset, rates, bar_width,
                label=dsl, color=PALETTE.get(dsl, None), edgecolor="white",
            )
        ax.set_xticks(list(xs))
        ax.set_xticklabels(ablation_labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(model, fontsize=10)
        if m_idx == 0:
            ax.set_ylabel("success rate")
        if m_idx == n_models - 1:
            ax.legend(title="DSL", loc="lower right", frameon=False, fontsize=8)

    metric_rows: list[dict[str, Any]] = []
    for r_idx, model in enumerate(models):
        model_succ = _filter(succ_df, _col(succ_df, "model") == model)
        for c_idx, metric in enumerate(METRICS):
            ax = fig.add_subplot(inter_gs[r_idx, c_idx])
            for fs_val, color, label in (
                (False, PALETTE["few_shot_off"], "few_shot=0"),
                (True, PALETTE["few_shot_on"], "few_shot=1"),
            ):
                x_pos = [0, 1]
                ms: list[float] = []
                los: list[float] = []
                his: list[float] = []
                for syn_val in (False, True):
                    sub = _filter(
                        model_succ,
                        (_col(model_succ, "ablation_syntax").fillna(False) == syn_val)
                        & (_col(model_succ, "ablation_few_shot").fillna(False) == fs_val),
                    )
                    if metric == "structural_matching":
                        sub = _filter(sub, _col(sub, "dsl") == "dcpl")
                    vals = _col(sub, metric).to_numpy(dtype=float)
                    m, lo, hi = _bootstrap_mean_ci(vals)
                    ms.append(m); los.append(lo); his.append(hi)
                    metric_rows.append({
                        "figure": "fig1_metric",
                        "model": model,
                        "metric": metric,
                        "ablation_syntax": syn_val,
                        "ablation_few_shot": fs_val,
                        "n": int((~np.isnan(vals)).sum()),
                        "mean": m,
                        "ci_lo": lo,
                        "ci_hi": hi,
                    })
                lower = [m - lo for m, lo in zip(ms, los)]
                upper = [hi - m for m, hi in zip(ms, his)]
                ax.errorbar(
                    x_pos, ms, yerr=[lower, upper],
                    marker="o", capsize=3, linewidth=1.5,
                    color=color, label=label,
                )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["syn=0", "syn=1"], fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            if r_idx == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=10)
            if c_idx == 0:
                ax.set_ylabel(f"{model}\nmean (95% CI)", fontsize=9)
            if r_idx == 0 and c_idx == len(METRICS) - 1:
                ax.legend(loc="lower right", frameon=False, fontsize=7)

    fig.suptitle(
        "Figure 1 — Ablation effects (syntax × few_shot), per model",
        y=0.995, fontsize=12,
    )
    summary = pd.DataFrame(succ_rows + metric_rows)
    return fig, summary


# --------------------------------------------------------------------------- #
# Figure 2 - DSL comparison                                                   #
# --------------------------------------------------------------------------- #

def _figure_dsl(df: pd.DataFrame) -> tuple[Figure, pd.DataFrame]:
    """Faceted by model (rows) × metric (cols).

    Each panel shows `dcpl` vs `odrl` box plots for one model and one
    metric, restricted to the best-supported ablation cell (syntax=1,
    few_shot=1) and to successful round-trips. The `matching` column
    renders `n/a` for `odrl` (dict-vs-dict; nothing to pair).
    """
    models = sorted(_col(df, "model").dropna().unique())
    n_models = len(models)
    dsls = sorted(_col(df, "dsl").dropna().unique())

    best = _successful(df)
    best = _filter(
        best,
        (_col(best, "ablation_syntax").fillna(False) == True)  # noqa: E712
        & (_col(best, "ablation_few_shot").fillna(False) == True),  # noqa: E712
    )

    fig, axes_obj = plt.subplots(
        n_models, len(METRICS),
        figsize=(13.5, 2.8 * n_models),
        squeeze=False,
    )
    rng = np.random.default_rng(seed=42)
    rows: list[dict[str, Any]] = []

    for r_idx, model in enumerate(models):
        model_best = _filter(best, _col(best, "model") == model)
        for c_idx, metric in enumerate(METRICS):
            ax = axes_obj[r_idx, c_idx]
            data_per_dsl: list[np.ndarray] = []
            for dsl in dsls:
                vals = _col(
                    _filter(model_best, _col(model_best, "dsl") == dsl),
                    metric,
                ).to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                data_per_dsl.append(vals)
                rows.append({
                    "figure": "fig2",
                    "model": model,
                    "metric": metric,
                    "dsl": dsl,
                    "n": int(len(vals)),
                    "median": float(np.median(vals)) if len(vals) else float("nan"),
                    "q1": float(np.quantile(vals, 0.25)) if len(vals) else float("nan"),
                    "q3": float(np.quantile(vals, 0.75)) if len(vals) else float("nan"),
                    "mean": float(vals.mean()) if len(vals) else float("nan"),
                })
            positions: list[int] = []
            plotted: list[np.ndarray] = []
            plotted_dsls: list[str] = []
            for i, vals in enumerate(data_per_dsl):
                if len(vals) > 0:
                    positions.append(i + 1)
                    plotted.append(vals)
                    plotted_dsls.append(dsls[i])
            if plotted:
                bp = ax.boxplot(
                    plotted, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False,
                )
                for patch, dsl in zip(bp["boxes"], plotted_dsls):
                    patch.set_facecolor(PALETTE.get(dsl, "#cccccc"))
                    patch.set_alpha(0.55)
                    patch.set_edgecolor("black")
                for med in bp["medians"]:
                    med.set_color("black")
                for vals, pos in zip(plotted, positions):
                    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                    ax.scatter(
                        np.full_like(vals, pos, dtype=float) + jitter, vals,
                        s=8, alpha=0.35, color="black",
                    )
            for i, vals in enumerate(data_per_dsl):
                if len(vals) == 0:
                    ax.text(
                        i + 1, 0.5, "n/a",
                        ha="center", va="center",
                        fontsize=8, color="gray", style="italic",
                    )
            ax.set_xticks(list(range(1, len(dsls) + 1)))
            ax.set_xticklabels(dsls, fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            if r_idx == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=10)
            if c_idx == 0:
                ax.set_ylabel(f"{model}\nscore", fontsize=9)

    fig.suptitle(
        "Figure 2 — DSL comparison (syntax=1, few_shot=1; successful round-trips), per model",
        y=1.0, fontsize=11,
    )
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure 3 - Structural vs semantic                                           #
# --------------------------------------------------------------------------- #

def _structural_composite(df: pd.DataFrame) -> pd.Series:
    cols = [f"structural_{f}" for f in STRUCTURAL_FIELDS]
    return df[cols].mean(axis=1, skipna=True)


def _figure_correlation(df: pd.DataFrame) -> tuple[Figure, pd.DataFrame]:
    """Faceted by DSL (rows) × model (cols).

    Each panel scatters per-cell structural composite (x) against semantic
    similarity (y), with points colored by ablation cell, and is annotated
    with Spearman ρ and Pearson r.
    """
    succ = _successful(df).copy()
    succ["structural_composite"] = _structural_composite(succ)
    succ = succ.dropna(subset=["structural_composite", "semantic"])
    dsls = sorted(_col(succ, "dsl").dropna().unique())
    models = sorted(_col(succ, "model").dropna().unique())

    fig, axes_obj = plt.subplots(
        len(dsls), len(models),
        figsize=(3.5 * len(models), 3.2 * len(dsls)),
        squeeze=False, sharex=True, sharey=True,
    )

    ablation_cells: list[tuple[bool, bool]] = [
        (False, False), (True, False), (False, True), (True, True),
    ]
    ablation_colors = {
        (False, False): PALETTE["ab_00"],
        (True, False): PALETTE["ab_10"],
        (False, True): PALETTE["ab_01"],
        (True, True): PALETTE["ab_11"],
    }
    rows: list[dict[str, Any]] = []
    for r_idx, dsl in enumerate(dsls):
        for c_idx, model in enumerate(models):
            ax = axes_obj[r_idx, c_idx]
            sub = _filter(
                succ,
                (_col(succ, "dsl") == dsl) & (_col(succ, "model") == model),
            )
            for (s, f) in ablation_cells:
                cell = _filter(
                    sub,
                    (_col(sub, "ablation_syntax").fillna(False) == s)
                    & (_col(sub, "ablation_few_shot").fillna(False) == f),
                )
                if len(cell) == 0:
                    continue
                ax.scatter(
                    _col(cell, "structural_composite"), _col(cell, "semantic"),
                    s=18, alpha=0.6, color=ablation_colors[(s, f)],
                    label=f"syn={int(s)},fs={int(f)}",
                    edgecolor="none",
                )
            x = _col(sub, "structural_composite").to_numpy(dtype=float)
            y = _col(sub, "semantic").to_numpy(dtype=float)
            if len(x) >= 3:
                sp = stats.spearmanr(x, y)
                pe = stats.pearsonr(x, y)
                sp_rho = float(cast(Any, sp).statistic)
                sp_p = float(cast(Any, sp).pvalue)
                pe_r = float(cast(Any, pe).statistic)
                pe_p = float(cast(Any, pe).pvalue)
                ax.text(
                    0.04, 0.96,
                    f"n={len(x)}\n"
                    f"ρ={sp_rho:.3f} (p={sp_p:.1e})\n"
                    f"r={pe_r:.3f} (p={pe_p:.1e})",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor="lightgray"),
                )
                rows.append({
                    "figure": "fig3",
                    "dsl": dsl,
                    "model": model,
                    "n": int(len(x)),
                    "spearman_rho": sp_rho,
                    "spearman_p": sp_p,
                    "pearson_r": pe_r,
                    "pearson_p": pe_p,
                })
            else:
                ax.text(
                    0.5, 0.5, f"n={len(x)} (too few)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="gray",
                )
                rows.append({
                    "figure": "fig3",
                    "dsl": dsl,
                    "model": model,
                    "n": int(len(x)),
                    "spearman_rho": float("nan"),
                    "spearman_p": float("nan"),
                    "pearson_r": float("nan"),
                    "pearson_p": float("nan"),
                })
            ax.plot([0, 1], [0, 1], color="lightgray", linewidth=0.8, linestyle=":")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            if r_idx == len(dsls) - 1:
                ax.set_xlabel("structural composite", fontsize=9)
            if c_idx == 0:
                ax.set_ylabel(f"{dsl}\nsemantic similarity", fontsize=9)
            if r_idx == 0:
                ax.set_title(model, fontsize=10)

    handles, labels = axes_obj[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center",
            ncol=len(ablation_cells), frameon=False, fontsize=8,
            title="ablation", bbox_to_anchor=(0.5, -0.02),
        )
    fig.suptitle(
        "Figure 3 — Structural vs semantic agreement, faceted by DSL × model",
        y=1.0, fontsize=11,
    )
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def _atomic_savefig(fig: Figure, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(tmp, format=path.suffix.lstrip("."))
    os.replace(tmp, path)


def _atomic_save_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_visualize(input_path: Path, output_dir: Path) -> None:
    """Render all three figures + summary stats CSV from a scores CSV."""
    # Scoped to this phase: matplotlib's PDF backend logs every glyph it
    # subsets at INFO, which drowns out our own progress lines. Raising the
    # threshold for the duration of the render keeps the noise out without
    # touching the root logger config.
    for name in ("matplotlib", "fontTools", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_scores(input_path)
    logging.info(f"Loaded {len(df)} rows from {input_path}")
    _paper_style()

    fig1, s1 = _figure_ablation(df)
    _atomic_savefig(fig1, output_dir / "fig1_ablation.pdf")
    plt.close(fig1)

    fig2, s2 = _figure_dsl(df)
    _atomic_savefig(fig2, output_dir / "fig2_dsl.pdf")
    plt.close(fig2)

    fig3, s3 = _figure_correlation(df)
    _atomic_savefig(fig3, output_dir / "fig3_correlation.pdf")
    plt.close(fig3)

    summary = pd.concat([s1, s2, s3], ignore_index=True)
    _atomic_save_csv(summary, output_dir / "summary_stats.csv")
    logging.info(
        f"Wrote fig1_ablation.pdf, fig2_dsl.pdf, fig3_correlation.pdf, "
        f"summary_stats.csv to {output_dir}"
    )
