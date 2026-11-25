"""Compare P3 adaptive bandits with paired explore-then-commit baselines by revenue."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
BANDIT_CHECKPOINT_NAME = "NONSTATIONARY_P3_R2_BANDITS_50x6.csv"
ETC_CHECKPOINT_NAME = "NONSTATIONARY_P3_R2_ETC_50x4.csv"
DEFAULT_OUTPUT_DIR = (
    ROOT / "analysis" / "outputs" / "daily10000_logit_switch_f400_p3_bandits_vs_traditional_etc_50_current"
)
BANDIT_METHODS = (
    "TS_HTIX",
    "TS_SIMPLE",
    "EXP3_HTIX",
    "EXP3_SIMPLE",
    "HEDGE_HTIX",
    "HEDGE_SIMPLE",
)
TRADITIONAL_METHODS = ("ETC_ABN_U05", "ETC_ABN_U10", "ETC_ABN_U14", "ETC_ABN_U24")
METHODS = BANDIT_METHODS + TRADITIONAL_METHODS
METHOD_LABELS = {
    "TS_HTIX": "Beta-TS + HT-IX",
    "TS_SIMPLE": "Beta-TS + Simple Avg",
    "EXP3_HTIX": "EXP3 + HT-IX",
    "EXP3_SIMPLE": "EXP3 + Simple Avg",
    "HEDGE_HTIX": "Hedge + HT-IX",
    "HEDGE_SIMPLE": "Hedge + Simple Avg",
    "ETC_ABN_U05": "Traditional ETC 10%",
    "ETC_ABN_U10": "Traditional ETC 20%",
    "ETC_ABN_U14": "Traditional ETC 30%",
    "ETC_ABN_U24": "Traditional ETC 50%",
}
METHOD_COLORS = {
    "TS_HTIX": "#0072B2",
    "TS_SIMPLE": "#56B4E9",
    "EXP3_HTIX": "#D55E00",
    "EXP3_SIMPLE": "#E69F00",
    "HEDGE_HTIX": "#009E73",
    "HEDGE_SIMPLE": "#CC79A7",
    "ETC_ABN_U05": "#4C78A8",
    "ETC_ABN_U10": "#72B7B2",
    "ETC_ABN_U14": "#54A24B",
    "ETC_ABN_U24": "#EECA3B",
}
PHASES = (
    (1, 0, 25200, "00:00-07:00"),
    (2, 25200, 36000, "07:00-10:00"),
    (3, 36000, 57600, "10:00-16:00"),
    (4, 57600, 72000, "16:00-20:00"),
    (5, 72000, 86400, "20:00-24:00"),
)


def as_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def summarize(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    count = len(clean)
    if count == 0:
        raise ValueError("No realized-revenue values were available for summary.")
    mean = float(np.mean(clean))
    if count == 1:
        return {"samples": 1, "mean": mean, "ci95_lower": float("nan"), "ci95_upper": float("nan"), "ci95_half_width": float("nan")}
    half_width = float(student_t.ppf(0.975, count - 1) * np.std(clean, ddof=1) / np.sqrt(count))
    return {
        "samples": int(count),
        "mean": mean,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "ci95_half_width": half_width,
    }


def load_checkpoint_rows(checkpoint: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(checkpoint)
    if "source_scenario_config" not in rows.columns:
        raise ValueError(f"Checkpoint has no source_scenario_config column: {checkpoint}")
    metadata_frames: list[pd.DataFrame] = []
    for source in rows["source_scenario_config"].dropna().astype(str).unique():
        config = Path(source)
        if not config.is_file():
            raise FileNotFoundError(f"Scenario configuration is unavailable: {config}")
        frame = pd.read_csv(config)
        required = {"scenario_name", "sample_id", "method_id"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Scenario configuration {config} is missing {sorted(missing)}")
        metadata_frames.append(frame.loc[:, ["scenario_name", "sample_id", "method_id"]])
    if not metadata_frames:
        raise ValueError(f"No scenario metadata is available for {checkpoint}")
    metadata = pd.concat(metadata_frames, ignore_index=True).drop_duplicates("scenario_name")
    metadata = metadata.rename(columns={"sample_id": "sample_id_config", "method_id": "method_id_config"})
    merged = rows.merge(metadata, on="scenario_name", how="left", validate="many_to_one")
    for column in ("sample_id", "method_id"):
        config_column = f"{column}_config"
        merged[column] = merged[column].where(merged[column].notna(), merged[config_column]) if column in merged else merged[config_column]
    return merged, metadata.rename(columns={"sample_id_config": "sample_id", "method_id_config": "method_id"})


def completed_runs(checkpoint: Path, methods: tuple[str, ...]) -> dict[int, dict[str, Path]]:
    rows, metadata = load_checkpoint_rows(checkpoint)
    expected = metadata.loc[metadata["method_id"].isin(methods)].copy()
    expected["sample_id"] = pd.to_numeric(expected["sample_id"], errors="raise").astype(int)
    expected_ids = [
        sample_id
        for sample_id, group in expected.groupby("sample_id", sort=True)
        if set(group["method_id"]).issuperset(methods)
    ]
    if not expected_ids:
        raise ValueError(f"No complete paired design is defined by {checkpoint}")
    completed = rows.loc[rows["status"].eq("completed") & rows["method_id"].isin(methods)].copy()
    completed["sample_id"] = pd.to_numeric(completed["sample_id"], errors="raise").astype(int)
    runs: dict[int, dict[str, Path]] = {}
    missing: list[str] = []
    for sample_id in expected_ids:
        group = completed.loc[completed["sample_id"].eq(sample_id)]
        by_method: dict[str, Path] = {}
        for method in methods:
            candidates = group.loc[group["method_id"].eq(method)]
            if len(candidates) != 1:
                missing.append(f"sample {sample_id}: {method} ({len(candidates)} completed)")
                continue
            result_dir = Path(candidates.iloc[0]["result_dir"])
            required = (result_dir / "standard_eval.csv", result_dir / "mab_spatial_request_log.csv")
            if not all(path.is_file() for path in required):
                missing.append(f"sample {sample_id}: {method} missing result output")
                continue
            by_method[method] = result_dir
        if len(by_method) == len(methods):
            runs[sample_id] = by_method
    if missing:
        raise RuntimeError(f"Batch is incomplete: {len(runs)}/{len(expected_ids)} paired samples. {'; '.join(missing[:10])}")
    return runs


def merge_runs(bandit_checkpoint: Path, traditional_checkpoint: Path) -> dict[int, dict[str, Path]]:
    bandits = completed_runs(bandit_checkpoint, BANDIT_METHODS)
    traditional = completed_runs(traditional_checkpoint, TRADITIONAL_METHODS)
    common_ids = sorted(set(bandits).intersection(traditional))
    if not common_ids:
        raise RuntimeError("No completed paired samples are common to the bandit and traditional checkpoints.")
    return {sample_id: {**bandits[sample_id], **traditional[sample_id]} for sample_id in common_ids}


def standard_eval_revenue(result_dir: Path) -> float:
    path = result_dir / "standard_eval.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0].strip() == "mod revenue":
                return float(row[1])
    raise KeyError(f"mod revenue is missing from {path}")


def phase_rows_for_run(sample_id: int, method: str, result_dir: Path) -> list[dict[str, object]]:
    path = result_dir / "mab_spatial_request_log.csv"
    frame = pd.read_csv(path, low_memory=False).drop_duplicates(subset=["request_id"], keep="last").copy()
    required = {"pricing_episode_start", "request_reward_amount", "request_reward_observed"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    frame["episode_start"] = pd.to_numeric(frame["pricing_episode_start"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["request_reward_amount"], errors="coerce").fillna(0.0)
    frame["observed"] = as_bool(frame["request_reward_observed"])
    frame = frame.dropna(subset=["episode_start"])
    if not frame["observed"].all():
        unobserved = int((~frame["observed"]).sum())
        raise ValueError(f"{path} has {unobserved} requests without finalized revenue outcomes")
    rows: list[dict[str, object]] = []
    for phase_id, start, end, label in PHASES:
        phase = frame.loc[frame["episode_start"].ge(start) & frame["episode_start"].lt(end)]
        if phase.empty:
            raise ValueError(f"{path} has no requests in phase {phase_id} ({label})")
        revenue = float(phase["amount"].clip(lower=0.0).sum())
        rows.append({
            "sample_id": sample_id,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "phase_id": phase_id,
            "phase_label": label,
            "request_count": int(len(phase)),
            "phase_realized_operator_revenue": revenue,
            "realized_operator_revenue_per_request": revenue / len(phase),
        })
    return rows


def build_total_metrics(runs: dict[int, dict[str, Path]]) -> pd.DataFrame:
    rows = [
        {
            "sample_id": sample_id,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "realized_operator_revenue": standard_eval_revenue(result_dir),
        }
        for sample_id, methods in sorted(runs.items())
        for method, result_dir in methods.items()
    ]
    return pd.DataFrame(rows).sort_values(["sample_id", "method"]).reset_index(drop=True)


def build_phase_metrics(runs: dict[int, dict[str, Path]]) -> pd.DataFrame:
    rows = [
        row
        for sample_id, methods in sorted(runs.items())
        for method, result_dir in methods.items()
        for row in phase_rows_for_run(sample_id, method, result_dir)
    ]
    expected = len(runs) * len(METHODS) * len(PHASES)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} phase rows, found {len(rows)}")
    return pd.DataFrame(rows)


def total_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        result = summarize(metrics.loc[metrics["method"].eq(method), "realized_operator_revenue"])
        rows.append({
            "method": method,
            "method_label": METHOD_LABELS[method],
            "samples": result["samples"],
            "mean_realized_operator_revenue": result["mean"],
            "ci95_lower": result["ci95_lower"],
            "ci95_upper": result["ci95_upper"],
            "ci95_half_width": result["ci95_half_width"],
        })
    return pd.DataFrame(rows)


def paired_lifts(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot(index="sample_id", columns="method", values="realized_operator_revenue")
    rows = []
    for bandit in BANDIT_METHODS:
        for traditional in TRADITIONAL_METHODS:
            result = summarize(wide[bandit] - wide[traditional])
            rows.append({
                "bandit_method": bandit,
                "bandit_label": METHOD_LABELS[bandit],
                "traditional_method": traditional,
                "traditional_label": METHOD_LABELS[traditional],
                "samples": result["samples"],
                "mean_paired_revenue_lift": result["mean"],
                "ci95_lower": result["ci95_lower"],
                "ci95_upper": result["ci95_upper"],
                "ci95_half_width": result["ci95_half_width"],
            })
    return pd.DataFrame(rows)


def paired_performance_improvements(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired percentage improvements relative to each ETC baseline."""
    wide = metrics.pivot(index="sample_id", columns="method", values="realized_operator_revenue")
    rows = []
    for bandit in BANDIT_METHODS:
        for traditional in TRADITIONAL_METHODS:
            baseline = wide[traditional]
            if (baseline <= 0.0).any():
                raise ValueError(f"{traditional} has a nonpositive paired revenue value.")
            improvement = 100.0 * (wide[bandit] - baseline) / baseline
            result = summarize(improvement)
            rows.append({
                "bandit_method": bandit,
                "bandit_label": METHOD_LABELS[bandit],
                "traditional_method": traditional,
                "traditional_label": METHOD_LABELS[traditional],
                "samples": result["samples"],
                "mean_performance_improvement_pct": result["mean"],
                "ci95_lower_pct": result["ci95_lower"],
                "ci95_upper_pct": result["ci95_upper"],
                "ci95_half_width_pct": result["ci95_half_width"],
            })
    return pd.DataFrame(rows)


def phase_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (phase_id, phase_label, method, method_label), group in metrics.groupby(
        ["phase_id", "phase_label", "method", "method_label"], sort=True
    ):
        total = summarize(group["phase_realized_operator_revenue"])
        per_request = summarize(group["realized_operator_revenue_per_request"])
        rows.append({
            "phase_id": phase_id,
            "phase_label": phase_label,
            "method": method,
            "method_label": method_label,
            "mean_phase_realized_operator_revenue": total["mean"],
            "revenue_ci95_lower": total["ci95_lower"],
            "revenue_ci95_upper": total["ci95_upper"],
            "revenue_ci95_half_width": total["ci95_half_width"],
            "mean_revenue_per_request": per_request["mean"],
            "per_request_ci95_lower": per_request["ci95_lower"],
            "per_request_ci95_upper": per_request["ci95_upper"],
            "per_request_ci95_half_width": per_request["ci95_half_width"],
            "samples": total["samples"],
        })
    return pd.DataFrame(rows).sort_values(["phase_id", "method"]).reset_index(drop=True)


def render_boxplot(metrics: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(14.0, 6.2))
    values = [metrics.loc[metrics["method"].eq(method), "realized_operator_revenue"] for method in METHODS]
    boxes = axis.boxplot(values, patch_artist=True, widths=0.6, showfliers=False, medianprops={"color": "#222222", "linewidth": 1.3})
    for patch, method in zip(boxes["boxes"], METHODS):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.78)
    axis.set_xticks(range(1, len(METHODS) + 1), [METHOD_LABELS[method] for method in METHODS], rotation=24, ha="right")
    axis.set_ylabel("Realized operator revenue")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_bandit_minus_baseline_boxplot(
    metrics: pd.DataFrame,
    output: Path,
    baseline_method: str,
    *,
    relative: bool = False,
) -> None:
    wide = metrics.pivot(index="sample_id", columns="method", values="realized_operator_revenue")
    baseline = wide[baseline_method]
    if relative:
        values = [100.0 * (wide[method] - baseline) / baseline for method in BANDIT_METHODS]
        ylabel = "Paired performance improvement (%)"
    else:
        values = [wide[method] - baseline for method in BANDIT_METHODS]
        ylabel = f"Paired revenue difference vs {METHOD_LABELS[baseline_method]}"
    figure, axis = plt.subplots(figsize=(11.5, 6.2))
    boxes = axis.boxplot(values, patch_artist=True, widths=0.62, showfliers=False, medianprops={"color": "#222222", "linewidth": 1.4})
    for patch, method in zip(boxes["boxes"], BANDIT_METHODS):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.78)
    if relative:
        summaries = [summarize(value) for value in values]
        positions = np.arange(1, len(BANDIT_METHODS) + 1, dtype=float)
        axis.errorbar(
            positions,
            [summary["mean"] for summary in summaries],
            yerr=[summary["ci95_half_width"] for summary in summaries],
            fmt="D",
            color="#151515",
            markerfacecolor="white",
            markeredgewidth=1.1,
            markersize=5.2,
            capsize=4,
            linewidth=1.2,
            label="Mean and 95% CI",
            zorder=5,
        )
        axis.legend(loc="upper right", frameon=True, fontsize=8.5)
    axis.axhline(0, color="#555555", linewidth=1.0)
    axis.set_xticks(range(1, len(BANDIT_METHODS) + 1), [METHOD_LABELS[method] for method in BANDIT_METHODS], rotation=24, ha="right")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_bandit_vs_traditional_percent_grid(metrics: pd.DataFrame, output: Path) -> None:
    wide = metrics.pivot(index="sample_id", columns="method", values="realized_operator_revenue")
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), sharey=True)
    for axis, baseline_method in zip(axes.flat, TRADITIONAL_METHODS):
        baseline = wide[baseline_method]
        values = [100.0 * (wide[method] - baseline) / baseline for method in BANDIT_METHODS]
        boxes = axis.boxplot(values, patch_artist=True, widths=0.62, showfliers=False, medianprops={"color": "#222222", "linewidth": 1.3})
        for patch, method in zip(boxes["boxes"], BANDIT_METHODS):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.78)
        axis.axhline(0, color="#555555", linewidth=1.0)
        axis.set_title(METHOD_LABELS[baseline_method])
        axis.set_xticks(range(1, len(BANDIT_METHODS) + 1), [METHOD_LABELS[method] for method in BANDIT_METHODS], rotation=24, ha="right", fontsize=8.4)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    for axis in axes[:, 0]:
        axis.set_ylabel("Paired revenue difference (%)")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_means(summary: pd.DataFrame, output: Path) -> None:
    ordered = summary.set_index("method").loc[list(METHODS)].reset_index()
    positions = np.arange(len(METHODS), dtype=float)
    figure, axis = plt.subplots(figsize=(14.0, 6.2))
    axis.bar(positions, ordered["mean_realized_operator_revenue"], color=[METHOD_COLORS[method] for method in METHODS], width=0.62)
    axis.errorbar(positions, ordered["mean_realized_operator_revenue"], yerr=ordered["ci95_half_width"], fmt="none", color="#222222", capsize=4, linewidth=1.2)
    axis.set_xticks(positions, [METHOD_LABELS[method] for method in METHODS], rotation=24, ha="right")
    axis.set_ylabel("Mean realized operator revenue (95% CI)")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_phase_means(summary: pd.DataFrame, output: Path) -> None:
    positions = np.arange(len(PHASES), dtype=float)
    figure, axis = plt.subplots(figsize=(14.0, 6.5))
    for method in METHODS:
        subset = summary.loc[summary["method"].eq(method)].sort_values("phase_id")
        means = subset["mean_phase_realized_operator_revenue"].to_numpy(dtype=float)
        ci = subset["revenue_ci95_half_width"].to_numpy(dtype=float)
        axis.plot(positions, means, marker="o", linewidth=1.8, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axis.fill_between(positions, means - ci, means + ci, color=METHOD_COLORS[method], alpha=0.08)
    axis.set_xticks(positions, [label for _, _, _, label in PHASES])
    axis.set_xlabel("Pre-specified demand phase")
    axis.set_ylabel("Mean realized operator revenue")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleetpy-root", type=Path, help="FleetPy checkout used to locate any unspecified checkpoint.")
    parser.add_argument("--bandit-checkpoint", type=Path, help="Bandit checkpoint CSV path, overriding --fleetpy-root.")
    parser.add_argument("--traditional-checkpoint", type=Path, help="ETC checkpoint CSV path, overriding --fleetpy-root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.fleetpy_root is not None:
        checkpoint_dir = args.fleetpy_root / "experiments" / "checkpoints"
        args.bandit_checkpoint = args.bandit_checkpoint or checkpoint_dir / BANDIT_CHECKPOINT_NAME
        args.traditional_checkpoint = args.traditional_checkpoint or checkpoint_dir / ETC_CHECKPOINT_NAME
    if args.bandit_checkpoint is None or args.traditional_checkpoint is None:
        parser.error("provide --fleetpy-root or both --bandit-checkpoint and --traditional-checkpoint")

    runs = merge_runs(args.bandit_checkpoint, args.traditional_checkpoint)
    totals = build_total_metrics(runs)
    phases = build_phase_metrics(runs)
    summary = total_summary(totals)
    lifts = paired_lifts(totals)
    improvements = paired_performance_improvements(totals)
    phase_results = phase_summary(phases)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    totals.to_csv(args.output_dir / "paired_total_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    lifts.to_csv(args.output_dir / "bandit_vs_traditional_paired_revenue_lift.csv", index=False)
    improvements.to_csv(args.output_dir / "bandit_vs_traditional_paired_performance_improvement.csv", index=False)
    phases.to_csv(args.output_dir / "phase_revenue_metrics.csv", index=False)
    phase_results.to_csv(args.output_dir / "phase_revenue_summary.csv", index=False)
    boxplot = args.output_dir / "total_revenue_boxplot.png"
    bandit_minus_baseline_boxplot = args.output_dir / "bandit_minus_traditional_etc10_revenue_boxplot.png"
    bandit_minus_baseline_percent_boxplot = args.output_dir / "bandit_minus_traditional_etc10_percent_boxplot.png"
    bandit_minus_etc20_percent_boxplot = args.output_dir / "bandit_minus_traditional_etc20_percent_boxplot.png"
    bandit_minus_etc30_percent_boxplot = args.output_dir / "bandit_minus_traditional_etc30_percent_boxplot.png"
    bandit_vs_traditional_percent_grid = args.output_dir / "bandit_vs_all_traditional_percent_boxplots.png"
    means = args.output_dir / "mean_revenue_95ci.png"
    phase_figure = args.output_dir / "mean_phase_revenue_95ci.png"
    render_boxplot(totals, boxplot)
    render_bandit_minus_baseline_boxplot(totals, bandit_minus_baseline_boxplot, "ETC_ABN_U05")
    render_bandit_minus_baseline_boxplot(totals, bandit_minus_baseline_percent_boxplot, "ETC_ABN_U05", relative=True)
    render_bandit_minus_baseline_boxplot(totals, bandit_minus_etc20_percent_boxplot, "ETC_ABN_U10", relative=True)
    render_bandit_minus_baseline_boxplot(totals, bandit_minus_etc30_percent_boxplot, "ETC_ABN_U14", relative=True)
    render_bandit_vs_traditional_percent_grid(totals, bandit_vs_traditional_percent_grid)
    render_means(summary, means)
    render_phase_means(phase_results, phase_figure)
    payload = {
        "bandit_checkpoint": str(args.bandit_checkpoint),
        "traditional_checkpoint": str(args.traditional_checkpoint),
        "paired_samples": len(runs),
        "outputs": {
            "paired_total_metrics": str(args.output_dir / "paired_total_metrics.csv"),
            "method_summary": str(args.output_dir / "method_summary.csv"),
            "bandit_vs_traditional_paired_revenue_lift": str(args.output_dir / "bandit_vs_traditional_paired_revenue_lift.csv"),
            "bandit_vs_traditional_paired_performance_improvement": str(args.output_dir / "bandit_vs_traditional_paired_performance_improvement.csv"),
            "phase_revenue_summary": str(args.output_dir / "phase_revenue_summary.csv"),
            "total_revenue_boxplot": str(boxplot),
            "bandit_minus_traditional_etc10_revenue_boxplot": str(bandit_minus_baseline_boxplot),
            "bandit_minus_traditional_etc10_percent_boxplot": str(bandit_minus_baseline_percent_boxplot),
            "bandit_minus_traditional_etc20_percent_boxplot": str(bandit_minus_etc20_percent_boxplot),
            "bandit_minus_traditional_etc30_percent_boxplot": str(bandit_minus_etc30_percent_boxplot),
            "bandit_vs_all_traditional_percent_boxplots": str(bandit_vs_traditional_percent_grid),
            "mean_revenue_95ci": str(means),
            "mean_phase_revenue_95ci": str(phase_figure),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
