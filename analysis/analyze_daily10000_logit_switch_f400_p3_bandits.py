"""Summarize paired realized-revenue outcomes for six P3 adaptive bandits."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "experiments"
    / "checkpoints"
    / "LOGIT_SWITCH_D10000_F400_R1500_P3_BANDITS_50x6.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "analysis"
    / "outputs"
    / "daily10000_logit_switch_f400_p3_bandits_50_current"
)
METHODS = (
    "TS_HTIX",
    "TS_SIMPLE",
    "EXP3_HTIX",
    "EXP3_SIMPLE",
    "HEDGE_HTIX",
    "HEDGE_SIMPLE",
)
METHOD_LABELS = {
    "TS_HTIX": "Beta-TS + HT-IX",
    "TS_SIMPLE": "Beta-TS + Simple Avg",
    "EXP3_HTIX": "EXP3 + HT-IX",
    "EXP3_SIMPLE": "EXP3 + Simple Avg",
    "HEDGE_HTIX": "Hedge + HT-IX",
    "HEDGE_SIMPLE": "Hedge + Simple Avg",
}
METHOD_COLORS = {
    "TS_HTIX": "#0072B2",
    "TS_SIMPLE": "#56B4E9",
    "EXP3_HTIX": "#D55E00",
    "EXP3_SIMPLE": "#E69F00",
    "HEDGE_HTIX": "#009E73",
    "HEDGE_SIMPLE": "#CC79A7",
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
    count = int(len(clean))
    if count == 0:
        return {
            "samples": 0,
            "mean": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "ci95_half_width": float("nan"),
        }
    mean = float(np.mean(clean))
    if count == 1:
        return {
            "samples": 1,
            "mean": mean,
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "ci95_half_width": float("nan"),
        }
    half_width = float(student_t.ppf(0.975, count - 1) * np.std(clean, ddof=1) / np.sqrt(count))
    return {
        "samples": count,
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
    metadata = metadata.rename(
        columns={"sample_id": "sample_id_config", "method_id": "method_id_config"}
    )
    merged = rows.merge(metadata, on="scenario_name", how="left", validate="many_to_one")
    for column in ("sample_id", "method_id"):
        config_column = f"{column}_config"
        if column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna(), merged[config_column])
        else:
            merged[column] = merged[config_column]
    return merged, metadata.rename(
        columns={"sample_id_config": "sample_id", "method_id_config": "method_id"}
    )


def completed_runs(checkpoint: Path) -> dict[int, dict[str, Path]]:
    rows, metadata = load_checkpoint_rows(checkpoint)
    expected = metadata.loc[metadata["method_id"].isin(METHODS)].copy()
    expected["sample_id"] = pd.to_numeric(expected["sample_id"], errors="raise").astype(int)
    expected_sample_ids = sorted(
        sample_id
        for sample_id, group in expected.groupby("sample_id", sort=True)
        if set(group["method_id"]).issuperset(METHODS)
    )
    if not expected_sample_ids:
        raise ValueError(f"No complete six-method design is defined by {checkpoint}")

    completed = rows.loc[rows["status"].eq("completed") & rows["method_id"].isin(METHODS)].copy()
    completed["sample_id"] = pd.to_numeric(completed["sample_id"], errors="raise").astype(int)
    runs: dict[int, dict[str, Path]] = {}
    missing: list[str] = []
    for sample_id in expected_sample_ids:
        group = completed.loc[completed["sample_id"].eq(sample_id)]
        by_method: dict[str, Path] = {}
        for method in METHODS:
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
        if len(by_method) == len(METHODS):
            runs[sample_id] = by_method
    if missing:
        preview = "; ".join(missing[:10])
        raise RuntimeError(
            f"Adaptive P3 batch is incomplete: {len(runs)}/{len(expected_sample_ids)} paired samples. {preview}"
        )
    return runs


def standard_eval_revenue(result_dir: Path) -> float:
    path = result_dir / "standard_eval.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0].strip() == "mod revenue":
                return float(row[1])
    raise KeyError(f"mod revenue is missing from {path}")


def phase_rows_for_run(sample_id: int, method: str, result_dir: Path) -> list[dict[str, object]]:
    path = result_dir / "mab_spatial_request_log.csv"
    frame = pd.read_csv(path, low_memory=False)
    required = {"pricing_episode_start", "request_reward_amount", "request_reward_observed"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    frame = frame.drop_duplicates(subset=["request_id"], keep="last").copy()
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
        request_count = int(len(phase))
        revenue = float(phase["amount"].clip(lower=0.0).sum())
        rows.append(
            {
                "sample_id": sample_id,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "phase_id": phase_id,
                "phase_label": label,
                "request_count": request_count,
                "phase_realized_operator_revenue": revenue,
                "realized_operator_revenue_per_request": revenue / request_count,
            }
        )
    return rows


def build_total_metrics(runs: dict[int, dict[str, Path]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, methods in sorted(runs.items()):
        for method in METHODS:
            rows.append(
                {
                    "sample_id": sample_id,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "realized_operator_revenue": standard_eval_revenue(methods[method]),
                }
            )
    return pd.DataFrame(rows).sort_values(["sample_id", "method"]).reset_index(drop=True)


def build_phase_metrics(runs: dict[int, dict[str, Path]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, methods in sorted(runs.items()):
        for method in METHODS:
            rows.extend(phase_rows_for_run(sample_id, method, methods[method]))
    expected = len(runs) * len(METHODS) * len(PHASES)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} phase rows, found {len(rows)}")
    return pd.DataFrame(rows)


def total_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        subset = metrics.loc[metrics["method"].eq(method)]
        result = summarize(subset["realized_operator_revenue"])
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "samples": result["samples"],
                "mean_realized_operator_revenue": result["mean"],
                "ci95_lower": result["ci95_lower"],
                "ci95_upper": result["ci95_upper"],
                "ci95_half_width": result["ci95_half_width"],
            }
        )
    return pd.DataFrame(rows)


def paired_lifts(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot(index="sample_id", columns="method", values="realized_operator_revenue")
    rows: list[dict[str, object]] = []
    for left, right in combinations(METHODS, 2):
        result = summarize(wide[left] - wide[right])
        rows.append(
            {
                "method_a": left,
                "method_b": right,
                "contrast": f"{METHOD_LABELS[left]} minus {METHOD_LABELS[right]}",
                "samples": result["samples"],
                "mean_paired_revenue_lift": result["mean"],
                "ci95_lower": result["ci95_lower"],
                "ci95_upper": result["ci95_upper"],
                "ci95_half_width": result["ci95_half_width"],
            }
        )
    return pd.DataFrame(rows)


def phase_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (phase_id, phase_label, method, method_label), group in metrics.groupby(
        ["phase_id", "phase_label", "method", "method_label"], sort=True
    ):
        total = summarize(group["phase_realized_operator_revenue"])
        per_request = summarize(group["realized_operator_revenue_per_request"])
        rows.append(
            {
                "phase_id": phase_id,
                "phase_label": phase_label,
                "method": method,
                "method_label": method_label,
                "samples": total["samples"],
                "mean_phase_realized_operator_revenue": total["mean"],
                "revenue_ci95_lower": total["ci95_lower"],
                "revenue_ci95_upper": total["ci95_upper"],
                "revenue_ci95_half_width": total["ci95_half_width"],
                "mean_revenue_per_request": per_request["mean"],
                "per_request_ci95_lower": per_request["ci95_lower"],
                "per_request_ci95_upper": per_request["ci95_upper"],
                "per_request_ci95_half_width": per_request["ci95_half_width"],
            }
        )
    return pd.DataFrame(rows).sort_values(["phase_id", "method"]).reset_index(drop=True)


def render_boxplot(metrics: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    values = [
        metrics.loc[metrics["method"].eq(method), "realized_operator_revenue"].to_numpy(dtype=float)
        for method in METHODS
    ]
    figure, axis = plt.subplots(figsize=(11.8, 5.8))
    boxes = axis.boxplot(values, patch_artist=True, widths=0.62, showfliers=False, medianprops={"color": "#222222", "linewidth": 1.4})
    for patch, method in zip(boxes["boxes"], METHODS):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.78)
    axis.set_xticks(range(1, len(METHODS) + 1), [METHOD_LABELS[method] for method in METHODS], rotation=18, ha="right")
    axis.set_ylabel("Realized operator revenue")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_means(summary: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    ordered = summary.set_index("method").loc[list(METHODS)].reset_index()
    positions = np.arange(len(METHODS), dtype=float)
    figure, axis = plt.subplots(figsize=(11.8, 5.8))
    axis.bar(positions, ordered["mean_realized_operator_revenue"], color=[METHOD_COLORS[method] for method in METHODS], width=0.62)
    axis.errorbar(
        positions,
        ordered["mean_realized_operator_revenue"],
        yerr=ordered["ci95_half_width"],
        fmt="none",
        color="#222222",
        capsize=4,
        linewidth=1.25,
    )
    axis.set_xticks(positions, [METHOD_LABELS[method] for method in METHODS], rotation=18, ha="right")
    axis.set_ylabel("Mean realized operator revenue")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_phase_means(summary: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    positions = np.arange(len(PHASES), dtype=float)
    figure, axis = plt.subplots(figsize=(11.8, 5.8))
    for method in METHODS:
        subset = summary.loc[summary["method"].eq(method)].sort_values("phase_id")
        means = subset["mean_phase_realized_operator_revenue"].to_numpy(dtype=float)
        ci = subset["revenue_ci95_half_width"].to_numpy(dtype=float)
        axis.plot(positions, means, marker="o", linewidth=2.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axis.fill_between(positions, means - ci, means + ci, color=METHOD_COLORS[method], alpha=0.11)
    axis.set_xticks(positions, [label for _, _, _, label in PHASES])
    axis.set_xlabel("Pre-specified demand phase")
    axis.set_ylabel("Mean realized operator revenue")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = completed_runs(args.checkpoint)
    total_metrics = build_total_metrics(runs)
    phase_metrics = build_phase_metrics(runs)
    method_summary = total_summary(total_metrics)
    lifts = paired_lifts(total_metrics)
    phases = phase_summary(phase_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_metrics.to_csv(args.output_dir / "paired_total_metrics.csv", index=False)
    method_summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    lifts.to_csv(args.output_dir / "paired_revenue_lifts.csv", index=False)
    phase_metrics.to_csv(args.output_dir / "phase_revenue_metrics.csv", index=False)
    phases.to_csv(args.output_dir / "phase_revenue_summary.csv", index=False)
    boxplot = args.output_dir / "total_revenue_boxplot.png"
    means = args.output_dir / "mean_revenue_95ci.png"
    phase_figure = args.output_dir / "mean_phase_revenue_95ci.png"
    render_boxplot(total_metrics, boxplot)
    render_means(method_summary, means)
    render_phase_means(phases, phase_figure)

    payload = {
        "checkpoint": str(args.checkpoint),
        "paired_samples": len(runs),
        "methods": method_summary.to_dict(orient="records"),
        "outputs": {
            "paired_total_metrics": str(args.output_dir / "paired_total_metrics.csv"),
            "method_summary": str(args.output_dir / "method_summary.csv"),
            "paired_revenue_lifts": str(args.output_dir / "paired_revenue_lifts.csv"),
            "phase_revenue_summary": str(args.output_dir / "phase_revenue_summary.csv"),
            "total_revenue_boxplot": str(boxplot),
            "mean_revenue_95ci": str(means),
            "mean_phase_revenue_95ci": str(phase_figure),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
