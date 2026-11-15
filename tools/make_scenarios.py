import argparse
import csv
import os
from pathlib import Path


PREFIX = "P3_R2_LOGIT_SWITCH_D10000_F400_R1500_"
BANDIT_METHODS = (
    ("TS_HTIX", "ThompsonSampling", "HT_IX", True),
    ("TS_SIMPLE", "ThompsonSampling", "SimpleAverage", False),
    ("EXP3_HTIX", "EXP3", "HT_IX", True),
    ("EXP3_SIMPLE", "EXP3", "SimpleAverage", False),
    ("HEDGE_HTIX", "HedgeTSEXP3", "HT_IX", True),
    ("HEDGE_SIMPLE", "HedgeTSEXP3", "SimpleAverage", False),
)
ETC_UPDATES = (5, 10, 14, 24)
FIELDS = (
    "scenario_name",
    "replication_id",
    "sample_id",
    "method_id",
    "rq_file",
    "op_forecast_f",
    "random_seed",
    "op_mab_random_seed",
    "op_mab_algorithm",
    "op_mab_estimator",
    "op_mab_period_policy_mode",
    "op_mab_reward_event",
    "op_mab_reward_target",
    "op_mab_feedback_mode",
    "op_mab_use_exposure_prob",
    "op_mab_htix_self_normalized",
    "op_mab_reward_normalization",
    "op_mab_fixed_allocation_probs",
    "op_mab_fixed_arm",
    "op_mab_reward_lower_bound",
    "op_mab_reward_upper_bound",
    "op_mab_grid_beta_reward_upper_bound",
    "priority_group",
    "op_mab_beta",
    "op_mab_ts_mc_samples",
    "op_mab_hedge_gamma",
    "op_mab_htix_diagnostics",
)


def write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def base_row(sample_id):
    demand = f"2018-11-12_accepted_daily10000_boundarycorridor70_sample{sample_id:03d}_with_fare.csv"
    return {
        "replication_id": sample_id - 1,
        "sample_id": sample_id,
        "rq_file": demand,
        "op_forecast_f": f"agg_{demand}",
        "random_seed": 999 + sample_id,
        "op_mab_random_seed": 41 + sample_id,
        "op_mab_period_policy_mode": "cached",
        "op_mab_reward_event": "acceptance",
        "op_mab_reward_target": "total_fare",
        "op_mab_feedback_mode": "event_matured",
        "op_mab_htix_self_normalized": "False",
        "op_mab_reward_normalization": "fixed",
        "op_mab_fixed_allocation_probs": "",
        "op_mab_fixed_arm": "",
        "op_mab_reward_lower_bound": "0.0",
        "op_mab_reward_upper_bound": "3953.0",
        "op_mab_grid_beta_reward_upper_bound": "3953.0",
        "priority_group": sample_id,
        "op_mab_beta": "0.1",
        "op_mab_ts_mc_samples": "2000",
        "op_mab_hedge_gamma": "0.02",
    }


def build_bandits(samples):
    rows = []
    for sample_id in range(1, samples + 1):
        for method_id, algorithm, estimator, htix in BANDIT_METHODS:
            row = base_row(sample_id)
            row.update(
                {
                    "scenario_name": f"{PREFIX}{method_id}_SAMPLE{sample_id:03d}",
                    "method_id": method_id,
                    "op_mab_algorithm": algorithm,
                    "op_mab_estimator": estimator,
                    "op_mab_use_exposure_prob": str(htix),
                    "op_mab_htix_diagnostics": str(htix),
                }
            )
            rows.append(row)
    return rows


def build_etc(samples):
    rows = []
    for sample_id in range(1, samples + 1):
        for updates in ETC_UPDATES:
            method_id = f"ETC_ABN_U{updates:02d}"
            row = base_row(sample_id)
            row.update(
                {
                    "scenario_name": f"{PREFIX}{method_id}_SAMPLE{sample_id:03d}",
                    "method_id": method_id,
                    "op_mab_algorithm": "ExploreThenCommit",
                    "op_mab_estimator": "SimpleAverage",
                    "op_mab_use_exposure_prob": "False",
                    "op_mab_htix_diagnostics": "False",
                    "op_mab_etc_exploration_updates": updates,
                }
            )
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleetpy-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.samples <= 50:
        raise ValueError("samples must be between 1 and 50.")
    output = args.fleetpy_root.resolve() / "experiments" / "configs"
    bandits = output / "daily10000_logit_switch_f400_p3_r2_bandits_50x6.csv"
    etc = output / "daily10000_logit_switch_f400_p3_r2_traditional_etc_50x4.csv"
    write_rows(bandits, FIELDS, build_bandits(args.samples))
    write_rows(etc, (*FIELDS, "op_mab_etc_exploration_updates"), build_etc(args.samples))
    print(f"Wrote {bandits}")
    print(f"Wrote {etc}")


if __name__ == "__main__":
    main()
