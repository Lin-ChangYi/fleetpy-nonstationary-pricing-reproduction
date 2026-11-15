import importlib
import json
import logging
import os
import re
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from src.fleetctrl.pricing.DynamicPricingBase import DynamicPricingBase
from src.misc.globals import *

LOG = logging.getLogger(__name__)


def _load_mab_algo(
    name: str,
    n_arms: int,
    seed: int,
    T: int,
    eta: float = None,
    exp3_eta: float = None,
    exp3_exploration_epsilon: float = None,
    exp3_discount_factor: float = 1.0,
    exp3_fixed_share_alpha: float = 0.0,
    ts_mc_samples: int = 2000,
    hedge_eta: float = None,
    hedge_gamma: float = 0.02,
    hedge_risk_lambda: float = 1.0,
    fixed_share_alpha: float = None,
    discount_factor: float = 1.0,
    etc_exploration_updates: int = 1,
):
    algorithms = {
        "ThompsonSampling": (
            "mab_algo.thompson",
            "ThompsonSampling",
            {"n_arms": n_arms, "seed": seed, "mc_draws": ts_mc_samples},
        ),
        "EXP3": (
            "mab_algo.exp3",
            "EXP3",
            {
                "n_arms": n_arms,
                "seed": seed,
                "T": T,
                "eta": exp3_eta,
                "exploration_epsilon": exp3_exploration_epsilon,
                "discount_factor": exp3_discount_factor,
                "fixed_share_alpha": exp3_fixed_share_alpha,
            },
        ),
        "HedgeTSEXP3": (
            "mab_algo.hedge",
            "HedgeTSEXP3",
            {
                "n_arms": n_arms,
                "seed": seed,
                "T": T,
                "mc_draws": ts_mc_samples,
                "hedge_eta": hedge_eta,
                "hedge_gamma": hedge_gamma,
                "exp3_eta": exp3_eta,
                "exp3_exploration_epsilon": exp3_exploration_epsilon,
                "exp3_discount_factor": exp3_discount_factor,
                "exp3_fixed_share_alpha": exp3_fixed_share_alpha,
            },
        ),
        "ExploreThenCommit": (
            "mab_algo.explore_then_commit",
            "ExploreThenCommit",
            {"n_arms": n_arms, "seed": seed, "exploration_updates": etc_exploration_updates},
        ),
    }
    if name not in algorithms:
        raise ValueError(f"[MABSpatialPricing] Unknown MAB algorithm: {name}")
    module_path, class_name, kwargs = algorithms[name]
    return getattr(importlib.import_module(module_path), class_name)(**kwargs)

def _load_estimator(name: str, beta: float = 0.1):
    from mab_algo.estimators import get_estimator

    kwargs = {"beta": beta} if name == "HT_IX" else {}
    return get_estimator(name, **kwargs)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return default


def _optional_float(value: Any, default: float, parameter_name: str) -> float:
    """Parse a scenario CSV float while treating an empty optional cell as unset."""
    if value is None:
        return float(default)
    if isinstance(value, str) and value.strip().lower() in {"", "none", "nan"}:
        return float(default)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[MABSpatialPricing] {parameter_name} must be a finite number."
        ) from exc
    return float(default) if np.isnan(numeric) else numeric


def _nullable_float(value: Any, parameter_name: str) -> Optional[float]:
    """Parse a finite optional float while preserving an unset CSV cell."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "nan"}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[MABSpatialPricing] {parameter_name} must be a finite number."
        ) from exc
    if not np.isfinite(numeric):
        raise ValueError(
            f"[MABSpatialPricing] {parameter_name} must be a finite number."
        )
    return numeric


def _parse_phase_reset_times(
    value: Any,
    sim_start: int,
    sim_end: int,
    mab_interval: int,
) -> tuple[int, ...]:
    """Validate optional pre-specified EXP3 reset boundaries."""
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "nan"}):
        return ()
    if isinstance(value, str):
        try:
            raw_times = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "[MABSpatialPricing] op_mab_phase_reset_times must be a JSON array."
            ) from exc
    else:
        raw_times = value
    if not isinstance(raw_times, (list, tuple)):
        raise ValueError(
            "[MABSpatialPricing] op_mab_phase_reset_times must be a JSON array."
        )
    parsed_times = []
    for raw_time in raw_times:
        try:
            numeric_time = float(raw_time)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "[MABSpatialPricing] phase reset times must be finite integer seconds."
            ) from exc
        if not np.isfinite(numeric_time) or not numeric_time.is_integer():
            raise ValueError(
                "[MABSpatialPricing] phase reset times must be finite integer seconds."
            )
        reset_time = int(numeric_time)
        if not sim_start < reset_time < sim_end:
            raise ValueError(
                "[MABSpatialPricing] phase reset times must lie strictly within the simulation horizon."
            )
        if (reset_time - sim_start) % mab_interval:
            raise ValueError(
                "[MABSpatialPricing] phase reset times must align with op_mab_interval."
            )
        parsed_times.append(reset_time)
    if len(set(parsed_times)) != len(parsed_times):
        raise ValueError("[MABSpatialPricing] phase reset times must be unique.")
    return tuple(sorted(parsed_times))


def _optional_positive_int(value: Any, default: int, parameter_name: str) -> int:
    """Parse an optional positive integer from CSV-backed scenario settings."""
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "nan"}):
        return int(default)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"[MABSpatialPricing] {parameter_name} must be a positive integer.") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        raise ValueError(f"[MABSpatialPricing] {parameter_name} must be a positive integer.")
    return int(numeric)


INPUT_PARAMETERS_MABSpatialPricing = {
    "doc": """
    MAB-based spatial dynamic pricing.
    Parameters:
        op_mab_algorithm:           ThompsonSampling | EXP3 | HedgeTSEXP3 | ExploreThenCommit
        op_mab_estimator:           SimpleAverage | HT_IX
        op_mab_beta:                IX parameter for HT_IX (default: 0.1)
        op_mab_eta:                 EXP3 learning rate (default: auto)
        op_mab_exp3_eta:            optional EXP3 learning rate, separate from exploration
        op_mab_exp3_exploration_epsilon:
                                    optional EXP3 uniform exploration mass in [0, 1]
        op_mab_exp3_discount_factor:
                                    optional EXP3 exponential memory factor in (0, 1]
        op_mab_exp3_fixed_share_alpha:
                                    optional EXP3 fixed-share mass in [0, 1]
        op_mab_phase_reset_times:   optional JSON array of pre-specified EXP3 reset times
        op_mab_hedge_exp3_reward_upper_bound:
                                    optional fixed upper reward bound for Hedge's inner EXP3;
                                    TS and the Hedge meta-update retain the primary reward bound
        op_mab_hedge_eta:           Hedge meta learning rate (default: sqrt(2 log 2 / T))
        op_mab_hedge_gamma:         minimum exploration added to the mixed Hedge policy
        op_mab_etc_exploration_updates:
                                    number of observed 30-minute feedback batches to
                                    allocate uniformly before ExploreThenCommit locks
                                    the empirically best arm for the remaining horizon
        interference radius:        automatically set to one quarter of the current grid edge length
        op_mab_assignment_level:    zone | passenger (default: zone)
        op_mab_fixed_allocation_probs:
                                    optional fixed arm-probability vector; when set, grid cells
                                    are randomized from this vector every period and no learner
                                    update is performed
        op_mab_ts_mc_samples:       Monte Carlo draws used to approximate Thompson action distribution
        op_mab_period_policy_mode:  cached | legacy_resample
                                    cached constructs one probability vector per period;
                                    legacy_resample reproduces the historical per-grid calls
        op_mab_use_exposure_prob:   use clean-exposure probability in HT/HT-IX denominator
        op_mab_htix_self_normalized:
                                    use sum(wr)/sum(w) for delayed-observed HT/HT-IX feedback
        op_mab_reward_target:       total_fare | request_zero_average | request_zero_snipw
        op_mab_request_reward_delay:
                                    seconds to wait before treating unresolved priced requests as zero reward
        op_mab_reward_event:        offer | acceptance | boarding
        op_mab_feedback_mode:       arrival_batch | delayed_observed | event_matured
                                    delayed_observed updates only when boarding feedback has arrived,
                                    using the original pricing-time treatment context
                                    event_matured retains every assigned request; acceptance uses the
                                    accepted quoted fare, while rejection or cancellation contributes
                                    zero; boarding uses realized fare and unresolved requests mature to zero
        op_mab_htix_diagnostics:    save per-request clean-exposure diagnostics
        op_mab_interference_radius_override:
                                    optional fixed interference radius override used for
                                    legacy reproduction experiments; default is auto grid/4
        op_mab_reward_normalization:
                                    fixed | period_max learner-input normalization
        op_mab_reward_lower_bound:  fixed lower bound for learner-input normalization
        op_mab_reward_upper_bound:  fixed upper bound for learner-input normalization
        op_mab_learning_reward_mode:
                                    aggregate_legacy | request_fixed_scale.  The latter
                                    scales each request before HT-IX aggregation and
                                    leaves the aggregate estimate un-clipped.
        op_mab_grid_beta_reward_upper_bound:
                                    fixed per-request upper bound used to construct each
                                    active grid's fractional-Beta reward
        op_mab_zone_system:        optional pricing/assignment zone system. When omitted,
                                    op_zone_system is used for both operations and MAB pricing.
        op_mab_grid_schedule:       optional list of time-of-day grid entries, each with
                                    start_time, end_time, zone_system
    """,
    "inherit": "DynamicPricingBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        "op_mab_interval",
        "op_mab_arms",
        "op_mab_fixed_arm",
        "op_mab_random_assignment",
        "op_mab_random_seed",
        "op_mab_algorithm",
        "op_mab_estimator",
        "op_mab_beta",
        "op_mab_eta",
        "op_mab_exp3_eta",
        "op_mab_exp3_exploration_epsilon",
        "op_mab_exp3_discount_factor",
        "op_mab_exp3_fixed_share_alpha",
        "op_mab_phase_reset_times",
        "op_mab_hedge_exp3_reward_upper_bound",
        "op_mab_hedge_eta",
        "op_mab_hedge_gamma",
        "op_mab_hedge_risk_lambda",
        "op_mab_fixed_share_alpha",
        "op_mab_discount_factor",
        "op_mab_etc_exploration_updates",
        "op_mab_assignment_level",
        "op_mab_fixed_allocation_probs",
        "op_mab_ts_mc_samples",
        "op_mab_period_policy_mode",
        "op_mab_use_exposure_prob",
        "op_mab_htix_self_normalized",
        "op_mab_reward_target",
        "op_mab_request_reward_delay",
        "op_mab_feedback_mode",
        "op_mab_htix_diagnostics",
        "op_mab_interference_radius_override",
        "op_mab_reward_normalization",
        "op_mab_reward_lower_bound",
        "op_mab_reward_upper_bound",
        "op_mab_learning_reward_mode",
        "op_mab_grid_beta_reward_upper_bound",
        "op_mab_zone_system",
        "op_mab_grid_schedule",
        "op_mab_reward_event",
    ],
    "mandatory_modules": [],
    "optional_modules": [],
}

DEFAULT_ARMS = [(0.8, 0.8), (1.0, 1.0), (1.2, 1.2)]


def _infer_grid_edge_meters(zone_system_name: Any) -> float:
    match = re.search(r"(\d+)m", str(zone_system_name))
    if not match:
        raise ValueError(
            "[MABSpatialPricing] Could not infer grid edge from zone system "
            f"'{zone_system_name}'. Expected a name like Manhattan_Grid_700m."
        )
    return float(match.group(1))


def _resolve_interference_radius(
    grid_edge_m: float,
    configured_override: Optional[Any],
) -> float:
    if configured_override is None or str(configured_override).strip() == "":
        return float(grid_edge_m) / 4.0
    return float(configured_override)


class MABSpatialPricing(DynamicPricingBase):
    def __init__(self, fleetctrl, operator_attributes, solver="Gurobi"):
        super().__init__(fleetctrl, operator_attributes, solver)

        self.fleetctrl = fleetctrl
        self.mab_interval = int(operator_attributes.get("op_mab_interval", 1800))
        self.arms = operator_attributes.get("op_mab_arms", DEFAULT_ARMS)
        self.n_arms = len(self.arms)
        raw_fixed_arm = operator_attributes.get("op_mab_fixed_arm", None)
        if raw_fixed_arm is None:
            self.fixed_arm = None
        else:
            try:
                numeric_fixed_arm = float(raw_fixed_arm)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "[MABSpatialPricing] op_mab_fixed_arm must be an integer arm index."
                ) from exc
            if np.isnan(numeric_fixed_arm):
                self.fixed_arm = None
            elif not np.isfinite(numeric_fixed_arm) or not numeric_fixed_arm.is_integer():
                raise ValueError(
                    "[MABSpatialPricing] op_mab_fixed_arm must be an integer arm index."
                )
            else:
                self.fixed_arm = int(numeric_fixed_arm)
        if self.fixed_arm is not None and not 0 <= self.fixed_arm < self.n_arms:
            raise ValueError(
                "[MABSpatialPricing] op_mab_fixed_arm must be a valid arm index."
            )
        raw_fixed_probs = operator_attributes.get("op_mab_fixed_allocation_probs", None)
        self.fixed_allocation_probs = None
        if raw_fixed_probs is not None:
            fixed_probs = np.asarray(list(raw_fixed_probs), dtype=float)
            if len(fixed_probs) != self.n_arms:
                raise ValueError(
                    "[MABSpatialPricing] op_mab_fixed_allocation_probs must match "
                    "the number of pricing arms."
                )
            if np.any(fixed_probs < 0) or float(fixed_probs.sum()) <= 0:
                raise ValueError(
                    "[MABSpatialPricing] fixed-allocation probabilities must be "
                    "non-negative and sum to a positive value."
                )
            self.fixed_allocation_probs = fixed_probs / fixed_probs.sum()
        if self.fixed_arm is not None and self.fixed_allocation_probs is not None:
            raise ValueError(
                "[MABSpatialPricing] op_mab_fixed_arm and "
                "op_mab_fixed_allocation_probs are mutually exclusive."
            )
        self.reward_event = str(operator_attributes.get("op_mab_reward_event", "offer")).lower()
        if self.reward_event not in {"offer", "acceptance", "boarding"}:
            raise ValueError(
                "[MABSpatialPricing] op_mab_reward_event must be 'offer', "
                "'acceptance', or 'boarding'."
            )
        self.reward_target = str(operator_attributes.get("op_mab_reward_target", "total_fare")).lower()
        if self.reward_target not in {"total_fare", "request_zero_average", "request_zero_snipw"}:
            raise ValueError(
                "[MABSpatialPricing] op_mab_reward_target must be one of "
                "'total_fare', 'request_zero_average', or 'request_zero_snipw'."
            )
        if self.reward_target != "total_fare" and self.reward_event != "boarding":
            raise ValueError(
                "[MABSpatialPricing] request-level reward targets require "
                "op_mab_reward_event='boarding'."
            )
        self.random_assignment = bool(operator_attributes.get("op_mab_random_assignment", False))
        if self.random_assignment and self.fixed_allocation_probs is not None:
            raise ValueError(
                "[MABSpatialPricing] checkerboard random assignment and fixed-allocation "
                "probabilities are mutually exclusive."
            )
        self.assignment_level = operator_attributes.get("op_mab_assignment_level", "zone")
        operational_zone_system_name = operator_attributes.get(
            "op_zone_system",
            operator_attributes.get("zone_system_name", ""),
        )
        pricing_zone_system_name = operator_attributes.get(
            "op_mab_zone_system",
            operational_zone_system_name,
        )
        self._operational_zone_system_name = operational_zone_system_name
        self._zone_system_name = pricing_zone_system_name
        operator_attributes["op_mab_zone_system_resolved"] = pricing_zone_system_name
        self._grid_edge_m = _infer_grid_edge_meters(pricing_zone_system_name)
        self._configured_interference_radius_override = operator_attributes.get(
            "op_mab_interference_radius_override",
            None,
        )
        self._interference_radius = _resolve_interference_radius(
            self._grid_edge_m,
            self._configured_interference_radius_override,
        )
        operator_attributes["op_mab_interference_radius_auto"] = self._interference_radius
        self._seed = int(operator_attributes.get("op_mab_random_seed", 42))
        self.rng = np.random.default_rng(self._seed)
        self.reward_normalization_mode = str(
            operator_attributes.get("op_mab_reward_normalization", "fixed")
        ).strip().lower()
        if self.reward_normalization_mode not in {"fixed", "period_max"}:
            raise ValueError(
                "[MABSpatialPricing] op_mab_reward_normalization must be "
                "'fixed' or 'period_max'."
            )
        self.reward_lower_bound = _optional_float(
            operator_attributes.get("op_mab_reward_lower_bound"),
            0.0,
            "op_mab_reward_lower_bound",
        )
        self.reward_upper_bound = _optional_float(
            operator_attributes.get("op_mab_reward_upper_bound"),
            13636.0,
            "op_mab_reward_upper_bound",
        )
        if not np.isfinite(self.reward_lower_bound) or not np.isfinite(self.reward_upper_bound):
            raise ValueError("[MABSpatialPricing] reward-normalization bounds must be finite.")
        if self.reward_upper_bound <= self.reward_lower_bound:
            raise ValueError(
                "[MABSpatialPricing] op_mab_reward_upper_bound must exceed "
                "op_mab_reward_lower_bound."
            )
        self.grid_beta_reward_upper_bound = _optional_float(
            operator_attributes.get("op_mab_grid_beta_reward_upper_bound"),
            self.reward_upper_bound,
            "op_mab_grid_beta_reward_upper_bound",
        )
        if (
            not np.isfinite(self.grid_beta_reward_upper_bound)
            or self.grid_beta_reward_upper_bound <= self.reward_lower_bound
        ):
            raise ValueError(
                "[MABSpatialPricing] op_mab_grid_beta_reward_upper_bound must "
                "be finite and exceed op_mab_reward_lower_bound."
            )
        operator_attributes["op_mab_reward_lower_bound_resolved"] = self.reward_lower_bound
        operator_attributes["op_mab_reward_upper_bound_resolved"] = self.reward_upper_bound
        operator_attributes[
            "op_mab_grid_beta_reward_upper_bound_resolved"
        ] = self.grid_beta_reward_upper_bound
        operator_attributes[
            "op_mab_reward_normalization_resolved"
        ] = self.reward_normalization_mode
        self._last_reward_normalization_denominator = np.nan
        self._last_reward_standard_errors = {}
        self.learning_reward_mode = str(
            operator_attributes.get("op_mab_learning_reward_mode", "aggregate_legacy")
        ).strip().lower()
        if self.learning_reward_mode not in {
            "aggregate_legacy",
            "request_fixed_scale",
        }:
            raise ValueError(
                "[MABSpatialPricing] op_mab_learning_reward_mode must be "
                "'aggregate_legacy' or 'request_fixed_scale'."
            )
        if (
            self.learning_reward_mode == "request_fixed_scale"
            and self.reward_lower_bound != 0.0
        ):
            raise ValueError(
                "[MABSpatialPricing] request_fixed_scale requires a zero "
                "op_mab_reward_lower_bound."
            )
        operator_attributes[
            "op_mab_learning_reward_mode_resolved"
        ] = self.learning_reward_mode

        if self.assignment_level not in ("zone", "passenger"):
            raise ValueError(f"[MABSpatialPricing] Unknown assignment level: {self.assignment_level}")
        if self.assignment_level == "passenger" and not self.random_assignment:
            raise ValueError(
                "[MABSpatialPricing] passenger assignment currently requires op_mab_random_assignment=True"
            )
        if self.assignment_level == "zone" and self.random_assignment and self.n_arms != 2:
            raise ValueError(
                "[MABSpatialPricing] checkerboard random assignment currently supports exactly 2 arms"
            )

        algo_name = operator_attributes.get("op_mab_algorithm", "ThompsonSampling")
        estimator_name = operator_attributes.get("op_mab_estimator", "SimpleAverage")
        if (
            self.learning_reward_mode == "request_fixed_scale"
            and algo_name
            not in {"JiaEXP3", "DiscountedJiaEXP3", "HedgeGaussianTSJiaEXP3"}
        ):
            raise ValueError(
                "[MABSpatialPricing] request_fixed_scale is reserved for "
                "JiaEXP3, DiscountedJiaEXP3, and HedgeGaussianTSJiaEXP3 so the JiaEXP3 "
                "aggregate HT-IX signal is not clipped by an incompatible learner."
            )
        eta_val = operator_attributes.get("op_mab_eta", None)
        exp3_eta_val = _nullable_float(
            operator_attributes.get("op_mab_exp3_eta"),
            "op_mab_exp3_eta",
        )
        if exp3_eta_val is not None and exp3_eta_val <= 0.0:
            raise ValueError(
                "[MABSpatialPricing] op_mab_exp3_eta must be positive when set."
            )
        exp3_exploration_epsilon_val = _nullable_float(
            operator_attributes.get("op_mab_exp3_exploration_epsilon"),
            "op_mab_exp3_exploration_epsilon",
        )
        if (
            exp3_exploration_epsilon_val is not None
            and not 0.0 <= exp3_exploration_epsilon_val <= 1.0
        ):
            raise ValueError(
                "[MABSpatialPricing] op_mab_exp3_exploration_epsilon must lie in [0, 1]."
            )
        exp3_discount_factor_val = _optional_float(
            operator_attributes.get("op_mab_exp3_discount_factor"),
            1.0,
            "op_mab_exp3_discount_factor",
        )
        if not 0.0 < exp3_discount_factor_val <= 1.0:
            raise ValueError(
                "[MABSpatialPricing] op_mab_exp3_discount_factor must lie in (0, 1]."
            )
        exp3_fixed_share_alpha_val = _optional_float(
            operator_attributes.get("op_mab_exp3_fixed_share_alpha"),
            0.0,
            "op_mab_exp3_fixed_share_alpha",
        )
        if not 0.0 <= exp3_fixed_share_alpha_val <= 1.0:
            raise ValueError(
                "[MABSpatialPricing] op_mab_exp3_fixed_share_alpha must lie in [0, 1]."
            )
        beta_val = float(operator_attributes.get("op_mab_beta", 0.1))
        ts_mc_samples = int(operator_attributes.get("op_mab_ts_mc_samples", 2000))
        self.use_exposure_prob = _as_bool(
            operator_attributes.get("op_mab_use_exposure_prob", False),
            default=False,
        )
        self.htix_self_normalized = _as_bool(
            operator_attributes.get("op_mab_htix_self_normalized", False),
            default=False,
        )
        self.period_policy_mode = str(
            operator_attributes.get("op_mab_period_policy_mode", "cached")
        ).strip().lower()
        if self.period_policy_mode not in {"cached", "legacy_resample"}:
            raise ValueError(
                "[MABSpatialPricing] op_mab_period_policy_mode must be "
                "'cached' or 'legacy_resample'."
            )
        operator_attributes["op_mab_period_policy_mode_resolved"] = self.period_policy_mode
        self.request_reward_delay = int(
            operator_attributes.get("op_mab_request_reward_delay", self.mab_interval)
        )
        self.feedback_mode = str(
            operator_attributes.get("op_mab_feedback_mode", "arrival_batch")
        ).strip().lower()
        if self.feedback_mode not in {"arrival_batch", "delayed_observed", "event_matured"}:
            raise ValueError(
                "[MABSpatialPricing] op_mab_feedback_mode must be either "
                "'arrival_batch', 'delayed_observed', or 'event_matured'."
            )
        self.htix_diagnostics_enabled = _as_bool(
            operator_attributes.get("op_mab_htix_diagnostics", False),
            default=False,
        )
        hedge_eta_val = operator_attributes.get("op_mab_hedge_eta", None)
        hedge_gamma_val = float(operator_attributes.get("op_mab_hedge_gamma", 0.02))
        hedge_risk_lambda_val = _optional_float(
            operator_attributes.get("op_mab_hedge_risk_lambda"),
            1.0,
            "op_mab_hedge_risk_lambda",
        )
        if hedge_risk_lambda_val < 0.0:
            raise ValueError(
                "[MABSpatialPricing] op_mab_hedge_risk_lambda must be non-negative."
            )
        fixed_share_alpha_val = operator_attributes.get("op_mab_fixed_share_alpha", None)
        discount_factor_val = _optional_float(
            operator_attributes.get("op_mab_discount_factor"),
            1.0,
            "op_mab_discount_factor",
        )
        if not 0.0 < discount_factor_val <= 1.0:
            raise ValueError(
                "[MABSpatialPricing] op_mab_discount_factor must lie in (0, 1]."
            )
        sim_start = int(operator_attributes.get("start_time", 0))
        sim_end = int(operator_attributes.get("end_time", 86400))
        T_episodes = max(1, int((sim_end - sim_start) / max(self.mab_interval, 1)))
        phase_reset_times = _parse_phase_reset_times(
            operator_attributes.get("op_mab_phase_reset_times"),
            sim_start,
            sim_end,
            self.mab_interval,
        )
        if phase_reset_times and algo_name != "EXP3":
            raise ValueError(
                "[MABSpatialPricing] op_mab_phase_reset_times is supported only for EXP3."
            )
        hedge_exp3_reward_upper_bound_val = _nullable_float(
            operator_attributes.get("op_mab_hedge_exp3_reward_upper_bound"),
            "op_mab_hedge_exp3_reward_upper_bound",
        )
        if hedge_exp3_reward_upper_bound_val is not None:
            if algo_name != "HedgeTSEXP3":
                raise ValueError(
                    "[MABSpatialPricing] op_mab_hedge_exp3_reward_upper_bound "
                    "is supported only for HedgeTSEXP3."
                )
            if self.reward_normalization_mode != "fixed":
                raise ValueError(
                    "[MABSpatialPricing] separate Hedge EXP3 reward scaling "
                    "requires fixed reward normalization."
                )
            if self.learning_reward_mode != "aggregate_legacy":
                raise ValueError(
                    "[MABSpatialPricing] separate Hedge EXP3 reward scaling "
                    "requires aggregate_legacy learning rewards."
                )
            if hedge_exp3_reward_upper_bound_val <= self.reward_lower_bound:
                raise ValueError(
                    "[MABSpatialPricing] op_mab_hedge_exp3_reward_upper_bound "
                    "must exceed op_mab_reward_lower_bound."
                )
        etc_exploration_updates = _optional_positive_int(
            operator_attributes.get("op_mab_etc_exploration_updates", None),
            1,
            "op_mab_etc_exploration_updates",
        )
        if algo_name == "ExploreThenCommit" and etc_exploration_updates >= T_episodes:
            raise ValueError(
                "[MABSpatialPricing] op_mab_etc_exploration_updates must be shorter "
                "than the simulation horizon for ExploreThenCommit."
            )
        self._algo_name = algo_name
        self._eta_val = eta_val
        self._exp3_eta_val = exp3_eta_val
        self._exp3_exploration_epsilon_val = exp3_exploration_epsilon_val
        self._exp3_discount_factor_val = float(exp3_discount_factor_val)
        self._exp3_fixed_share_alpha_val = float(exp3_fixed_share_alpha_val)
        operator_attributes[
            "op_mab_exp3_fixed_share_alpha_resolved"
        ] = self._exp3_fixed_share_alpha_val
        self._hedge_exp3_reward_upper_bound = hedge_exp3_reward_upper_bound_val
        operator_attributes[
            "op_mab_hedge_exp3_reward_upper_bound_resolved"
        ] = (
            float(hedge_exp3_reward_upper_bound_val)
            if hedge_exp3_reward_upper_bound_val is not None
            else float(self.reward_upper_bound)
        )
        self._beta_val = beta_val
        self._ts_mc_samples = ts_mc_samples
        self._hedge_eta_val = hedge_eta_val
        self._hedge_gamma_val = hedge_gamma_val
        self._hedge_risk_lambda_val = hedge_risk_lambda_val
        self._fixed_share_alpha_val = (
            float(fixed_share_alpha_val) if fixed_share_alpha_val is not None else None
        )
        self._discount_factor_val = float(discount_factor_val)
        self._etc_exploration_updates = etc_exploration_updates
        self._T_episodes = T_episodes
        self._phase_reset_times = phase_reset_times
        self._applied_phase_reset_times = set()
        self._phase_reset_count = 0
        self._last_phase_reset_time = None
        operator_attributes["op_mab_phase_reset_times_resolved"] = list(
            self._phase_reset_times
        )
        self.estimator = _load_estimator(estimator_name, beta=beta_val)
        self.algo_name = algo_name
        if self._algo_name in {"GridBetaThompsonSampling", "HedgeGridBetaEXP3"}:
            if self.assignment_level != "zone":
                raise ValueError(
                    "[MABSpatialPricing] Grid-Beta methods require "
                    "zone-level assignment."
                )
            if self.reward_normalization_mode != "fixed":
                raise ValueError(
                    "[MABSpatialPricing] Grid-Beta methods require a "
                    "fixed reward scale, not period_max normalization."
                )
            if not self._uses_event_matured_feedback():
                raise ValueError(
                    "[MABSpatialPricing] Grid-Beta methods require "
                    "event_matured acceptance or boarding feedback."
                )
        self._grid_schedule = self._parse_grid_schedule(operator_attributes)
        self._dynamic_grid_enabled = bool(self._grid_schedule)
        if self._phase_reset_times and self._dynamic_grid_enabled:
            raise ValueError(
                "[MABSpatialPricing] phase-reset diagnostics require a fixed pricing grid."
            )
        self._pricing_states = {}
        self._active_state_key = None
        self._checkerboard_base = None
        self._checkerboard_phase = 0
        self._rid_to_arm = {}
        self._fare_records = []
        self._recorded_rids = set()
        self._accepted_reward_rids = set()
        self._period_probs = None
        self._period_algo_debug_state = {}

        if self._dynamic_grid_enabled:
            self._init_dynamic_grid_states(sim_start)
        else:
            self.node_to_zone, self.n_zones, node_zone_path = self._load_zone_mapping()
            self._node_xy = None
            self._kdtree = None
            self._node_zone_by_node_id = None
            self._load_network_nodes(node_zone_path)
            self.global_algo = self._new_algo(self._seed)
            if self.period_policy_mode == "cached":
                self._refresh_period_policy()
            self.zone_neighbors = self._compute_zone_neighbors()
            self.zone_algos = [self.global_algo] * self.n_zones

            self._episode_count = 0
            self.ucb_counts = np.zeros((self.n_zones, self.n_arms), dtype=float)
            self.ucb_rewards = np.zeros((self.n_zones, self.n_arms), dtype=float)
            self.zone_pulls = np.zeros(self.n_zones, dtype=int)

            if self.assignment_level == "passenger":
                self.current_arm = np.full(self.n_zones, -1, dtype=int)
            elif self.random_assignment:
                self.current_arm = self._balanced_random_assignment()
            else:
                self.current_arm = self._per_zone_sample()
            self.episode_start_time = None
            self.zone_episode_revenue = np.zeros(self.n_zones, dtype=float)
            self.zone_revenue = np.zeros(self.n_zones, dtype=float)
        if self._dynamic_grid_enabled:
            self._set_active_state_refs(self._pricing_states[self._active_state_key])

        self.mab_log = []
        self.mab_log_path = os.path.join(self.output_dir, "mab_spatial_pricing_log.csv")
        self.request_treatment_records = {}
        self._request_outcomes = {}
        self._request_outcome_times = {}
        self._request_outcome_types = {}
        self._request_outcome_reasons = {}
        self._processed_request_reward_rids = set()
        self._processed_delayed_feedback_rids = set()
        self._processed_grid_beta_feedback_rids = set()
        self._late_request_outcome_count = 0
        self._last_request_reward_debug = {}
        self._last_delayed_feedback_debug = {}
        self._last_grid_beta_debug = {}
        self.request_log_path = os.path.join(self.output_dir, "mab_spatial_request_log.csv")
        self._htix_diagnostics = []
        self.htix_diagnostics_path = os.path.join(
            self.output_dir, "mab_spatial_htix_diagnostics.csv"
        )

        print(
            f"[MABSpatialPricing] Initialized: zones={self.n_zones}, arms={self.n_arms}, "
            f"interval={self.mab_interval}s, algo={algo_name}, estimator={self.estimator.name()}, "
            f"random_assignment={self.random_assignment}, assignment_level={self.assignment_level}, "
            f"reward_event={self.reward_event}, reward_target={self.reward_target}, "
            f"feedback_mode={self.feedback_mode}, use_exposure_prob={self.use_exposure_prob}, "
            f"reward_normalization={self.reward_normalization_mode}, "
            f"reward_bounds=({self.reward_lower_bound}, {self.reward_upper_bound})"
        )
        print(f"[MABSpatialPricing] Arms: {self.arms}")
        if self._dynamic_grid_enabled:
            schedule_desc = ", ".join(
                f"{entry['zone_system']}[{int(entry['start_time'])},{int(entry['end_time']) if entry['end_time'] is not None else 'end'})"
                for entry in self._grid_schedule
            )
            print(f"[MABSpatialPricing] Dynamic grid schedule enabled: {schedule_desc}")
        else:
            if str(self._operational_zone_system_name) != str(self._zone_system_name):
                print(
                    "[MABSpatialPricing] Split zone systems: "
                    f"operational={self._operational_zone_system_name}, "
                    f"pricing={self._zone_system_name}"
                )
            print(
                f"[MABSpatialPricing] Derived interference radius from grid edge: "
                f"edge={self._grid_edge_m:.1f}m -> r={self._interference_radius:.1f}m"
            )





    @staticmethod
    def _normalize_factors(factors: Iterable[Any]):
        factors = list(factors)
        if len(factors) != 2:
            raise ValueError(f"[MABSpatialPricing] Pricing arm factors must have length 2, got {factors}")
        return float(factors[0]), float(factors[1])

    @classmethod
    def _arm_label(cls, arm_spec: Any, arm_idx: int) -> str:
        if isinstance(arm_spec, dict):
            return str(arm_spec.get("label", f"arm_{arm_idx}"))
        try:
            base_factor, dist_factor = cls._normalize_factors(arm_spec)
            return f"arm_{arm_idx}({base_factor:.3f},{dist_factor:.3f})"
        except Exception:
            return f"arm_{arm_idx}"

    @staticmethod
    def _schedule_entry_matches(sim_time: float, entry: dict) -> bool:
        start_time = float(entry.get("start_time", 0))
        end_time = entry.get("end_time")
        if end_time is None:
            return sim_time >= start_time
        end_time = float(end_time)
        if start_time <= end_time:
            return start_time <= sim_time < end_time
        return sim_time >= start_time or sim_time < end_time

    @classmethod
    def _resolve_arm_at_time(cls, arm_spec: Any, sim_time: float, arm_idx: int):
        if isinstance(arm_spec, dict):
            default_factors = cls._normalize_factors(arm_spec.get("default_factors", (1.0, 1.0)))
            label = cls._arm_label(arm_spec, arm_idx)
            for entry in arm_spec.get("schedule", []):
                if cls._schedule_entry_matches(sim_time, entry):
                    factors = cls._normalize_factors(entry.get("factors", default_factors))
                    return factors[0], factors[1], label
            return default_factors[0], default_factors[1], label

        base_factor, dist_factor = cls._normalize_factors(arm_spec)
        return base_factor, dist_factor, cls._arm_label(arm_spec, arm_idx)

    def _record_request_treatment(
        self,
        sim_time,
        rid,
        zone_id,
        arm_idx,
        base_factor,
        dist_factor,
        arm_label,
        zone_system_name=None,
        grid_edge_m=None,
        arm_prob=None,
        node_id=None,
        pricing_episode_start=None,
        algo_debug_state=None,
    ):
        if rid is None:
            return
        rid_str = str(rid)
        if rid_str in self.request_treatment_records:
            return
        if pricing_episode_start is None:
            pricing_episode_start = self._pricing_episode_start_for_time(sim_time)
        if algo_debug_state is None:
            algo_debug_state = dict(self._period_algo_debug_state)
        treatment_name = (
            f"arm{arm_idx}|{arm_label}|bf={base_factor:.3f}|df={dist_factor:.3f}|"
            f"level={self.assignment_level}"
        )
        self.request_treatment_records[rid_str] = {
            "request_id": rid_str,
            "pricing_time": sim_time,
            "pricing_episode_start": float(pricing_episode_start),
            "zone_id": int(zone_id) if zone_id is not None else -1,
            "arm": int(arm_idx),
            "arm_label": arm_label,
            "treatment_name": treatment_name,
            "base_factor": float(base_factor),
            "dist_factor": float(dist_factor),
            "general_factor": 1.0,
            "assignment_level": self.assignment_level,
            "zone_system_name": zone_system_name,
            "grid_edge_m": float(grid_edge_m) if grid_edge_m is not None else np.nan,
            "arm_prob": float(arm_prob) if arm_prob is not None else np.nan,
            "period_probs": np.asarray(self._period_probs, dtype=float).tolist()
            if self._period_probs is not None
            else None,
            "node_id": int(node_id) if node_id is not None else -1,
            "pricing_ts_probs": algo_debug_state.get("ts_probs"),
            "pricing_exp3_probs": algo_debug_state.get("exp3_probs"),
            "pricing_hedge_expert_trust": algo_debug_state.get("expert_trust"),
        }

    def _uses_request_reward_target(self) -> bool:
        return self.reward_target in {"request_zero_average", "request_zero_snipw"}

    def _uses_delayed_observed_feedback(self) -> bool:
        return self.feedback_mode == "delayed_observed" and self.reward_event == "boarding"

    def _uses_event_matured_feedback(self) -> bool:
        return self.feedback_mode == "event_matured" and self.reward_event in {
            "acceptance",
            "boarding",
        }

    def _uses_grid_beta_evidence(self) -> bool:
        return bool(getattr(self.global_algo, "grid_beta_evidence", False))

    def _uses_grid_beta_hedge(self) -> bool:
        return bool(getattr(self.global_algo, "grid_beta_with_exp3", False))

    def _estimate_grid_beta_hedge_companion_feedback(
        self,
        sim_time: float,
        zone_system_name: Optional[str] = None,
    ) -> tuple[dict, dict]:
        """Return the EXP3/Hedge signal paired with Grid-Beta TS evidence.

        Grid-Beta TS has its own per-grid fractional update.  The companion
        EXP3 and Hedge comparison receive the configured request-level
        estimator on the same fixed [0, 1] reward scale.
        """
        if not self._uses_event_matured_feedback():
            raise RuntimeError(
                "Grid-Beta Hedge requires event-matured acceptance or boarding feedback."
            )
        raw_rewards = self._estimate_delayed_observed_feedback(
            sim_time,
            zone_system_name=zone_system_name,
        )
        return raw_rewards, self._normalize_learning_rewards(raw_rewards)

    def _pricing_episode_start_for_time(self, sim_time: float) -> float:
        if self.episode_start_time is not None:
            try:
                if float(sim_time) >= float(self.episode_start_time):
                    return float(self.episode_start_time)
            except (TypeError, ValueError):
                pass
        interval = max(float(self.mab_interval), 1.0)
        return float(sim_time) - (float(sim_time) % interval)

    def _record_request_outcome(
        self,
        rid_key: str,
        amount: float,
        sim_time=None,
        outcome_type: str = "reward",
        reason: Optional[str] = None,
    ) -> None:
        if rid_key is None:
            return
        if rid_key not in self._request_outcomes and (
            rid_key in self._processed_request_reward_rids
            or rid_key in self._processed_delayed_feedback_rids
        ):
            self._late_request_outcome_count += 1
        self._request_outcomes[rid_key] = float(amount)
        if sim_time is not None:
            self._request_outcome_times[rid_key] = float(sim_time)
        self._request_outcome_types[rid_key] = str(outcome_type)
        if reason is not None:
            self._request_outcome_reasons[rid_key] = str(reason)

    def record_request_failure(self, rid=None, sim_time=None, reason: str = "request_failure") -> None:
        """Record a matured zero-reward request outcome for event-driven feedback."""
        if not self._uses_event_matured_feedback() or rid is None:
            return
        rid_key = str(rid)
        if (
            rid_key in self._processed_request_reward_rids
            or rid_key in self._processed_delayed_feedback_rids
        ):
            self._late_request_outcome_count += 1
            return
        if rid_key in self._recorded_rids:
            return
        if rid_key not in self.request_treatment_records:
            return
        self._recorded_rids.add(rid_key)
        self._record_request_outcome(
            rid_key,
            0.0,
            sim_time=sim_time,
            outcome_type="zero",
            reason=reason,
        )

    def record_request_acceptance(self, rid=None, amount=None, sim_time=None) -> None:
        """Store an accepted quote pending event-matured learning.

        A later cancellation may overwrite this provisional positive outcome with
        zero before it is processed. This avoids negative correction updates,
        which are not valid inputs for every learner used in the experiment.
        """
        if (
            self.reward_event != "acceptance"
            or not self._uses_event_matured_feedback()
            or rid is None
            or amount is None
        ):
            return
        rid_key = str(rid)
        if (
            rid_key in self._processed_request_reward_rids
            or rid_key in self._processed_delayed_feedback_rids
        ):
            self._late_request_outcome_count += 1
            return
        if (
            rid_key in self._recorded_rids
            or rid_key in self._accepted_reward_rids
            or rid_key not in self.request_treatment_records
        ):
            return
        self._accepted_reward_rids.add(rid_key)
        self._record_request_outcome(
            rid_key,
            float(amount),
            sim_time=sim_time,
            outcome_type="acceptance",
            reason="accepted_quote",
        )

    def _estimate_request_reward_target(self, sim_time: float, zone_system_name: Optional[str] = None) -> dict:
        sums = np.zeros(self.n_arms, dtype=float)
        counts = np.zeros(self.n_arms, dtype=float)
        weighted_sums = np.zeros(self.n_arms, dtype=float)
        weights = np.zeros(self.n_arms, dtype=float)
        served_counts = np.zeros(self.n_arms, dtype=float)
        matured_count = 0
        pending_count = 0
        cutoff = float(sim_time) - float(self.request_reward_delay)

        for rid_key, record in self.request_treatment_records.items():
            if rid_key in self._processed_request_reward_rids:
                continue
            if zone_system_name is not None and record.get("zone_system_name") != zone_system_name:
                continue
            try:
                pricing_time = float(record.get("pricing_time"))
            except (TypeError, ValueError):
                continue
            if pricing_time > cutoff:
                pending_count += 1
                continue
            try:
                arm = int(record.get("arm"))
            except (TypeError, ValueError):
                self._processed_request_reward_rids.add(rid_key)
                continue
            if arm < 0 or arm >= self.n_arms:
                self._processed_request_reward_rids.add(rid_key)
                continue

            amount = float(self._request_outcomes.get(rid_key, 0.0))
            arm_prob = record.get("arm_prob")
            try:
                arm_prob = float(arm_prob)
            except (TypeError, ValueError):
                arm_prob = np.nan
            if not np.isfinite(arm_prob) or arm_prob <= 0:
                arm_prob = float(self._assignment_probability_for_arm(arm))
            if not np.isfinite(arm_prob) or arm_prob <= 0:
                arm_prob = 1.0 / max(1, self.n_arms)

            denom = max(float(arm_prob) + float(self._beta_val), 1e-9)
            weight = 1.0 / denom
            sums[arm] += amount
            counts[arm] += 1.0
            weighted_sums[arm] += amount * weight
            weights[arm] += weight
            if amount > 0:
                served_counts[arm] += 1.0
            matured_count += 1
            self._processed_request_reward_rids.add(rid_key)

        if self.reward_target == "request_zero_snipw":
            rewards_arr = np.divide(
                weighted_sums,
                weights,
                out=np.zeros(self.n_arms, dtype=float),
                where=weights > 0,
            )
        else:
            rewards_arr = np.divide(
                sums,
                counts,
                out=np.zeros(self.n_arms, dtype=float),
                where=counts > 0,
            )

        self._last_request_reward_debug = {
            "request_reward_target": self.reward_target,
            "request_reward_delay": int(self.request_reward_delay),
            "matured_request_count": int(matured_count),
            "pending_request_count": int(pending_count),
            "late_request_outcome_count": int(self._late_request_outcome_count),
            "request_counts_by_arm": counts.astype(float).tolist(),
            "served_counts_by_arm": served_counts.astype(float).tolist(),
            "request_reward_sums_by_arm": sums.astype(float).tolist(),
            "request_reward_weights_by_arm": weights.astype(float).tolist(),
        }
        return {arm: float(rewards_arr[arm]) for arm in range(self.n_arms)}

    def _records_in_pricing_episode(self, zone_system_name: Optional[str], episode_start: float):
        records = []
        for request_id, record in self.request_treatment_records.items():
            if zone_system_name is not None and record.get("zone_system_name") != zone_system_name:
                continue
            try:
                record_episode_start = float(record.get("pricing_episode_start"))
            except (TypeError, ValueError):
                continue
            if abs(record_episode_start - float(episode_start)) > 1e-6:
                continue
            try:
                node_id = int(record.get("node_id", -1))
                arm = int(record.get("arm", -1))
                zone_id = int(record.get("zone_id", -1))
            except (TypeError, ValueError):
                continue
            if node_id < 0 or arm < 0 or zone_id < 0:
                continue
            if self._node_xy is None or node_id >= len(self._node_xy):
                continue
            records.append((str(request_id), record, node_id, arm, zone_id))
        return records

    def _clean_exposure_details(
        self,
        rid_key: str,
        record: dict,
        zone_system_name: Optional[str],
    ) -> dict:
        try:
            node_id = int(record.get("node_id", -1))
            arm = int(record.get("arm", -1))
            zone_id = int(record.get("zone_id", -1))
            episode_start = float(record.get("pricing_episode_start"))
        except (TypeError, ValueError):
            return {"clean": False, "reason": "invalid_treatment_record"}

        if self._node_xy is None or node_id < 0 or node_id >= len(self._node_xy):
            return {"clean": False, "reason": "missing_node_location"}

        episode_records = self._records_in_pricing_episode(zone_system_name, episode_start)
        if not episode_records:
            return {"clean": False, "reason": "empty_pricing_episode"}

        episode_node_ids = np.array([item[2] for item in episode_records], dtype=int)
        episode_xy = self._node_xy[episode_node_ids]
        episode_tree = KDTree(episode_xy)
        query_xy = self._node_xy[node_id]
        neighbor_kd_idxs = episode_tree.query_ball_point(
            query_xy,
            r=self._interference_radius,
            p=np.inf,
        )

        neighbor_ids = []
        neighbor_arms = []
        neighbor_zones = []
        interfering_request_ids = []
        same_arm_count = 0
        different_arm_count = 0
        distinct_neighbor_zones = set()

        for kd_idx in neighbor_kd_idxs:
            neighbor_rid, _neighbor_record, neighbor_node_id, neighbor_arm, neighbor_zone = (
                episode_records[kd_idx]
            )
            if neighbor_rid == rid_key:
                continue
            neighbor_ids.append(neighbor_rid)
            neighbor_arms.append(int(neighbor_arm))
            neighbor_zones.append(int(neighbor_zone))
            distinct_neighbor_zones.add(int(neighbor_zone))
            if int(neighbor_arm) == arm:
                same_arm_count += 1
            else:
                different_arm_count += 1
                interfering_request_ids.append(neighbor_rid)

        clean = different_arm_count == 0
        active_exposure_zones = set(distinct_neighbor_zones)
        active_exposure_zones.add(int(zone_id))
        x_coord, y_coord = query_xy
        return {
            "clean": bool(clean),
            "reason": "clean" if clean else "neighbor_different_arm",
            "node_id": int(node_id),
            "x": float(x_coord),
            "y": float(y_coord),
            "zone_id": int(zone_id),
            "arm": int(arm),
            "pricing_episode_start": float(episode_start),
            "neighbor_count": int(len(neighbor_ids)),
            "same_arm_neighbor_count": int(same_arm_count),
            "different_arm_neighbor_count": int(different_arm_count),
            "distinct_neighbor_zone_count": int(len(distinct_neighbor_zones)),
            "active_exposure_zone_count": int(len(active_exposure_zones)),
            "neighbor_arms": ";".join(str(value) for value in neighbor_arms),
            "neighbor_zones": ";".join(str(value) for value in neighbor_zones),
            "interfering_request_ids": ";".join(interfering_request_ids),
        }

    def _scale_request_learning_reward(self, amount: float) -> float:
        """Return the request-level reward used by the selected learner input.

        request_fixed_scale implements r_u,t = min(y_u,t, B) / B before any
        Simple Average or HT-IX aggregation. The legacy path deliberately
        preserves existing experiments that scaled only after aggregation.
        """
        numeric_amount = float(amount)
        if getattr(self, "learning_reward_mode", "aggregate_legacy") != "request_fixed_scale":
            return numeric_amount
        upper_bound = float(self.reward_upper_bound)
        if not np.isfinite(upper_bound) or upper_bound <= 0.0:
            raise ValueError(
                "[MABSpatialPricing] request_fixed_scale requires a finite "
                "positive op_mab_reward_upper_bound."
            )
        return float(np.clip(numeric_amount, 0.0, upper_bound) / upper_bound)

    def _estimate_delayed_observed_feedback(
        self,
        sim_time: float,
        zone_system_name: Optional[str] = None,
    ) -> dict:
        estimator_name = self.estimator.__class__.__name__
        use_ht_style = estimator_name in {"HT", "HT_IX"}
        beta = float(getattr(self.estimator, "beta", 0.0)) if estimator_name == "HT_IX" else 0.0
        include_unobserved_as_zero = self._uses_event_matured_feedback()

        numerators = np.zeros(self.n_arms, dtype=float)
        contribution_square_sums = np.zeros(self.n_arms, dtype=float)
        reward_weights = np.zeros(self.n_arms, dtype=float)
        weight_square_sums = np.zeros(self.n_arms, dtype=float)
        weighted_reward_square_sums = np.zeros(self.n_arms, dtype=float)
        weighted_reward_weight_square_sums = np.zeros(self.n_arms, dtype=float)
        eligible_counts = np.zeros(self.n_arms, dtype=float)
        observed_counts = np.zeros(self.n_arms, dtype=float)
        zero_counts = np.zeros(self.n_arms, dtype=float)
        used_counts = np.zeros(self.n_arms, dtype=float)
        discarded_counts = np.zeros(self.n_arms, dtype=float)
        clean_counts = np.zeros(self.n_arms, dtype=float)
        pending_count = 0
        processed_count = 0
        kappa_values = [[] for _ in range(self.n_arms)]
        exposure_prob_values = [[] for _ in range(self.n_arms)]

        for rid_key, record in self.request_treatment_records.items():
            if rid_key in self._processed_delayed_feedback_rids:
                continue
            if zone_system_name is not None and record.get("zone_system_name") != zone_system_name:
                continue

            try:
                episode_start = float(record.get("pricing_episode_start"))
            except (TypeError, ValueError):
                self._processed_delayed_feedback_rids.add(rid_key)
                continue

            if include_unobserved_as_zero:
                maturity_time = (
                    episode_start
                    + float(self.mab_interval)
                    + float(self.request_reward_delay)
                )
                if float(sim_time) < maturity_time:
                    pending_count += 1
                    continue
                amount = float(self._request_outcomes.get(rid_key, 0.0))
                outcome_time = float(self._request_outcome_times.get(rid_key, maturity_time))
                outcome_type = self._request_outcome_types.get(rid_key, "matured_zero")
                default_reason = (
                    "no_accepted_quote_by_maturity"
                    if self.reward_event == "acceptance"
                    else "no_realized_boarding_by_maturity"
                )
                outcome_reason = self._request_outcome_reasons.get(rid_key, default_reason)
            else:
                if rid_key not in self._request_outcomes:
                    pending_count += 1
                    continue
                outcome_time = float(self._request_outcome_times.get(rid_key, sim_time))
                if outcome_time > float(sim_time):
                    pending_count += 1
                    continue
                amount = float(self._request_outcomes[rid_key])
                outcome_type = self._request_outcome_types.get(rid_key, "observed")
                outcome_reason = self._request_outcome_reasons.get(rid_key, "observed")

            learning_reward = self._scale_request_learning_reward(amount)

            try:
                arm = int(record.get("arm"))
            except (TypeError, ValueError):
                self._processed_delayed_feedback_rids.add(rid_key)
                continue
            if arm < 0 or arm >= self.n_arms:
                self._processed_delayed_feedback_rids.add(rid_key)
                continue

            eligible_counts[arm] += 1.0
            if rid_key in self._request_outcomes:
                observed_counts[arm] += 1.0
            if amount <= 0:
                zero_counts[arm] += 1.0
            details = self._clean_exposure_details(rid_key, record, zone_system_name)
            clean = bool(details.get("clean", False))
            arm_prob = record.get("arm_prob")
            try:
                arm_prob = float(arm_prob)
            except (TypeError, ValueError):
                arm_prob = np.nan
            if not np.isfinite(arm_prob) or arm_prob <= 0:
                arm_prob = float(self._assignment_probability_for_arm(arm))
            if not np.isfinite(arm_prob) or arm_prob <= 0:
                arm_prob = 1.0 / max(1, self.n_arms)

            if self.use_exposure_prob:
                kappa = max(1, int(details.get("active_exposure_zone_count", 1)))
                exposure_prob = arm_prob ** kappa
            else:
                kappa = 1
                exposure_prob = arm_prob
            kappa_values[arm].append(float(kappa))
            exposure_prob_values[arm].append(float(exposure_prob))

            used_by_estimator = True
            contribution = learning_reward
            weight = 1.0
            if use_ht_style:
                used_by_estimator = clean
                if used_by_estimator:
                    denom = max(float(exposure_prob) + beta, 1e-9)
                    weight = 1.0 / denom
                    contribution = learning_reward * weight
                    reward_weights[arm] += weight
                    clean_counts[arm] += 1.0
                else:
                    discarded_counts[arm] += 1.0
                    contribution = 0.0
            else:
                reward_weights[arm] += 1.0

            if used_by_estimator:
                numerators[arm] += contribution
                contribution_square_sums[arm] += contribution * contribution
                if use_ht_style:
                    weight_square_sums[arm] += weight * weight
                    weighted_reward_square_sums[arm] += (
                        weight * learning_reward
                    ) ** 2
                    weighted_reward_weight_square_sums[arm] += (
                        weight * weight * learning_reward
                    )
                used_counts[arm] += 1.0

            processed_count += 1
            self._processed_delayed_feedback_rids.add(rid_key)

            if self.htix_diagnostics_enabled:
                pricing_time = float(record.get("pricing_time", np.nan))
                diagnostic_row = {
                    "update_time": float(sim_time),
                    "request_id": rid_key,
                    "pricing_time": pricing_time,
                    "outcome_time": outcome_time,
                    "outcome_type": outcome_type,
                    "outcome_reason": outcome_reason,
                    "feedback_delay": outcome_time - pricing_time
                    if np.isfinite(pricing_time)
                    else np.nan,
                    "zone_system_name": record.get("zone_system_name"),
                    "estimator": estimator_name,
                    "amount": amount,
                    "learning_reward": learning_reward,
                    "learning_reward_mode": getattr(
                        self, "learning_reward_mode", "aggregate_legacy"
                    ),
                    "arm_prob": float(arm_prob),
                    "exposure_prob": float(exposure_prob),
                    "htix_beta": float(beta),
                    "used_by_estimator": bool(used_by_estimator),
                    "would_be_used_by_htix": bool(clean),
                    "discard_reason": details.get("reason"),
                }
                diagnostic_row.update(details)
                self._htix_diagnostics.append(diagnostic_row)

        if use_ht_style and self.htix_self_normalized:
            valid_arms = reward_weights > 0
            rewards_arr = np.divide(
                numerators,
                reward_weights,
                out=np.full(self.n_arms, np.nan, dtype=float),
                where=valid_arms,
            )
        elif use_ht_style:
            total_eligible = float(eligible_counts.sum())
            if total_eligible > 0:



                valid_arms = np.ones(self.n_arms, dtype=bool)
                rewards_arr = numerators / total_eligible
            else:
                valid_arms = np.zeros(self.n_arms, dtype=bool)
                rewards_arr = np.full(self.n_arms, np.nan, dtype=float)
        else:
            valid_arms = eligible_counts > 0
            rewards_arr = np.divide(
                numerators,
                eligible_counts,
                out=np.full(self.n_arms, np.nan, dtype=float),
                where=valid_arms,
            )

        standard_errors = np.full(self.n_arms, np.nan, dtype=float)
        if use_ht_style and self.htix_self_normalized:
            for arm in range(self.n_arms):
                weight_sum = float(reward_weights[arm])
                if weight_sum <= 0.0 or not np.isfinite(rewards_arr[arm]):
                    continue
                mean_reward = float(rewards_arr[arm])
                residual_square_sum = (
                    weighted_reward_square_sums[arm]
                    - 2.0 * mean_reward * weighted_reward_weight_square_sums[arm]
                    + mean_reward * mean_reward * weight_square_sums[arm]
                )
                standard_errors[arm] = np.sqrt(
                    max(float(residual_square_sum), 0.0)
                ) / weight_sum
        else:
            denominator_by_arm = (
                np.full(self.n_arms, float(eligible_counts.sum()), dtype=float)
                if use_ht_style
                else eligible_counts
            )
            for arm in range(self.n_arms):
                count = float(denominator_by_arm[arm])
                if count <= 1.0 or not np.isfinite(rewards_arr[arm]):
                    continue
                centered_square_sum = (
                    contribution_square_sums[arm]
                    - (numerators[arm] * numerators[arm]) / count
                )
                standard_errors[arm] = np.sqrt(
                    max(float(centered_square_sum), 0.0) / (count * (count - 1.0))
                )

        def _summary(values_by_arm, reducer):
            result = []
            for values in values_by_arm:
                result.append(float(reducer(values)) if values else np.nan)
            return result

        self._last_delayed_feedback_debug = {
            "feedback_mode": self.feedback_mode,
            "htix_self_normalized": bool(self.htix_self_normalized),
            "learning_reward_mode": getattr(
                self, "learning_reward_mode", "aggregate_legacy"
            ),
            "processed_feedback_count": int(processed_count),
            "pending_feedback_count": int(pending_count),
            "eligible_count": int(eligible_counts.sum()),
            "eligible_counts_by_arm": eligible_counts.astype(float).tolist(),
            "observed_counts_by_arm": observed_counts.astype(float).tolist(),
            "zero_counts_by_arm": zero_counts.astype(float).tolist(),
            "used_counts_by_arm": used_counts.astype(float).tolist(),
            "discarded_counts_by_arm": discarded_counts.astype(float).tolist(),
            "clean_counts_by_arm": clean_counts.astype(float).tolist(),
            "clean_rates_by_arm": np.divide(
                clean_counts,
                eligible_counts,
                out=np.zeros(self.n_arms, dtype=float),
                where=eligible_counts > 0,
            ).tolist(),
            "estimator_numerators_by_arm": numerators.astype(float).tolist(),
            "estimator_contribution_square_sums_by_arm": contribution_square_sums.astype(float).tolist(),
            "estimator_weight_sums_by_arm": reward_weights.astype(float).tolist(),
            "estimator_weight_square_sums_by_arm": weight_square_sums.astype(float).tolist(),
            "estimator_standard_errors_by_arm": standard_errors.astype(float).tolist(),
            "kappa_min_by_arm": _summary(kappa_values, min),
            "kappa_mean_by_arm": _summary(kappa_values, np.mean),
            "kappa_max_by_arm": _summary(kappa_values, max),
            "exposure_prob_min_by_arm": _summary(exposure_prob_values, min),
            "exposure_prob_mean_by_arm": _summary(exposure_prob_values, np.mean),
            "exposure_prob_max_by_arm": _summary(exposure_prob_values, max),
            "estimated_rewards_by_arm": rewards_arr.astype(float).tolist(),
            "valid_estimated_reward_count": int(np.sum(valid_arms)),
        }
        self._last_request_reward_debug = self._last_delayed_feedback_debug
        self._last_reward_standard_errors = {
            arm: float(standard_errors[arm])
            for arm in range(self.n_arms)
            if np.isfinite(standard_errors[arm])
        }
        return {arm: float(rewards_arr[arm]) for arm in range(self.n_arms) if bool(valid_arms[arm])}

    def _estimate_grid_beta_evidence(
        self,
        sim_time: float,
        zone_system_name: Optional[str] = None,
    ) -> dict:
        """Aggregate event-matured request outcomes into active grid evidence.

        An informative grid is a grid/period pair with at least one matured
        priced request.  Its reward is the mean of the fixed-scale request
        rewards, so each active grid contributes one fractional Beta trial.
        Empty grids are deliberately not counted as failures: they have no
        passenger outcome from which a pricing reward can be observed.
        """
        if not self._uses_event_matured_feedback():
            raise RuntimeError(
                "Grid-Beta evidence requires event-matured acceptance or boarding feedback."
            )

        grid_rewards: dict[tuple[float, int, int], list[float]] = {}
        pending_count = 0
        processed_count = 0
        clipped_count = 0
        lower = float(self.reward_lower_bound)
        scale = float(self.grid_beta_reward_upper_bound - lower)

        for rid_key, record in self.request_treatment_records.items():
            if rid_key in self._processed_grid_beta_feedback_rids:
                continue
            if zone_system_name is not None and record.get("zone_system_name") != zone_system_name:
                continue
            try:
                episode_start = float(record.get("pricing_episode_start"))
                zone_id = int(record.get("zone_id", -1))
                arm = int(record.get("arm", -1))
            except (TypeError, ValueError):
                self._processed_grid_beta_feedback_rids.add(rid_key)
                continue
            if zone_id < 0 or arm < 0 or arm >= self.n_arms:
                self._processed_grid_beta_feedback_rids.add(rid_key)
                continue

            maturity_time = (
                episode_start
                + float(self.mab_interval)
                + float(self.request_reward_delay)
            )
            if float(sim_time) < maturity_time:
                pending_count += 1
                continue

            amount = float(self._request_outcomes.get(rid_key, 0.0))
            scaled = (amount - lower) / scale
            if scaled < 0.0 or scaled > 1.0:
                clipped_count += 1
            scaled = float(np.clip(scaled, 0.0, 1.0))
            grid_rewards.setdefault((episode_start, zone_id, arm), []).append(scaled)
            self._processed_grid_beta_feedback_rids.add(rid_key)
            processed_count += 1

        reward_sums = np.zeros(self.n_arms, dtype=float)
        trial_counts = np.zeros(self.n_arms, dtype=float)
        request_counts = np.zeros(self.n_arms, dtype=float)
        for (_episode_start, _zone_id, arm), rewards in grid_rewards.items():
            reward_sums[arm] += float(np.mean(rewards))
            trial_counts[arm] += 1.0
            request_counts[arm] += float(len(rewards))

        mean_rewards = np.divide(
            reward_sums,
            trial_counts,
            out=np.full(self.n_arms, np.nan, dtype=float),
            where=trial_counts > 0,
        )
        self._last_reward_normalization_denominator = float(scale)
        self._last_grid_beta_debug = {
            "feedback_mode": self.feedback_mode,
            "grid_beta_reward_lower_bound": lower,
            "grid_beta_reward_upper_bound": float(self.grid_beta_reward_upper_bound),
            "grid_beta_reward_scale": scale,
            "processed_feedback_count": int(processed_count),
            "pending_feedback_count": int(pending_count),
            "clipped_request_reward_count": int(clipped_count),
            "informative_grid_count": int(trial_counts.sum()),
            "grid_trial_counts_by_arm": trial_counts.astype(float).tolist(),
            "grid_reward_sums_by_arm": reward_sums.astype(float).tolist(),
            "grid_mean_rewards_by_arm": mean_rewards.astype(float).tolist(),
            "matured_request_counts_by_arm": request_counts.astype(float).tolist(),
        }
        self._last_request_reward_debug = self._last_grid_beta_debug
        return {
            "reward_sums": reward_sums,
            "trial_counts": trial_counts,
            "mean_rewards": {
                arm: float(mean_rewards[arm])
                for arm in range(self.n_arms)
                if bool(trial_counts[arm] > 0)
            },
        }

    def _assignment_probability_for_arm(self, arm_idx: int) -> float:
        if arm_idx is None:
            return np.nan
        arm_idx = int(arm_idx)
        probs = getattr(self, "_period_probs", None)
        if probs is not None and 0 <= arm_idx < len(probs):
            return float(probs[arm_idx])
        return np.nan

    def _normalize_probability_vector(self, probs) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        if probs.shape != (self.n_arms,):
            raise ValueError(
                "[MABSpatialPricing] policy probability vector must have one entry per arm."
            )
        if np.any(~np.isfinite(probs)) or np.any(probs < 0):
            raise ValueError(
                "[MABSpatialPricing] policy probabilities must be finite and non-negative."
            )
        total = float(probs.sum())
        if total <= 0:
            raise ValueError("[MABSpatialPricing] policy probabilities must sum to a positive value.")
        return probs / total

    def _compute_period_policy(self, algo=None):
        algo = self.global_algo if algo is None else algo
        if self.fixed_arm is not None:
            probs = np.zeros(self.n_arms, dtype=float)
            probs[int(self.fixed_arm)] = 1.0
            debug_state = {"policy_mode": "fixed_arm", "fixed_arm": int(self.fixed_arm)}
        elif self.fixed_allocation_probs is not None:
            probs = self.fixed_allocation_probs.copy()
            debug_state = {
                "policy_mode": "fixed_allocation",
                "fixed_allocation_probs": probs.astype(float).tolist(),
            }
        elif self.assignment_level == "passenger" or self.random_assignment:
            probs = np.full(self.n_arms, 1.0 / max(1, self.n_arms), dtype=float)
            debug_state = {"policy_mode": "uniform_random"}
        else:
            probs = np.asarray(algo.get_probs(), dtype=float)
            debug_state = {}
            if hasattr(algo, "get_debug_state"):
                candidate = algo.get_debug_state()
                if isinstance(candidate, dict):
                    debug_state = candidate
            debug_state = dict(debug_state)
            debug_state["policy_mode"] = "adaptive"
        debug_state = self._annotate_phase_reset_debug(debug_state)
        return self._normalize_probability_vector(probs), debug_state

    def _refresh_period_policy(self, algo=None):
        probs, debug_state = self._compute_period_policy(algo=algo)
        self._period_probs = probs
        self._period_algo_debug_state = debug_state
        return probs

    def _normalize_learning_rewards(
        self, estimated_rewards: dict, record_denominator: bool = True
    ) -> dict:
        finite_rewards = {}
        for arm, reward in estimated_rewards.items():
            try:
                numeric_reward = float(reward)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_reward):
                finite_rewards[int(arm)] = numeric_reward

        mode = getattr(self, "reward_normalization_mode", "fixed")
        if mode == "period_max":
            if not finite_rewards:
                if record_denominator:
                    self._last_reward_normalization_denominator = np.nan
                return {}
            denominator = float(max(finite_rewards.values()))
            if record_denominator:
                self._last_reward_normalization_denominator = denominator
            if denominator <= 0.0:
                return {arm: 0.0 for arm in finite_rewards}
            return {
                arm: float(np.clip(reward / denominator, 0.0, 1.0))
                for arm, reward in finite_rewards.items()
            }

        scale = self.reward_upper_bound - self.reward_lower_bound
        if record_denominator:
            self._last_reward_normalization_denominator = float(scale)
        return {
            arm: float(
                np.clip((reward - self.reward_lower_bound) / scale, 0.0, 1.0)
            )
            for arm, reward in finite_rewards.items()
        }

    def _prepare_learning_rewards(self, estimated_rewards: dict) -> dict:
        """Apply only the learner-input transformation selected by the scenario."""
        if getattr(self, "learning_reward_mode", "aggregate_legacy") != "request_fixed_scale":
            return self._normalize_learning_rewards(estimated_rewards)

        finite_rewards = {}
        for arm, reward in estimated_rewards.items():
            try:
                numeric_reward = float(reward)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_reward):
                finite_rewards[int(arm)] = numeric_reward



        self._last_reward_normalization_denominator = float(self.reward_upper_bound)
        return finite_rewards

    def _prepare_hedge_exp3_rewards(self, estimated_rewards: dict) -> Optional[dict]:
        """Build an opt-in, separately scaled reward vector for Hedge's EXP3 expert."""
        upper_bound = getattr(self, "_hedge_exp3_reward_upper_bound", None)
        if upper_bound is None:
            return None
        lower_bound = float(self.reward_lower_bound)
        scale = float(upper_bound) - lower_bound
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "[MABSpatialPricing] Invalid separate Hedge EXP3 reward scale."
            )
        normalized = {}
        for arm, reward in estimated_rewards.items():
            try:
                numeric_reward = float(reward)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_reward):
                normalized[int(arm)] = float(
                    np.clip((numeric_reward - lower_bound) / scale, 0.0, 1.0)
                )
        return normalized

    def _prepare_learning_reward_standard_errors(self) -> dict:
        """Put the current request-level standard errors on the learner's scale.

        The uncertainty is computed before reward normalization from the same
        matured request population used for the corresponding arm estimate.
        It is only consumed by the risk-aware Hedge meta-policy; the TS and
        EXP3 expert updates continue to receive the unchanged reward vector.
        """
        raw_standard_errors = getattr(self, "_last_reward_standard_errors", {})
        if not raw_standard_errors:
            return {}

        if getattr(self, "learning_reward_mode", "aggregate_legacy") == "request_fixed_scale":
            denominator = 1.0
        else:
            denominator = float(self._last_reward_normalization_denominator)
        if not np.isfinite(denominator) or denominator <= 0.0:
            return {}

        prepared = {}
        for arm, raw_standard_error in raw_standard_errors.items():
            try:
                numeric_standard_error = float(raw_standard_error)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric_standard_error) or numeric_standard_error < 0.0:
                continue
            prepared[int(arm)] = float(
                np.clip(numeric_standard_error / denominator, 0.0, 1.0)
            )
        return prepared

    def _update_global_algorithm(
        self,
        normalized_rewards: dict,
        arm_standard_errors: Optional[dict] = None,
        hedge_exp3_rewards: Optional[dict] = None,
    ) -> None:
        """Route online uncertainty only to learners that explicitly use it."""
        if hedge_exp3_rewards is not None:
            update_experts = getattr(self.global_algo, "update_experts", None)
            if not callable(update_experts):
                raise RuntimeError(
                    "[MABSpatialPricing] Separate Hedge EXP3 rewards require "
                    "an algorithm with update_experts()."
                )
            update_experts(normalized_rewards, hedge_exp3_rewards, normalized_rewards)
            return
        if bool(getattr(self.global_algo, "uses_reward_uncertainty", False)):
            self.global_algo.update_all(
                normalized_rewards,
                arm_standard_errors=arm_standard_errors or {},
            )
            return
        self.global_algo.update_all(normalized_rewards)

    def _get_algo_debug_state(self) -> dict:
        try:
            if hasattr(self.global_algo, "get_debug_state"):
                debug_state = self.global_algo.get_debug_state()
                if isinstance(debug_state, dict):
                    return self._annotate_phase_reset_debug(debug_state)
        except Exception as exc:
            LOG.warning("[MABSpatialPricing] Could not read algo debug state: %s", exc)
        return self._annotate_phase_reset_debug({})

    def _annotate_phase_reset_debug(self, debug_state: dict) -> dict:
        """Attach opt-in reset provenance to per-period diagnostic records."""
        annotated = dict(debug_state)
        last_reset_time = getattr(self, "_last_phase_reset_time", None)
        if last_reset_time is not None:
            annotated["phase_reset_time"] = int(last_reset_time)
        annotated["phase_reset_count"] = int(
            getattr(self, "_phase_reset_count", 0)
        )
        return annotated

    def _record_learning_update_debug(
        self,
        estimated_rewards: dict,
        normalized_rewards: dict,
        skipped_update_reason: Optional[str] = None,
    ):
        if not self.mab_log:
            return
        update_debug_state = self._get_algo_debug_state()
        normalization_denominator = (
            float(self._last_reward_normalization_denominator)
            if normalized_rewards
            else np.nan
        )
        self.mab_log[-1].update(
            {
                "estimated_rewards": [
                    float(estimated_rewards.get(arm, np.nan)) for arm in range(self.n_arms)
                ],
                "normalized_rewards": [
                    float(normalized_rewards.get(arm, np.nan)) for arm in range(self.n_arms)
                ],
                "post_update_algo_debug_state": update_debug_state,
                "post_update_hedge_meta_weights": update_debug_state.get("meta_weights"),
                "post_update_hedge_expert_trust": update_debug_state.get("expert_trust"),
                "hedge_last_meta_rewards": update_debug_state.get("last_meta_rewards"),
                "hedge_last_meta_scores": update_debug_state.get("last_meta_scores"),
                "hedge_last_meta_standard_errors": update_debug_state.get(
                    "last_meta_standard_errors"
                ),
                "hedge_last_arm_standard_errors": update_debug_state.get(
                    "last_arm_standard_errors"
                ),
                "hedge_risk_lambda": update_debug_state.get("hedge_risk_lambda"),
                "hedge_last_reward_vec": update_debug_state.get("last_reward_vec"),
                "hedge_last_ts_reward_vec": update_debug_state.get("last_ts_reward_vec"),
                "hedge_last_exp3_reward_vec": update_debug_state.get("last_exp3_reward_vec"),
                "post_update_hedge_fixed_share_alpha": update_debug_state.get("fixed_share_alpha"),
                "request_reward_debug": self._last_request_reward_debug,
                "delayed_feedback_debug": self._last_delayed_feedback_debug,
                "grid_beta_feedback_debug": self._last_grid_beta_debug,
                "skipped_update_reason": skipped_update_reason,
                "reward_normalization_mode": self.reward_normalization_mode,
                "learning_reward_mode": getattr(
                    self, "learning_reward_mode", "aggregate_legacy"
                ),
                "post_estimator_reward_clipping": (
                    getattr(self, "learning_reward_mode", "aggregate_legacy")
                    != "request_fixed_scale"
                ),
                "reward_normalization_denominator": normalization_denominator,
                "reward_normalization_lower_bound": float(self.reward_lower_bound),
                "reward_normalization_upper_bound": float(self.reward_upper_bound),
            }
        )
        self._save_log()

    def _start_next_period_assignment(self) -> None:
        """Construct the next period's assignment under the selected audit mode."""
        self._maybe_reset_for_phase()
        if self.period_policy_mode == "cached":
            self._refresh_period_policy()
        else:
            self._period_probs = None
            self._period_algo_debug_state = {"policy_mode": "legacy_resample"}
        if self.assignment_level != "zone":
            return
        if self.random_assignment and self.fixed_arm is None and self.fixed_allocation_probs is None:
            self.current_arm = self._balanced_random_assignment()
        else:
            self.current_arm = self._per_zone_sample()

    def _is_non_learning_policy(self) -> bool:
        return self.fixed_arm is not None or self.fixed_allocation_probs is not None

    def _maybe_reset_for_phase(self) -> None:
        """Reset explicitly configured EXP3 diagnostics at known phase boundaries."""
        reset_times = getattr(self, "_phase_reset_times", ())
        episode_start_time = getattr(self, "episode_start_time", None)
        if not reset_times or episode_start_time is None:
            return
        reset_time = int(round(float(episode_start_time)))
        if (
            abs(float(episode_start_time) - reset_time) > 1e-6
            or reset_time not in reset_times
            or reset_time in self._applied_phase_reset_times
        ):
            return
        self.global_algo.reset()
        self.zone_algos = [self.global_algo] * self.n_zones
        self._period_probs = None
        self._applied_phase_reset_times.add(reset_time)
        self._phase_reset_count += 1
        self._last_phase_reset_time = reset_time
        self._period_algo_debug_state = self._annotate_phase_reset_debug(
            {"phase_reset_applied": True}
        )
        LOG.info(
            "[MABSpatialPricing] Reset EXP3 at pre-specified phase boundary t=%s.",
            reset_time,
        )

    def _new_algo(self, seed: int):
        return _load_mab_algo(
            self._algo_name,
            self.n_arms,
            seed,
            self._T_episodes,
            eta=self._eta_val,
            exp3_eta=self._exp3_eta_val,
            exp3_exploration_epsilon=self._exp3_exploration_epsilon_val,
            exp3_discount_factor=self._exp3_discount_factor_val,
            exp3_fixed_share_alpha=self._exp3_fixed_share_alpha_val,
            ts_mc_samples=self._ts_mc_samples,
            hedge_eta=self._hedge_eta_val,
            hedge_gamma=self._hedge_gamma_val,
            hedge_risk_lambda=self._hedge_risk_lambda_val,
            fixed_share_alpha=self._fixed_share_alpha_val,
            discount_factor=self._discount_factor_val,
            etc_exploration_updates=self._etc_exploration_updates,
        )

    def _parse_grid_schedule(self, operator_attributes):
        raw_schedule = operator_attributes.get("op_mab_grid_schedule")
        if not raw_schedule:
            return []
        if self.random_assignment or self.assignment_level != "zone" or self.fixed_arm is not None:
            raise ValueError(
                "[MABSpatialPricing] op_mab_grid_schedule currently supports only zone-level adaptive MAB runs."
            )
        schedule = []
        for idx, entry in enumerate(raw_schedule):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"[MABSpatialPricing] Invalid op_mab_grid_schedule entry at index {idx}: {entry}"
                )
            zone_system = entry.get("zone_system") or entry.get("op_zone_system")
            if not zone_system:
                raise ValueError(
                    "[MABSpatialPricing] Each op_mab_grid_schedule entry requires 'zone_system'."
                )
            start_time = float(entry.get("start_time", 0))
            end_time = entry.get("end_time")
            if end_time is not None:
                end_time = float(end_time)
            schedule.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "zone_system": str(zone_system),
                }
            )
        schedule.sort(key=lambda item: item["start_time"])
        return schedule

    def _zone_dir_for_system(self, zone_system_name: str) -> str:
        network_name = os.path.basename(self.fleetctrl.dir_names.get(G_DIR_NETWORK, ""))
        if not network_name:
            raise ValueError("[MABSpatialPricing] Could not infer network name for dynamic grid zone loading.")
        zone_dir = os.path.join(
            self.fleetctrl.dir_names[G_DIR_DATA],
            "zones",
            zone_system_name,
            network_name,
        )
        if not os.path.isdir(zone_dir):
            raise FileNotFoundError(
                f"[MABSpatialPricing] Zone directory not found for MAB pricing grid: {zone_dir}"
            )
        return zone_dir

    def _load_zone_mapping_from_dir(self, zone_dir: str):
        node_zone_path = os.path.join(zone_dir, "node_zone_info.csv")
        if not os.path.exists(node_zone_path):
            raise FileNotFoundError(f"[MABSpatialPricing] node_zone_info.csv not found: {node_zone_path}")

        zone_df = pd.read_csv(node_zone_path, index_col=0)
        node_to_zone = zone_df["zone_id"].to_dict()
        n_zones = int(zone_df["zone_id"].max()) + 1
        print(f"[MABSpatialPricing] Loaded {n_zones} zones from {node_zone_path}")
        return node_to_zone, n_zones, node_zone_path

    def _build_node_level_context(self, node_zone_path: str, interference_radius: float):
        node_xy = None
        kdtree = None
        node_zone_by_node_id = None
        try:
            network_dir = self.fleetctrl.dir_names.get(G_DIR_NETWORK, None)
            if network_dir is None:
                raise ValueError("G_DIR_NETWORK not found")

            nodes_path = os.path.join(network_dir, "base", "nodes.csv")
            nodes_df = pd.read_csv(nodes_path)[["node_index", "pos_x", "pos_y"]]
            node_zone_df = pd.read_csv(node_zone_path)[["node_index", "zone_id"]]
            merged = nodes_df.merge(node_zone_df, on="node_index", how="left")
            merged["zone_id"] = merged["zone_id"].fillna(-1).astype(int)

            max_idx = int(merged["node_index"].max())
            node_xy = np.full((max_idx + 1, 2), np.nan)
            node_zone_arr = np.full(max_idx + 1, -1, dtype=int)

            for row in merged.itertuples(index=False):
                idx = int(row.node_index)
                node_xy[idx] = [row.pos_x, row.pos_y]
                node_zone_arr[idx] = int(row.zone_id)

            valid_mask = ~np.isnan(node_xy[:, 0])
            valid_node_ids = np.where(valid_mask)[0]
            kdtree = KDTree(node_xy[valid_node_ids])
            node_zone_by_node_id = node_zone_arr
            LOG.info(
                "[MABSpatialPricing] Loaded %s valid nodes for node-level exposure, radius=%sm",
                len(valid_node_ids),
                interference_radius,
            )
            print(
                f"[MABSpatialPricing] Loaded {len(valid_node_ids)} valid nodes for exposure, "
                f"r={interference_radius}m"
            )
        except Exception as exc:
            LOG.warning("[MABSpatialPricing] Could not build node-level exposure mapping: %s", exc)
            print(f"[MABSpatialPricing] Could not build node-level exposure mapping: {exc}")
        return node_xy, kdtree, node_zone_by_node_id

    def _compute_zone_neighbors_for_dir(self, zone_dir: str) -> dict:
        try:
            geojson_path = os.path.join(os.path.dirname(zone_dir), "polygon_definition.geojson")
            if not os.path.exists(geojson_path):
                return {}

            with open(geojson_path, encoding="utf-8") as handle:
                data = json.load(handle)

            xy_to_zone = {}
            zone_to_xy = {}
            for feature in data["features"]:
                zone_id = int(feature["properties"]["zone_id"])
                x_bin, y_bin = map(int, feature["properties"]["zone_name"].split(";"))
                xy_to_zone[(x_bin, y_bin)] = zone_id
                zone_to_xy[zone_id] = (x_bin, y_bin)

            neighbors = {}
            for zone_id, (x_bin, y_bin) in zone_to_xy.items():
                neighbor_ids = []
                for delta_x, delta_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    candidate = (x_bin + delta_x, y_bin + delta_y)
                    if candidate in xy_to_zone:
                        neighbor_ids.append(xy_to_zone[candidate])
                neighbors[zone_id] = neighbor_ids
            return neighbors
        except Exception as exc:
            LOG.warning("[MABSpatialPricing] Could not compute zone neighbors: %s", exc)
            return {}

    def _create_pricing_state(self, zone_system_name: str, state_seed: int) -> dict:
        zone_dir = self._zone_dir_for_system(zone_system_name)
        node_to_zone, n_zones, node_zone_path = self._load_zone_mapping_from_dir(zone_dir)
        grid_edge_m = _infer_grid_edge_meters(zone_system_name)
        interference_radius = _resolve_interference_radius(
            grid_edge_m,
            self._configured_interference_radius_override,
        )
        node_xy, kdtree, node_zone_by_node_id = self._build_node_level_context(
            node_zone_path,
            interference_radius,
        )
        global_algo = self._new_algo(state_seed)
        period_probs, period_debug_state = self._compute_period_policy(algo=global_algo)
        state = {
            "zone_system_name": zone_system_name,
            "grid_edge_m": grid_edge_m,
            "interference_radius": interference_radius,
            "node_to_zone": node_to_zone,
            "n_zones": n_zones,
            "node_xy": node_xy,
            "kdtree": kdtree,
            "node_zone_by_node_id": node_zone_by_node_id,
            "zone_neighbors": self._compute_zone_neighbors_for_dir(zone_dir),
            "global_algo": global_algo,
            "episode_count": 0,
            "ucb_counts": np.zeros((n_zones, self.n_arms), dtype=float),
            "ucb_rewards": np.zeros((n_zones, self.n_arms), dtype=float),
            "zone_pulls": np.zeros(n_zones, dtype=int),
            "zone_revenue": np.zeros(n_zones, dtype=float),
            "zone_episode_revenue": np.zeros(n_zones, dtype=float),
            "episode_start_time": None,
            "fare_records": [],
            "period_probs": period_probs,
            "period_algo_debug_state": period_debug_state,
        }
        if self.random_assignment:
            state["current_arm"] = self._balanced_random_assignment()
        else:
            if self.fixed_arm is not None:
                state["current_arm"] = np.full(n_zones, int(self.fixed_arm), dtype=int)
            elif getattr(global_algo, "gridwise_sampling", False):
                state["current_arm"] = np.array(
                    [global_algo.sample_arm() for _ in range(n_zones)],
                    dtype=int,
                )
            else:
                state["current_arm"] = self.rng.choice(
                    self.n_arms,
                    size=n_zones,
                    p=period_probs,
                )
        return state

    def _current_schedule_entry(self, sim_time: float):
        for entry in self._grid_schedule:
            if self._schedule_entry_matches(sim_time, entry):
                return entry
        return None

    def _set_active_state_refs(self, state: dict):
        self.node_to_zone = state["node_to_zone"]
        self.n_zones = state["n_zones"]
        self._node_xy = state["node_xy"]
        self._kdtree = state["kdtree"]
        self._node_zone_by_node_id = state["node_zone_by_node_id"]
        self.zone_neighbors = state["zone_neighbors"]
        self.global_algo = state["global_algo"]
        self._episode_count = state["episode_count"]
        self.ucb_counts = state["ucb_counts"]
        self.ucb_rewards = state["ucb_rewards"]
        self.zone_pulls = state["zone_pulls"]
        self.zone_revenue = state["zone_revenue"]
        self.zone_episode_revenue = state["zone_episode_revenue"]
        self.episode_start_time = state["episode_start_time"]
        self.current_arm = state["current_arm"]
        self._fare_records = state["fare_records"]
        self._period_probs = state["period_probs"]
        self._period_algo_debug_state = state["period_algo_debug_state"]
        self._grid_edge_m = state["grid_edge_m"]
        self._interference_radius = state["interference_radius"]
        self._zone_system_name = state["zone_system_name"]

    def _sync_active_state_refs(self, state: dict):
        state["episode_count"] = self._episode_count
        state["episode_start_time"] = self.episode_start_time
        state["current_arm"] = self.current_arm
        state["fare_records"] = self._fare_records
        state["period_probs"] = self._period_probs
        state["period_algo_debug_state"] = self._period_algo_debug_state

    def _activate_dynamic_state(self, sim_time: float):
        entry = self._current_schedule_entry(sim_time)
        if entry is None:
            raise ValueError(f"[MABSpatialPricing] No grid schedule entry covers sim_time={sim_time}")
        state_key = entry["zone_system"]
        if self._active_state_key == state_key:
            return self._pricing_states[state_key]

        if self._active_state_key is not None:
            old_state = self._pricing_states[self._active_state_key]
            if old_state["episode_start_time"] is not None and sim_time - old_state["episode_start_time"] >= self.mab_interval:
                self._update_active_state_episode(old_state, sim_time)
            else:
                self._set_active_state_refs(old_state)
                self._sync_active_state_refs(old_state)

        new_state = self._pricing_states[state_key]
        self._active_state_key = state_key
        self._set_active_state_refs(new_state)
        if self.episode_start_time is None or sim_time - self.episode_start_time >= self.mab_interval:
            self.episode_start_time = sim_time
            self.zone_episode_revenue = self.zone_revenue.copy()
            self._sync_active_state_refs(new_state)
        return new_state

    def _init_dynamic_grid_states(self, sim_start: int):
        unique_zone_systems = []
        for entry in self._grid_schedule:
            if entry["zone_system"] not in unique_zone_systems:
                unique_zone_systems.append(entry["zone_system"])
        for idx, zone_system_name in enumerate(unique_zone_systems):
            self._pricing_states[zone_system_name] = self._create_pricing_state(
                zone_system_name,
                self._seed + 1000 * (idx + 1),
            )
        active_state = self._activate_dynamic_state(sim_start)
        self.zone_algos = [self.global_algo] * self.n_zones
        LOG.info(
            "[MABSpatialPricing] Dynamic grid states initialized: %s",
            list(self._pricing_states.keys()),
        )





    def _load_zone_mapping(self):
        zone_dir = self._zone_dir_for_system(self._zone_system_name)
        return self._load_zone_mapping_from_dir(zone_dir)

    def _load_network_nodes(self, node_zone_path: str):
        self._node_xy, self._kdtree, self._node_zone_by_node_id = self._build_node_level_context(
            node_zone_path,
            self._interference_radius,
        )





    def _compute_zone_neighbors(self) -> dict:
        zone_dir = self._zone_dir_for_system(self._zone_system_name)
        return self._compute_zone_neighbors_for_dir(zone_dir)

    def _compute_checkerboard(self):
        try:
            zone_dir = self._zone_dir_for_system(self._zone_system_name)
            geojson_path = os.path.join(os.path.dirname(zone_dir), "polygon_definition.geojson")
            if not os.path.exists(geojson_path):
                raise FileNotFoundError(geojson_path)

            with open(geojson_path, encoding="utf-8") as handle:
                data = json.load(handle)

            base = np.zeros(self.n_zones, dtype=int)
            for feature in data["features"]:
                zone_id = int(feature["properties"]["zone_id"])
                x_bin, y_bin = map(int, feature["properties"]["zone_name"].split(";"))
                if zone_id < self.n_zones:
                    base[zone_id] = (x_bin + y_bin) % 2
            return base
        except Exception as exc:
            LOG.warning("[MABSpatialPricing] Checkerboard construction failed (%s), using shuffled fallback", exc)
            base = np.zeros(self.n_zones, dtype=int)
            base[self.n_zones // 2 :] = 1
            self.rng.shuffle(base)
            return base

    def _balanced_random_assignment(self):
        if self._checkerboard_base is None:
            self._checkerboard_base = self._compute_checkerboard()
            self._checkerboard_phase = 0
        else:
            self._checkerboard_phase = 1 - self._checkerboard_phase
        return ((self._checkerboard_base + self._checkerboard_phase) % 2).copy()

    def _per_zone_sample(self) -> np.ndarray:
        if self.fixed_arm is not None:
            return np.full(self.n_zones, int(self.fixed_arm), dtype=int)
        if getattr(self.global_algo, "gridwise_sampling", False):
            return np.array(
                [self.global_algo.sample_arm() for _ in range(self.n_zones)],
                dtype=int,
            )
        if self.period_policy_mode == "legacy_resample":
            return np.array(
                [self.global_algo.sample_arm() for _ in range(self.n_zones)],
                dtype=int,
            )
        if self._period_probs is None:
            self._refresh_period_policy()
        return self.rng.choice(
            self.n_arms,
            size=self.n_zones,
            p=np.asarray(self._period_probs, dtype=float),
        )

    def _passenger_arm(self, rid) -> int:
        if self.fixed_arm is not None:
            return int(self.fixed_arm)
        if rid is None:
            raise ValueError("[MABSpatialPricing] passenger-level assignment requires rid")
        rid_key = str(rid)
        if rid_key not in self._rid_to_arm:
            if self._period_probs is None:
                self._refresh_period_policy()
            self._rid_to_arm[rid_key] = int(
                self.rng.choice(
                    self.n_arms,
                    p=np.asarray(self._period_probs, dtype=float),
                )
            )
        return self._rid_to_arm[rid_key]





    def _record_fare_to_state(
        self,
        state: dict,
        amount: float,
        zone_id: int,
        node_id: Optional[int],
        rid=None,
        arm_idx: Optional[int] = None,
        pricing_time: Optional[float] = None,
        arm_prob: Optional[float] = None,
    ):
        if zone_id >= 0 and zone_id < state["n_zones"]:
            state["zone_revenue"][zone_id] += float(amount)
            if state["kdtree"] is not None:
                if self.assignment_level == "zone" and node_id is not None:
                    record_arm = arm_idx
                    if record_arm is None:
                        current_arms = state.get("current_arm")
                        if current_arms is not None and zone_id < len(current_arms):
                            record_arm = int(current_arms[zone_id])
                    if record_arm is None:
                        state["fare_records"].append((node_id, float(amount)))
                    else:
                        record_pricing_time = np.nan if pricing_time is None else float(pricing_time)
                        record_arm_prob = np.nan if arm_prob is None else float(arm_prob)
                        state["fare_records"].append(
                            (
                                node_id,
                                float(amount),
                                int(record_arm),
                                int(zone_id),
                                record_pricing_time,
                                record_arm_prob,
                            )
                        )
                elif self.assignment_level == "passenger" and node_id is not None:
                    arm = arm_idx
                    if arm is None:
                        arm = self._rid_to_arm.get(str(rid)) if rid is not None else None
                    if arm is not None:
                        record_pricing_time = np.nan if pricing_time is None else float(pricing_time)
                        record_arm_prob = np.nan if arm_prob is None else float(arm_prob)
                        state["fare_records"].append(
                            [
                                str(rid),
                                node_id,
                                float(amount),
                                int(arm),
                                int(zone_id),
                                record_pricing_time,
                                record_arm_prob,
                            ]
                        )
            return
        state["zone_revenue"][0] += float(amount)

    def record_fare_at_offer(self) -> bool:
        return self.reward_event == "offer" and not self._uses_request_reward_target()

    def record_fare_at_boarding(self) -> bool:
        return self.reward_event == "boarding"

    def _update_active_state_episode(self, state: dict, sim_time: float):
        self._set_active_state_refs(state)
        if self.episode_start_time is None:
            self.episode_start_time = sim_time
            self.zone_episode_revenue = self.zone_revenue.copy()
            self._sync_active_state_refs(state)
            return

        if sim_time - self.episode_start_time < self.mab_interval:
            self._sync_active_state_refs(state)
            return

        episode_rewards = self.zone_revenue - self.zone_episode_revenue
        if self.period_policy_mode == "legacy_resample":
            current_probs = np.asarray(self.global_algo.get_probs(), dtype=float).tolist()
            algo_debug_state = self._get_algo_debug_state()
        else:
            current_probs = np.asarray(self._period_probs, dtype=float).tolist()
            algo_debug_state = dict(self._period_algo_debug_state)

        self.mab_log.append(
            {
                "sim_time_start": self.episode_start_time,
                "sim_time_end": sim_time,
                "zone_system_name": state["zone_system_name"],
                "grid_edge_m": state["grid_edge_m"],
                "interference_radius": state["interference_radius"],
                "zone_arms": self.current_arm.tolist(),
                "zone_rewards": episode_rewards.tolist(),
                "ucb_counts": self.ucb_counts.tolist(),
                "ucb_rewards": self.ucb_rewards.tolist(),
                "assignment_level": self.assignment_level,
                "arms": self.arms,
                "fixed_arm": self.fixed_arm,
                "fixed_allocation_probs": None
                if self.fixed_allocation_probs is None
                else self.fixed_allocation_probs.astype(float).tolist(),
                "use_exposure_prob": self.use_exposure_prob,
                "period_policy_mode": self.period_policy_mode,
                "htix_self_normalized": self.htix_self_normalized,
                "reward_event": self.reward_event,
                "reward_target": self.reward_target,
                "feedback_mode": self.feedback_mode,
                "request_reward_delay": self.request_reward_delay,
                "reward_normalization_mode": self.reward_normalization_mode,
                "reward_normalization_denominator": np.nan,
                "reward_normalization_lower_bound": float(self.reward_lower_bound),
                "reward_normalization_upper_bound": float(self.reward_upper_bound),
                "passenger_records": self._fare_records if self.assignment_level == "passenger" else [],
                "episode": self._episode_count,
                "arm_probs": current_probs,
                "algo_debug_state": algo_debug_state,
                "hedge_meta_weights": algo_debug_state.get("meta_weights"),
                "hedge_expert_trust": algo_debug_state.get("expert_trust"),
                "hedge_ts_probs": algo_debug_state.get("ts_probs"),
                "hedge_exp3_probs": algo_debug_state.get("exp3_probs"),
                "hedge_eta": algo_debug_state.get("hedge_eta"),
                "hedge_gamma": algo_debug_state.get("hedge_gamma"),
                "hedge_fixed_share_alpha": algo_debug_state.get("fixed_share_alpha"),
            }
        )
        self._save_log()

        self.episode_start_time = sim_time
        self.zone_episode_revenue = self.zone_revenue.copy()

        if self._episode_count == 0:
            self._fare_records = []
            self._record_learning_update_debug(
                {},
                {},
                skipped_update_reason="warm_start_no_matured_feedback",
            )
            self._episode_count += 1
            self._start_next_period_assignment()
            self._sync_active_state_refs(state)
            LOG.debug(
                "[MABSpatialPricing] ep0 skipped update to avoid warm-start bias (%s)",
                state["zone_system_name"],
            )
            return

        if self.random_assignment and not self._is_non_learning_policy():
            self._fare_records = []
            self._episode_count += 1
            self._start_next_period_assignment()
            self._sync_active_state_refs(state)
            return

        arm_probs = (
            np.asarray(self.global_algo.get_probs(), dtype=float)
            if self.period_policy_mode == "legacy_resample"
            else np.asarray(self._period_probs, dtype=float)
        )
        skip_update_reason = None
        grid_beta_evidence = None
        companion_normalized_rewards = None
        if self._uses_grid_beta_evidence():
            grid_beta_evidence = self._estimate_grid_beta_evidence(
                sim_time,
                zone_system_name=state["zone_system_name"],
            )
            if self._uses_grid_beta_hedge():
                companion_rewards, companion_normalized_rewards = (
                    self._estimate_grid_beta_hedge_companion_feedback(
                        sim_time,
                        zone_system_name=state["zone_system_name"],
                    )
                )
                estimated_rewards = (
                    companion_rewards
                    if companion_normalized_rewards
                    else grid_beta_evidence["mean_rewards"]
                )
            else:
                estimated_rewards = grid_beta_evidence["mean_rewards"]
            if (
                self._last_grid_beta_debug.get("processed_feedback_count", 0) == 0
                and not companion_normalized_rewards
            ):
                skip_update_reason = "no_matured_grid_beta_feedback"
            elif (
                self._last_grid_beta_debug.get("informative_grid_count", 0) == 0
                and not companion_normalized_rewards
            ):
                skip_update_reason = "no_informative_grid_beta_feedback"
        elif self._uses_delayed_observed_feedback() or self._uses_event_matured_feedback():
            estimated_rewards = self._estimate_delayed_observed_feedback(
                sim_time,
                zone_system_name=state["zone_system_name"],
            )
            if self._last_delayed_feedback_debug.get("processed_feedback_count", 0) == 0:
                skip_update_reason = "no_observed_delayed_feedback"
            elif self._last_delayed_feedback_debug.get("valid_estimated_reward_count", 1) == 0:
                skip_update_reason = "no_valid_htix_feedback"
        elif self._uses_request_reward_target():
            estimated_rewards = self._estimate_request_reward_target(
                sim_time,
                zone_system_name=state["zone_system_name"],
            )
        else:
            estimated_rewards = self.estimator.estimate(
                zone_arms=self.current_arm,
                zone_revenues=episode_rewards,
                arm_probs=arm_probs,
                zone_neighbors=self.zone_neighbors if self.zone_neighbors else None,
                fare_records=self._fare_records if self._kdtree is not None else None,
                node_xy=self._node_xy,
                node_zone=self._node_zone_by_node_id,
                kdtree=self._kdtree,
                interference_radius=self._interference_radius,
                use_exposure_prob=self.use_exposure_prob,
                self_normalized=self.htix_self_normalized,
            )

        self._fare_records = []
        if skip_update_reason is not None:
            self._record_learning_update_debug({}, {}, skipped_update_reason=skip_update_reason)
            self._episode_count += 1
            self._start_next_period_assignment()
            self._sync_active_state_refs(state)
            return

        if grid_beta_evidence is not None:
            normalized_rewards = (
                companion_normalized_rewards
                if companion_normalized_rewards
                else estimated_rewards
            )
            arm_standard_errors = {}
            hedge_exp3_rewards = None
        else:
            normalized_rewards = self._prepare_learning_rewards(estimated_rewards)
            arm_standard_errors = self._prepare_learning_reward_standard_errors()
            hedge_exp3_rewards = self._prepare_hedge_exp3_rewards(estimated_rewards)
        if self._is_non_learning_policy():
            self._record_learning_update_debug(
                estimated_rewards,
                normalized_rewards,
                skipped_update_reason="fixed_policy_no_learning",
            )
        else:
            if grid_beta_evidence is not None:
                if self._uses_grid_beta_hedge():
                    self.global_algo.update_grid_beta_experts(
                        grid_beta_evidence["reward_sums"],
                        grid_beta_evidence["trial_counts"],
                        exp3_rewards=companion_normalized_rewards or None,
                        meta_rewards=companion_normalized_rewards or None,
                    )
                else:
                    self.global_algo.update_grid_evidence(
                        grid_beta_evidence["reward_sums"],
                        grid_beta_evidence["trial_counts"],
                    )
            else:
                self._update_global_algorithm(
                    normalized_rewards,
                    arm_standard_errors,
                    hedge_exp3_rewards=hedge_exp3_rewards,
                )
            self._record_learning_update_debug(estimated_rewards, normalized_rewards)

        for zone_id in range(self.n_zones):
            arm_idx = int(self.current_arm[zone_id])
            self.ucb_counts[zone_id, arm_idx] += 1.0
            self.ucb_rewards[zone_id, arm_idx] += float(episode_rewards[zone_id])
        self.zone_pulls += 1

        self._episode_count += 1
        self._start_next_period_assignment()

        new_probs = (
            np.asarray(self.global_algo.get_probs(), dtype=float)
            if self.period_policy_mode == "legacy_resample"
            else np.asarray(self._period_probs, dtype=float)
        )
        arm_counts = {arm: int(np.sum(self.current_arm == arm)) for arm in range(self.n_arms)}
        LOG.debug(
            "[MABSpatialPricing] t=%s | ep=%s | grid=%s | algo=%s | probs=%s | arm_counts=%s",
            sim_time,
            self._episode_count,
            state["zone_system_name"],
            self.algo_name,
            [f"{value:.3f}" for value in new_probs],
            arm_counts,
        )
        self._sync_active_state_refs(state)

    def record_fare(self, amount: float, o_pos=None, rid=None, sim_time=None):
        rid_key = str(rid) if rid is not None else None
        if rid is not None:
            if rid_key in self._recorded_rids:
                return
            self._recorded_rids.add(rid_key)
        if (
            self._uses_request_reward_target()
            or self._uses_delayed_observed_feedback()
            or self._uses_event_matured_feedback()
        ):
            self._record_request_outcome(
                rid_key,
                float(amount),
                sim_time=sim_time,
                outcome_type="boarding",
                reason="boarding",
            )

        node_id = int(o_pos[0]) if o_pos is not None else None
        record = self.request_treatment_records.get(rid_key) if rid_key is not None else None
        if record is not None:
            record_zone_id = int(record.get("zone_id", -1))
            record_arm = record.get("arm")
            record_arm = int(record_arm) if record_arm is not None else None
            record_pricing_time = record.get("pricing_time")
            record_pricing_time = float(record_pricing_time) if record_pricing_time is not None else None
            record_arm_prob = record.get("arm_prob")
            record_arm_prob = float(record_arm_prob) if record_arm_prob is not None else None

            if self._dynamic_grid_enabled:
                state_key = record.get("zone_system_name")
                state = self._pricing_states.get(state_key)
                if state is not None:
                    self._record_fare_to_state(
                        state,
                        amount,
                        record_zone_id,
                        node_id,
                        rid=rid,
                        arm_idx=record_arm,
                        pricing_time=record_pricing_time,
                        arm_prob=record_arm_prob,
                    )
                    return
            else:
                self._record_fare_to_state(
                    {
                        "n_zones": self.n_zones,
                        "zone_revenue": self.zone_revenue,
                        "kdtree": self._kdtree,
                        "fare_records": self._fare_records,
                        "current_arm": self.current_arm,
                    },
                    amount,
                    record_zone_id,
                    node_id,
                    rid=rid,
                    arm_idx=record_arm,
                    pricing_time=record_pricing_time,
                    arm_prob=record_arm_prob,
                )
                return

        if self._dynamic_grid_enabled:
            if self._active_state_key is None:
                self._activate_dynamic_state(0)
            state = self._pricing_states[self._active_state_key]
            zone_id = state["node_to_zone"].get(node_id, -1) if node_id is not None else -1
            self._record_fare_to_state(state, amount, zone_id, node_id, rid=rid)
            return

        if o_pos is not None:
            zone_id = self.node_to_zone.get(node_id, -1)
            self._record_fare_to_state(
                {
                    "n_zones": self.n_zones,
                    "zone_revenue": self.zone_revenue,
                    "kdtree": self._kdtree,
                    "fare_records": self._fare_records,
                    "current_arm": self.current_arm,
                },
                amount,
                zone_id,
                node_id,
                rid=rid,
            )
            return

        self.zone_revenue[0] += float(amount)

    def get_elastic_price_factors(self, sim_time, expected_pu_time=None, o_pos=None, d_pos=None, rid=None):
        if self._dynamic_grid_enabled:
            self._activate_dynamic_state(sim_time)
        zone_id = -1
        if o_pos is not None:
            zone_id = self.node_to_zone.get(int(o_pos[0]), -1)

        if self.assignment_level == "passenger":
            arm_idx = self._passenger_arm(rid)
            base_factor, dist_factor, arm_label = self._resolve_arm_at_time(
                self.arms[arm_idx], sim_time, arm_idx
            )
            arm_prob = self._assignment_probability_for_arm(arm_idx)
            algo_debug_state = dict(self._period_algo_debug_state)
            self._record_request_treatment(
                sim_time,
                rid,
                zone_id,
                arm_idx,
                base_factor,
                dist_factor,
                arm_label,
                zone_system_name=self._zone_system_name,
                grid_edge_m=self._grid_edge_m,
                arm_prob=arm_prob,
                node_id=int(o_pos[0]) if o_pos is not None else None,
                pricing_episode_start=self._pricing_episode_start_for_time(sim_time),
                algo_debug_state=algo_debug_state,
            )
            return float(base_factor), float(dist_factor), 1.0

        if zone_id >= 0:
            arm_idx = int(self.current_arm[zone_id])
            base_factor, dist_factor, arm_label = self._resolve_arm_at_time(
                self.arms[arm_idx], sim_time, arm_idx
            )
            arm_prob = self._assignment_probability_for_arm(arm_idx)
            algo_debug_state = dict(self._period_algo_debug_state)
            self._record_request_treatment(
                sim_time,
                rid,
                zone_id,
                arm_idx,
                base_factor,
                dist_factor,
                arm_label,
                zone_system_name=self._zone_system_name,
                grid_edge_m=self._grid_edge_m,
                arm_prob=arm_prob,
                node_id=int(o_pos[0]) if o_pos is not None else None,
                pricing_episode_start=self._pricing_episode_start_for_time(sim_time),
                algo_debug_state=algo_debug_state,
            )
            return float(base_factor), float(dist_factor), 1.0

        return 1.0, 1.0, 1.0

    def record_unserved_request(
        self,
        sim_time,
        rid=None,
        o_pos=None,
        d_pos=None,
        reason: str = "operator_rejection",
    ) -> None:
        """Assign an otherwise unpriced request and retain it as a zero outcome."""
        if rid is None or not self._uses_event_matured_feedback():
            return
        rid_key = str(rid)
        if rid_key not in self.request_treatment_records:
            self.get_elastic_price_factors(
                sim_time,
                expected_pu_time=None,
                o_pos=o_pos,
                d_pos=d_pos,
                rid=rid,
            )
        self.record_request_failure(
            rid=rid,
            sim_time=sim_time,
            reason=reason,
        )

    def update_current_price_factors(self, sim_time):
        if self._dynamic_grid_enabled:
            state = self._activate_dynamic_state(sim_time)
            self._update_active_state_episode(state, sim_time)
            return
        if self.episode_start_time is None:
            self.episode_start_time = sim_time
            self.zone_episode_revenue = self.zone_revenue.copy()
            return

        if sim_time - self.episode_start_time < self.mab_interval:
            return

        episode_rewards = self.zone_revenue - self.zone_episode_revenue
        if self.period_policy_mode == "legacy_resample":
            current_probs = np.asarray(self.global_algo.get_probs(), dtype=float).tolist()
            algo_debug_state = self._get_algo_debug_state()
        else:
            current_probs = np.asarray(self._period_probs, dtype=float).tolist()
            algo_debug_state = dict(self._period_algo_debug_state)

        self.mab_log.append(
            {
                "sim_time_start": self.episode_start_time,
                "sim_time_end": sim_time,
                "zone_arms": self.current_arm.tolist(),
                "zone_rewards": episode_rewards.tolist(),
                "ucb_counts": self.ucb_counts.tolist(),
                "ucb_rewards": self.ucb_rewards.tolist(),
                "assignment_level": self.assignment_level,
                "arms": self.arms,
                "fixed_arm": self.fixed_arm,
                "fixed_allocation_probs": None
                if self.fixed_allocation_probs is None
                else self.fixed_allocation_probs.astype(float).tolist(),
                "use_exposure_prob": self.use_exposure_prob,
                "period_policy_mode": self.period_policy_mode,
                "htix_self_normalized": self.htix_self_normalized,
                "reward_event": self.reward_event,
                "reward_target": self.reward_target,
                "feedback_mode": self.feedback_mode,
                "request_reward_delay": self.request_reward_delay,
                "reward_normalization_mode": self.reward_normalization_mode,
                "reward_normalization_denominator": np.nan,
                "reward_normalization_lower_bound": float(self.reward_lower_bound),
                "reward_normalization_upper_bound": float(self.reward_upper_bound),
                "passenger_records": self._fare_records if self.assignment_level == "passenger" else [],
                "episode": self._episode_count,
                "arm_probs": current_probs,
                "algo_debug_state": algo_debug_state,
                "hedge_meta_weights": algo_debug_state.get("meta_weights"),
                "hedge_expert_trust": algo_debug_state.get("expert_trust"),
                "hedge_ts_probs": algo_debug_state.get("ts_probs"),
                "hedge_exp3_probs": algo_debug_state.get("exp3_probs"),
                "hedge_eta": algo_debug_state.get("hedge_eta"),
                "hedge_gamma": algo_debug_state.get("hedge_gamma"),
                "hedge_fixed_share_alpha": algo_debug_state.get("fixed_share_alpha"),
            }
        )
        self._save_log()

        self.episode_start_time = sim_time
        self.zone_episode_revenue = self.zone_revenue.copy()

        if self.random_assignment:
            self._fare_records = []
            self._episode_count += 1
            self._start_next_period_assignment()
            return

        arm_probs = (
            np.asarray(self.global_algo.get_probs(), dtype=float)
            if self.period_policy_mode == "legacy_resample"
            else np.asarray(self._period_probs, dtype=float)
        )

        if self._episode_count == 0:
            self._fare_records = []
            self._record_learning_update_debug(
                {},
                {},
                skipped_update_reason="warm_start_no_matured_feedback",
            )
            self._episode_count += 1
            self._start_next_period_assignment()
            LOG.debug("[MABSpatialPricing] ep0 skipped update to avoid warm-start bias")
            return

        skip_update_reason = None
        grid_beta_evidence = None
        companion_normalized_rewards = None
        if self._uses_grid_beta_evidence():
            grid_beta_evidence = self._estimate_grid_beta_evidence(sim_time)
            if self._uses_grid_beta_hedge():
                companion_rewards, companion_normalized_rewards = (
                    self._estimate_grid_beta_hedge_companion_feedback(sim_time)
                )
                estimated_rewards = (
                    companion_rewards
                    if companion_normalized_rewards
                    else grid_beta_evidence["mean_rewards"]
                )
            else:
                estimated_rewards = grid_beta_evidence["mean_rewards"]
            if (
                self._last_grid_beta_debug.get("processed_feedback_count", 0) == 0
                and not companion_normalized_rewards
            ):
                skip_update_reason = "no_matured_grid_beta_feedback"
            elif (
                self._last_grid_beta_debug.get("informative_grid_count", 0) == 0
                and not companion_normalized_rewards
            ):
                skip_update_reason = "no_informative_grid_beta_feedback"
        elif self._uses_delayed_observed_feedback() or self._uses_event_matured_feedback():
            estimated_rewards = self._estimate_delayed_observed_feedback(sim_time)
            if self._last_delayed_feedback_debug.get("processed_feedback_count", 0) == 0:
                skip_update_reason = "no_observed_delayed_feedback"
            elif self._last_delayed_feedback_debug.get("valid_estimated_reward_count", 1) == 0:
                skip_update_reason = "no_valid_htix_feedback"
        elif self._uses_request_reward_target():
            estimated_rewards = self._estimate_request_reward_target(sim_time)
        else:
            estimated_rewards = self.estimator.estimate(
                zone_arms=self.current_arm,
                zone_revenues=episode_rewards,
                arm_probs=arm_probs,
                zone_neighbors=self.zone_neighbors if self.zone_neighbors else None,
                fare_records=self._fare_records if self._kdtree is not None else None,
                node_xy=self._node_xy,
                node_zone=self._node_zone_by_node_id,
                kdtree=self._kdtree,
                interference_radius=self._interference_radius,
                use_exposure_prob=self.use_exposure_prob,
                self_normalized=self.htix_self_normalized,
            )

        self._fare_records = []
        if skip_update_reason is not None:
            self._record_learning_update_debug({}, {}, skipped_update_reason=skip_update_reason)
            self._episode_count += 1
            self._start_next_period_assignment()
            return

        if grid_beta_evidence is not None:
            normalized_rewards = (
                companion_normalized_rewards
                if companion_normalized_rewards
                else estimated_rewards
            )
            arm_standard_errors = {}
            hedge_exp3_rewards = None
        else:
            normalized_rewards = self._prepare_learning_rewards(estimated_rewards)
            arm_standard_errors = self._prepare_learning_reward_standard_errors()
            hedge_exp3_rewards = self._prepare_hedge_exp3_rewards(estimated_rewards)
        if self._is_non_learning_policy():
            self._record_learning_update_debug(
                estimated_rewards,
                normalized_rewards,
                skipped_update_reason="fixed_policy_no_learning",
            )
        else:
            if grid_beta_evidence is not None:
                if self._uses_grid_beta_hedge():
                    self.global_algo.update_grid_beta_experts(
                        grid_beta_evidence["reward_sums"],
                        grid_beta_evidence["trial_counts"],
                        exp3_rewards=companion_normalized_rewards or None,
                        meta_rewards=companion_normalized_rewards or None,
                    )
                else:
                    self.global_algo.update_grid_evidence(
                        grid_beta_evidence["reward_sums"],
                        grid_beta_evidence["trial_counts"],
                    )
            else:
                self._update_global_algorithm(
                    normalized_rewards,
                    arm_standard_errors,
                    hedge_exp3_rewards=hedge_exp3_rewards,
                )
            self._record_learning_update_debug(estimated_rewards, normalized_rewards)

        for zone_id in range(self.n_zones):
            arm_idx = int(self.current_arm[zone_id])
            self.ucb_counts[zone_id, arm_idx] += 1.0
            self.ucb_rewards[zone_id, arm_idx] += float(episode_rewards[zone_id])
        self.zone_pulls += 1

        self._episode_count += 1
        self._start_next_period_assignment()

        new_probs = (
            np.asarray(self.global_algo.get_probs(), dtype=float)
            if self.period_policy_mode == "legacy_resample"
            else np.asarray(self._period_probs, dtype=float)
        )
        arm_counts = {arm: int(np.sum(self.current_arm == arm)) for arm in range(self.n_arms)}
        LOG.debug(
            "[MABSpatialPricing] t=%s | ep=%s | algo=%s | probs=%s | arm_counts=%s",
            sim_time,
            self._episode_count,
            self.algo_name,
            [f"{value:.3f}" for value in new_probs],
            arm_counts,
        )





    def _save_log(self):
        try:
            pd.DataFrame(self.mab_log).to_csv(self.mab_log_path, index=False)
        except Exception as exc:
            LOG.warning("[MABSpatialPricing] Could not save MAB log: %s", exc)
        try:
            request_df = pd.DataFrame(self.request_treatment_records.values())
            if not request_df.empty:
                request_df["request_reward_amount"] = (
                    request_df["request_id"].astype(str).map(self._request_outcomes).fillna(0.0)
                )
                request_df["request_reward_observed"] = (
                    request_df["request_id"].astype(str).isin(self._request_outcomes)
                )
                request_df["request_reward_observed_time"] = (
                    request_df["request_id"].astype(str).map(self._request_outcome_times)
                )
                request_df["request_reward_outcome_type"] = (
                    request_df["request_id"].astype(str).map(self._request_outcome_types)
                )
                request_df["request_reward_outcome_reason"] = (
                    request_df["request_id"].astype(str).map(self._request_outcome_reasons)
                )
                request_df["request_reward_processed"] = (
                    request_df["request_id"].astype(str).isin(self._processed_request_reward_rids)
                )
                request_df["delayed_feedback_processed"] = (
                    request_df["request_id"].astype(str).isin(self._processed_delayed_feedback_rids)
                )
                request_df = request_df.sort_values(["pricing_time", "request_id"]).reset_index(drop=True)
            request_df.to_csv(self.request_log_path, index=False)
        except Exception as exc:
            LOG.warning("[MABSpatialPricing] Could not save request treatment log: %s", exc)
        if self.htix_diagnostics_enabled:
            try:
                pd.DataFrame(self._htix_diagnostics).to_csv(
                    self.htix_diagnostics_path,
                    index=False,
                )
            except Exception as exc:
                LOG.warning("[MABSpatialPricing] Could not save HT-IX diagnostics: %s", exc)

    def final_log(self):
        self._save_log()
        if self._dynamic_grid_enabled:
            print("[MABSpatialPricing] ===== Final Dynamic-Grid Summary =====")
            for zone_system_name, state in self._pricing_states.items():
                total_pulls = int(state["zone_pulls"].sum())
                total_revenue = float(np.sum(state["zone_revenue"]))
                print(
                    f"  Grid {zone_system_name}: edge={state['grid_edge_m']:.0f}m, "
                    f"episodes={state['episode_count']}, total_pulls={total_pulls}, revenue={total_revenue:.2f}"
                )
            print(f"  Episode log saved to: {self.mab_log_path}")
            print(f"  Request log saved to: {self.request_log_path}")
            return
        if self.random_assignment:
            print("[MABSpatialPricing] Random assignment mode - no convergence summary.")
            return
        if self.zone_pulls.sum() == 0:
            print("[MABSpatialPricing] No episodes completed.")
            return

        print("[MABSpatialPricing] ===== Final Summary =====")
        avg_rewards = np.divide(
            self.ucb_rewards,
            self.ucb_counts,
            out=np.zeros((self.n_zones, self.n_arms), dtype=float),
            where=self.ucb_counts > 0,
        )
        best_arms = np.argmax(avg_rewards, axis=1)
        for zone_id in range(self.n_zones):
            best_arm = int(best_arms[zone_id])
            print(
                f"  Zone {zone_id}: best_arm={best_arm} {self.arms[best_arm]}, "
                f"pulls={self.zone_pulls[zone_id]}, total_revenue={self.zone_revenue[zone_id]:.2f}"
            )
        print(f"  Episode log saved to: {self.mab_log_path}")
        print(f"  Request log saved to: {self.request_log_path}")
        if self.htix_diagnostics_enabled:
            print(f"  HT-IX diagnostics saved to: {self.htix_diagnostics_path}")
