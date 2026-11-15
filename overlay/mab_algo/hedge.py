import numpy as np

from .base import MABAlgorithm
from .exp3 import EXP3, JiaEXP3
from .thompson import GaussianThompsonSampling, GridBetaThompsonSampling, ThompsonSampling


class HedgeTSEXP3(MABAlgorithm):
    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        T: int = 100,
        mc_draws: int = 2000,
        hedge_eta: float = None,
        hedge_gamma: float = 0.02,
        exp3_eta: float = None,
        exp3_exploration_epsilon: float = None,
        exp3_discount_factor: float = 1.0,
        exp3_fixed_share_alpha: float = 0.0,
    ):
        super().__init__(n_arms, seed)
        self.n_experts = 2
        default_eta = np.sqrt(2.0 * np.log(self.n_experts) / max(T, 1))
        self.hedge_eta = float(hedge_eta) if hedge_eta is not None else float(default_eta)
        self.hedge_gamma = float(np.clip(hedge_gamma, 0.0, 1.0))

        self.meta_weights = np.ones(self.n_experts, dtype=float)
        self.last_meta_rewards = np.zeros(self.n_experts, dtype=float)
        self.last_reward_vec = np.zeros(self.n_arms, dtype=float)
        self.last_ts_reward_vec = np.zeros(self.n_arms, dtype=float)
        self.last_exp3_reward_vec = np.zeros(self.n_arms, dtype=float)
        self._last_ts_probs = None
        self._last_exp3_probs = None
        self._last_mixed_probs = None
        self.ts = ThompsonSampling(
            n_arms=n_arms,
            seed=seed + 101,
            mc_draws=mc_draws,
        )
        self.exp3 = EXP3(
            n_arms=n_arms,
            seed=seed + 202,
            T=T,
            eta=exp3_eta,
            exploration_epsilon=exp3_exploration_epsilon,
            discount_factor=exp3_discount_factor,
            fixed_share_alpha=exp3_fixed_share_alpha,
        )

    @staticmethod
    def _normalize_reward(reward: float) -> float:
        return float(np.clip(reward, 0.0, 1.0))

    def _uses_jia_exp3(self) -> bool:
        return isinstance(self.exp3, JiaEXP3)

    def _exp3_reward_vector(self, rewards: dict | None) -> np.ndarray:
        """Return the precise vector consumed by the EXP3 expert.

        Legacy EXP3 expects a bounded reward and keeps its historical clipping.
        JiaEXP3 instead receives the request-level-scaled aggregate unchanged:
        HT-IX estimates can exceed one after inverse-probability weighting.
        """
        if rewards is None:
            return np.zeros(self.n_arms, dtype=float)
        values = []
        for arm in range(self.n_arms):
            try:
                reward = float(rewards.get(arm, 0.0))
            except (AttributeError, TypeError, ValueError):
                reward = 0.0
            if not np.isfinite(reward):
                reward = 0.0
            if not self._uses_jia_exp3():
                reward = self._normalize_reward(reward)
            values.append(reward)
        return np.asarray(values, dtype=float)

    def _expert_trust(self) -> np.ndarray:
        weights = self.meta_weights / max(self.meta_weights.sum(), 1e-12)
        return weights

    def _expert_probs(self):
        return self.ts.get_probs(), self.exp3.get_probs()

    def _invalidate_policy_cache(self):
        self._last_ts_probs = None
        self._last_exp3_probs = None
        self._last_mixed_probs = None

    def _period_policy(self):
        if self._last_mixed_probs is None:
            self.get_probs()
        return (
            self._last_ts_probs.copy(),
            self._last_exp3_probs.copy(),
            self._last_mixed_probs.copy(),
        )

    def get_debug_state(self) -> dict:
        trust = self._expert_trust()
        exp3_debug = (
            self.exp3.get_debug_state()
            if hasattr(self.exp3, "get_debug_state")
            else {}
        )
        return {
            "meta_weights": self.meta_weights.astype(float).tolist(),
            "expert_trust": trust.astype(float).tolist(),
            "ts_probs": None
            if self._last_ts_probs is None
            else self._last_ts_probs.astype(float).tolist(),
            "exp3_probs": None
            if self._last_exp3_probs is None
            else self._last_exp3_probs.astype(float).tolist(),
            "mixed_probs": None
            if self._last_mixed_probs is None
            else self._last_mixed_probs.astype(float).tolist(),
            "hedge_eta": float(self.hedge_eta),
            "hedge_gamma": float(self.hedge_gamma),
            "last_meta_rewards": self.last_meta_rewards.astype(float).tolist(),
            "last_reward_vec": self.last_reward_vec.astype(float).tolist(),
            "last_ts_reward_vec": self.last_ts_reward_vec.astype(float).tolist(),
            "last_exp3_reward_vec": self.last_exp3_reward_vec.astype(float).tolist(),
            "exp3_expert_algorithm": exp3_debug.get(
                "algorithm", self.exp3.__class__.__name__
            ),
            "exp3_forced_uniform_exploration": bool(
                exp3_debug.get(
                    "forced_uniform_exploration",
                    not self._uses_jia_exp3(),
                )
            ),
            "exp3_aggregate_reward_clipping": bool(
                exp3_debug.get(
                    "aggregate_reward_clipping",
                    not self._uses_jia_exp3(),
                )
            ),
            "exp3_eta": exp3_debug.get("eta"),
            "exp3_exploration_epsilon": exp3_debug.get(
                "exploration_epsilon"
            ),
            "exp3_discount_factor": exp3_debug.get("discount_factor"),
            "exp3_fixed_share_alpha": exp3_debug.get("fixed_share_alpha"),
            "exp3_effective_memory_periods": exp3_debug.get(
                "effective_memory_periods"
            ),
        }

    def get_probs(self) -> np.ndarray:
        trust = self._expert_trust()
        ts_probs, exp3_probs = self._expert_probs()
        mixed = trust[0] * ts_probs + trust[1] * exp3_probs
        final = (1.0 - self.hedge_gamma) * mixed + self.hedge_gamma / self.n_arms
        final = np.asarray(final, dtype=float)
        final = final / max(final.sum(), 1e-12)
        self._last_ts_probs = np.asarray(ts_probs, dtype=float).copy()
        self._last_exp3_probs = np.asarray(exp3_probs, dtype=float).copy()
        self._last_mixed_probs = final.copy()
        return final.copy()

    def _stabilize_meta_weights(self):
        max_w = np.max(self.meta_weights)
        if max_w > 0:
            self.meta_weights = self.meta_weights / max_w

    def _apply_meta_update(self, meta_rewards: np.ndarray):
        meta_rewards = np.asarray(meta_rewards, dtype=float)
        self.meta_weights *= np.exp(self.hedge_eta * meta_rewards)
        self._stabilize_meta_weights()

    def update(self, arm: int, reward: float):
        self.t += 1
        arm = int(arm)
        observed_reward = self._normalize_reward(reward)
        exp3_reward = float(reward) if self._uses_jia_exp3() else observed_reward
        if not np.isfinite(exp3_reward):
            exp3_reward = 0.0

        ts_probs, exp3_probs, mixed_probs = self._period_policy()
        denom = max(float(mixed_probs[arm]), 1e-9)
        reward_est = observed_reward / denom

        ts_meta_reward = float(np.clip(ts_probs[arm] * reward_est, 0.0, 1.0))
        exp3_meta_reward = float(np.clip(exp3_probs[arm] * reward_est, 0.0, 1.0))

        self.last_reward_vec = np.zeros(self.n_arms, dtype=float)
        self.last_reward_vec[arm] = observed_reward
        self.last_exp3_reward_vec = np.zeros(self.n_arms, dtype=float)
        self.last_exp3_reward_vec[arm] = exp3_reward
        self.last_meta_rewards = np.array([ts_meta_reward, exp3_meta_reward], dtype=float)
        self._apply_meta_update(self.last_meta_rewards)

        self.ts.update(arm, observed_reward)
        self.exp3.update(arm, exp3_reward)
        self._invalidate_policy_cache()

    def update_experts(
        self,
        ts_rewards: dict | None,
        exp3_rewards: dict | None,
        meta_rewards: dict | None,
    ):
        """Update each expert from its designated signal.

        ``meta_rewards`` must be a common reward vector when the two experts use
        different internal estimators. This keeps the Hedge comparison on one
        scale while allowing TS and EXP3 to maintain estimator-appropriate
        internal states.
        """
        self.t += 1

        def reward_vector(rewards: dict | None) -> np.ndarray:
            if rewards is None:
                return np.zeros(self.n_arms, dtype=float)
            return np.array(
                [
                    self._normalize_reward(rewards.get(arm, 0.0))
                    for arm in range(self.n_arms)
                ],
                dtype=float,
            )

        ts_reward_vec = reward_vector(ts_rewards)
        exp3_reward_vec = self._exp3_reward_vector(exp3_rewards)
        meta_reward_vec = reward_vector(meta_rewards)

        self.last_ts_reward_vec = ts_reward_vec.copy()
        self.last_exp3_reward_vec = exp3_reward_vec.copy()
        self.last_reward_vec = meta_reward_vec.copy()

        if meta_rewards is not None:
            ts_probs, exp3_probs, _mixed_probs = self._period_policy()
            ts_meta_reward = float(np.dot(ts_probs, meta_reward_vec))
            exp3_meta_reward = float(np.dot(exp3_probs, meta_reward_vec))
            self.last_meta_rewards = np.array(
                [ts_meta_reward, exp3_meta_reward], dtype=float
            )
            self._apply_meta_update(self.last_meta_rewards)
        else:
            self.last_meta_rewards = np.zeros(self.n_experts, dtype=float)

        if ts_rewards is not None:
            self.ts.update_all(
                {arm: float(ts_reward_vec[arm]) for arm in range(self.n_arms)}
            )
        if exp3_rewards is not None:
            self.exp3.update_all(
                {arm: float(exp3_reward_vec[arm]) for arm in range(self.n_arms)}
            )
        self._invalidate_policy_cache()

    def update_all(self, rewards: dict):
        self.update_experts(rewards, rewards, rewards)

    def reset(self):
        super().reset()
        self.meta_weights = np.ones(self.n_experts, dtype=float)
        self.last_meta_rewards = np.zeros(self.n_experts, dtype=float)
        self.last_reward_vec = np.zeros(self.n_arms, dtype=float)
        self.last_ts_reward_vec = np.zeros(self.n_arms, dtype=float)
        self.last_exp3_reward_vec = np.zeros(self.n_arms, dtype=float)
        self._invalidate_policy_cache()
        self.ts.reset()
        self.exp3.reset()


class RiskAwareHedgeTSEXP3(HedgeTSEXP3):
    """Hedge(TS, EXP3) with an online policy-level uncertainty penalty.

    The inner TS and EXP3 experts receive exactly the same period reward vector
    as ``HedgeTSEXP3``.  Only the outer Hedge score changes: it subtracts a
    pre-specified multiple of the contemporaneous estimator standard error.
    This keeps risk control causal and avoids using variation across completed
    simulation replications as an online signal.
    """

    uses_reward_uncertainty = True

    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        T: int = 100,
        mc_draws: int = 2000,
        hedge_eta: float = None,
        hedge_gamma: float = 0.02,
        hedge_risk_lambda: float = 1.0,
    ):
        super().__init__(
            n_arms=n_arms,
            seed=seed,
            T=T,
            mc_draws=mc_draws,
            hedge_eta=hedge_eta,
            hedge_gamma=hedge_gamma,
        )
        if not np.isfinite(hedge_risk_lambda) or hedge_risk_lambda < 0.0:
            raise ValueError("hedge_risk_lambda must be finite and non-negative.")
        self.hedge_risk_lambda = float(hedge_risk_lambda)
        self.last_arm_standard_errors = np.zeros(self.n_arms, dtype=float)
        self.last_meta_standard_errors = np.zeros(self.n_experts, dtype=float)
        self.last_meta_scores = np.zeros(self.n_experts, dtype=float)

    def _arm_standard_error_vector(
        self,
        arm_standard_errors: dict | None,
    ) -> np.ndarray:
        if arm_standard_errors is None:
            return np.zeros(self.n_arms, dtype=float)
        values = []
        for arm in range(self.n_arms):
            try:
                value = float(arm_standard_errors.get(arm, 0.0))
            except (AttributeError, TypeError, ValueError):
                value = 0.0
            values.append(float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0)
        return np.asarray(values, dtype=float)

    @staticmethod
    def _policy_standard_error(policy: np.ndarray, arm_standard_errors: np.ndarray) -> float:
        return float(
            np.sqrt(
                np.sum(
                    np.square(np.asarray(policy, dtype=float))
                    * np.square(np.asarray(arm_standard_errors, dtype=float))
                )
            )
        )

    def update_experts(
        self,
        ts_rewards: dict | None,
        exp3_rewards: dict | None,
        meta_rewards: dict | None,
        arm_standard_errors: dict | None = None,
    ):
        self.t += 1

        def reward_vector(rewards: dict | None) -> np.ndarray:
            if rewards is None:
                return np.zeros(self.n_arms, dtype=float)
            return np.array(
                [
                    self._normalize_reward(rewards.get(arm, 0.0))
                    for arm in range(self.n_arms)
                ],
                dtype=float,
            )

        ts_reward_vec = reward_vector(ts_rewards)
        exp3_reward_vec = self._exp3_reward_vector(exp3_rewards)
        meta_reward_vec = reward_vector(meta_rewards)
        arm_standard_error_vec = self._arm_standard_error_vector(arm_standard_errors)

        self.last_ts_reward_vec = ts_reward_vec.copy()
        self.last_exp3_reward_vec = exp3_reward_vec.copy()
        self.last_reward_vec = meta_reward_vec.copy()
        self.last_arm_standard_errors = arm_standard_error_vec.copy()

        if meta_rewards is not None:
            ts_probs, exp3_probs, _mixed_probs = self._period_policy()
            raw_meta_rewards = np.array(
                [
                    float(np.dot(ts_probs, meta_reward_vec)),
                    float(np.dot(exp3_probs, meta_reward_vec)),
                ],
                dtype=float,
            )
            meta_standard_errors = np.array(
                [
                    self._policy_standard_error(ts_probs, arm_standard_error_vec),
                    self._policy_standard_error(exp3_probs, arm_standard_error_vec),
                ],
                dtype=float,
            )
            meta_scores = np.clip(
                raw_meta_rewards - self.hedge_risk_lambda * meta_standard_errors,
                0.0,
                1.0,
            )
            self.last_meta_rewards = raw_meta_rewards
            self.last_meta_standard_errors = meta_standard_errors
            self.last_meta_scores = meta_scores
            self._apply_meta_update(meta_scores)
        else:
            self.last_meta_rewards = np.zeros(self.n_experts, dtype=float)
            self.last_meta_standard_errors = np.zeros(self.n_experts, dtype=float)
            self.last_meta_scores = np.zeros(self.n_experts, dtype=float)

        if ts_rewards is not None:
            self.ts.update_all(
                {arm: float(ts_reward_vec[arm]) for arm in range(self.n_arms)}
            )
        if exp3_rewards is not None:
            self.exp3.update_all(
                {arm: float(exp3_reward_vec[arm]) for arm in range(self.n_arms)}
            )
        self._invalidate_policy_cache()

    def update_all(
        self,
        rewards: dict,
        arm_standard_errors: dict | None = None,
    ):
        self.update_experts(
            rewards,
            rewards,
            rewards,
            arm_standard_errors=arm_standard_errors,
        )

    def get_debug_state(self) -> dict:
        state = super().get_debug_state()
        state.update(
            {
                "algorithm": "RiskAwareHedgeTSEXP3",
                "hedge_risk_lambda": float(self.hedge_risk_lambda),
                "risk_signal": "online_policy_standard_error",
                "last_arm_standard_errors": self.last_arm_standard_errors.astype(float).tolist(),
                "last_meta_standard_errors": self.last_meta_standard_errors.astype(float).tolist(),
                "last_meta_scores": self.last_meta_scores.astype(float).tolist(),
            }
        )
        return state

    def reset(self):
        super().reset()
        self.last_arm_standard_errors = np.zeros(self.n_arms, dtype=float)
        self.last_meta_standard_errors = np.zeros(self.n_experts, dtype=float)
        self.last_meta_scores = np.zeros(self.n_experts, dtype=float)


class HedgeGaussianTSEXP3(HedgeTSEXP3):
    """Hedge between Normal-Inverse-Gamma Gaussian TS and EXP3."""

    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        T: int = 100,
        mc_draws: int = 2000,
        hedge_eta: float = None,
        hedge_gamma: float = 0.02,
    ):
        super().__init__(
            n_arms=n_arms,
            seed=seed,
            T=T,
            mc_draws=mc_draws,
            hedge_eta=hedge_eta,
            hedge_gamma=hedge_gamma,
        )
        self.ts = GaussianThompsonSampling(
            n_arms=n_arms,
            seed=seed + 101,
            mc_draws=mc_draws,
        )

    def get_debug_state(self) -> dict:
        state = super().get_debug_state()
        state["ts_expert_algorithm"] = "GaussianThompsonSampling"
        return state


class HedgeGaussianTSJiaEXP3(HedgeGaussianTSEXP3):
    """Hedge between Gaussian TS and JiaEXP3 request-level estimates."""

    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        T: int = 100,
        mc_draws: int = 2000,
        eta: float = None,
        hedge_eta: float = None,
        hedge_gamma: float = 0.02,
    ):
        super().__init__(
            n_arms=n_arms,
            seed=seed,
            T=T,
            mc_draws=mc_draws,
            hedge_eta=hedge_eta,
            hedge_gamma=hedge_gamma,
        )
        self.exp3 = JiaEXP3(
            n_arms=n_arms,
            seed=seed + 202,
            eta=eta,
            T=T,
        )

    def get_debug_state(self) -> dict:
        state = super().get_debug_state()
        state["algorithm"] = "HedgeGaussianTSJiaEXP3"
        state["jia_exp3_request_level_signal"] = True
        return state


class FixedShareHedgeTSEXP3(HedgeTSEXP3):
    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        T: int = 100,
        mc_draws: int = 2000,
        hedge_eta: float = None,
        hedge_gamma: float = 0.02,
        fixed_share_alpha: float = None,
        exp3_eta: float = None,
        exp3_exploration_epsilon: float = None,
        exp3_discount_factor: float = 1.0,
        exp3_fixed_share_alpha: float = 0.0,
    ):
        super().__init__(
            n_arms=n_arms,
            seed=seed,
            T=T,
            mc_draws=mc_draws,
            hedge_eta=hedge_eta,
            hedge_gamma=hedge_gamma,
            exp3_eta=exp3_eta,
            exp3_exploration_epsilon=exp3_exploration_epsilon,
            exp3_discount_factor=exp3_discount_factor,
            exp3_fixed_share_alpha=exp3_fixed_share_alpha,
        )
        default_alpha = 1.0 / max(T, 1)
        self.fixed_share_alpha = float(
            np.clip(default_alpha if fixed_share_alpha is None else fixed_share_alpha, 0.0, 1.0)
        )

    def get_debug_state(self) -> dict:
        state = super().get_debug_state()
        state["fixed_share_alpha"] = float(self.fixed_share_alpha)
        state["meta_algorithm"] = "FixedShareHedgeTSEXP3"
        return state

    def _apply_meta_update(self, meta_rewards: np.ndarray):
        meta_rewards = np.asarray(meta_rewards, dtype=float)
        updated = self.meta_weights * np.exp(self.hedge_eta * meta_rewards)

        alpha = self.fixed_share_alpha
        if self.n_experts > 1 and alpha > 0.0:
            total_weight = float(updated.sum())
            shared_from_others = (total_weight - updated) / (self.n_experts - 1)
            updated = (1.0 - alpha) * updated + alpha * shared_from_others

        self.meta_weights = updated
        self._stabilize_meta_weights()


class HedgeGridBetaEXP3(HedgeTSEXP3):
    """Hedge between grid-wise fractional-Beta TS and EXP3.

    The TS expert makes one posterior draw per grid cell.  The EXP3 expert
    remains a period-level policy.  Hedge samples the expert mixture per grid
    cell, so the TS part of the policy retains the independent grid-level
    exploration requested by the Grid-Beta construction.
    """

    gridwise_sampling = True
    grid_beta_evidence = True
    grid_beta_with_exp3 = True

    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        T: int = 100,
        mc_draws: int = 2000,
        hedge_eta: float = None,
        hedge_gamma: float = 0.02,
    ):
        super().__init__(
            n_arms=n_arms,
            seed=seed,
            T=T,
            mc_draws=mc_draws,
            hedge_eta=hedge_eta,
            hedge_gamma=hedge_gamma,
        )
        self.ts = GridBetaThompsonSampling(
            n_arms=n_arms,
            seed=seed + 101,
            mc_draws=mc_draws,
        )

    def sample_arm(self) -> int:
        """Sample one grid assignment from the Hedge mixture.

        The implementation mirrors ``get_probs`` in expectation: Hedge's
        exploration mass is uniform over arms; otherwise the selected expert
        supplies the action.  Selecting the TS expert calls its independent
        one-draw Grid-Beta sampler rather than sampling an MC probability.
        """
        _ts_probs, exp3_probs, _mixed_probs = self._period_policy()
        if self.rng.random() < self.hedge_gamma:
            return int(self.rng.integers(self.n_arms))
        if self.rng.random() < self._expert_trust()[0]:
            return self.ts.sample_arm()
        return int(self.rng.choice(self.n_arms, p=np.asarray(exp3_probs, dtype=float)))

    def update_grid_beta_experts(
        self,
        reward_sums,
        trial_counts,
        exp3_rewards: dict | None,
        meta_rewards: dict | None,
    ) -> None:
        """Update Grid-Beta TS and the period-level EXP3/Hedge signals.

        ``reward_sums`` and ``trial_counts`` are the fractional evidence from
        informative grid cells.  ``exp3_rewards`` and ``meta_rewards`` are on
        a common [0, 1] fixed scale; they may be absent when the companion
        estimator has no valid feedback in a period.
        """
        self.t += 1
        reward_sums = np.asarray(reward_sums, dtype=float)
        trial_counts = np.asarray(trial_counts, dtype=float)
        if reward_sums.shape != (self.n_arms,) or trial_counts.shape != (self.n_arms,):
            raise ValueError("Grid evidence must provide one sum and count per arm.")

        ts_means = np.divide(
            reward_sums,
            trial_counts,
            out=np.zeros(self.n_arms, dtype=float),
            where=trial_counts > 0.0,
        )

        def reward_vector(rewards: dict | None) -> np.ndarray:
            if rewards is None:
                return np.zeros(self.n_arms, dtype=float)
            return np.array(
                [self._normalize_reward(rewards.get(arm, 0.0)) for arm in range(self.n_arms)],
                dtype=float,
            )

        exp3_reward_vec = self._exp3_reward_vector(exp3_rewards)
        meta_reward_vec = reward_vector(meta_rewards)
        self.last_ts_reward_vec = ts_means.copy()
        self.last_exp3_reward_vec = exp3_reward_vec.copy()
        self.last_reward_vec = meta_reward_vec.copy()

        if meta_rewards is not None:
            ts_probs, exp3_probs, _mixed_probs = self._period_policy()
            self.last_meta_rewards = np.array(
                [
                    float(np.dot(ts_probs, meta_reward_vec)),
                    float(np.dot(exp3_probs, meta_reward_vec)),
                ],
                dtype=float,
            )
            self._apply_meta_update(self.last_meta_rewards)
        else:
            self.last_meta_rewards = np.zeros(self.n_experts, dtype=float)

        if float(trial_counts.sum()) > 0.0:
            self.ts.update_grid_evidence(reward_sums, trial_counts)
        if exp3_rewards is not None:
            self.exp3.update_all(
                {arm: float(exp3_reward_vec[arm]) for arm in range(self.n_arms)}
            )
        self._invalidate_policy_cache()

    def get_debug_state(self) -> dict:
        state = super().get_debug_state()
        state.update(
            {
                "ts_expert_algorithm": "GridBetaThompsonSampling",
                "grid_beta_evidence": True,
                "ts_grid_beta_state": self.ts.get_debug_state(),
            }
        )
        return state
