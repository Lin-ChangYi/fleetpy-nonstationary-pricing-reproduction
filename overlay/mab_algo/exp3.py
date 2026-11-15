import numpy as np

from .base import MABAlgorithm


class EXP3(MABAlgorithm):
    """EXP3 with optional decoupled learning and exploration controls.

    When the optional controls are omitted, this class retains the historical
    EXP3 update exactly: ``gamma`` determines both the uniform exploration
    mass and the exponential-weight learning rate.  The optional controls are
    available only for explicitly configured diagnostic experiments.
    """

    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        gamma: float = None,
        T: int = 100,
        eta: float = None,
        exploration_epsilon: float = None,
        discount_factor: float = 1.0,
        fixed_share_alpha: float = 0.0,
    ):
        super().__init__(n_arms, seed)
        default_gamma = np.sqrt(self.n_arms * np.log(self.n_arms) / max(T, 1))
        self.gamma = float(gamma) if gamma is not None else min(1.0, default_gamma)
        if not np.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("EXP3 gamma must be finite and in [0, 1].")
        self.eta = (
            float(eta) if eta is not None else float(self.gamma / self.n_arms)
        )
        if not np.isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("EXP3 eta must be a finite positive number.")
        self.exploration_epsilon = (
            float(exploration_epsilon)
            if exploration_epsilon is not None
            else float(self.gamma)
        )
        if (
            not np.isfinite(self.exploration_epsilon)
            or not 0.0 <= self.exploration_epsilon <= 1.0
        ):
            raise ValueError(
                "EXP3 exploration_epsilon must be finite and in [0, 1]."
            )
        self.discount_factor = float(discount_factor)
        if not np.isfinite(self.discount_factor) or not 0.0 < self.discount_factor <= 1.0:
            raise ValueError(
                "EXP3 discount_factor must be finite and in (0, 1]."
            )
        self.fixed_share_alpha = float(fixed_share_alpha)
        if (
            not np.isfinite(self.fixed_share_alpha)
            or not 0.0 <= self.fixed_share_alpha <= 1.0
        ):
            raise ValueError(
                "EXP3 fixed_share_alpha must be finite and in [0, 1]."
            )
        self._legacy_update_rule = (
            eta is None
            and exploration_epsilon is None
            and self.discount_factor == 1.0
        )
        self.weights = np.ones(self.n_arms, dtype=float)

    def get_probs(self) -> np.ndarray:
        normalized = self.weights / self.weights.sum()
        if self._legacy_update_rule:
            return (1.0 - self.gamma) * normalized + self.gamma / self.n_arms
        return (
            (1.0 - self.exploration_epsilon) * normalized
            + self.exploration_epsilon / self.n_arms
        )

    def update(self, arm: int, reward: float):
        self.t += 1
        self._discount_history()
        self._bandit_update(int(arm), float(reward))
        self._stabilize_weights()
        self._apply_fixed_share()

    def update_all(self, rewards: dict):
        self.t += 1


        self._discount_history()
        for arm, reward in rewards.items():
            self._full_information_update(int(arm), float(reward))
        self._stabilize_weights()
        self._apply_fixed_share()

    @staticmethod
    def _normalize_reward(reward: float) -> float:
        return float(np.clip(reward, 0.0, 1.0))

    def _bandit_update(self, arm: int, reward: float):
        probs = self.get_probs()
        scaled_reward = self._normalize_reward(reward)
        if self._legacy_update_rule:
            exponent = self.gamma * (scaled_reward / probs[arm]) / self.n_arms
        else:
            exponent = self.eta * (scaled_reward / probs[arm])
        self.weights[arm] *= np.exp(np.clip(exponent, -700.0, 700.0))

    def _full_information_update(self, arm: int, reward: float):
        scaled_reward = self._normalize_reward(reward)
        if self._legacy_update_rule:
            exponent = self.gamma * scaled_reward / self.n_arms
        else:
            exponent = self.eta * scaled_reward
        self.weights[arm] *= np.exp(np.clip(exponent, -700.0, 700.0))

    def _discount_history(self) -> None:
        if self.discount_factor < 1.0:
            self.weights **= self.discount_factor

    def _stabilize_weights(self) -> None:
        maximum = float(np.max(self.weights))
        if np.isfinite(maximum) and maximum > 1e100:
            self.weights /= maximum

    def _apply_fixed_share(self) -> None:
        """Re-inject probability mass so prior regimes cannot lock an arm forever."""
        if self.fixed_share_alpha <= 0.0:
            return
        total = float(self.weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            self.weights = np.ones(self.n_arms, dtype=float)
            return
        normalized = self.weights / total
        self.weights = (
            (1.0 - self.fixed_share_alpha) * normalized
            + self.fixed_share_alpha / self.n_arms
        )

    def get_debug_state(self) -> dict:
        return {
            "algorithm": "EXP3",
            "gamma": float(self.gamma),
            "eta": float(self.eta),
            "exploration_epsilon": float(self.exploration_epsilon),
            "discount_factor": float(self.discount_factor),
            "fixed_share_alpha": float(self.fixed_share_alpha),
            "effective_memory_periods": float(1.0 / (1.0 - self.discount_factor))
            if self.discount_factor < 1.0
            else float("inf"),
            "weights": self.weights.astype(float).tolist(),
            "forced_uniform_exploration": bool(self.exploration_epsilon > 0.0),
            "aggregate_reward_clipping": True,
        }

    def _update_stats(self, arm: int, reward: float):
        self._bandit_update(int(arm), float(reward))

    def reset(self):
        super().reset()
        self.weights = np.ones(self.n_arms, dtype=float)


class JiaEXP3(MABAlgorithm):
    """Jia-style exponential weights for a vector of HT-IX estimates.

    The caller is responsible for putting each request-level reward on [0, 1]
    before it constructs the HT-IX estimate.  In particular, the aggregate
    estimate is intentionally not clipped here: HT-IX values may exceed one.
    """

    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        eta: float = None,
        T: int = 100,
    ):
        super().__init__(n_arms, seed)
        default_eta = np.sqrt(
            np.log(self.n_arms) / max(self.n_arms * max(int(T), 1), 1)
        )
        self.eta = float(eta) if eta is not None else float(default_eta)
        if not np.isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("JiaEXP3 eta must be a finite positive number.")
        self.log_weights = np.zeros(self.n_arms, dtype=float)

    def get_probs(self) -> np.ndarray:
        centered = self.log_weights - np.max(self.log_weights)
        weights = np.exp(centered)
        return weights / weights.sum()

    def update(self, arm: int, reward: float):
        self.t += 1
        self._update_stats(int(arm), float(reward))

    def update_all(self, rewards: dict):
        self.t += 1
        for arm, reward in rewards.items():
            self._update_stats(int(arm), float(reward))

    def _update_stats(self, arm: int, reward: float):
        if not np.isfinite(reward):
            return
        self.log_weights[arm] += self.eta * reward

    def get_debug_state(self) -> dict:
        return {
            "algorithm": "JiaEXP3",
            "eta": float(self.eta),
            "log_weights": self.log_weights.astype(float).tolist(),
            "forced_uniform_exploration": False,
            "aggregate_reward_clipping": False,
        }

    def reset(self):
        super().reset()
        self.log_weights = np.zeros(self.n_arms, dtype=float)


class DiscountedJiaEXP3(JiaEXP3):
    """Short-memory Jia-style exponential weights.

    Before each pricing-period update, the accumulated log weights are
    multiplied by ``discount_factor``.  This is the exponentially discounted
    version of the existing full-information Jia update, not a claim that the
    algorithm has the bandit-only EXP3.S feedback model.
    """

    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        eta: float = None,
        T: int = 100,
        discount_factor: float = 0.95,
    ):
        super().__init__(n_arms=n_arms, seed=seed, eta=eta, T=T)
        self.discount_factor = float(discount_factor)
        if not np.isfinite(self.discount_factor) or not 0.0 < self.discount_factor <= 1.0:
            raise ValueError(
                "DiscountedJiaEXP3 discount_factor must be finite and in (0, 1]."
            )

    def _discount_history(self) -> None:
        self.log_weights *= self.discount_factor

    def update(self, arm: int, reward: float):
        self.t += 1
        self._discount_history()
        self._update_stats(int(arm), float(reward))

    def update_all(self, rewards: dict):
        self.t += 1


        self._discount_history()
        for arm, reward in rewards.items():
            self._update_stats(int(arm), float(reward))

    def get_debug_state(self) -> dict:
        state = super().get_debug_state()
        state.update(
            {
                "algorithm": "DiscountedJiaEXP3",
                "discount_factor": float(self.discount_factor),
                "effective_memory_periods": float(1.0 / (1.0 - self.discount_factor))
                if self.discount_factor < 1.0
                else float("inf"),
            }
        )
        return state
