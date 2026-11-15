import numpy as np

from .base import MABAlgorithm


class ThompsonSampling(MABAlgorithm):
    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        mc_draws: int = 2000,
    ):
        super().__init__(n_arms, seed)
        self.mc_draws = max(1, int(mc_draws))
        self.alpha = np.ones(self.n_arms, dtype=float)
        self.beta = np.ones(self.n_arms, dtype=float)

    def get_probs(self) -> np.ndarray:
        posterior_samples = self.rng.beta(
            self.alpha[None, :],
            self.beta[None, :],
            size=(self.mc_draws, self.n_arms),
        )
        winners = np.argmax(posterior_samples, axis=1)
        counts = np.bincount(winners, minlength=self.n_arms).astype(float)
        probs = counts / counts.sum()
        return probs

    def get_debug_state(self) -> dict:
        return {
            "alpha": self.alpha.astype(float).tolist(),
            "beta": self.beta.astype(float).tolist(),
            "posterior_mean": (self.alpha / (self.alpha + self.beta)).astype(float).tolist(),
            "mc_draws": int(self.mc_draws),
            "t": int(self.t),
        }

    def update(self, arm: int, reward: float):
        self.t += 1
        self._update_stats(int(arm), float(reward))

    def update_all(self, rewards: dict):
        self.t += 1
        for arm, reward in rewards.items():
            self._update_stats(int(arm), float(reward))

    def _update_stats(self, arm: int, reward: float):
        clipped_reward = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm] += clipped_reward
        self.beta[arm] += 1.0 - clipped_reward

    def reset(self):
        super().reset()
        self.alpha = np.ones(self.n_arms, dtype=float)
        self.beta = np.ones(self.n_arms, dtype=float)


class GaussianThompsonSampling(MABAlgorithm):
    """Normal-Inverse-Gamma Thompson sampling for continuous rewards.

    Each arm has a normal likelihood with unknown mean and variance.  The
    Normal-Inverse-Gamma prior yields posterior draws of a mean without
    requiring rewards to be Bernoulli or fractional-Beta evidence.
    """

    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        mc_draws: int = 2000,
        prior_mean: float = 0.10,
        prior_kappa: float = 1.0,
        prior_alpha: float = 2.0,
        prior_beta: float = 0.01,
    ):
        super().__init__(n_arms, seed)
        if prior_kappa <= 0.0 or prior_alpha <= 0.0 or prior_beta <= 0.0:
            raise ValueError("Gaussian TS prior_kappa, prior_alpha, and prior_beta must be positive.")
        self.mc_draws = max(1, int(mc_draws))
        self.prior_mean = float(prior_mean)
        self.prior_kappa = float(prior_kappa)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.counts = np.zeros(self.n_arms, dtype=float)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self.reward_square_sums = np.zeros(self.n_arms, dtype=float)

    def _posterior_parameters(self):
        counts = self.counts
        kappa = self.prior_kappa + counts
        sample_mean = np.divide(
            self.reward_sums,
            counts,
            out=np.zeros(self.n_arms, dtype=float),
            where=counts > 0.0,
        )
        centered_sum_squares = np.maximum(
            self.reward_square_sums - counts * np.square(sample_mean),
            0.0,
        )
        mean = (self.prior_kappa * self.prior_mean + self.reward_sums) / kappa
        alpha = self.prior_alpha + 0.5 * counts
        beta = (
            self.prior_beta
            + 0.5 * centered_sum_squares
            + 0.5
            * self.prior_kappa
            * counts
            * np.square(sample_mean - self.prior_mean)
            / kappa
        )
        return mean, kappa, alpha, np.maximum(beta, np.finfo(float).tiny)

    def _draw_posterior_means(self, draws: int) -> np.ndarray:
        mean, kappa, alpha, beta = self._posterior_parameters()
        precision = self.rng.gamma(
            shape=alpha[None, :],
            scale=1.0 / beta[None, :],
            size=(draws, self.n_arms),
        )
        variance = 1.0 / np.maximum(precision, np.finfo(float).tiny)
        return self.rng.normal(
            loc=mean[None, :],
            scale=np.sqrt(variance / kappa[None, :]),
        )

    def get_probs(self) -> np.ndarray:
        posterior_samples = self._draw_posterior_means(self.mc_draws)
        winners = np.argmax(posterior_samples, axis=1)
        counts = np.bincount(winners, minlength=self.n_arms).astype(float)
        return counts / counts.sum()

    def get_debug_state(self) -> dict:
        mean, kappa, alpha, beta = self._posterior_parameters()
        predictive_sd = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        return {
            "posterior_family": "Normal-Inverse-Gamma",
            "posterior_mean": mean.astype(float).tolist(),
            "posterior_predictive_sd": predictive_sd.astype(float).tolist(),
            "counts": self.counts.astype(float).tolist(),
            "reward_sums": self.reward_sums.astype(float).tolist(),
            "reward_square_sums": self.reward_square_sums.astype(float).tolist(),
            "prior_mean": float(self.prior_mean),
            "prior_kappa": float(self.prior_kappa),
            "prior_alpha": float(self.prior_alpha),
            "prior_beta": float(self.prior_beta),
            "mc_draws": int(self.mc_draws),
            "t": int(self.t),
        }

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
        self.counts[arm] += 1.0
        self.reward_sums[arm] += reward
        self.reward_square_sums[arm] += reward * reward

    def reset(self):
        super().reset()
        self.counts = np.zeros(self.n_arms, dtype=float)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self.reward_square_sums = np.zeros(self.n_arms, dtype=float)


class GridBetaThompsonSampling(ThompsonSampling):
    """Thompson sampling with one posterior draw for every assigned grid cell.

    This class implements fractional Beta evidence at the grid level.  A grid
    with scaled reward ``r`` in ``[0, 1]`` contributes ``r`` to alpha and
    ``1 - r`` to beta.  Multiple informative grids assigned to the same arm
    contribute independently within a pricing period.
    """

    gridwise_sampling = True
    grid_beta_evidence = True

    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        mc_draws: int = 2000,
    ):
        super().__init__(n_arms=n_arms, seed=seed, mc_draws=mc_draws)


        self._policy_rng = np.random.default_rng(int(seed) + 1_000_003)
        self.last_grid_reward_sums = np.zeros(self.n_arms, dtype=float)
        self.last_grid_trial_counts = np.zeros(self.n_arms, dtype=float)

    def get_probs(self) -> np.ndarray:
        """Estimate posterior winner probabilities for logging only."""
        posterior_samples = self._policy_rng.beta(
            self.alpha[None, :],
            self.beta[None, :],
            size=(self.mc_draws, self.n_arms),
        )
        winners = np.argmax(posterior_samples, axis=1)
        counts = np.bincount(winners, minlength=self.n_arms).astype(float)
        return counts / counts.sum()

    def sample_arm(self) -> int:
        """Make one independent posterior draw and select its winning arm."""
        posterior_sample = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(posterior_sample))

    def update_grid_evidence(self, reward_sums, trial_counts) -> None:
        """Apply the screenshot-style fractional-Beta grid evidence update."""
        reward_sums = np.asarray(reward_sums, dtype=float)
        trial_counts = np.asarray(trial_counts, dtype=float)
        if reward_sums.shape != (self.n_arms,) or trial_counts.shape != (self.n_arms,):
            raise ValueError("Grid evidence must provide one sum and count per arm.")
        if np.any(~np.isfinite(reward_sums)) or np.any(~np.isfinite(trial_counts)):
            raise ValueError("Grid evidence must be finite.")
        if np.any(trial_counts < 0.0):
            raise ValueError("Grid trial counts must be non-negative.")
        if np.any(reward_sums < -1e-9) or np.any(reward_sums > trial_counts + 1e-9):
            raise ValueError("Each grid reward sum must lie between zero and its trial count.")

        reward_sums = np.clip(reward_sums, 0.0, trial_counts)
        self.t += 1
        self.alpha += reward_sums
        self.beta += trial_counts - reward_sums
        self.last_grid_reward_sums = reward_sums.copy()
        self.last_grid_trial_counts = trial_counts.copy()

    def reset(self):
        super().reset()
        self.last_grid_reward_sums = np.zeros(self.n_arms, dtype=float)
        self.last_grid_trial_counts = np.zeros(self.n_arms, dtype=float)

    def get_debug_state(self) -> dict:
        state = super().get_debug_state()
        state.update(
            {
                "gridwise_sampling": True,
                "grid_beta_evidence": True,
                "last_grid_reward_sums": self.last_grid_reward_sums.astype(float).tolist(),
                "last_grid_trial_counts": self.last_grid_trial_counts.astype(float).tolist(),
            }
        )
        return state
