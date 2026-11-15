"""A fixed-allocation explore-then-commit policy for conventional A/B/n tests."""

from __future__ import annotations

import numpy as np

from .base import MABAlgorithm


class ExploreThenCommit(MABAlgorithm):
    """Explore uniformly for a fixed number of feedback batches, then commit.

    The policy intentionally has no adaptive allocation during exploration: every
    arm receives the same probability.  It accumulates the controller-provided
    arm estimates, fixes the empirically best arm after ``exploration_updates``
    valid feedback batches, and keeps that arm for the rest of the horizon.
    """

    def __init__(
        self,
        n_arms: int,
        seed: int = 42,
        exploration_updates: int = 1,
    ):
        super().__init__(n_arms=n_arms, seed=seed)
        self.exploration_updates = int(exploration_updates)
        if self.exploration_updates < 1:
            raise ValueError("exploration_updates must be at least one.")
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self.reward_counts = np.zeros(self.n_arms, dtype=float)
        self.committed_arm: int | None = None

    def _commit_if_ready(self) -> None:
        if self.committed_arm is not None or self.t < self.exploration_updates:
            return
        means = np.divide(
            self.reward_sums,
            self.reward_counts,
            out=np.full(self.n_arms, -np.inf, dtype=float),
            where=self.reward_counts > 0.0,
        )

        self.committed_arm = int(np.argmax(means))

    def get_probs(self) -> np.ndarray:
        self._commit_if_ready()
        if self.committed_arm is None:
            return np.full(self.n_arms, 1.0 / self.n_arms, dtype=float)
        probs = np.zeros(self.n_arms, dtype=float)
        probs[self.committed_arm] = 1.0
        return probs

    def update(self, arm: int, reward: float):
        if self.committed_arm is not None:
            return
        self.t += 1
        self._record_reward(int(arm), float(reward))
        self._commit_if_ready()

    def update_all(self, rewards: dict):
        if self.committed_arm is not None:
            return
        self.t += 1
        for arm, reward in rewards.items():
            self._record_reward(int(arm), float(reward))
        self._commit_if_ready()

    def _record_reward(self, arm: int, reward: float) -> None:
        if not 0 <= arm < self.n_arms or not np.isfinite(reward):
            return
        self.reward_sums[arm] += float(reward)
        self.reward_counts[arm] += 1.0

    def get_debug_state(self) -> dict:
        means = np.divide(
            self.reward_sums,
            self.reward_counts,
            out=np.full(self.n_arms, np.nan, dtype=float),
            where=self.reward_counts > 0.0,
        )
        return {
            "policy_mode": (
                "explore_then_commit_exploitation"
                if self.committed_arm is not None
                else "explore_then_commit_exploration"
            ),
            "exploration_updates": int(self.exploration_updates),
            "observed_updates": int(self.t),
            "committed_arm": self.committed_arm,
            "exploration_reward_sums": self.reward_sums.astype(float).tolist(),
            "exploration_reward_counts": self.reward_counts.astype(float).tolist(),
            "exploration_mean_rewards": means.astype(float).tolist(),
        }

    def reset(self):
        super().reset()
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self.reward_counts = np.zeros(self.n_arms, dtype=float)
        self.committed_arm = None
