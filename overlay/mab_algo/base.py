import numpy as np


class MABAlgorithm:
    def __init__(self, n_arms: int, seed: int = 42):
        self.n_arms = int(n_arms)
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def get_probs(self) -> np.ndarray:
        raise NotImplementedError

    def sample_arm(self) -> int:
        probs = np.asarray(self.get_probs(), dtype=float)
        probs = probs / probs.sum()
        return int(self.rng.choice(self.n_arms, p=probs))

    def select_arm(self) -> int:
        return int(np.argmax(self.get_probs()))

    def update(self, arm: int, reward: float):
        raise NotImplementedError

    def update_all(self, rewards: dict):
        self.t += 1
        for arm, reward in rewards.items():
            self._update_stats(int(arm), float(reward))

    def _update_stats(self, arm: int, reward: float):
        raise NotImplementedError

    def reset(self):
        self.t = 0
