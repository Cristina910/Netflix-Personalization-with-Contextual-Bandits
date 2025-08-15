from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

@dataclass
class Environment:
    user_dim: int
    item_dim: int
    n_users: int
    n_items: int
    noise: float = 0.1
    seed: int = 42

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # Hidden user preference vectors (d)
        self.user_theta = rng.normal(0, 1, size=(self.n_users, self.item_dim))
        # Item feature matrix (K x d)
        self.item_X = rng.normal(0, 1, size=(self.n_items, self.item_dim))
        # Normalize items
        self.item_X = self.item_X / np.linalg.norm(self.item_X, axis=1, keepdims=True)

    def step(self, user_idx: int, action_idx: int) -> tuple[float, float]:
        """Return (click, expected_click_prob)."""
        theta = self.user_theta[user_idx]
        x = self.item_X[action_idx]
        p = sigmoid(theta @ x + self.noise * np.random.randn())
        click = float(np.random.rand() < p)
        return click, p

    def best_action(self, user_idx: int) -> int:
        theta = self.user_theta[user_idx]
        scores = self.item_X @ theta
        return int(np.argmax(scores))
