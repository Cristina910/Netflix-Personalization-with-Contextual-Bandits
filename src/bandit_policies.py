from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class Action:
    index: int
    x: np.ndarray  # feature vector for item/arm

class Policy:
    def select(self, context_matrix: np.ndarray) -> int:
        raise NotImplementedError
    def update(self, action: int, reward: float, x: np.ndarray):
        pass

class EpsilonGreedy(Policy):
    def __init__(self, n_actions: int, epsilon: float = 0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.counts = np.zeros(n_actions)
        self.values = np.zeros(n_actions)

    def select(self, context_matrix: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.values))

    def update(self, action: int, reward: float, x: np.ndarray):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] = value + (reward - value) / n

class LinUCB(Policy):
    """Linear UCB with shared d across arms (per-arm A, b)."""
    def __init__(self, n_actions: int, d: int, alpha: float = 1.0):
        self.n_actions = n_actions
        self.d = d
        self.alpha = alpha
        self.A = [np.eye(d) for _ in range(n_actions)]  # design matrices
        self.b = [np.zeros((d, 1)) for _ in range(n_actions)]

    def _theta(self, a: int):
        A_inv = np.linalg.inv(self.A[a])
        return A_inv @ self.b[a], A_inv

    def select(self, context_matrix: np.ndarray) -> int:
        p = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            x = context_matrix[a].reshape(-1,1)
            theta, A_inv = self._theta(a)
            mu = float((theta.T @ x))
            ucb = self.alpha * float(np.sqrt(x.T @ A_inv @ x))
            p[a] = mu + ucb
        return int(np.argmax(p))

    def update(self, action: int, reward: float, x: np.ndarray):
        x = x.reshape(-1,1)
        self.A[action] += x @ x.T
        self.b[action] += reward * x

class ThompsonSamplingLinear(Policy):
    """Bayesian linear bandit with Gaussian prior per arm."""
    def __init__(self, n_actions: int, d: int, v: float = 0.1):
        self.n_actions = n_actions
        self.d = d
        self.v = v
        self.B = [np.eye(d) for _ in range(n_actions)]  # precision
        self.f = [np.zeros((d,1)) for _ in range(n_actions)]

    def select(self, context_matrix: np.ndarray) -> int:
        samples = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            B_inv = np.linalg.inv(self.B[a])
            mu = B_inv @ self.f[a]
            theta_sample = np.random.multivariate_normal(mu.flatten(), self.v**2 * B_inv)
            samples[a] = float(theta_sample @ context_matrix[a])
        return int(np.argmax(samples))

    def update(self, action: int, reward: float, x: np.ndarray):
        x = x.reshape(-1,1)
        self.B[action] += x @ x.T
        self.f[action] += reward * x
