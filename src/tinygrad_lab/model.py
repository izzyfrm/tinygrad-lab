"""A tiny two-layer neural network implemented with NumPy only."""
import numpy as np


class NeuralNetwork:
    def __init__(self, input_dim=2, hidden_dim=32, seed=42):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = rng.normal(0, np.sqrt(2 / hidden_dim), (hidden_dim, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, x):
        self.x = np.asarray(x, dtype=float)
        self.z1 = self.x @ self.w1 + self.b1
        self.a1 = np.maximum(self.z1, 0)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.p = 1 / (1 + np.exp(-np.clip(self.z2, -50, 50)))
        return self.p

    def train_step(self, x, y, lr=0.05):
        y = np.asarray(y).reshape(-1, 1)
        p = self.forward(x)
        eps = 1e-10
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        n = len(x)
        dz2 = (p - y) / n
        dw2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        dz1 = (dz2 @ self.w2.T) * (self.z1 > 0)
        dw1 = self.x.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        return float(loss)

    def predict(self, x):
        return (self.forward(x) >= 0.5).astype(int).ravel()
