import numpy as np


class LinearModel:
    def __init__(self, inp_dim: int, out_dim: int = 1) -> None:
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.W = np.random.randn(inp_dim, out_dim) * (1 / np.sqrt(inp_dim))
        self.b = np.zeros(out_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def __repr__(self) -> str:
        return f'LinearModel({self.inp_dim}, {self.out_dim})'


class LogisticRegression(LinearModel):
    def __init__(self, inp_dim: int) -> None:
        super().__init__(inp_dim, 1)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x @ self.W + self.b)


class SoftmaxRegression(LinearModel):
    def __init__(self, inp_dim: int, out_dim: int = 10) -> None:

        super().__init__(inp_dim, out_dim)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # substracting 'max' for stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        logits = np.dot(x, self.W) + self.b
        dist = self._softmax(logits)
        return dist
    