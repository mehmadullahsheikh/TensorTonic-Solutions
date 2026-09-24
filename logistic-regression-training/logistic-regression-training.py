import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 1000
) -> tuple[np.ndarray, float]:

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N, D = X.shape

    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        # Forward pass
        logits = X @ w + b
        preds = _sigmoid(logits)

        # Gradients
        grad_w = (X.T @ (preds - y)) / N
        grad_b = np.mean(preds - y)

        # Gradient descent
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b