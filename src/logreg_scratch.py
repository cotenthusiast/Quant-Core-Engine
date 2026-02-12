import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w, b): 
    z = X @ w + b
    return sigmoid(z)

def log_loss(y, p, l2=0.0, w=None):
    p = np.clip(p, 1e-15, 1 - 1e-15)  # Clip p to avoid log(0)
    loss = - ( y * np.log(p) + (1 - y) * np.log(1 - p) )
    if w is not None:
        loss += (l2 / 2) * np.sum(w**2) # L2 regularization term return loss.mean()
    return loss.mean()

def gradients(X, y, p, l2=0.0, w=None):
    e = p - y
    grad_W = (X.T @ e) / len(y) + l2*w
    grad_b = e.mean()
    return grad_W, grad_b

