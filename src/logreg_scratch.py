# src/logreg_scratch.py
import os
import time
import numpy as np


class LogisticRegression:
    """
    Quant-Ready Logistic Regression Engine with:
    - Standardization (Learned mu/sd on train set)
    - Vectorized Mini-batch Gradient Descent
    - L2 (Ridge) Regularization
    - Training Latency Benchmarking (perf_counter)
    """

    def __init__(self, lr=0.1, l2=0.01, epochs=100, batch_size=32, seed=None):
        # Hyperparameters
        self.lr = float(lr)
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = seed  # can be None

        # Learned parameters / state
        self._w = None
        self._b = None
        self._mu = None
        self._sd = None
        self._loss_history = None
        self._train_time = None  # seconds (float)

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, X_train, y_train, log_every=10, model_path="artifacts/model.npz"):
        """
        Trains the model and benchmarks training latency.
        """
        X = np.array(X_train, dtype=np.float64, copy=True)
        y = np.array(y_train, dtype=np.float64, copy=True).reshape(-1)

        # Basic validation
        if X.ndim != 2:
            raise ValueError("X_train must be a 2D array (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y_train must be a 1D array (n_samples,).")
        if len(X) != len(y):
            raise ValueError("X_train and y_train must have the same number of rows.")
        if len(X) == 0:
            raise ValueError("X_train is empty.")

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.seed)

        # Start timer (includes standardization + training loop; consistent is what matters)
        t0 = time.perf_counter()

        # Standardize using training stats
        self._standardize_fit(X)
        Xs = self._standardize_apply(X)

        # Init params
        self._w = np.zeros(n_features, dtype=np.float64)
        self._b = 0.0
        self._loss_history = np.zeros(self.epochs, dtype=np.float64)

        # Mini-batch GD
        for epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = Xs[indices]
            y_shuffled = y[indices]

            epoch_loss_sum = 0.0

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                p = self._predict_proba_internal(X_batch)  # expects scaled
                batch_loss = self._log_loss(y_batch, p, w=self._w)
                epoch_loss_sum += batch_loss * len(y_batch)

                grad_w, grad_b = self._gradients(X_batch, y_batch, p, w=self._w)

                self._w -= self.lr * grad_w
                self._b -= self.lr * grad_b

            self._loss_history[epoch] = epoch_loss_sum / n_samples

            if log_every and (epoch % int(log_every) == 0):
                print(f"Epoch {epoch} | Avg Loss: {self._loss_history[epoch]:.4f}")

        self._train_time = time.perf_counter() - t0
        print(f"Training Complete | Latency: {self._train_time:.6f} seconds")

        if model_path:
            self.save(model_path)

        return self

    def predict_proba(self, X):
        """
        Returns P(y=1 | X). IMPORTANT: takes RAW X and standardizes internally.
        """
        self._check_is_fitted()
        X = np.array(X, dtype=np.float64)
        Xs = self._standardize_apply(X)
        return self._predict_proba_internal(Xs)

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        return (p >= float(threshold)).astype(np.int64)

    def save(self, model_path="artifacts/model.npz"):
        """
        Save parameters + training stats to NPZ.
        """
        self._check_is_fitted()

        dir_name = os.path.dirname(model_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # if model was loaded and never fit, keep train_time as NaN rather than crashing
        train_time = np.nan if self._train_time is None else float(self._train_time)

        np.savez(
            model_path,
            w=self._w,
            b=np.array(self._b, dtype=np.float64),
            mu=self._mu,
            sd=self._sd,
            loss_history=self._loss_history if self._loss_history is not None else np.array([]),
            train_time=np.array(train_time, dtype=np.float64),
            lr=np.array(self.lr, dtype=np.float64),
            l2=np.array(self.l2, dtype=np.float64),
            epochs=np.array(self.epochs, dtype=np.int64),
            batch_size=np.array(self.batch_size, dtype=np.int64),
            seed=np.array(-1 if self.seed is None else int(self.seed), dtype=np.int64),
        )
        return model_path

    @classmethod
    def load(cls, model_path="artifacts/model.npz"):
        """
        Load model from NPZ.
        Backwards-compatible: if some keys are missing, uses defaults.
        """
        data = np.load(model_path, allow_pickle=False)

        lr = float(data["lr"]) if "lr" in data else 0.1
        l2 = float(data["l2"]) if "l2" in data else 0.01
        epochs = int(data["epochs"]) if "epochs" in data else 100
        batch_size = int(data["batch_size"]) if "batch_size" in data else 32

        seed = None
        if "seed" in data:
            s = int(data["seed"])
            seed = None if s == -1 else s

        model = cls(lr=lr, l2=l2, epochs=epochs, batch_size=batch_size, seed=seed)

        model._w = data["w"]
        model._b = float(data["b"])
        model._mu = data["mu"]
        model._sd = data["sd"]

        model._loss_history = data["loss_history"] if "loss_history" in data else None

        if "train_time" in data:
            tt = float(data["train_time"])
            model._train_time = None if np.isnan(tt) else tt
        else:
            model._train_time = None

        return model

    def summary(self):
        print("=" * 40)
        print("QUANT-CORE LOGISTIC REGRESSION ENGINE")
        print("=" * 40)
        if self._w is None:
            print("Status: Not fitted")
        else:
            if self._train_time is not None:
                print(f"Latency:      {self._train_time:.6f}s")
            else:
                print("Latency:      (not recorded)")
            if self._loss_history is not None and len(self._loss_history) > 0:
                print(f"Final Loss:   {self._loss_history[-1]:.6f}")
            else:
                print("Final Loss:   (no history)")
            print(f"Features:     {len(self._w)}")
            print(f"Hyperparams:  LR={self.lr}, L2={self.l2}, epochs={self.epochs}, batch={self.batch_size}")
        print("=" * 40)

    # -------------------------
    # Properties (safe access)
    # -------------------------
    @property
    def w(self):
        self._check_is_fitted()
        return self._w

    @property
    def b(self):
        self._check_is_fitted()
        return self._b

    @property
    def loss_history(self):
        self._check_is_fitted()
        return self._loss_history

    @property
    def train_time(self):
        # train_time can be None if loaded from older artifact / not recorded
        return self._train_time

    # -------------------------
    # Internal helpers
    # -------------------------
    def _check_is_fitted(self):
        if self._w is None or self._mu is None or self._sd is None:
            raise RuntimeError("Model is not fitted (or not loaded) — missing parameters.")

    def _sigmoid(self, z):
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _predict_proba_internal(self, X_scaled):
        # expects standardized X
        return self._sigmoid(X_scaled @ self._w + self._b)

    def _log_loss(self, y, p, w=None):
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        data_loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        if w is not None:
            data_loss += (self.l2 / 2.0) * np.sum(w * w)
        return data_loss

    def _gradients(self, X, y, p, w=None):
        m = len(y)
        error = p - y
        grad_w = (X.T @ error) / m
        if w is not None:
            grad_w += self.l2 * w
        grad_b = float(np.mean(error))
        return grad_w, grad_b

    def _standardize_fit(self, X):
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0) + 1e-12

    def _standardize_apply(self, X):
        return (X - self._mu) / self._sd