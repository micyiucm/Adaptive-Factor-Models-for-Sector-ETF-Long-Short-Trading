import jax
import jax.numpy as jnp
from jax import jit, vmap
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from src.models.base import FactorModel

# Enable 64-bit precision for Kalman stability
jax.config.update("jax_enable_x64", True)


@jit
def run_kalman_static(Y, X, delta, ve):
    """
    JIT-compiled Kalman Filter engine.

    Parameters:
        Y: Asset returns matrix, shape (T, N)
        X: Factor matrix, shape (T, K)
        delta: State transition noise
        ve: Observation noise variance

    Returns:
        residuals_matrix: Shape (T, N)
    """
    T, N = Y.shape
    K = X.shape[1]

    initial_beta = jnp.zeros(K)
    initial_P = jnp.eye(K)

    def kalman_step(carry, data_t):
        beta_prev, P_prev = carry
        y_t, x_t = data_t

        beta_pred = beta_prev
        P_pred = P_prev + delta * jnp.eye(K)

        y_pred = jnp.dot(x_t, beta_pred)
        e_t = y_t - y_pred

        Q_t = jnp.dot(jnp.dot(x_t, P_pred), x_t.T) + ve
        K_gain = jnp.dot(P_pred, x_t.T) / Q_t

        beta_new = beta_pred + K_gain * e_t
        P_new = P_pred - jnp.dot(jnp.outer(K_gain, x_t), P_pred)

        return (beta_new, P_new), e_t

    def scan_single_asset(y_series, x_matrix):
        _, residuals = jax.lax.scan(
            kalman_step,
            (initial_beta, initial_P),
            (y_series, x_matrix),
        )
        return residuals

    residuals_matrix = vmap(scan_single_asset, in_axes=(1, None))(Y, X)
    return residuals_matrix.T


class KalmanFactorModel(FactorModel):
    """Kalman filter with time-varying betas, using PCA factors as inputs."""

    def __init__(self, n_components: int = 3, delta: float = 1e-3, ve: float = 1e-3):
        self.n_components = n_components
        self.delta = delta
        self.ve = ve
        self._pca = None

    def fit(self, train_returns: pd.DataFrame) -> "KalmanFactorModel":
        k = min(self.n_components, train_returns.shape[1])
        self._pca = PCA(n_components=k)
        self._pca.fit(train_returns)
        return self

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Run Kalman filter on returns using fitted PCA factors."""
        factors = self._pca.transform(returns)
        factors_jax = jnp.array(factors)
        centered = returns.values - self._pca.mean_
        Y = jnp.array(centered)

        residuals_matrix = run_kalman_static(Y, factors_jax, self.delta, self.ve)

        return pd.DataFrame(
            np.array(residuals_matrix),
            index=returns.index,
            columns=returns.columns,
        )

    def fit_transform(self, train_returns: pd.DataFrame, test_returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Override: fit PCA on train, run Kalman on concat(train, test) for warm-up,
        return test-period residuals only.
        """
        self.fit(train_returns)

        if test_returns is None:
            return self.transform(train_returns)

        combined = pd.concat([train_returns, test_returns])
        combined_residuals = self.transform(combined)
        return combined_residuals.loc[test_returns.index]
