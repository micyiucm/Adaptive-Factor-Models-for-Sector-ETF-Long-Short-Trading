import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.models.base import FactorModel


class FixedPCAModel(FactorModel):
    """PCA with fixed loadings: fit on train, apply to test."""

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self._pca = None

    def fit(self, train_returns: pd.DataFrame) -> "FixedPCAModel":
        k = min(self.n_components, train_returns.shape[1])
        self._pca = PCA(n_components=k)
        self._pca.fit(train_returns)
        return self

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        factors = self._pca.transform(returns)
        reconstructed = np.dot(factors, self._pca.components_) + self._pca.mean_
        residuals = returns.values - reconstructed
        return pd.DataFrame(residuals, index=returns.index, columns=returns.columns)


class RollingPCAModel(FactorModel):
    """PCA with rolling window: refits at each time step."""

    def __init__(self, n_components: int = 3, window: int = 60):
        self.n_components = n_components
        self.window = window

    def fit(self, train_returns: pd.DataFrame) -> "RollingPCAModel":
        # Rolling model refits every step; no stored state from training.
        return self

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        data = returns.values
        n_samples, n_assets = data.shape
        k = min(self.n_components, n_assets)

        residuals = np.full_like(data, np.nan, dtype=float)

        for t in range(self.window, n_samples):
            window_data = data[t - self.window : t]
            pca = PCA(n_components=k)
            pca.fit(window_data)
            current_return = data[t].reshape(1, -1)
            factors = pca.transform(current_return)
            expected_return = np.dot(factors, pca.components_) + pca.mean_
            residuals[t] = current_return - expected_return

        return pd.DataFrame(residuals, index=returns.index, columns=returns.columns)

    def fit_transform(self, train_returns: pd.DataFrame, test_returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Override: run rolling PCA on concat(train, test) so the rolling window
        has history, then return test-period residuals only.
        """
        self.fit(train_returns)

        if test_returns is None:
            return self.transform(train_returns)

        combined = pd.concat([train_returns, test_returns])
        combined_residuals = self.transform(combined)
        return combined_residuals.loc[test_returns.index]
