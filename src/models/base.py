from abc import ABC, abstractmethod
import pandas as pd


class FactorModel(ABC):
    """Abstract base class for factor models that produce residuals."""

    @abstractmethod
    def fit(self, train_returns: pd.DataFrame) -> "FactorModel":
        """
        Fit factor model on training data.

        Parameters:
            train_returns: DataFrame of log returns (T_train x N_assets)
        Returns:
            self (for method chaining)
        """
        ...

    @abstractmethod
    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate residuals for given returns using fitted model.

        Parameters:
            returns: DataFrame of log returns (T x N_assets)
        Returns:
            DataFrame of residuals (same shape as input)
        """
        ...

    def fit_transform(self, train_returns: pd.DataFrame, test_returns: pd.DataFrame = None) -> pd.DataFrame:
        """Fit on train, transform test (or train if test is None)."""
        self.fit(train_returns)
        return self.transform(test_returns if test_returns is not None else train_returns)
