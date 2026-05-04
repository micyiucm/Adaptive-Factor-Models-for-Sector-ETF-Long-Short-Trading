from abc import ABC, abstractmethod
import pandas as pd


class Signal(ABC):
    """Abstract base class for trading signals derived from residuals.

    Subclasses must set `higher_is_long`:
        True  = high score -> long (e.g. momentum)
        False = low score -> long (e.g. mean-reversion)
    """

    higher_is_long: bool = True

    @abstractmethod
    def generate(self, residuals: pd.DataFrame) -> pd.DataFrame:
        """
        Generate raw signal scores from residuals.

        Parameters:
            residuals: DataFrame of factor model residuals (T x N_assets)
        Returns:
            DataFrame of signal scores (same shape)
        """
        ...
