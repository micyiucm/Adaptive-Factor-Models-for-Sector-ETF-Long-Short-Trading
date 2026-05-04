import numpy as np
import pandas as pd

from src.signals.base import Signal


class MeanReversionSignal(Signal):
    """
    Mean-reversion signal: EWMA-standardized residual level.
    Low score = residual is low (expect reversion upward -> long).

    Note: This signal uses reverse=True in the portfolio constructor
    (low score -> long, high score -> short).
    """

    higher_is_long = False  # Low score = long conviction (mean-reversion)

    def __init__(self, vol_halflife: int = 20, z_clip: float = 5.0):
        self.vol_halflife = vol_halflife
        self.z_clip = z_clip

    def generate(self, residuals: pd.DataFrame) -> pd.DataFrame:
        vol = residuals.ewm(halflife=self.vol_halflife, adjust=False).std()
        vol = vol.clip(lower=1e-8)
        Z = (residuals / vol).replace([np.inf, -np.inf], np.nan).clip(-self.z_clip, self.z_clip)
        return Z
