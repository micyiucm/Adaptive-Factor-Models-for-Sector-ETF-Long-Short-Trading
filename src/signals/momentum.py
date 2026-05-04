import numpy as np
import pandas as pd

from src.signals.base import Signal


class MomentumSignal(Signal):
    """
    Residual momentum signal: L-day diff of EWMA-standardized residuals.
    Positive score = residual has drifted upward (long conviction).
    """

    higher_is_long = True  # High score = long conviction (momentum)

    def __init__(self, lookback: int = 20, vol_halflife: int = 20, z_clip: float = 5.0):
        self.lookback = lookback
        self.vol_halflife = vol_halflife
        self.z_clip = z_clip

    def generate(self, residuals: pd.DataFrame) -> pd.DataFrame:
        vol = residuals.ewm(halflife=self.vol_halflife, adjust=False).std()
        vol = vol.clip(lower=1e-8)
        Z = (residuals / vol).replace([np.inf, -np.inf], np.nan).clip(-self.z_clip, self.z_clip)
        return Z.diff(self.lookback)
