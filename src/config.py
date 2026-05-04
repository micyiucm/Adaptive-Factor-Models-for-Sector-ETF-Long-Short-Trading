from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Paths
    data_path: str = "data/etf_returns.csv"
    config_path: str = "data/walk_forward_config.csv"
    pca_fixed_res_path: str = "data/pca_residuals_fixed_etf.csv"
    pca_rolling_res_path: str = "data/pca_residuals_rolling_etf.csv"

    # Strategy parameters
    cost_bps: float = 0.0002
    n_factors: int = 3
    rolling_window: int = 60

    # Kalman grid search
    kalman_deltas: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    kalman_ves: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])

    # Signal defaults
    vol_halflife: int = 20
    mom_lookback: int = 20
    z_clip: float = 5.0

    # Portfolio defaults
    long_q: float = 0.7
    short_q: float = 0.3
    rebalance: str = "W-FRI"
    beta_neutral: bool = True
    beta_window: int = 60
    market_col: str = "SPY"
    trade_market: bool = False
    min_beta_obs: int = 10
