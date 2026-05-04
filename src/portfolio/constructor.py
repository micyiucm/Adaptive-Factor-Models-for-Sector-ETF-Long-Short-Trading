import numpy as np
import pandas as pd


class PortfolioConstructor:
    """
    Converts signal scores to portfolio weights.
    Applies quantile-based long/short selection and optional beta-neutral projection.
    """

    def __init__(
        self,
        long_q: float = 0.7,
        short_q: float = 0.3,
        rebalance: str = "W-FRI",
        beta_neutral: bool = True,
        beta_window: int = 60,
        market_col: str = "SPY",
        trade_market: bool = False,
        min_beta_obs: int = 10,
    ):
        self.long_q = float(long_q)
        self.short_q = float(short_q)
        self.rebalance = rebalance
        self.beta_neutral = bool(beta_neutral)
        self.beta_window = int(beta_window)
        self.market_col = market_col
        self.trade_market = bool(trade_market)
        self.min_beta_obs = int(min_beta_obs)

    @staticmethod
    def _project_out_constraints(w: np.ndarray, betas: np.ndarray) -> np.ndarray:
        """
        Project weights to satisfy dollar-neutrality and beta-neutrality.
        w_adj = w - A (A^T A)^+ A^T w,  A = [1, beta]
        """
        n = w.shape[0]
        A = np.column_stack([np.ones(n), betas])
        inv_ATA = np.linalg.pinv(A.T @ A)
        return w - A @ (inv_ATA @ (A.T @ w))

    def construct(self, scores: pd.DataFrame, returns: pd.DataFrame, higher_is_long: bool = True) -> pd.DataFrame:
        """
        Build target weight matrix from signal scores.

        Parameters:
            scores: DataFrame of signal scores (T x N).
            returns: DataFrame of asset returns (T x N, includes market for beta calc).
            higher_is_long: If True, high score = long. If False, low score = long.

        Returns:
            DataFrame of target weights (T x N_tradable), populated on rebalance dates only.
        """
        # Determine tradable columns (exclude market if configured)
        tradable_cols = list(scores.columns)
        if (not self.trade_market) and (self.market_col in tradable_cols):
            tradable_cols.remove(self.market_col)

        scores_tr = scores[tradable_cols]
        returns_tr = returns[tradable_cols] if all(c in returns.columns for c in tradable_cols) else returns.reindex(columns=tradable_cols)

        # Market for beta estimation
        market = returns[self.market_col] if self.market_col in returns.columns else None

        # Rebalance dates
        reb_dates = scores_tr.resample(self.rebalance).last().index
        reb_dates = reb_dates.intersection(scores_tr.index)

        target_w = pd.DataFrame(0.0, index=scores_tr.index, columns=tradable_cols)

        for dt in reb_dates:
            score = scores_tr.loc[dt].dropna()
            if score.empty:
                continue

            ranks = score.rank(pct=True)

            if higher_is_long:
                longs = ranks[ranks >= self.long_q].index
                shorts = ranks[ranks <= self.short_q].index
            else:
                # Mean-reversion: low score -> long, high score -> short
                longs = ranks[ranks <= self.short_q].index
                shorts = ranks[ranks >= self.long_q].index

            if len(longs) == 0 or len(shorts) == 0:
                continue

            w = pd.Series(0.0, index=tradable_cols)
            w.loc[longs] = 1.0 / len(longs)
            w.loc[shorts] = -1.0 / len(shorts)

            # Beta-neutral projection
            if self.beta_neutral and market is not None:
                idx = returns_tr.index.get_loc(dt)
                start = max(0, idx - self.beta_window)

                if idx - start >= self.min_beta_obs:
                    win_r = returns_tr.iloc[start:idx]
                    win_m = market.iloc[start:idx]
                    var_m = win_m.var()

                    if var_m > 0:
                        covs = win_r.apply(lambda s: s.cov(win_m), axis=0)
                        betas_full = (covs / var_m).reindex(tradable_cols).fillna(0.0).values

                        w_arr = w.values
                        active = (w_arr != 0.0)

                        if active.sum() >= 3:
                            w_act = w_arr[active]
                            b_act = betas_full[active]
                            w_act_adj = self._project_out_constraints(w_act, b_act)

                            w_new = np.zeros_like(w_arr)
                            w_new[active] = w_act_adj
                            w = pd.Series(w_new, index=tradable_cols)

            # Renormalize to unit gross exposure
            gross = w.abs().sum()
            if gross > 0:
                w = w / gross
                target_w.loc[dt] = w

        return target_w
