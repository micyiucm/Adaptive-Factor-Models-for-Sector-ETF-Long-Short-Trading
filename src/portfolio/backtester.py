import numpy as np
import pandas as pd


class Backtester:
    """
    Pure execution engine: target weights + returns -> PnL.
    Handles forward-fill, execution lag, transaction costs, and PnL arithmetic.
    """

    def __init__(self, cost_bps: float = 0.0):
        self.cost = float(cost_bps)

    def run(
        self,
        target_weights: pd.DataFrame,
        returns: pd.DataFrame,
        return_components: bool = False,
    ):
        """
        Simulate execution of a weight schedule.

        Parameters:
            target_weights: DataFrame of target weights (T x N).
                           Non-zero only on rebalance dates; ffilled internally.
            returns: DataFrame of log returns (T x N), aligned with weights.
            return_components: If True, return (net_pnl, components_dict)

        Returns:
            net_pnl: Series of daily net log returns
            components (optional): dict with gross_pnl, net_pnl, turnover, costs
        """
        # Align columns
        common_cols = target_weights.columns.intersection(returns.columns)
        weights = target_weights[common_cols]
        rets = returns[common_cols]

        # Forward-fill between rebalances + execution lag
        positions = weights.ffill().fillna(0.0)
        positions = positions.shift(1).fillna(0.0)

        # PnL calculation
        simple_ret = np.expm1(rets)
        gross_simple = (positions * simple_ret).sum(axis=1)

        # Turnover and costs
        turnover = 0.5 * positions.diff().abs().sum(axis=1).fillna(0.0)
        costs = turnover * self.cost

        net_simple = (gross_simple - costs).clip(lower=-0.999999)

        gross_log = np.log1p(gross_simple.clip(lower=-0.999999))
        net_log = np.log1p(net_simple)

        if return_components:
            components = {
                "gross_pnl": gross_log,
                "net_pnl": net_log,
                "turnover": turnover,
                "costs": costs,
            }
            return net_log, components

        return net_log
