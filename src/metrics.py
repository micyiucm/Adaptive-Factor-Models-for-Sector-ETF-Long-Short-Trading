import numpy as np
import pandas as pd


def calc_sharpe(log_returns: pd.Series) -> float:
    """Annualized Sharpe ratio from daily log returns."""
    if log_returns.std() == 0:
        return 0.0
    return np.sqrt(252) * log_returns.mean() / log_returns.std()


def calc_max_drawdown(wealth: pd.Series) -> float:
    """Maximum drawdown from cumulative wealth series."""
    peaks = wealth.cummax()
    drawdowns = (wealth - peaks) / peaks
    return drawdowns.min()


def calc_cagr(log_returns: pd.Series) -> float:
    """Compound Annual Growth Rate from log returns."""
    years = len(log_returns) / 252
    if years == 0:
        return 0.0
    return np.exp(log_returns.sum() / years) - 1


def calc_var_es(simple_returns: pd.Series, confidence: float = 0.95) -> tuple:
    """Value at Risk and Expected Shortfall on simple returns."""
    if simple_returns.empty:
        return 0.0, 0.0
    var = simple_returns.quantile(1 - confidence)
    tail_losses = simple_returns[simple_returns <= var]
    es = tail_losses.mean() if not tail_losses.empty else var
    return var, es


def calc_metrics(net_pnl: pd.Series, gross_pnl: pd.Series, turnover: pd.Series) -> dict:
    """Calculate comprehensive performance metrics from PnL series."""
    if net_pnl.empty:
        return {
            "Total_Return": 0.0,
            "Sharpe": 0.0,
            "MaxDD": 0.0,
            "VaR_95": 0.0,
            "ES_95": 0.0,
            "Avg_Turnover": 0.0,
            "Cost_Impact": 0.0,
        }

    wealth = np.exp(net_pnl.cumsum())
    total_return = wealth.iloc[-1] - 1
    sharpe = calc_sharpe(net_pnl)
    max_dd = calc_max_drawdown(wealth)
    avg_turnover = turnover.mean() if not turnover.empty else 0.0

    net_simple = np.expm1(net_pnl)
    var_95, es_95 = calc_var_es(net_simple, confidence=0.95)

    gross_return = np.exp(gross_pnl.sum()) - 1 if not gross_pnl.empty else total_return
    cost_impact = gross_return - total_return

    return {
        "Total_Return": total_return,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "VaR_95": var_95,
        "ES_95": es_95,
        "Avg_Turnover": avg_turnover,
        "Cost_Impact": cost_impact,
    }
