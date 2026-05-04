"""Walk-forward validation orchestrator."""

import pandas as pd
import numpy as np
import os

from src.config import Config
from src.models.static_pca import FixedPCAModel, RollingPCAModel
from src.models.kalman import KalmanFactorModel
from src.signals.momentum import MomentumSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.portfolio.constructor import PortfolioConstructor
from src.portfolio.backtester import Backtester
from src.metrics import calc_metrics, calc_cagr
from src.plotting import plot_wealth_curves


def load_data(cfg: Config):
    """Load returns and walk-forward configuration."""
    returns = pd.read_csv(cfg.data_path, index_col=0, parse_dates=True)
    config = pd.read_csv(
        cfg.config_path,
        parse_dates=["Train_Start", "Train_End", "Test_Start", "Test_End"],
    )
    return returns, config


def run_single_fold(model, signal, constructor, backtester, train_returns, test_returns):
    """Execute one walk-forward fold."""
    residuals = model.fit_transform(train_returns, test_returns)
    scores = signal.generate(residuals)
    weights = constructor.construct(scores, test_returns, higher_is_long=signal.higher_is_long)
    return backtester.run(weights, test_returns, return_components=True)


def run_walk_forward_strategy(
    name, returns, config_df, model_factory, signal, constructor, backtester,
    cfg, print_folds=True, print_summary=True,
):
    """Iterate over all folds, stitch results."""
    print(f"\n--- Running Walk-Forward: {name} ---")

    stitched_pnl = []
    stitched_gross = []
    stitched_turnover = []
    fold_rows = []

    for _, row in config_df.iterrows():
        train_start, train_end = row["Train_Start"], row["Train_End"]
        test_start, test_end = row["Test_Start"], row["Test_End"]

        train_data = returns.loc[train_start:train_end]
        test_data = returns.loc[test_start:test_end]

        if len(train_data) < cfg.rolling_window or len(test_data) == 0:
            continue

        # Fresh model per fold
        model = model_factory()

        pnl, components = run_single_fold(
            model, signal, constructor, backtester, train_data, test_data
        )
        gross_pnl = components["gross_pnl"]
        turnover = components["turnover"]
        stitched_pnl.append(pnl)
        stitched_gross.append(gross_pnl)
        stitched_turnover.append(turnover)

        metrics = calc_metrics(pnl, gross_pnl, turnover)
        fold_rows.append({
            "Fold": row["Fold"],
            "Start": test_start.date().isoformat(),
            "End": test_end.date().isoformat(),
            "Total_Return": metrics["Total_Return"],
            "Sharpe": metrics["Sharpe"],
            "MaxDD": metrics["MaxDD"],
            "VaR_95": metrics["VaR_95"],
            "ES_95": metrics["ES_95"],
            "Avg_Turnover": metrics["Avg_Turnover"],
            "Cost_Impact": metrics["Cost_Impact"],
        })
        if print_folds:
            print(f"Fold {row['Fold']} [{test_start.date()} -> {test_end.date()}]: {metrics['Total_Return']:.2%}")

    if not stitched_pnl:
        empty = pd.Series(dtype=float)
        metrics = calc_metrics(empty, empty, empty)
        metrics["CAGR"] = 0.0
        return empty, metrics, pd.DataFrame()

    net_all = pd.concat(stitched_pnl)
    gross_all = pd.concat(stitched_gross)
    turnover_all = pd.concat(stitched_turnover)
    agg_metrics = calc_metrics(net_all, gross_all, turnover_all)
    cagr = calc_cagr(net_all)

    fold_table = pd.DataFrame(fold_rows)
    if print_folds:
        print("\nFold Summary:")
        print(
            fold_table.to_string(
                index=False,
                formatters={
                    "Total_Return": "{:.2%}".format,
                    "Sharpe": "{:.2f}".format,
                    "MaxDD": "{:.2%}".format,
                    "VaR_95": "{:.2%}".format,
                    "ES_95": "{:.2%}".format,
                    "Avg_Turnover": "{:.3f}".format,
                    "Cost_Impact": "{:.2%}".format,
                },
            )
        )

    if print_summary:
        print("\nAggregate Metrics:")
        print(f"Total Return:   {agg_metrics['Total_Return']:.2%}")
        print(f"CAGR:           {cagr:.2%}")
        print(f"Sharpe:         {agg_metrics['Sharpe']:.2f}")
        print(f"Max Drawdown:   {agg_metrics['MaxDD']:.2%}")
        print(f"VaR 95%:        {agg_metrics['VaR_95']:.2%}")
        print(f"ES 95%:         {agg_metrics['ES_95']:.2%}")
        print(f"Avg Turnover:   {agg_metrics['Avg_Turnover']:.3f}")
        print(f"Cost Impact:    {agg_metrics['Cost_Impact']:.2%}")

    agg_metrics["CAGR"] = cagr
    return net_all, agg_metrics, fold_table


def signal_label(signal_name: str) -> str:
    if signal_name == "mean_reversion":
        return "MeanRev"
    return "Momentum"


def build_experiments(cfg: Config):
    """Build experiment grid."""
    experiments = []

    for pca_mode in ["fixed", "rolling"]:
        for signal_name in ["mean_reversion", "momentum"]:
            label = signal_label(signal_name)
            experiments.append({
                "name": f"{pca_mode.title()} PCA | {label}",
                "model_factory": lambda mode=pca_mode: (
                    FixedPCAModel(n_components=cfg.n_factors) if mode == "fixed"
                    else RollingPCAModel(n_components=cfg.n_factors, window=cfg.rolling_window)
                ),
                "signal_name": signal_name,
                "print_folds": True,
                "print_summary": True,
            })

    for delta in cfg.kalman_deltas:
        for ve in cfg.kalman_ves:
            for signal_name in ["mean_reversion", "momentum"]:
                label = signal_label(signal_name)
                experiments.append({
                    "name": f"Fixed PCA + Kalman | {label} | d={delta:g}, ve={ve:g}",
                    "model_factory": lambda d=delta, v=ve: KalmanFactorModel(
                        n_components=cfg.n_factors, delta=d, ve=v
                    ),
                    "signal_name": signal_name,
                    "print_folds": False,
                    "print_summary": False,
                })

    return experiments


def main():
    cfg = Config()
    returns, config_df = load_data(cfg)

    # Build shared components
    constructor = PortfolioConstructor(
        long_q=cfg.long_q,
        short_q=cfg.short_q,
        rebalance=cfg.rebalance,
        beta_neutral=cfg.beta_neutral,
        beta_window=cfg.beta_window,
        market_col=cfg.market_col,
        trade_market=cfg.trade_market,
        min_beta_obs=cfg.min_beta_obs,
    )
    backtester = Backtester(cost_bps=cfg.cost_bps)

    # Build signal instances
    signals = {
        "mean_reversion": MeanReversionSignal(vol_halflife=cfg.vol_halflife, z_clip=cfg.z_clip),
        "momentum": MomentumSignal(lookback=cfg.mom_lookback, vol_halflife=cfg.vol_halflife, z_clip=cfg.z_clip),
    }

    experiments = build_experiments(cfg)
    summary_rows = []
    exp_results = {}

    for exp in experiments:
        signal = signals[exp["signal_name"]]
        net_pnl, metrics, _ = run_walk_forward_strategy(
            exp["name"], returns, config_df,
            model_factory=exp["model_factory"],
            signal=signal,
            constructor=constructor,
            backtester=backtester,
            cfg=cfg,
            print_folds=exp.get("print_folds", True),
            print_summary=exp.get("print_summary", True),
        )

        exp_results[exp["name"]] = {"pnl": net_pnl, "metrics": metrics}

        summary_rows.append({
            "Strategy": exp["name"],
            "Total_Return": metrics["Total_Return"],
            "CAGR": metrics["CAGR"],
            "Sharpe": metrics["Sharpe"],
            "MaxDD": metrics["MaxDD"],
            "VaR_95": metrics["VaR_95"],
            "ES_95": metrics["ES_95"],
            "Avg_Turnover": metrics["Avg_Turnover"],
            "Cost_Impact": metrics["Cost_Impact"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\nExperiment Summary:")
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "Total_Return": "{:.2%}".format,
                "CAGR": "{:.2%}".format,
                "Sharpe": "{:.2f}".format,
                "MaxDD": "{:.2%}".format,
                "VaR_95": "{:.2%}".format,
                "ES_95": "{:.2%}".format,
                "Avg_Turnover": "{:.3f}".format,
                "Cost_Impact": "{:.2%}".format,
            },
        )
    )
    summary_df.to_csv("data/experiment_results.csv", index=False)
    print("Results saved to data/experiment_results.csv")

    # Plotting
    base_plot_names = [
        "Fixed PCA | MeanRev",
        "Fixed PCA | Momentum",
        "Rolling PCA | MeanRev",
        "Rolling PCA | Momentum",
    ]

    selected_names = [n for n in base_plot_names if n in exp_results and not exp_results[n]["pnl"].empty]

    # Add best/worst Kalman by Sharpe
    kalman_rows = [r for r in summary_rows if "Kalman" in r["Strategy"]]
    if kalman_rows:
        kalman_df = pd.DataFrame(kalman_rows)
        best_name = kalman_df.loc[kalman_df["Sharpe"].idxmax(), "Strategy"]
        worst_name = kalman_df.loc[kalman_df["Sharpe"].idxmin(), "Strategy"]
        for name in (best_name, worst_name):
            if name in exp_results and not exp_results[name]["pnl"].empty and name not in selected_names:
                selected_names.append(name)

    if selected_names:
        plot_wealth_curves(exp_results, selected_names)


if __name__ == "__main__":
    main()
