import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def short_label(name: str) -> str:
    """Shorten strategy name for plot legend."""
    name = name.replace("Fixed PCA |", "PCA")
    name = name.replace("Rolling PCA |", "RollPCA")
    name = name.replace("Fixed PCA + Kalman |", "PCA+KF")
    name = name.replace("MeanRev", "MR")
    name = name.replace("Momentum", "MOM")
    name = name.replace("ve=", "R=")
    name = name.replace("d=", "Q=")
    return name


def plot_wealth_curves(
    exp_results: dict,
    selected_names: list,
    save_path: str = "figures/wealth_curves.png",
    show: bool = True,
) -> None:
    """Plot cumulative wealth curves for selected strategies."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    for name in selected_names:
        res = exp_results[name]
        pnl = res["pnl"]
        if pnl.empty:
            continue
        wealth = np.exp(pnl.cumsum())
        ax.plot(wealth.index, wealth, label=short_label(name), linewidth=1.8)

    ax.set_title("Walk-forward cumulative wealth (selected strategies)")
    ax.set_ylabel("Cumulative wealth")
    ax.grid(True, alpha=0.25)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")

    if show:
        plt.show()
