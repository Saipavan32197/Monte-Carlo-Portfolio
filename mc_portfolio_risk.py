"""
Monte Carlo portfolio risk
--------------------------
1. Load 5 years of daily prices for a few stocks
2. Measure volatility, correlations and estimate expected returns
3. Simulate 10,000 possible 1-year futures (correlated, fat-tailed daily shocks)
4. Report risk: VaR, CVaR, probability of loss, drawdowns + charts

Run:  python mc_portfolio_risk.py            (uses data/prices.csv)
      python mc_portfolio_risk.py --yahoo    (downloads fresh data with yfinance)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- settings ----------------
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOG", "RMD"]
WEIGHTS = np.array([0.20, 0.20, 0.20, 0.20, 0.20])   # equal weight
START_VALUE = 100_000          # dollars invested
N_SIMS = 10_000                # number of possible futures
HORIZON_DAYS = 252             # 1 trading year
PRIOR_RETURN = 0.08            # long-run equity return used to temper history
SHRINK = 0.5                   # 50% history, 50% prior
T_DOF = 5                      # Student-t degrees of freedom (fat tails); None = normal
SEED = 42
TD = 252                       # trading days per year

OUT = Path("output")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ---------------- 1. data ----------------
def load_prices(use_yahoo: bool) -> pd.DataFrame:
    if use_yahoo:
        import yfinance as yf
        df = yf.download(TICKERS, period="5y", auto_adjust=True, progress=False)["Close"]
        return df[TICKERS].dropna()
    df = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")
    return df[TICKERS].dropna()


# ---------------- 2. estimates ----------------
def estimate(prices: pd.DataFrame):
    rets = np.log(prices / prices.shift(1)).dropna()
    mu_d, cov_d = rets.mean().values, rets.cov().values

    vol = np.sqrt(np.diag(cov_d) * TD)
    hist_ret = np.exp(mu_d * TD + 0.5 * np.diag(cov_d) * TD) - 1      # arithmetic annual
    exp_ret = SHRINK * hist_ret + (1 - SHRINK) * PRIOR_RETURN          # tempered forecast
    corr = rets.corr()

    stats = pd.DataFrame({
        "Annual volatility": vol,
        "Historical annual return": hist_ret,
        "Expected return (used)": exp_ret,
        "Sharpe (hist, rf=0)": hist_ret / vol,
    }, index=TICKERS)
    # daily log drift that reproduces the expected arithmetic return
    mu_sim = np.log1p(exp_ret) / TD - 0.5 * np.diag(cov_d)
    return rets, stats, corr, mu_sim, cov_d


# ---------------- 3. simulation ----------------
def simulate_daily_returns(mu_d, cov_d, rng, n_sims=N_SIMS, t_dof=T_DOF):
    """Correlated daily log returns, shape (sims, days, assets)."""
    n = len(mu_d)
    L = np.linalg.cholesky(cov_d)
    z = rng.standard_normal((n_sims, HORIZON_DAYS, n))
    if t_dof:  # multivariate Student-t, rescaled so variance matches history
        chi = rng.chisquare(t_dof, size=(n_sims, HORIZON_DAYS, 1))
        z = z / np.sqrt(chi / t_dof) * np.sqrt((t_dof - 2) / t_dof)
    return mu_d + z @ L.T


def to_portfolio_paths(daily):
    """Buy-and-hold portfolio value paths from daily log returns, shape (sims, days+1)."""
    growth = np.exp(np.cumsum(daily, axis=1))               # (sims, days, assets)
    paths = growth @ (START_VALUE * WEIGHTS)
    return np.hstack([np.full((len(paths), 1), START_VALUE), paths])


def simulate(mu_d, cov_d, rng):
    return to_portfolio_paths(simulate_daily_returns(mu_d, cov_d, rng))


# ---------------- 4. risk metrics ----------------
def risk_report(paths):
    end = paths[:, -1]
    pnl = end - START_VALUE
    r = end / START_VALUE - 1
    peak = np.maximum.accumulate(paths, axis=1)
    mdd = ((paths - peak) / peak).min(axis=1)

    def var(a): return -np.percentile(pnl, 100 - a)
    def cvar(a): return -pnl[pnl <= -var(a)].mean()

    rows = {
        "Mean ending value": end.mean(),
        "Median ending value": np.median(end),
        "5th percentile ending value": np.percentile(end, 5),
        "95th percentile ending value": np.percentile(end, 95),
        "Expected return": r.mean(),
        "Probability of a loss": (end < START_VALUE).mean(),
        "Probability of losing >20%": (r < -0.20).mean(),
        "VaR 95% (1y)": var(95),
        "CVaR 95% (1y)": cvar(95),
        "VaR 99% (1y)": var(99),
        "CVaR 99% (1y)": cvar(99),
        "Median max drawdown": np.median(mdd),
        "Worst-5% max drawdown": np.percentile(mdd, 5),
    }
    return pd.Series(rows), mdd


# ---------------- charts ----------------
def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(stats, corr, paths, report):
    OUT.mkdir(exist_ok=True)
    money = matplotlib.ticker.FuncFormatter(lambda x, _: f"${x/1000:,.0f}k")

    # correlation heatmap (sequential single hue)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURF)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", "#2a78d6", "#0d366b"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(TICKERS)), TICKERS); ax.set_yticks(range(len(TICKERS)), TICKERS)
    for i in range(len(TICKERS)):
        for j in range(len(TICKERS)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if v > 0.55 else INK)
    ax.set_title("Daily return correlations (5y)", loc="left", color=INK, fontweight="bold")
    ax.tick_params(length=0, colors=INK2)
    for s in ax.spines.values(): s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(OUT / "1_correlations.png", dpi=160); plt.close(fig)

    # volatility vs expected return
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURF)
    style(ax, "Risk vs expected return (annual)")
    for k, t in enumerate(TICKERS):
        x, y = stats.loc[t, "Annual volatility"], stats.loc[t, "Expected return (used)"]
        ax.scatter(x, y, s=90, color=COLORS[k], edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(t, (x, y), xytext=(8, 4), textcoords="offset points", color=INK, fontsize=10)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Volatility", color=INK2); ax.set_ylabel("Expected return", color=INK2)
    fig.tight_layout(); fig.savefig(OUT / "2_risk_return.png", dpi=160); plt.close(fig)

    # fan chart of simulated paths
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURF)
    style(ax, f"{N_SIMS:,} simulated futures for a ${START_VALUE:,} portfolio")
    days = np.arange(paths.shape[1])
    for lo, hi, a in [(5, 95, 0.18), (25, 75, 0.35)]:
        ax.fill_between(days, np.percentile(paths, lo, 0), np.percentile(paths, hi, 0),
                        color="#2a78d6", alpha=a, linewidth=0,
                        label=f"{lo}th-{hi}th percentile")
    for p in paths[:40]:
        ax.plot(days, p, color=INK2, alpha=0.12, linewidth=0.7)
    ax.plot(days, np.median(paths, 0), color="#184f95", linewidth=2, label="Median")
    ax.axhline(START_VALUE, color=INK2, linewidth=1, linestyle="--")
    ax.yaxis.set_major_formatter(money)
    ax.set_xlabel("Trading days ahead", color=INK2)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "3_simulated_paths.png", dpi=160); plt.close(fig)

    # ending value distribution with VaR / CVaR
    end = paths[:, -1]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURF)
    style(ax, "Portfolio value after 1 year")
    bins = np.linspace(np.percentile(end, 0.2), np.percentile(end, 99.5), 80)
    cut = START_VALUE - report["VaR 95% (1y)"]
    ax.hist(end[end >= cut], bins=bins, color="#2a78d6", edgecolor=SURF, linewidth=0.5)
    ax.hist(end[end < cut], bins=bins, color="#d03b3b", edgecolor=SURF, linewidth=0.5,
            label=f"Worst 5% (VaR 95% = ${report['VaR 95% (1y)']:,.0f})")
    ax.axvline(START_VALUE, color=INK2, linestyle="--", linewidth=1)
    ax.text(START_VALUE, ax.get_ylim()[1] * 0.95, " starting value", color=INK2, fontsize=9)
    ax.xaxis.set_major_formatter(money)
    ax.set_ylabel("Number of simulations", color=INK2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "4_ending_value_distribution.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yahoo", action="store_true", help="download fresh prices via yfinance")
    args = ap.parse_args()

    prices = load_prices(args.yahoo)
    rets, stats, corr, mu_sim, cov_d = estimate(prices)
    paths = simulate(mu_sim, cov_d, np.random.default_rng(SEED))
    report, _ = risk_report(paths)

    pd.set_option("display.width", 120)
    print(f"Data: {prices.index[0].date()} -> {prices.index[-1].date()}  ({len(rets)} daily returns)\n")
    print("=== Per-stock estimates ===")
    shown = stats.map(lambda v: f"{v:.1%}")
    shown["Sharpe (hist, rf=0)"] = stats["Sharpe (hist, rf=0)"].map(lambda v: f"{v:.2f}")
    print(shown.to_string(), "\n")
    print("=== Correlation matrix ===")
    print(corr.round(2).to_string(), "\n")
    pv = np.sqrt(WEIGHTS @ (cov_d * TD) @ WEIGHTS)
    print(f"Portfolio volatility: {pv:.1%}   (avg single-stock vol: {stats['Annual volatility'].mean():.1%})\n")
    print(f"=== {N_SIMS:,} simulations, {HORIZON_DAYS}-day horizon, ${START_VALUE:,} start ===")
    for k, v in report.items():
        if "value" in k or "VaR" in k:
            print(f"{k:32s} ${v:,.0f}")
        else:
            print(f"{k:32s} {v:.1%}")

    OUT.mkdir(exist_ok=True)
    stats.to_csv(OUT / "stock_estimates.csv")
    corr.to_csv(OUT / "correlations.csv")
    report.to_csv(OUT / "risk_report.csv", header=["value"])
    charts(stats, corr, paths, report)


if __name__ == "__main__":
    main()
