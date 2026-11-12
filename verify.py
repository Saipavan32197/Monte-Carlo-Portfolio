"""
Independent checks that the Monte Carlo model does what it claims.
Run:  python verify.py      (exits non-zero if any check fails)
"""
import numpy as np
import pandas as pd

import mc_portfolio_risk as m

results = []


def check(name, ok, detail):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


prices = m.load_prices(False)
rets, stats, corr, mu_d, cov_d = m.estimate(prices)
TD, H, W, V0 = m.TD, m.HORIZON_DAYS, m.WEIGHTS, m.START_VALUE

# 1. Inputs: estimates recomputed a different way (simple pandas std, no shared code)
vol_alt = np.log(prices).diff().std().values * np.sqrt(TD)
check("Volatility estimate", np.allclose(vol_alt, stats["Annual volatility"].values, atol=1e-9),
      f"max diff {np.abs(vol_alt - stats['Annual volatility'].values).max():.1e}")
corr_alt = np.corrcoef(np.log(prices).diff().dropna().values.T)
check("Correlation estimate", np.allclose(corr_alt, corr.values, atol=1e-9), "matches numpy corrcoef")
check("Covariance is positive definite", np.all(np.linalg.eigvalsh(cov_d) > 0),
      f"min eigenvalue {np.linalg.eigvalsh(cov_d).min():.2e}")
check("Weights sum to 1", abs(W.sum() - 1) < 1e-12, f"sum = {W.sum()}")

# 2. Simulated daily returns reproduce the inputs
daily = m.simulate_daily_returns(mu_d, cov_d, np.random.default_rng(1))
flat = daily.reshape(-1, daily.shape[-1])
sim_vol = flat.std(0) * np.sqrt(TD)
err = np.abs(sim_vol / stats["Annual volatility"].values - 1).max()
check("Simulated volatility = historical", err < 0.01, f"max relative error {err:.2%}")
sim_corr = np.corrcoef(flat.T)
err = np.abs(sim_corr - corr.values).max()
check("Simulated correlations = historical", err < 0.01, f"max abs error {err:.3f}")
k = pd.Series(flat[:, 0]).kurt()
check("Fat tails present (excess kurtosis > 0)", k > 1, f"AAPL daily excess kurtosis {k:.1f}")

# 3. Expected return comes out as intended
paths = m.to_portfolio_paths(daily)
exp_port = (W * stats["Expected return (used)"].values).sum()
sim_port = paths[:, -1].mean() / V0 - 1
asset_mean = np.exp(daily.sum(1)).mean(0) - 1
err = np.abs(asset_mean - stats["Expected return (used)"].values).max()
check("Per-stock simulated mean return = target", err < 0.01, f"max abs error {err:.2%}")
check("Portfolio mean return = weighted target", abs(sim_port - exp_port) < 0.01,
      f"sim {sim_port:.2%} vs target {exp_port:.2%}")

# 4. Risk metrics recomputed independently (sort-based)
report, mdd = m.risk_report(paths)
loss = np.sort(V0 - paths[:, -1])[::-1]             # biggest loss first
n5 = int(0.05 * len(loss))
var95_alt, cvar95_alt = loss[n5], loss[:n5].mean()
check("VaR 95% recomputed", abs(var95_alt - report["VaR 95% (1y)"]) / var95_alt < 0.01,
      f"{report['VaR 95% (1y)']:,.0f} vs {var95_alt:,.0f}")
check("CVaR 95% recomputed", abs(cvar95_alt - report["CVaR 95% (1y)"]) / cvar95_alt < 0.01,
      f"{report['CVaR 95% (1y)']:,.0f} vs {cvar95_alt:,.0f}")
check("CVaR >= VaR (both levels)", report["CVaR 95% (1y)"] >= report["VaR 95% (1y)"]
      and report["CVaR 99% (1y)"] >= report["VaR 99% (1y)"], "ordering holds")
p0 = paths[0]; dd = (p0 / np.maximum.accumulate(p0) - 1).min()
check("Max drawdown recomputed (path 0)", abs(dd - mdd[0]) < 1e-12, f"{dd:.2%}")

# 5. Stability: different random seeds give the same answer
rows = []
for s in range(5):
    r, _ = m.risk_report(m.simulate(mu_d, cov_d, np.random.default_rng(100 + s)))
    rows.append(r)
seeds = pd.DataFrame(rows)
spread = (seeds["VaR 95% (1y)"].max() - seeds["VaR 95% (1y)"].min()) / seeds["VaR 95% (1y)"].mean()
check("VaR 95% stable across 5 seeds", spread < 0.05,
      f"range ${seeds['VaR 95% (1y)'].min():,.0f}-${seeds['VaR 95% (1y)'].max():,.0f}")
pl = seeds["Probability of a loss"]
check("P(loss) stable across 5 seeds", pl.max() - pl.min() < 0.02, f"range {pl.min():.1%}-{pl.max():.1%}")

# 6. Model comparison: normal shocks and historical bootstrap (resampling real days)
normal = m.to_portfolio_paths(m.simulate_daily_returns(mu_d, cov_d, np.random.default_rng(7), t_dof=None))
rng = np.random.default_rng(8)
idx = rng.integers(0, len(rets), size=(m.N_SIMS, H))
boot = rets.values[idx] - rets.values.mean(0) + mu_d      # real days, same drift as the model
boot_paths = m.to_portfolio_paths(boot)
cmp = pd.DataFrame({
    "Student-t (main)": report,
    "Normal": m.risk_report(normal)[0],
    "Historical bootstrap": m.risk_report(boot_paths)[0],
}).loc[["Probability of a loss", "VaR 95% (1y)", "CVaR 95% (1y)", "VaR 99% (1y)", "CVaR 99% (1y)"]]
print("\nModel comparison:\n" + cmp.map(lambda v: f"{v:.1%}" if v < 1 else f"${v:,.0f}").to_string())
d = abs(cmp.loc["VaR 95% (1y)", "Student-t (main)"] / cmp.loc["VaR 95% (1y)", "Historical bootstrap"] - 1)
check("Main model VaR within 10% of bootstrap", d < 0.10, f"diff {d:.1%}")

# 7. Reality check: actual rolling 1-year returns of this portfolio over the 5 years
port = (prices / prices.iloc[0] * W).sum(axis=1)
r1y = (port / port.shift(TD) - 1).dropna()
print(f"\nHistorical rolling 1y returns ({len(r1y)} windows): loss in {(r1y < 0).mean():.1%}, "
      f"worst {r1y.min():.1%}, best {r1y.max():.1%}")

print(f"\n{sum(ok for _, ok in results)}/{len(results)} checks passed")
raise SystemExit(0 if all(ok for _, ok in results) else 1)
