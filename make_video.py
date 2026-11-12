"""
Render a ~30s LinkedIn clip (1080x1350, 30 fps) of the Monte Carlo results.
Run:  python make_video.py        -> output/monte_carlo_results.mp4
Every number on screen is computed from the same simulation as mc_portfolio_risk.py.
"""
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mc_portfolio_risk as m

plt.rcParams["font.family"] = "Inter"
plt.rcParams["text.parse_math"] = False   # "$" is money, not math
W, H, FPS = 1080, 1350, 30
BG, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, DARK_BLUE, RED, GREEN = "#2a78d6", "#184f95", "#d03b3b", "#0ca30c"
OUT = Path("output")
TMP = Path("/tmp/mcvideo")

# ---------- data (same seed as the report) ----------
prices = m.load_prices(False)
rets, stats, corr, mu_d, cov_d = m.estimate(prices)
paths = m.simulate(mu_d, cov_d, np.random.default_rng(m.SEED))
report, mdd = m.risk_report(paths)
end = paths[:, -1]
pcts = {p: np.percentile(paths, p, axis=0) for p in (5, 25, 50, 75, 95)}
port_vol = np.sqrt(m.WEIGHTS @ (cov_d * m.TD) @ m.WEIGHTS)
port = (prices / prices.iloc[0] * m.WEIGHTS).sum(axis=1)
hist_loss = ((port / port.shift(m.TD) - 1).dropna() < 0).mean()


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def fig_new():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    return fig


def header(fig, kicker, title, a=1.0):
    fig.text(0.08, 0.915, kicker, color=BLUE, fontsize=22, fontweight="bold", alpha=a)
    fig.text(0.08, 0.865, title, color=INK, fontsize=36, fontweight="bold", alpha=a, va="top")


def footer(fig):
    fig.text(0.08, 0.035, "Monte Carlo portfolio risk  ·  AAPL MSFT NVDA GOOG RMD  ·  data to Nov 11, 2026",
             color=MUTED, fontsize=14)


def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=17, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


money = matplotlib.ticker.FuncFormatter(lambda x, _: f"${x/1000:,.0f}k")


def fade(t, dur, fin=0.4, fout=0.4):
    return min(1.0, t / fin, (dur - t) / fout)


# ---------- scenes: each is (duration_s, draw(t) -> fig) ----------
def s_title(t, dur=4.0):
    fig = fig_new()
    a = ease(t / 0.8)
    fig.text(0.08, 0.68, "I simulated", color=INK2, fontsize=46, alpha=a)
    n = int(10_000 * ease((t - 0.3) / 1.6))
    fig.text(0.08, 0.53, f"{n:,}", color=BLUE, fontsize=150, fontweight="bold", alpha=a)
    fig.text(0.08, 0.44, "possible futures", color=INK, fontsize=56, fontweight="bold", alpha=a)
    a2 = ease((t - 1.6) / 0.7)
    fig.text(0.08, 0.37, "for a $100,000 stock portfolio.", color=INK, fontsize=40, alpha=a2)
    fig.text(0.08, 0.28, "How much could it lose in a year?", color=INK2, fontsize=34, alpha=ease((t - 2.4) / 0.7))
    footer(fig)
    return fig


def s_inputs(t, dur=6.0):
    fig = fig_new()
    header(fig, "STEP 1  ·  5 YEARS OF DAILY PRICES", "How risky is each stock?")
    ax = fig.add_axes([0.2, 0.47, 0.72, 0.29])
    style_ax(ax); ax.grid(False); ax.spines["bottom"].set_visible(False)
    tick = list(stats.index)[::-1]
    vols = stats["Annual volatility"].values[::-1]
    for i, (tk, v) in enumerate(zip(tick, vols)):
        g = ease((t - 0.3 - 0.15 * (4 - i)) / 0.9)
        ax.barh(i, v * g, height=0.55, color=BLUE)
        ax.text(-0.01, i, tk, ha="right", va="center", fontsize=22, color=INK, fontweight="bold")
        if g > 0.95:
            ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=20, color=INK2)
    ax.set_xlim(0, 0.62); ax.set_yticks([]); ax.set_xticks([])
    fig.text(0.08, 0.785, "Annual volatility", color=INK2, fontsize=20)
    b = ease((t - 2.4) / 0.7)
    fig.text(0.08, 0.37, "Tech stocks move together", color=INK, fontsize=30, fontweight="bold", alpha=b)
    tech = corr.values[:4, :4][np.triu_indices(4, 1)]
    rmd = corr.values[4, :4]
    fig.text(0.08, 0.325, f"correlation {tech.min():.2f}–{tech.max():.2f}.  ResMed only {rmd.min():.2f}–{rmd.max():.2f}.",
             color=INK2, fontsize=24, alpha=b)
    c = ease((t - 3.6) / 0.7)
    fig.text(0.08, 0.23, f"Mixed together, the portfolio's volatility is", color=INK2, fontsize=24, alpha=c)
    fig.text(0.08, 0.16, f"{port_vol:.0%}", color=BLUE, fontsize=54, fontweight="bold", alpha=c)
    fig.text(0.08, 0.125, f"vs {stats['Annual volatility'].mean():.0%} for an average single stock",
             color=INK2, fontsize=24, alpha=c)
    footer(fig)
    return fig


N_LINES = 60


def s_paths(t, dur=8.0):
    fig = fig_new()
    header(fig, "STEP 2  ·  10,000 SIMULATED YEARS", "Every line is one possible future")
    ax = fig.add_axes([0.14, 0.24, 0.8, 0.55])
    style_ax(ax)
    d = int(1 + 252 * ease((t - 0.3) / 5.0))
    x = np.arange(d)
    ax.fill_between(x, pcts[5][:d], pcts[95][:d], color=BLUE, alpha=0.16, lw=0)
    ax.fill_between(x, pcts[25][:d], pcts[75][:d], color=BLUE, alpha=0.32, lw=0)
    for p in paths[:N_LINES]:
        ax.plot(x, p[:d], color=INK2, alpha=0.13, lw=1)
    ax.plot(x, pcts[50][:d], color=DARK_BLUE, lw=3)
    ax.axhline(m.START_VALUE, color=INK2, lw=1.2, ls="--")
    ax.set_xlim(0, 252); ax.set_ylim(45_000, 260_000)
    ax.yaxis.set_major_formatter(money)
    ax.set_xlabel("Trading days ahead", color=INK2, fontsize=18)
    n = int(10_000 * ease((t - 0.3) / 5.0))
    fig.text(0.08, 0.80, f"{n:,} simulations", color=INK2, fontsize=20)
    a = ease((t - 5.6) / 0.7)
    fig.text(0.08, 0.15, f"Middle 50% of outcomes:  ${pcts[25][-1]/1000:,.0f}k – ${pcts[75][-1]/1000:,.0f}k",
             color=INK, fontsize=26, fontweight="bold", alpha=a)
    fig.text(0.08, 0.11, f"Middle 90%:  ${pcts[5][-1]/1000:,.0f}k – ${pcts[95][-1]/1000:,.0f}k",
             color=INK2, fontsize=24, alpha=a)
    footer(fig)
    return fig


bins = np.linspace(np.percentile(end, 0.2), np.percentile(end, 99.5), 60)
counts, _ = np.histogram(end, bins)
centers, width = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
cut = m.START_VALUE - report["VaR 95% (1y)"]


def s_dist(t, dur=7.0):
    fig = fig_new()
    header(fig, "STEP 3  ·  WHERE THE YEAR ENDS", "The worst 5% of futures")
    ax = fig.add_axes([0.1, 0.36, 0.84, 0.42])
    style_ax(ax); ax.set_yticks([])
    g = ease((t - 0.2) / 1.5)
    r = ease((t - 2.0) / 0.8)
    cols = [matplotlib.colors.to_hex(np.array(matplotlib.colors.to_rgb(BLUE)) * (1 - r) +
                                     np.array(matplotlib.colors.to_rgb(RED)) * r) if c < cut else BLUE
            for c in centers]
    ax.bar(centers, counts * g, width=width * 0.9, color=cols)
    ax.axvline(m.START_VALUE, color=INK2, ls="--", lw=1.2)
    ax.text(m.START_VALUE, counts.max() * 1.02, " $100k start", color=INK2, fontsize=17)
    ax.set_ylim(0, counts.max() * 1.1)
    ax.xaxis.set_major_formatter(money)
    a = ease((t - 2.6) / 0.7)
    fig.text(0.08, 0.27, "95% Value at Risk", color=INK2, fontsize=26, alpha=a)
    fig.text(0.08, 0.195, f"${report['VaR 95% (1y)']:,.0f}", color=RED, fontsize=60, fontweight="bold", alpha=a)
    b = ease((t - 3.8) / 0.7)
    fig.text(0.08, 0.135, "In 1 of every 20 futures you lose at least this much.", color=INK, fontsize=24, alpha=b)
    fig.text(0.08, 0.095, f"When it happens, the average loss is ${report['CVaR 95% (1y)']:,.0f} (CVaR).",
             color=INK2, fontsize=24, alpha=b)
    footer(fig)
    return fig


def s_stats(t, dur=6.0):
    fig = fig_new()
    header(fig, "THE RESULTS", "One year, $100,000 invested")
    rows = [
        ("Median ending value", f"${report['Median ending value']:,.0f}", INK),
        ("Chance of losing money", f"{report['Probability of a loss']:.0%}", INK),
        ("Chance of losing more than 20%", f"{report['Probability of losing >20%']:.0%}", INK),
        ("Worst 1% (99% VaR)", f"−${report['VaR 99% (1y)']:,.0f}", RED),
        ("Typical drop from a high during the year", f"−{abs(report['Median max drawdown']):.0%}", INK),
    ]
    for i, (k, v, c) in enumerate(rows):
        a = ease((t - 0.3 - 0.55 * i) / 0.6)
        y = 0.72 - i * 0.115
        fig.text(0.08, y, k, color=INK2, fontsize=25, alpha=a)
        fig.text(0.08, y - 0.05, v, color=c, fontsize=46, fontweight="bold", alpha=a)
        fig.add_artist(plt.Line2D([0.08, 0.92], [y - 0.07, y - 0.07], color=GRID, lw=1, alpha=a))
    footer(fig)
    return fig


def s_verify(t, dur=6.0):
    fig = fig_new()
    header(fig, "DID I CHECK IT?", "Yes, 16 out of 16 checks passed")
    items = [
        "Simulated volatility & correlations match history (<1% error)",
        "VaR and CVaR recomputed independently",
        "Same answer across 5 random seeds",
        "3 different models agree on VaR within 5%",
    ]
    for i, s in enumerate(items):
        a = ease((t - 0.3 - 0.45 * i) / 0.5)
        y = 0.74 - i * 0.075
        fig.text(0.08, y, "✓", color=GREEN, fontsize=30, fontweight="bold", alpha=a)
        fig.text(0.13, y + 0.004, s, color=INK, fontsize=22, alpha=a)
    b = ease((t - 2.4) / 0.7)
    fig.text(0.08, 0.37, "Honest caveat", color=INK, fontsize=26, fontweight="bold", alpha=b)
    fig.text(0.08, 0.325, f"Over the last 5 years only {hist_loss:.0%} of 1-year windows lost money.",
             color=INK2, fontsize=22, alpha=b)
    fig.text(0.08, 0.29, "I used lower expected returns on purpose, so the model", color=INK2, fontsize=22, alpha=b)
    fig.text(0.08, 0.255, "doesn't assume the past 5 years' gains will repeat.", color=INK2, fontsize=22, alpha=b)
    c = ease((t - 3.8) / 0.6)
    fig.text(0.08, 0.15, "Python · NumPy · pandas · Matplotlib", color=BLUE, fontsize=26, fontweight="bold", alpha=c)
    fig.text(0.08, 0.105, "Code on GitHub (link in post)  ·  Not investment advice", color=INK2, fontsize=21, alpha=c)
    footer(fig)
    return fig


SCENES = [("01_title", s_title, 4.0), ("02_inputs", s_inputs, 6.5), ("03_paths", s_paths, 8.0),
          ("04_dist", s_dist, 7.0), ("05_stats", s_stats, 6.0), ("06_verify", s_verify, 6.5)]


def frame_rgb(fn, t, dur):
    fig = fn(t, dur)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    a = fade(t, dur)                      # fade from/to background inside each scene
    if a < 1:
        bg = np.array(matplotlib.colors.to_rgb(BG)) * 255
        img = (img * a + bg * (1 - a)).astype(np.uint8)
    return img


def render(name, fn, dur):
    TMP.mkdir(parents=True, exist_ok=True)
    tmp, final = TMP / f"{name}.part.mp4", TMP / f"{name}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "18", "-pix_fmt", "yuv420p", str(tmp)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(int(round(dur * FPS))):
        p.stdin.write(frame_rgb(fn, i / FPS, dur).tobytes())
    p.stdin.close()
    if p.wait() == 0:
        os.replace(tmp, final)


def stills():
    """Full-size frames at the busiest moments, for checking layout before rendering."""
    OUT.mkdir(exist_ok=True)
    for name, fn, dur in SCENES:
        plt.imsave(TMP / f"still_{name}.png", frame_rgb(fn, dur - 0.6, dur))


if __name__ == "__main__":
    TMP.mkdir(parents=True, exist_ok=True)
    if "--stills" in sys.argv:
        stills(); sys.exit()
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    for name, fn, dur in SCENES:
        if not only or name in only:
            render(name, fn, dur)
    if not only:
        (TMP / "list.txt").write_text("".join(f"file '{TMP / n}.mp4'\n" for n, _, _ in SCENES))
        OUT.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                        str(TMP / "list.txt"), "-c", "copy", "-movflags", "+faststart",
                        str(OUT / "monte_carlo_results.mp4")], check=True)
        print("wrote", OUT / "monte_carlo_results.mp4")
