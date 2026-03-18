"""
Backtest: XSP 1DTE VRP-Adjusted Iron Condor
────────────────────────────────────────────
Replicates the logic in app.py and simulates entering a 1DTE iron condor
every trading day at ~12:30 PST from January 2024 to present.

Data sources:
  - SPY daily prices  → spot price & 10-day realized volatility
  - ^VIX1D (1-Day VIX) → proxy for 1DTE ATM implied volatility
  - Next-day SPY close → settlement for P&L

Iron condor construction (per app.py):
  VRP = ATM_IV - 10D_RV
  adj_iv = max(rv, 5) if VRP > 0, else atm_iv
  expected_move = spot * (adj_iv/100) * sqrt(1/365)
  short_put  = round(spot - EM * 0.98)
  short_call = round(spot + EM * 0.98)
  wings = ±1 point
  est_credit = 0.15 + max(0, vrp/15) * 0.35
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")

# ── CONFIG (mirrors app.py) ─────────────────────────────────────
RV_WINDOW = 10
WING_WIDTH = 1.0
TARGET_COLLATERAL = 100
MIN_CREDIT = 0.20
START_DATE = "2024-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "backtest_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_data():
    """Download all required historical data."""
    print("Downloading SPY price data...")
    spy = yf.download("SPY", start="2023-12-01", end=END_DATE, progress=False)
    close_col = "Adj Close" if "Adj Close" in spy.columns else "Close"
    spy_close = spy[close_col]
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.squeeze()
    spy_close.name = "spy_close"

    print("Downloading VIX1D data (1-day implied volatility proxy)...")
    vix1d = yf.download("^VIX1D", start="2023-12-01", end=END_DATE, progress=False)
    vix1d_close = vix1d["Close"]
    if isinstance(vix1d_close, pd.DataFrame):
        vix1d_close = vix1d_close.squeeze()
    vix1d_close.name = "vix1d"

    print("Downloading VIX data (30-day IV fallback)...")
    vix = yf.download("^VIX", start="2023-12-01", end=END_DATE, progress=False)
    vix_close = vix["Close"]
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.squeeze()
    vix_close.name = "vix"

    df = pd.DataFrame({"spy_close": spy_close, "vix1d": vix1d_close, "vix": vix_close})
    df.index = pd.to_datetime(df.index)
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(0)
    df = df.sort_index()
    return df


def calculate_rv_series(prices, window=RV_WINDOW):
    """10-day realized volatility (annualized), matching app.py logic."""
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252) * 100
    return rv


def run_backtest(df):
    """
    For each trading day from Jan 2024 onward:
      1. Compute RV, ATM IV (VIX1D), VRP
      2. Compute adjusted expected move & strikes
      3. Determine P&L at next-day close
    """
    df = df.copy()
    df["rv_10d"] = calculate_rv_series(df["spy_close"], RV_WINDOW)
    df["atm_iv"] = df["vix1d"].fillna(df["vix"])

    mask = df.index >= START_DATE
    trade_dates = df.index[mask]

    results = []

    for i, date in enumerate(trade_dates):
        row = df.loc[date]
        spot = row["spy_close"]
        atm_iv = row["atm_iv"]
        rv = row["rv_10d"]

        if pd.isna(spot) or pd.isna(atm_iv) or pd.isna(rv):
            continue

        vrp = atm_iv - rv
        dte = 1

        adj_iv = max(atm_iv - vrp, 5) if vrp > 0 else atm_iv

        time_frac = dte / 365.0
        adj_em_pct = adj_iv / 100 * np.sqrt(time_frac) * 100
        adj_em_dol = spot * (adj_em_pct / 100)

        short_put = round(spot - adj_em_dol * 0.98)
        short_call = round(spot + adj_em_dol * 0.98)
        long_put = short_put - WING_WIDTH
        long_call = short_call + WING_WIDTH

        est_credit = 0.15 + max(0, vrp / 15) * 0.35

        if est_credit >= MIN_CREDIT and vrp > 5:
            signal = "LOAD"
        elif vrp > 0:
            signal = "THIN"
        else:
            signal = "SKIP"

        # Find next trading day for settlement
        future_dates = df.index[df.index > date]
        if len(future_dates) == 0:
            continue
        settle_date = future_dates[0]
        settle_price = df.loc[settle_date, "spy_close"]
        if pd.isna(settle_price):
            continue

        # Iron condor P&L at expiration
        put_spread_loss = max(min(short_put - settle_price, WING_WIDTH), 0)
        call_spread_loss = max(min(settle_price - short_call, WING_WIDTH), 0)
        pnl = est_credit - put_spread_loss - call_spread_loss

        # Max loss capped at wing_width - credit
        max_loss = -(WING_WIDTH - est_credit)
        pnl = max(pnl, max_loss)

        won = pnl > 0

        results.append({
            "trade_date": date,
            "settle_date": settle_date,
            "spot": spot,
            "settle_price": settle_price,
            "atm_iv": atm_iv,
            "rv_10d": rv,
            "vrp": vrp,
            "adj_iv": adj_iv,
            "adj_em_pct": adj_em_pct,
            "adj_em_dol": adj_em_dol,
            "short_put": short_put,
            "long_put": long_put,
            "short_call": short_call,
            "long_call": long_call,
            "est_credit": est_credit,
            "signal": signal,
            "put_spread_loss": put_spread_loss,
            "call_spread_loss": call_spread_loss,
            "pnl": pnl,
            "won": won,
        })

    return pd.DataFrame(results)


def print_stats(trades, label="ALL TRADES"):
    """Print summary statistics for a trade set."""
    if len(trades) == 0:
        print(f"\n{'='*60}")
        print(f"  {label}: No trades")
        print(f"{'='*60}")
        return

    n = len(trades)
    wins = trades["won"].sum()
    losses = n - wins
    win_rate = wins / n * 100

    total_pnl = trades["pnl"].sum()
    avg_pnl = trades["pnl"].mean()
    avg_win = trades.loc[trades["won"], "pnl"].mean() if wins > 0 else 0
    avg_loss = trades.loc[~trades["won"], "pnl"].mean() if losses > 0 else 0

    cumulative = trades["pnl"].cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    max_dd = drawdown.min()

    pnl_per_100 = total_pnl  # already per $1 wing = $100 notional
    pnl_per_100_dollars = total_pnl * 100  # scaled to $100 contracts

    # Per-contract: wing_width is $1, so max risk is $1 - credit per contract
    # In dollar terms (x100 multiplier for options), max risk per contract ≈ $100 - credit*100
    avg_max_risk = (WING_WIDTH - trades["est_credit"].mean()) * 100
    roi = total_pnl * 100 / (avg_max_risk * n) * 100 if avg_max_risk > 0 else 0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Period:           {trades['trade_date'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{trades['trade_date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Total trades:     {n}")
    print(f"  Wins / Losses:    {wins} / {losses}")
    print(f"  Win rate:         {win_rate:.1f}%")
    print(f"  ──────────────────────────────────────")
    print(f"  Total P&L:        ${total_pnl * 100:,.2f}  (per contract)")
    print(f"  Avg P&L/trade:    ${avg_pnl * 100:,.2f}")
    print(f"  Avg winner:       ${avg_win * 100:,.2f}")
    print(f"  Avg loser:        ${avg_loss * 100:,.2f}")
    print(f"  Best trade:       ${trades['pnl'].max() * 100:,.2f}")
    print(f"  Worst trade:      ${trades['pnl'].min() * 100:,.2f}")
    print(f"  ──────────────────────────────────────")
    print(f"  Max drawdown:     ${max_dd * 100:,.2f}")
    print(f"  Avg max risk:     ${avg_max_risk:,.2f}")
    print(f"  ROI (total P&L / total capital at risk): {roi:.2f}%")

    # Monthly breakdown
    trades_copy = trades.copy()
    trades_copy["month"] = trades_copy["trade_date"].dt.to_period("M")
    monthly = trades_copy.groupby("month").agg(
        trades_count=("pnl", "count"),
        wins_count=("won", "sum"),
        pnl_sum=("pnl", "sum"),
    )
    monthly["win_rate"] = monthly["wins_count"] / monthly["trades_count"] * 100
    monthly["pnl_dollars"] = monthly["pnl_sum"] * 100

    print(f"\n  Monthly Breakdown:")
    print(f"  {'Month':<10} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'P&L':>10}")
    print(f"  {'─'*42}")
    for period, row in monthly.iterrows():
        print(f"  {str(period):<10} {int(row['trades_count']):>7} "
              f"{int(row['wins_count']):>6} {row['win_rate']:>6.1f}% "
              f"${row['pnl_dollars']:>9,.2f}")
    print()


def generate_plots(trades):
    """Generate backtest visualization charts."""
    if len(trades) == 0:
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("XSP 1DTE VRP Iron Condor Backtest (Jan 2024 – Present)",
                 fontsize=14, fontweight="bold", y=0.98)

    dates = trades["trade_date"]

    # 1. Cumulative P&L (equity curve)
    ax = axes[0, 0]
    cum_pnl = trades["pnl"].cumsum() * 100
    ax.plot(dates, cum_pnl, color="#2196F3", linewidth=1.5)
    ax.fill_between(dates, 0, cum_pnl, alpha=0.15, color="#2196F3")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Cumulative P&L (per contract)")
    ax.set_ylabel("$ P&L")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # 2. Daily P&L bar chart
    ax = axes[0, 1]
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in trades["pnl"]]
    ax.bar(dates, trades["pnl"] * 100, color=colors, alpha=0.7, width=1)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Daily P&L")
    ax.set_ylabel("$ P&L")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # 3. P&L distribution histogram
    ax = axes[1, 0]
    ax.hist(trades["pnl"] * 100, bins=40, color="#9C27B0", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="gray", linewidth=1, linestyle="--")
    ax.axvline(trades["pnl"].mean() * 100, color="red", linewidth=1.5, linestyle="-",
               label=f"Mean: ${trades['pnl'].mean() * 100:.2f}")
    ax.set_title("P&L Distribution")
    ax.set_xlabel("$ P&L per trade")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. VRP over time with signal colors
    ax = axes[1, 1]
    signal_colors = {"LOAD": "#4CAF50", "THIN": "#FF9800", "SKIP": "#F44336"}
    for sig in ["LOAD", "THIN", "SKIP"]:
        mask = trades["signal"] == sig
        if mask.any():
            ax.scatter(dates[mask], trades.loc[mask, "vrp"],
                       c=signal_colors[sig], s=10, alpha=0.6, label=sig)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axhline(5, color="green", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_title("VRP Over Time (colored by signal)")
    ax.set_ylabel("VRP (%)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # 5. Rolling 20-trade win rate
    ax = axes[2, 0]
    rolling_wr = trades["won"].rolling(20).mean() * 100
    ax.plot(dates, rolling_wr, color="#FF5722", linewidth=1.5)
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Rolling 20-Trade Win Rate")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # 6. Cumulative P&L by signal type
    ax = axes[2, 1]
    for sig, color in signal_colors.items():
        sig_trades = trades[trades["signal"] == sig].copy()
        if len(sig_trades) > 0:
            sig_trades = sig_trades.reset_index(drop=True)
            ax.plot(sig_trades["trade_date"], sig_trades["pnl"].cumsum() * 100,
                    color=color, linewidth=1.5, label=f"{sig} ({len(sig_trades)} trades)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Cumulative P&L by Signal")
    ax.set_ylabel("$ P&L")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    chart_path = os.path.join(OUTPUT_DIR, "backtest_charts.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Charts saved to {chart_path}")

    # Monthly P&L heatmap
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    trades_copy = trades.copy()
    trades_copy["year"] = trades_copy["trade_date"].dt.year
    trades_copy["month_num"] = trades_copy["trade_date"].dt.month
    monthly_pnl = trades_copy.pivot_table(
        values="pnl", index="year", columns="month_num", aggfunc="sum"
    ) * 100
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_pnl.columns = [month_labels[m - 1] for m in monthly_pnl.columns]

    im = ax2.imshow(monthly_pnl.values, cmap="RdYlGn", aspect="auto")
    ax2.set_yticks(range(len(monthly_pnl.index)))
    ax2.set_yticklabels(monthly_pnl.index)
    ax2.set_xticks(range(len(monthly_pnl.columns)))
    ax2.set_xticklabels(monthly_pnl.columns)

    for i in range(monthly_pnl.shape[0]):
        for j in range(monthly_pnl.shape[1]):
            val = monthly_pnl.values[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f"${val:.0f}", ha="center", va="center",
                         fontsize=9, fontweight="bold",
                         color="white" if abs(val) > monthly_pnl.values[~np.isnan(monthly_pnl.values)].std() else "black")

    ax2.set_title("Monthly P&L Heatmap ($ per contract)", fontweight="bold")
    plt.colorbar(im, ax=ax2, label="$ P&L")
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "monthly_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {heatmap_path}")


def main():
    print("=" * 60)
    print("  XSP 1DTE VRP Iron Condor Backtest")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print("=" * 60)

    df = download_data()
    print(f"Data loaded: {len(df)} rows from {df.index[0].strftime('%Y-%m-%d')} "
          f"to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"SPY range: ${df['spy_close'].min():.2f} – ${df['spy_close'].max():.2f}")
    print(f"VIX1D range: {df['vix1d'].min():.2f} – {df['vix1d'].max():.2f}")

    trades = run_backtest(df)
    print(f"\nGenerated {len(trades)} trades")

    # Save full trade log
    csv_path = os.path.join(OUTPUT_DIR, "trade_log.csv")
    trades.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Trade log saved to {csv_path}")

    # Overall stats
    print_stats(trades, "ALL TRADES (every day, regardless of signal)")

    # Stats by signal
    for sig in ["LOAD", "THIN", "SKIP"]:
        sig_trades = trades[trades["signal"] == sig].reset_index(drop=True)
        print_stats(sig_trades, f"SIGNAL = {sig}")

    # Stats: only when signal is NOT skip
    active = trades[trades["signal"] != "SKIP"].reset_index(drop=True)
    print_stats(active, "ACTIVE TRADES (LOAD + THIN only)")

    # Generate charts
    print("\nGenerating charts...")
    generate_plots(trades)

    # Streak analysis
    print("\n" + "=" * 60)
    print("  STREAK ANALYSIS")
    print("=" * 60)
    streaks = []
    current_streak = 0
    streak_type = None
    for _, row in trades.iterrows():
        if row["won"]:
            if streak_type == "win":
                current_streak += 1
            else:
                if streak_type is not None:
                    streaks.append((streak_type, current_streak))
                current_streak = 1
                streak_type = "win"
        else:
            if streak_type == "loss":
                current_streak += 1
            else:
                if streak_type is not None:
                    streaks.append((streak_type, current_streak))
                current_streak = 1
                streak_type = "loss"
    if streak_type:
        streaks.append((streak_type, current_streak))

    win_streaks = [s[1] for s in streaks if s[0] == "win"]
    loss_streaks = [s[1] for s in streaks if s[0] == "loss"]
    print(f"  Longest win streak:   {max(win_streaks) if win_streaks else 0}")
    print(f"  Longest loss streak:  {max(loss_streaks) if loss_streaks else 0}")
    print(f"  Avg win streak:       {np.mean(win_streaks):.1f}" if win_streaks else "")
    print(f"  Avg loss streak:      {np.mean(loss_streaks):.1f}" if loss_streaks else "")

    # Drawdown analysis
    cum_pnl = trades["pnl"].cumsum() * 100
    peak = cum_pnl.cummax()
    dd = cum_pnl - peak
    max_dd_idx = dd.idxmin()
    peak_idx = cum_pnl[:max_dd_idx + 1].idxmax()

    print(f"\n  Max drawdown period:")
    print(f"    Peak:     {trades.loc[peak_idx, 'trade_date'].strftime('%Y-%m-%d')} "
          f"(${cum_pnl[peak_idx]:.2f})")
    print(f"    Trough:   {trades.loc[max_dd_idx, 'trade_date'].strftime('%Y-%m-%d')} "
          f"(${cum_pnl[max_dd_idx]:.2f})")
    print(f"    Drawdown: ${dd[max_dd_idx]:.2f}")

    print("\n" + "=" * 60)
    print("  BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
