"""
strategy_lab.py - Step 3: backtest Kalshi 15-minute Up/Down strategies on recorded data.

  python3 strategy_lab.py                       # uses ticks.csv, kalshi_ticks.csv, kalshi_settlements.csv
  python3 strategy_lab.py --dir data/gh-run-3   # or point at a folder holding those three files

Needs data from the ORDER-BOOK version of kalshi_recorder.py (Nov 11 2026 or later).
Older kalshi_ticks.csv files came from a cached endpoint (~30s stale) and give fake results.

Tests:
  1. Model calibration    - does our fair-value model predict real settlements?
  2. Market calibration   - do Kalshi prices predict settlements? (long-shot / favorite bias)
  3. "Take the gap"       - buy when model value beats the ask by X cents, pay taker fee, hold to settlement
  4. Late-window          - same, only in the last 3 minutes
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

SERIES = {"KXBTC15M": "BTC-USD", "KXETH15M": "ETH-USD", "KXSOL15M": "SOL-USD",
          "KXXRP15M": "XRP-USD", "KXDOGE15M": "DOGE-USD"}
CONTRACTS = 10   # order size used for fee rounding


def taker_fee_per_contract(p, n=CONTRACTS):
    """Kalshi taker fee: round up(0.07 * C * P * (1-P)) to the cent, per order."""
    return math.ceil(round(0.07 * n * p * (1 - p) * 100, 6)) / 100 / n


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def load(d):
    s = pd.read_csv(os.path.join(d, "ticks.csv"))
    s["ts"] = pd.to_datetime(s["exch_ts"], unit="s").dt.floor("1s")
    s["mid"] = np.where(s["best_bid"].notna() & s["best_ask"].notna(),
                        (s["best_bid"] + s["best_ask"]) / 2, s["price"])
    px = s.pivot_table(index="ts", columns="product", values="mid", aggfunc="last").resample("1s").last().ffill()
    vol = np.log(px).diff().rolling(900, min_periods=300).std()

    st = pd.read_csv(os.path.join(d, "kalshi_settlements.csv")).drop_duplicates("ticker")
    st["y"] = (st["result"] == "yes").astype(int)

    k = pd.read_csv(os.path.join(d, "kalshi_ticks.csv"))
    k["floor_strike"] = k["floor_strike"].fillna(k["ticker"].map(st.set_index("ticker")["floor_strike"]))
    k = k.dropna(subset=["yes_bid", "yes_ask", "floor_strike"])
    k = k[(k.yes_bid > 0) & (k.yes_ask < 1) & (k.yes_ask > k.yes_bid)]
    k["ts"] = pd.to_datetime(k["recv_ts"], unit="s").dt.floor("1s")
    k["close"] = pd.to_datetime(k["close_time"], utc=True).dt.tz_localize(None)
    k["secs_left"] = (k["close"] - k["ts"]).dt.total_seconds()
    k = k[k.secs_left > 0].drop_duplicates(["ticker", "ts"])
    k = k.merge(st[["ticker", "y"]], on="ticker", how="inner")      # only settled windows

    # fair value: P(average of last 60s >= strike) under a random walk from current spot
    fv = []
    for r in k.itertuples():
        p = SERIES.get(r.series)
        if p not in px or r.ts not in px.index:
            fv.append(np.nan); continue
        S, sig = px.at[r.ts, p], vol.at[r.ts, p]
        if not (np.isfinite(S) and np.isfinite(sig) and sig > 0):
            fv.append(np.nan); continue
        tau = max(r.secs_left - 30, 1)
        fv.append(ncdf(math.log(S / r.floor_strike) / (sig * math.sqrt(tau))))
    k["fv"] = fv
    k = k.dropna(subset=["fv"])
    k["mid"] = (k.yes_bid + k.yes_ask) / 2
    return k, st


def calib(df, col, label):
    b = pd.cut(df[col], [0, .05, .15, .3, .45, .55, .7, .85, .95, 1])
    t = df.groupby(b, observed=True).agg(n=("y", "size"), predicted=(col, "mean"), actual=("y", "mean"))
    print(f"\n== {label} (one snapshot per window per minute) ==")
    print(t.round(3).to_string())


def take_gap(k, thresholds, label, mask=None):
    """First signal per market per side; hold to settlement; taker fee."""
    d = k if mask is None else k[mask]
    print(f"\n== {label} ==")
    print("edge>=   trades  win%   avg_price  avg_model  P&L/contract  total($, 10 per trade)")
    for th in thresholds:
        trades = []
        for tk, g in d.groupby("ticker"):
            g = g.sort_values("ts")
            buy_yes = g[g.fv - g.yes_ask >= th]
            buy_no = g[(1 - g.fv) - (1 - g.yes_bid) >= th]
            if len(buy_yes):
                r = buy_yes.iloc[0]; p = r.yes_ask
                trades.append((p, r.fv, r.y, r.y - p - taker_fee_per_contract(p)))
            if len(buy_no):
                r = buy_no.iloc[0]; p = 1 - r.yes_bid
                trades.append((p, 1 - r.fv, 1 - r.y, (1 - r.y) - p - taker_fee_per_contract(p)))
        if not trades:
            print(f"{th*100:>5.0f}c   {0:>6}"); continue
        t = np.array(trades)
        pnl = t[:, 3]
        se = pnl.std(ddof=1) / math.sqrt(len(pnl)) if len(pnl) > 1 else float("nan")
        print(f"{th*100:>5.0f}c   {len(t):>6}  {100*t[:,2].mean():>4.0f}%   {t[:,0].mean():>8.2f}  "
              f"{t[:,1].mean():>9.2f}  {100*pnl.mean():>+8.1f}c ±{100*se:.1f}  {pnl.sum()*CONTRACTS:>+10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    a = ap.parse_args()
    k, st = load(a.dir)
    print(f"{st.shape[0]} settled windows, {k.ticker.nunique()} with order-book data, {len(k):,} snapshots")
    print(f"YES base rate: {st.y.mean():.3f}")
    print(f"Median spread: {100*(k.yes_ask-k.yes_bid).median():.1f}c   "
          f"median |model - mid|: {100*(k.fv-k.mid).abs().median():.1f}c")

    snap = k[k.ts.dt.second == 0]
    calib(snap, "fv", "1. Model calibration")
    calib(snap, "mid", "2. Kalshi price calibration")
    take_gap(k, [.02, .03, .05, .08, .12], "3. Take the gap (any time in window)")
    take_gap(k, [.02, .03, .05, .08], "4. Late window (last 3 minutes only)", k.secs_left <= 180)
    take_gap(k, [.03, .05, .08], "   Mid window only (3-12 min left)", (k.secs_left > 180) & (k.secs_left <= 720))
    print("\nHow to read: P&L/contract is after the taker fee. '±' is one standard error -")
    print("an edge is only believable when P&L is positive by more than ~2x the ± value,")
    print("across several days of data. A few hours is never enough.")


if __name__ == "__main__":
    main()
