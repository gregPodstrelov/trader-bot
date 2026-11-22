"""
strategy_lab.py - backtest Kalshi 15-minute Up/Down strategies on recorded data.

  python3 strategy_lab.py --dirs data/run-a data/run-b ...   # each dir holds ticks.csv, kalshi_ticks.csv, kalshi_settlements.csv
  python3 strategy_lab.py --dirs runs/* --out results.md     # also write a markdown report

Needs data from the ORDER-BOOK version of kalshi_recorder.py (commit c2fc7b9, Nov 11 2026, or later).
Each recording run is processed on its own so gaps between runs never create fake signals.

Sections:
  0. Data check           - is the Kalshi book really live (how often does the quote change)?
  1. Model calibration    - does the fair-value model predict real settlements?
  2. Market calibration   - do Kalshi prices predict settlements? (long-shot / favorite bias)
  3. Repricing speed      - after a sharp BTC move, how fast does the Kalshi book catch up to the model?
  4. Take-the-gap backtests - buy when model beats the price by X cents, filled ~1s LATER at whatever
                              the book shows then, taker fee paid, held to settlement.
"""
import argparse
import io
import math
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

SERIES = {"KXBTC15M": "BTC-USD", "KXETH15M": "ETH-USD", "KXSOL15M": "SOL-USD",
          "KXXRP15M": "XRP-USD", "KXDOGE15M": "DOGE-USD"}
CONTRACTS = 10   # order size used for fee rounding and the size check


def taker_fee_per_contract(p, n=CONTRACTS):
    """Kalshi taker fee: round up(0.07 * C * P * (1-P)) to the cent, per order."""
    return math.ceil(round(0.07 * n * p * (1 - p) * 100, 6)) / 100 / n


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def load_run(d):
    """Load one recording run; returns (kalshi snapshots with fair value, settlements, spot prices)."""
    s = pd.read_csv(os.path.join(d, "ticks.csv"))
    s["ts"] = pd.to_datetime(s["exch_ts"], unit="s").dt.floor("1s")
    s["mid"] = np.where(s["best_bid"].notna() & s["best_ask"].notna(),
                        (s["best_bid"] + s["best_ask"]) / 2, s["price"])
    px = s.pivot_table(index="ts", columns="product", values="mid", aggfunc="last").resample("1s").last()
    px = px.ffill(limit=30)                                   # never bridge long gaps
    vol = np.log(px).diff().rolling(900, min_periods=300).std()

    st = pd.read_csv(os.path.join(d, "kalshi_settlements.csv")).drop_duplicates("ticker")
    st = st[st.result.isin(["yes", "no"])].copy()
    st["y"] = (st["result"] == "yes").astype(int)

    k = pd.read_csv(os.path.join(d, "kalshi_ticks.csv"))
    k["floor_strike"] = k["floor_strike"].fillna(k["ticker"].map(st.set_index("ticker")["floor_strike"]))
    k = k.dropna(subset=["yes_bid", "yes_ask", "floor_strike"])
    k = k[(k.yes_bid > 0) & (k.yes_ask < 1) & (k.yes_ask > k.yes_bid)]
    k["ts"] = pd.to_datetime(k["recv_ts"], unit="s").dt.floor("1s")
    k["close"] = pd.to_datetime(k["close_time"], utc=True).dt.tz_localize(None)
    k["secs_left"] = (k["close"] - k["ts"]).dt.total_seconds()
    k = k[k.secs_left > 0].drop_duplicates(["ticker", "ts"])

    # vectorised fair value: P(avg of last 60s >= strike) under a random walk from current spot
    k["product"] = k.series.map(SERIES)
    S = np.full(len(k), np.nan); sig = np.full(len(k), np.nan)
    for prod, idx in k.groupby("product").groups.items():
        if prod not in px:
            continue
        ts = k.loc[idx, "ts"]
        S[k.index.get_indexer(idx)] = px[prod].reindex(ts).to_numpy()
        sig[k.index.get_indexer(idx)] = vol[prod].reindex(ts).to_numpy()
    tau = np.maximum(k.secs_left.to_numpy() - 30, 1)
    z = np.log(S / k.floor_strike.to_numpy()) / (sig * np.sqrt(tau))
    k["fv"] = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    k = k[np.isfinite(k.fv)]
    k["mid"] = (k.yes_bid + k.yes_ask) / 2

    # what the book shows ~1 second later (our realistic fill price)
    k = k.sort_values(["ticker", "ts"])
    g = k.groupby("ticker")
    for c in ["yes_bid", "yes_ask", "yes_bid_size", "yes_ask_size", "ts"]:
        k["next_" + c] = g[c].shift(-1)
    k["fill_ok"] = (k.next_ts - k.ts).dt.total_seconds().between(0.5, 3)
    return k, st, px


def load_all(dirs):
    ks, sts, pxs = [], [], []
    for d in dirs:
        try:
            k, st, px = load_run(d)
        except Exception as e:                       # a broken/empty run shouldn't stop the analysis
            print(f"  skipped {d}: {e!r}")
            continue
        k["run"] = os.path.basename(d.rstrip("/"))
        ks.append(k); sts.append(st); pxs.append((k["run"].iat[0] if len(k) else d, px, k))
    st = pd.concat(sts).drop_duplicates("ticker")
    k = pd.concat(ks).drop_duplicates(["ticker", "ts"])
    k = k.merge(st[["ticker", "y"]], on="ticker", how="inner")       # settled windows only
    return k, st, pxs


def data_check(k):
    print("\n## 0. Data check: is the Kalshi book live?")
    rows = []
    for s, g in k.groupby("series"):
        g = g.sort_values(["ticker", "ts"])
        chg = (g.groupby("ticker").yes_bid.diff().fillna(0) != 0) | (g.groupby("ticker").yes_ask.diff().fillna(0) != 0)
        rows.append((s, len(g), g.ticker.nunique(), 100 * chg.mean(),
                     100 * (g.yes_ask - g.yes_bid).median(), g.yes_ask_size.median()))
    t = pd.DataFrame(rows, columns=["series", "snapshots", "windows", "%snapshots w/ quote change",
                                    "median spread (c)", "median ask size"]).set_index("series")
    print(t.round(1).to_string())
    print("(The old cached data changed quotes on ~3% of snapshots. Live data should be far higher.)")


def calib(df, col, label):
    b = pd.cut(df[col], [0, .05, .15, .3, .45, .55, .7, .85, .95, 1])
    t = df.groupby(b, observed=True).agg(n=("y", "size"), predicted=(col, "mean"), actual=("y", "mean"))
    print(f"\n## {label}\n(one snapshot per window per minute)")
    print(t.round(3).to_string())


def repricing(pxs, move_bps=5):
    print(f"\n## 3. Repricing speed: BTC moves >= {move_bps} bps in 1s")
    print("% of the model's fair-value jump that the Kalshi mid has absorbed after N seconds")
    H = [0, 1, 2, 3, 5, 10, 20]
    fr = {s: {h: [] for h in H} for s in SERIES}
    left = {s: [] for s in SERIES}
    for run, px, k in pxs:
        if "BTC-USD" not in px or k.empty:
            continue
        b = np.log(px["BTC-USD"]).diff()
        ev = b[b.abs() >= move_bps / 1e4].index
        kept, last = [], None
        for t in ev:
            if last is None or (t - last).total_seconds() > 30:
                kept.append(t); last = t
        for s in SERIES:
            g = k[k.series == s].drop_duplicates("ts").set_index("ts")
            if g.empty:
                continue
            for t in kept:
                tp = t - pd.Timedelta(seconds=1)
                if tp not in g.index or t not in g.index or g.at[tp, "ticker"] != g.at[t, "ticker"]:
                    continue
                jump = g.at[t, "fv"] - g.at[tp, "fv"]
                if abs(jump) < 0.03:
                    continue
                for h in H:
                    th = t + pd.Timedelta(seconds=h)
                    if th in g.index and g.at[th, "ticker"] == g.at[t, "ticker"]:
                        fr[s][h].append((g.at[th, "mid"] - g.at[tp, "mid"]) / jump)
                th = t + pd.Timedelta(seconds=1)
                if th in g.index:
                    price = g.at[th, "yes_ask"] if jump > 0 else 1 - g.at[th, "yes_bid"]
                    model = g.at[t, "fv"] if jump > 0 else 1 - g.at[t, "fv"]
                    left[s].append(model - price - taker_fee_per_contract(price))
    print("series     " + "".join(f"{h:>6}s" for h in H) + "     n   edge left at +1s after fee (median, c)")
    for s in SERIES:
        n = len(fr[s][0])
        row = "".join(f"{100*np.median(fr[s][h]):>6.0f}%" if fr[s][h] else "      -" for h in H)
        e = f"{100*np.median(left[s]):+.1f}" if left[s] else "-"
        print(f"{s:<11}{row}  {n:>4}   {e}")


def take_gap(k, thresholds, label, mask=None, show_breakdown=False):
    """First signal per market per side. Fill at the NEXT snapshot's price (~1s later) if the
    book there still has CONTRACTS available; pay taker fee; hold to settlement."""
    d = k if mask is None else k[mask]
    d = d[d.fill_ok]
    print(f"\n## {label}")
    print("edge>=  trades  win%  avg_fill  avg_model  P&L/contract (±1 s.e.)  total $ @10/trade  t-stat")
    best = None
    for th in thresholds:
        trades = []
        for tk, g in d.groupby("ticker", sort=False):
            sig_yes = g[(g.fv - g.yes_ask >= th) & (g.next_yes_ask_size >= CONTRACTS)]
            sig_no = g[((1 - g.fv) - (1 - g.yes_bid) >= th) & (g.next_yes_bid_size >= CONTRACTS)]
            if len(sig_yes):
                r = sig_yes.iloc[0]; p = r.next_yes_ask
                trades.append((r.series, r.ts.date(), p, r.fv, r.y, r.y - p - taker_fee_per_contract(p)))
            if len(sig_no):
                r = sig_no.iloc[0]; p = 1 - r.next_yes_bid
                trades.append((r.series, r.ts.date(), p, 1 - r.fv, 1 - r.y, (1 - r.y) - p - taker_fee_per_contract(p)))
        if not trades:
            print(f"{th*100:>4.0f}c  {0:>6}"); continue
        t = pd.DataFrame(trades, columns=["series", "day", "price", "model", "win", "pnl"])
        se = t.pnl.std(ddof=1) / math.sqrt(len(t)) if len(t) > 1 else float("nan")
        tstat = t.pnl.mean() / se if se and se > 0 else float("nan")
        print(f"{th*100:>4.0f}c  {len(t):>6}  {100*t.win.mean():>3.0f}%  {t.price.mean():>8.2f}  {t.model.mean():>9.2f}"
              f"  {100*t.pnl.mean():>+7.2f}c ±{100*se:.2f}          {t.pnl.sum()*CONTRACTS:>+9.2f}   {tstat:>+5.1f}")
        if best is None or (np.isfinite(tstat) and tstat > best[0]):
            best = (tstat, th, t)
    if show_breakdown and best is not None:
        _, th, t = best
        print(f"\n  Breakdown at the strongest threshold ({th*100:.0f}c):")
        print("  by coin:  " + "  ".join(f"{s.replace('KX','').replace('15M','')} {100*g.pnl.mean():+.1f}c (n={len(g)})"
                                      for s, g in t.groupby("series")))
        daily = t.groupby("day").pnl.agg(["size", "mean"])
        print(f"  by day:   {int((daily['mean'] > 0).sum())} of {len(daily)} days positive; "
              f"worst day {100*daily['mean'].min():+.1f}c/contract, best {100*daily['mean'].max():+.1f}c")
        print("  " + " ".join(f"{str(dd)[5:]}:{100*m:+.0f}c" for dd, m in daily['mean'].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["."])
    ap.add_argument("--out", default=None, help="also write the report to this markdown file")
    a = ap.parse_args()

    buf = io.StringIO()
    with redirect_stdout(buf):
        k, st, pxs = load_all(sorted(a.dirs))
        print(f"# Strategy lab results\n")
        print(f"Runs: {k.run.nunique()}   settled windows with book data: {k.ticker.nunique()}   "
              f"snapshots: {len(k):,}   span: {k.ts.min()} -> {k.ts.max()} UTC")
        print(f"YES base rate: {k.drop_duplicates('ticker').y.mean():.3f}   "
              f"median |model - mid|: {100*(k.fv-k.mid).abs().median():.1f}c")
        print("\n```")
        data_check(k)
        snap = k[k.ts.dt.second == 0]
        calib(snap, "fv", "1. Model calibration")
        calib(snap, "mid", "2. Kalshi price calibration")
        repricing(pxs)
        take_gap(k, [.02, .03, .05, .08, .12], "4a. Take the gap, any time in the window", show_breakdown=True)
        take_gap(k, [.02, .03, .05, .08], "4b. Last 3 minutes only", k.secs_left <= 180, show_breakdown=True)
        take_gap(k, [.03, .05, .08], "4c. Mid window only (3-12 min left)", (k.secs_left > 180) & (k.secs_left <= 720))
        take_gap(k, [.03, .05, .08], "4d. Long shots only (fill price <= 20c)",
                 (k.yes_ask <= .2) | (k.yes_bid >= .8))
        print("```")
        print("\nHow to read: fills use the book ~1s after the signal, with the taker fee. "
              "An edge is believable only with t-stat > 2 AND most days positive.")
    out = buf.getvalue()
    print(out)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
