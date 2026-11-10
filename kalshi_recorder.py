"""
kalshi_recorder.py - Step 2: record Kalshi 15-minute crypto Up/Down prices, then
measure how fast Kalshi reprices after Bitcoin moves.

Run it at the SAME TIME as `lag_lab.py record` (two terminal windows):
  python3 kalshi_recorder.py record --minutes 240       # writes kalshi_ticks.csv + kalshi_settlements.csv
  python3 kalshi_recorder.py analyze                    # needs ticks.csv (from lag_lab) + kalshi_ticks.csv

Uses Kalshi's public market-data API (no account or API key needed for this step).
Install:  python3 -m pip install pandas numpy certifi
"""
import argparse
import csv
import json
import math
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

API = "https://external-api.kalshi.com/trade-api/v2"
# Kalshi series -> matching Coinbase product recorded by lag_lab.py
SERIES = {
    "KXBTC15M": "BTC-USD",
    "KXETH15M": "ETH-USD",
    "KXSOL15M": "SOL-USD",
    "KXXRP15M": "XRP-USD",
    "KXDOGE15M": "DOGE-USD",
}
COLS = ["recv_ts", "series", "ticker", "status", "open_time", "close_time", "floor_strike",
        "yes_bid", "yes_ask", "yes_bid_size", "yes_ask_size", "last_price", "volume"]


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CTX = _ctx()


def get(path, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API + path, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5, context=CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:              # rate limited: back off
                time.sleep(1 + attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    return None


def fetch_series(series):
    data = get(f"/markets?series_ticker={series}&status=open&limit=20")
    return series, (data or {}).get("markets", [])


# --------------------------------------------------------------------------- record
def record(minutes, ticks_file, settle_file):
    end = time.time() + minutes * 60
    pending = {}          # ticker -> close_time epoch, waiting for a result
    settled = set()
    n = 0
    last_report = 0

    # startup check
    print("Checking which Kalshi series are live...")
    with ThreadPoolExecutor(len(SERIES)) as pool:
        for s, mkts in pool.map(fetch_series, SERIES):
            print(f"  {s:<10} {len(mkts)} open market(s)" + (f"  e.g. {mkts[0]['ticker']}" if mkts else "  (none - skipped if it stays empty)"))

    tf = open(ticks_file, "a", newline="")
    sf = open(settle_file, "a", newline="")
    tw, sw = csv.writer(tf), csv.writer(sf)
    if tf.tell() == 0:
        tw.writerow(COLS)
    if sf.tell() == 0:
        sw.writerow(["ticker", "series", "close_time", "floor_strike", "expiration_value", "result"])

    pool = ThreadPoolExecutor(len(SERIES))
    print(f"Recording until {datetime.fromtimestamp(end):%H:%M:%S}  (Ctrl+C to stop early)")
    try:
        while time.time() < end:
            t0 = time.time()
            for series, mkts in pool.map(fetch_series, SERIES):
                for m in mkts:
                    tw.writerow([f"{t0:.3f}", series, m.get("ticker"), m.get("status"),
                                 m.get("open_time"), m.get("close_time"), m.get("floor_strike"),
                                 m.get("yes_bid_dollars"), m.get("yes_ask_dollars"),
                                 m.get("yes_bid_size_fp"), m.get("yes_ask_size_fp"),
                                 m.get("last_price_dollars"), m.get("volume_fp")])
                    n += 1
                    tk = m.get("ticker")
                    if tk and tk not in settled and m.get("close_time"):
                        ct = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()
                        pending[tk] = (series, ct)

            # look up results of markets that closed > 60s ago
            now = time.time()
            for tk, (series, ct) in list(pending.items()):
                if now > ct + 60:
                    d = get(f"/markets/{tk}") or {}
                    m = d.get("market", {})
                    if m.get("result") in ("yes", "no"):
                        sw.writerow([tk, series, m.get("close_time"), m.get("floor_strike"),
                                     m.get("expiration_value"), m.get("result")])
                        settled.add(tk); pending.pop(tk)
                    elif now > ct + 900:       # give up after 15 min
                        pending.pop(tk)

            if now - last_report > 30:
                tf.flush(); sf.flush()
                print(f"{datetime.now():%H:%M:%S}  {n} price rows, {len(settled)} settlements logged")
                last_report = now
            time.sleep(max(0, 1.0 - (time.time() - t0)))   # ~1 snapshot per second
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        tf.close(); sf.close(); pool.shutdown()
    print(f"Done. {n} rows -> {ticks_file}, {len(settled)} settlements -> {settle_file}")


# -------------------------------------------------------------------------- analyze
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def analyze(spot_file, kalshi_file, move_bps):
    import numpy as np
    import pandas as pd

    # Spot prices (Coinbase, from lag_lab.py) on a 1-second grid
    s = pd.read_csv(spot_file)
    s["ts"] = pd.to_datetime(s["exch_ts"], unit="s").dt.floor("1s")
    s["mid"] = np.where(s["best_bid"].notna() & s["best_ask"].notna(),
                        (s["best_bid"] + s["best_ask"]) / 2, s["price"])
    spot = s.pivot_table(index="ts", columns="product", values="mid", aggfunc="last")
    spot = spot.resample("1s").last().ffill()
    logspot = np.log(spot)
    vol = logspot.diff().rolling(900, min_periods=120).std()   # per-second volatility, 15-min window

    # Kalshi snapshots: keep the market closing soonest for each series each second
    k = pd.read_csv(kalshi_file)
    k = k.dropna(subset=["yes_bid", "yes_ask", "floor_strike"])
    k = k[(k["yes_bid"] > 0) & (k["yes_ask"] < 1)]
    k["ts"] = pd.to_datetime(k["recv_ts"], unit="s").dt.floor("1s")
    k["close"] = pd.to_datetime(k["close_time"], utc=True).dt.tz_localize(None)
    k = k[k["close"] > k["ts"]]
    k = k.sort_values("close").groupby(["series", "ts"]).first().reset_index()
    k["mid"] = (k["yes_bid"] + k["yes_ask"]) / 2
    k["spread"] = k["yes_ask"] - k["yes_bid"]
    k["tau"] = (k["close"] - k["ts"]).dt.total_seconds().clip(lower=1)

    print(f"\nSpot: {spot.index[0]} -> {spot.index[-1]}   Kalshi rows: {len(k):,}")
    print("\nTypical Kalshi spread (cents):")
    print((k.groupby("series")["spread"].median() * 100).round(1).to_string())

    # Fair value from spot: P(finish above strike) under a random walk
    def fair(row):
        p = SERIES.get(row.series)
        if p not in spot or row.ts not in spot.index:
            return np.nan
        S, sig = spot.at[row.ts, p], vol.at[row.ts, p]
        if not np.isfinite(S) or not np.isfinite(sig) or sig <= 0:
            return np.nan
        return norm_cdf(math.log(S / row.floor_strike) / (sig * math.sqrt(row.tau)))
    k["fv"] = [fair(r) for r in k.itertuples()]

    ok = k.dropna(subset=["fv"])
    if ok.empty:
        print("\nNo overlap between spot and Kalshi data yet - record both at the same time.")
        return
    err = (ok["fv"] - ok["mid"]).abs()
    print(f"\nModel vs Kalshi mid: median gap {100*err.median():.1f}c, 90th pct {100*err.quantile(.9):.1f}c")

    # Event study: BTC jumps, then how fast does each Kalshi market follow its new fair value?
    btc = logspot["BTC-USD"].diff()
    ev = btc[btc.abs() >= move_bps / 1e4].index
    kept, last = [], None
    for t in ev:
        if last is None or (t - last).total_seconds() > 30:
            kept.append(t); last = t
    print(f"\n=== {len(kept)} BTC moves >= {move_bps} bps in 1s ===")
    if not kept:
        print("No events - record longer or lower --move-bps.")
        return
    H = [0, 1, 2, 3, 5, 10, 20, 30]
    print("series".ljust(11) + "".join(f"{h:>6}s" for h in H) + "   n   | edge left at +1s/+2s (c, after half-spread)")
    for series in SERIES:
        g = ok[ok.series == series].set_index("ts")
        if g.empty:
            continue
        fracs = {h: [] for h in H}; edge1, edge2 = [], []
        for t in kept:
            t_prev = t - pd.Timedelta(seconds=1)
            if t_prev not in g.index or t not in g.index:
                continue
            jump = g.at[t, "fv"] - g.at[t_prev, "fv"]
            if abs(jump) < 0.02:                     # fair value barely moved: skip
                continue
            base = g.at[t_prev, "mid"]
            for h in H:
                th = t + pd.Timedelta(seconds=h)
                if th in g.index:
                    fracs[h].append((g.at[th, "mid"] - base) / jump)
            for h, bucket in ((1, edge1), (2, edge2)):
                th = t + pd.Timedelta(seconds=h)
                if th in g.index:
                    gap = np.sign(jump) * (g.at[t, "fv"] - g.at[th, "mid"]) - g.at[th, "spread"] / 2
                    bucket.append(gap)
        n = len(fracs[0])
        row = "".join(f"{100*np.median(fracs[h]):>6.0f}%" if fracs[h] else "     -" for h in H)
        e = (f"{100*np.median(edge1):>5.1f} / {100*np.median(edge2):.1f}" if edge1 and edge2 else "   -")
        print(series.ljust(11) + row + f"  {n:>3}   | {e}")
    print("\nRead it like this: % = how much of the fair-value jump Kalshi's price has absorbed.")
    print("If BTC/ETH markets show ~30% at 1s and ~90% at 5s, there's a few-second window.")
    print("'Edge left' must beat the taker fee (~1.5-2c near 50c) for a market order to pay.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    r = sp.add_parser("record")
    r.add_argument("--minutes", type=float, default=240)
    r.add_argument("--file", default="kalshi_ticks.csv")
    r.add_argument("--settle-file", default="kalshi_settlements.csv")
    a = sp.add_parser("analyze")
    a.add_argument("--spot", default="ticks.csv")
    a.add_argument("--kalshi", default="kalshi_ticks.csv")
    a.add_argument("--move-bps", type=float, default=10)
    args = ap.parse_args()
    if args.mode == "record":
        record(args.minutes, args.file, args.settle_file)
    else:
        analyze(args.spot, args.kalshi, args.move_bps)
