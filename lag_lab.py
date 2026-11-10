"""
lag_lab.py - Step 1 of a "alts trail Bitcoin" bot: MEASURE the lag before trading it.

Two modes:
  python lag_lab.py record  --minutes 120            # stream live ticks to ticks.csv
  python lag_lab.py analyze --file ticks.csv         # measure how alts follow BTC

Recording uses Coinbase's public market-data websocket (no API key needed).
Install:  pip install websockets pandas numpy certifi
"""
import argparse
import asyncio
import csv
import json
import time
from datetime import datetime

COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
LEADER = "BTC-USD"
FOLLOWERS = ["ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD", "AVAX-USD", "LINK-USD", "ADA-USD"]


# --------------------------------------------------------------------------- record
def _ssl_context():
    """Use certifi's CA bundle so python.org installs on macOS verify TLS correctly."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


async def record(outfile: str, minutes: float):
    import websockets

    products = [LEADER] + FOLLOWERS
    sub = {"type": "subscribe", "product_ids": products, "channels": ["ticker"]}
    end = time.time() + minutes * 60
    n = 0
    with open(outfile, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["recv_ts", "exch_ts", "product", "price", "best_bid", "best_ask"])
        while time.time() < end:
            try:
                async with websockets.connect(COINBASE_WS, ping_interval=20, ssl=_ssl_context()) as ws:
                    await ws.send(json.dumps(sub))
                    print(f"Connected. Recording {products} until {datetime.fromtimestamp(end)}")
                    while time.time() < end:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                        if msg.get("type") != "ticker":
                            continue
                        exch_ts = datetime.fromisoformat(msg["time"].replace("Z", "+00:00")).timestamp()
                        w.writerow([f"{time.time():.6f}", f"{exch_ts:.6f}", msg["product_id"],
                                    msg["price"], msg.get("best_bid"), msg.get("best_ask")])
                        n += 1
                        if n % 2000 == 0:
                            f.flush()
                            print(f"{n} ticks recorded")
            except Exception as e:  # reconnect on drops
                print(f"Websocket error: {e!r}; reconnecting in 2s")
                await asyncio.sleep(2)
    print(f"Done. {n} ticks written to {outfile}")


# -------------------------------------------------------------------------- analyze
def analyze(infile: str, bar_ms: int, move_bps: float, horizon_s: int):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(infile)
    df["ts"] = pd.to_datetime(df["exch_ts"], unit="s")
    # Use mid price where available (less bid/ask bounce noise than last trade)
    df["mid"] = np.where(df["best_bid"].notna() & df["best_ask"].notna(),
                         (df["best_bid"] + df["best_ask"]) / 2, df["price"])
    px = (df.pivot_table(index="ts", columns="product", values="mid", aggfunc="last")
            .resample(f"{bar_ms}ms").last().ffill().dropna())
    if LEADER not in px:
        raise SystemExit("No BTC data in file.")
    rets = np.log(px).diff().dropna()
    secs = rets.index.to_series().diff().dt.total_seconds().median()
    print(f"\n{len(px):,} bars of {bar_ms} ms covering {(px.index[-1]-px.index[0])}\n")

    btc = rets[LEADER]
    max_lag = int(horizon_s * 1000 / bar_ms)

    # 1) Lead-lag correlation: corr(BTC return now, alt return k bars later)
    print("=== Lead-lag correlation (BTC return at t vs alt return at t+lag) ===")
    lags = [0, 1, 2, 4, 8, 20, 40]
    lags = [l for l in lags if l <= max_lag]
    header = "product".ljust(10) + "".join(f"{l*bar_ms/1000:>8.1f}s" for l in lags) + "    beta"
    print(header)
    betas = {}
    for p in [c for c in rets.columns if c != LEADER]:
        alt = rets[p]
        row = [btc.corr(alt.shift(-l)) for l in lags]
        # beta on ~1-minute returns (captures full catch-up, not just instant reaction)
        k = max(1, int(60_000 / bar_ms))
        r1 = np.log(px).diff(k).dropna()
        betas[p] = r1[p].cov(r1[LEADER]) / r1[LEADER].var()
        print(p.ljust(10) + "".join(f"{v:>9.3f}" for v in row) + f"{betas[p]:>8.2f}")
    print("If correlation at lag>0 is near zero, the alt reacts instantly: no tradable lag.\n")

    # 2) Event study: after a sharp BTC move, how much of the expected alt move
    #    (beta x BTC move) has happened at each horizon?
    print(f"=== Event study: BTC moves >= {move_bps} bps within 1s ===")
    w = max(1, int(1000 / bar_ms))
    btc_move = np.log(px[LEADER]).diff(w)
    events = btc_move[btc_move.abs() >= move_bps / 1e4].index
    # de-cluster: keep events at least horizon apart
    kept, last = [], None
    for t in events:
        if last is None or (t - last).total_seconds() > horizon_s:
            kept.append(t); last = t
    print(f"{len(kept)} events found")
    if not kept:
        print("No events - record longer or lower --move-bps.")
        return
    checkpoints = [s for s in [0.5, 1, 2, 5, 10, 20, 30, 60] if s <= horizon_s]
    print("product".ljust(10) + "".join(f"{s:>7}s" for s in checkpoints) + "   (% of beta-implied move done)")
    logpx = np.log(px)
    for p in betas:
        fracs = {s: [] for s in checkpoints}
        for t in kept:
            i = px.index.get_loc(t)
            if i - w < 0:
                continue
            b_mv = logpx[LEADER].iloc[i] - logpx[LEADER].iloc[i - w]
            expected = betas[p] * b_mv
            if abs(expected) < 1e-6:
                continue
            for s in checkpoints:
                j = i + int(s * 1000 / bar_ms)
                if j < len(px):
                    fracs[s].append((logpx[p].iloc[j] - logpx[p].iloc[i - w]) / expected)
        print(p.ljust(10) + "".join(
            f"{100*np.median(fracs[s]):>7.0f}%" if fracs[s] else "      -" for s in checkpoints))
    print("\nRead it like this: if ETH shows 40% at 0.5s and 95% at 5s, the lag lasts a few")
    print("seconds. Your bot must detect, decide and fill well inside that window, AND the")
    print("remaining move must exceed fees + spread. Usually only thin alts pass that test.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    r = sp.add_parser("record"); r.add_argument("--file", default="ticks.csv"); r.add_argument("--minutes", type=float, default=60)
    a = sp.add_parser("analyze"); a.add_argument("--file", default="ticks.csv")
    a.add_argument("--bar-ms", type=int, default=250); a.add_argument("--move-bps", type=float, default=15)
    a.add_argument("--horizon-s", type=int, default=60)
    args = ap.parse_args()
    if args.mode == "record":
        asyncio.run(record(args.file, args.minutes))
    else:
        analyze(args.file, args.bar_ms, args.move_bps, args.horizon_s)
