"""
weather_backtest.py - backtest Kalshi daily-high temperature markets on HISTORY.

  python3 weather_backtest.py fetch   --days 90 --out wb     # download history (Kalshi + Open-Meteo)
  python3 weather_backtest.py analyze --out wb --report results/weather_backtest.md

Data (no API keys):
  Kalshi   settled brackets (live + /historical), hourly bid/ask candlesticks, official high (expiration_value)
  Open-Meteo Previous Runs API: what GFS / ECMWF / best_match forecast 1 and 2 days ahead, for every past day

Method (walk-forward, no look-ahead):
  forecast for day D = daily max of the model's hourly 2 m temperature, as forecast 24 h (lead 1) earlier
  for each city+model, bias and spread are learned ONLY from days before D (>= 14 days of history)
  P(bracket) = Normal(forecast + bias, spread), with whole-degree rounding
  decision times: 10 pm local the evening before (D-1), and 7 am local on D
  trade: buy YES if P - ask >= edge, buy NO if (1-P) - (1-bid) >= edge; taker fee; hold to settlement
"""
import argparse
import csv
import io
import json
import math
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

KALSHI = "https://external-api.kalshi.com/trade-api/v2"
UA = {"User-Agent": "trader-bot weather research", "Accept": "application/json"}
CITIES = {   # series: (station, name, lat, lon, tz)
    "KXHIGHNY":   ("CLINYC", "New York", 40.7789, -73.9692, "America/New_York"),
    "KXHIGHCHI":  ("CLIMDW", "Chicago", 41.7860, -87.7524, "America/Chicago"),
    "KXHIGHMIA":  ("CLIMIA", "Miami", 25.7906, -80.3164, "America/New_York"),
    "KXHIGHAUS":  ("CLIAUS", "Austin", 30.1945, -97.6699, "America/Chicago"),
    "KXHIGHDEN":  ("CLIDEN", "Denver", 39.8467, -104.6562, "America/Denver"),
    "KXHIGHPHIL": ("CLIPHL", "Philadelphia", 39.8683, -75.2311, "America/New_York"),
    "KXHIGHLAX":  ("CLILAX", "Los Angeles", 33.9425, -118.4081, "America/Los_Angeles"),
}
MODELS = ["gfs_seamless", "ecmwf_ifs025", "best_match"]
CONTRACTS = 10


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CTX = _ctx()
_last = [0.0]


def get(url, retries=4, pace=0.08):
    wait = pace - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(retries):
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30, context=CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1)); continue
            if e.code != 404:
                print(f"  HTTP {e.code}: {url[:120]}")
            return None
        except Exception as e:
            print(f"  {type(e).__name__}: {url[:120]}")
            time.sleep(1 + attempt)
    return None


def paged(path, key, params):
    out, cursor = [], None
    while True:
        p = dict(params, limit=1000)
        if cursor:
            p["cursor"] = cursor
        d = get(f"{KALSHI}{path}?{urllib.parse.urlencode(p)}") or {}
        out += d.get(key, [])
        cursor = d.get("cursor")
        if not cursor or not d.get(key):
            return out


def wcsv(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


# ===================================================================== FETCH
def fetch(days, out):
    now = int(time.time())
    start = now - days * 86400
    mrows, prows, frows = [], [], []
    for series, (_, name, lat, lon, tz) in CITIES.items():
        live = paged("/markets", "markets", {"series_ticker": series, "status": "settled", "min_close_ts": start})
        hist = paged("/historical/markets", "markets", {"series_ticker": series, "min_close_ts": start})
        mk = {m["ticker"]: (m, False) for m in hist}
        mk.update({m["ticker"]: (m, True) for m in live})
        ct = lambda m: int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()) if m.get("close_time") else 0
        mk = {tk: v for tk, v in mk.items() if ct(v[0]) >= start}     # historical endpoint ignores min_close_ts
        print(f"{series}: {len(live)} live + {len(hist)} historical settled brackets; {len(mk)} within {days} days")
        for tk, (m, is_live) in mk.items():
            if m.get("result") not in ("yes", "no"):
                continue
            close = ct(m)
            mrows.append([series, m.get("event_ticker"), tk, m.get("strike_type"), m.get("floor_strike"),
                          m.get("cap_strike"), m["close_time"], m["result"], m.get("expiration_value")])
            path = (f"/series/{series}/markets/{tk}/candlesticks" if is_live
                    else f"/historical/markets/{tk}/candlesticks")
            q = urllib.parse.urlencode({"start_ts": close - 4 * 86400, "end_ts": close, "period_interval": 60})
            c = get(f"{KALSHI}{path}?{q}") or {}
            for k in c.get("candlesticks", []):
                yb = (k.get("yes_bid") or {}).get("close_dollars")
                ya = (k.get("yes_ask") or {}).get("close_dollars")
                if yb is not None or ya is not None:
                    prows.append([tk, k["end_period_ts"], yb, ya, k.get("volume_fp")])

        # forecasts made 1 and 2 days ahead, hourly -> daily max in local time
        d0 = (datetime.now(timezone.utc) - timedelta(days=days + 3)).date()
        d1 = datetime.now(timezone.utc).date()
        for model in MODELS:
            q = urllib.parse.urlencode({
                "latitude": lat, "longitude": lon, "timezone": tz, "temperature_unit": "fahrenheit",
                "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
                "models": model, "start_date": d0.isoformat(), "end_date": d1.isoformat()})
            d = get(f"https://previous-runs-api.open-meteo.com/v1/forecast?{q}", pace=0.3) or {}
            h = d.get("hourly") or {}
            daily = {}
            for i, t in enumerate(h.get("time", [])):
                day = t[:10]
                for lead in (1, 2):
                    v = (h.get(f"temperature_2m_previous_day{lead}") or [None] * (i + 1))[i]
                    if v is not None:
                        daily.setdefault((day, lead), []).append(v)
            for (day, lead), vals in daily.items():
                if len(vals) >= 20:                      # need (almost) the full day
                    frows.append([series, day, model, lead, max(vals)])
        print(f"  prices so far: {len(prows)}  forecasts so far: {len(frows)}")

    wcsv(f"{out}/markets.csv", ["series", "event_ticker", "ticker", "strike_type", "floor_strike",
                                "cap_strike", "close_time", "result", "expiration_value"], mrows)
    wcsv(f"{out}/prices.csv", ["ticker", "end_ts", "yes_bid", "yes_ask", "volume"], prows)
    wcsv(f"{out}/forecasts.csv", ["series", "date", "model", "lead", "tmax_f"], frows)
    print(f"Saved {len(mrows)} brackets, {len(prows)} hourly prices, {len(frows)} daily forecasts to {out}/")


# =================================================================== ANALYZE
def fee(p, n=CONTRACTS):
    return math.ceil(round(0.07 * n * p * (1 - p) * 100, 6)) / 100 / n


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bracket_prob(st, lo, hi, mu, sd):
    """Official high is a whole degree. 'greater' floor f means >= f+1; 'less' cap c means <= c-1."""
    if st == "greater":
        return 1 - ncdf((lo + 0.5 - mu) / sd)
    if st == "less":
        return ncdf((hi - 0.5 - mu) / sd)
    return ncdf((hi + 0.5 - mu) / sd) - ncdf((lo - 0.5 - mu) / sd)


def analyze(out, report):
    import numpy as np
    import pandas as pd

    m = pd.read_csv(f"{out}/markets.csv")
    p = pd.read_csv(f"{out}/prices.csv")
    f = pd.read_csv(f"{out}/forecasts.csv")
    m["y"] = (m.result == "yes").astype(int)
    m["date"] = m.event_ticker.str.split("-").str[1].map(lambda s: datetime.strptime(s, "%y%b%d").date().isoformat())
    actual = (m.dropna(subset=["expiration_value"]).groupby(["series", "date"]).expiration_value.first().reset_index()
              .rename(columns={"expiration_value": "actual"}))

    # forecasts: per model + a blend of GFS and ECMWF
    f = f[f.lead == 1]
    fw = f.pivot_table(index=["series", "date"], columns="model", values="tmax_f").reset_index()
    if {"gfs_seamless", "ecmwf_ifs025"} <= set(fw.columns):
        fw["blend"] = fw[["gfs_seamless", "ecmwf_ifs025"]].mean(axis=1)
    srcs = [c for c in ["best_match", "gfs_seamless", "ecmwf_ifs025", "blend"] if c in fw.columns]
    fa = fw.merge(actual, on=["series", "date"], how="inner").sort_values("date")

    # walk-forward bias/spread per city+source using only earlier days
    params = []
    for series, g in fa.groupby("series"):
        g = g.sort_values("date").reset_index(drop=True)
        for s in srcs:
            err = (g.actual - g[s])
            for i in range(len(g)):
                past = err.iloc[max(0, i - 45):i].dropna()
                if len(past) >= 14 and pd.notna(g.at[i, s]):
                    params.append((series, g.at[i, "date"], s, g.at[i, s] + past.mean(), max(past.std(ddof=1), 1.2)))
    P = pd.DataFrame(params, columns=["series", "date", "src", "mu", "sd"])

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("# Weather backtest (Kalshi daily high temperature)\n")
        print(f"Cities: {', '.join(CITIES[s][1] for s in sorted(m.series.unique()))}   "
              f"days: {m.date.nunique()}   brackets: {len(m):,}   "
              f"span: {m.date.min()} -> {m.date.max()}\n")
        print("```")
        print("## 1. Forecast error (official high minus model, lead 1 day), all days")
        e = fa.melt(id_vars=["series", "date", "actual"], value_vars=srcs, var_name="src", value_name="fc").dropna()
        e["err"] = e.actual - e.fc
        t = e.groupby(["series", "src"]).err.agg(bias="mean", sd="std", mae=lambda x: x.abs().mean()).round(2)
        print(t.unstack("src").to_string())

        # decision-time prices
        p["end_ts"] = p.end_ts.astype(int)
        p = p.dropna(subset=["yes_bid", "yes_ask"])
        p = p[(p.yes_ask > p.yes_bid) & (p.yes_bid > 0) & (p.yes_ask < 1)].sort_values("end_ts")
        rows = []
        for r in m.itertuples():
            tz = ZoneInfo(CITIES[r.series][4])
            D = datetime.fromisoformat(r.date)
            for label, dt in (("eve-before 10pm", datetime(D.year, D.month, D.day, 22, tzinfo=tz) - timedelta(days=1)),
                              ("morning-of 7am", datetime(D.year, D.month, D.day, 7, tzinfo=tz))):
                rows.append((r.ticker, label, int(dt.timestamp())))
        dec = pd.DataFrame(rows, columns=["ticker", "when", "ts"]).sort_values("ts")
        q = pd.merge_asof(dec, p.rename(columns={"end_ts": "ts"}), on="ts", by="ticker",
                          direction="backward", tolerance=3 * 3600)
        q = q.dropna(subset=["yes_bid", "yes_ask"]).merge(m[["ticker", "series", "date", "strike_type",
                                                             "floor_strike", "cap_strike", "y"]], on="ticker")
        q["mid"] = (q.yes_bid + q.yes_ask) / 2
        print(f"\nBracket-times with a quoted market: {len(q):,}   median spread {100*(q.yes_ask-q.yes_bid).median():.1f}c")

        res = q.merge(P, on=["series", "date"])
        res["prob"] = [bracket_prob(r.strike_type, r.floor_strike, r.cap_strike, r.mu, r.sd)
                       for r in res.itertuples()]
        res["prob"] = res.prob.clip(0.001, 0.999)

        print("\n## 2. Who predicts better? Brier score (lower = better), same brackets")
        for when, g in res.groupby("when"):
            mk = g.drop_duplicates("ticker")
            line = f"{when:<16} Kalshi mid {((mk.mid - mk.y)**2).mean():.4f}"
            for s, gs in g.groupby("src"):
                line += f" | {s} {((gs.prob - gs.y)**2).mean():.4f}"
            print(line)

        print("\n## 3. Calibration of the best source vs Kalshi (eve-before)")
        best_src = res.groupby("src").apply(lambda g: ((g.prob - g.y)**2).mean()).idxmin()
        g = res[(res.src == best_src) & (res.when == "eve-before 10pm")]
        bins = [0, .05, .15, .3, .5, .7, .85, .95, 1]
        print(f"model = {best_src}")
        print(g.groupby(pd.cut(g.prob, bins), observed=True).agg(n=("y", "size"), model=("prob", "mean"), actual=("y", "mean")).round(3).to_string())
        print(g.groupby(pd.cut(g.mid, bins), observed=True).agg(n=("y", "size"), kalshi=("mid", "mean"), actual=("y", "mean")).round(3).to_string())

        print("\n## 4. Backtest: take the gap, taker fee, hold to settlement (one trade per bracket per decision time)")
        print("source         when             edge>=  trades  win%  avg_px  P&L/contract (±se)   t-stat  weeks+ ")
        for (s, when), g in res.groupby(["src", "when"]):
            for th in (0.05, 0.10, 0.15, 0.20):
                tr = []
                for r in g.itertuples():
                    if r.prob - r.yes_ask >= th:
                        px = r.yes_ask; tr.append((r.date, px, r.y, r.y - px - fee(px), r.series))
                    elif (1 - r.prob) - (1 - r.yes_bid) >= th:
                        px = 1 - r.yes_bid; tr.append((r.date, px, 1 - r.y, (1 - r.y) - px - fee(px), r.series))
                if len(tr) < 10:
                    continue
                t = pd.DataFrame(tr, columns=["date", "px", "win", "pnl", "series"])
                se = t.pnl.std(ddof=1) / math.sqrt(len(t))
                wk = t.groupby(pd.to_datetime(t.date).dt.isocalendar().week).pnl.mean()
                print(f"{s:<14} {when:<16} {th*100:>4.0f}c  {len(t):>6}  {100*t.win.mean():>3.0f}%  {t.px.mean():>6.2f}"
                      f"  {100*t.pnl.mean():>+7.2f}c ±{100*se:.2f}   {t.pnl.mean()/se:>+5.1f}   {(wk>0).sum()}/{len(wk)}")
        print("```")
        print("\nAn edge is believable only with t-stat > 2, most weeks positive, and similar results across cities.")
    txt = buf.getvalue()
    print(txt)
    if report:
        os.makedirs(os.path.dirname(report) or ".", exist_ok=True)
        open(report, "w").write(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fetch", "analyze"])
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--out", default="wb")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    fetch(a.days, a.out) if a.mode == "fetch" else analyze(a.out, a.report)
