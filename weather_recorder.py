"""
weather_recorder.py - snapshot Kalshi daily high-temperature markets + weather forecasts.

Run once per snapshot (GitHub Actions runs it every 30 min):
  python3 weather_recorder.py --out wdata

Writes, under <out>/<UTC date>/ :
  markets.csv      live order-book quotes for every open temperature bracket
  forecasts.csv    daily-high forecasts: NWS official + Open-Meteo ensemble members (GFS, ECMWF)
  settlements.csv  results of recently settled brackets (deduplicate when analyzing)

No API keys needed (Kalshi public market data, Open-Meteo, api.weather.gov).
"""
import argparse
import csv
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

KALSHI = "https://external-api.kalshi.com/trade-api/v2"
UA = {"User-Agent": "trader-bot weather research (github.com/gregPodstrelov/trader-bot)",
      "Accept": "application/json"}

# Kalshi series -> settlement station (from each market's rules) and its coordinates / timezone
CITIES = {
    "KXHIGHNY":   ("CLINYC", "New York Central Park", 40.7789, -73.9692, "America/New_York"),
    "KXHIGHCHI":  ("CLIMDW", "Chicago Midway",        41.7860, -87.7524, "America/Chicago"),
    "KXHIGHMIA":  ("CLIMIA", "Miami Intl",            25.7906, -80.3164, "America/New_York"),
    "KXHIGHAUS":  ("CLIAUS", "Austin Bergstrom",      30.1945, -97.6699, "America/Chicago"),
    "KXHIGHDEN":  ("CLIDEN", "Denver Intl",           39.8467, -104.6562, "America/Denver"),
    "KXHIGHPHIL": ("CLIPHL", "Philadelphia Intl",     39.8683, -75.2311, "America/New_York"),
    "KXHIGHLAX":  ("CLILAX", "Los Angeles Intl",      33.9425, -118.4081, "America/Los_Angeles"),
}


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CTX = _ctx()


def get(url, retries=3, timeout=20):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1)); continue
            print(f"  HTTP {e.code} for {url[:110]}")
            return None
        except Exception as e:
            print(f"  {type(e).__name__} for {url[:110]}")
            time.sleep(1 + attempt)
    return None


def append(path, header, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------ Kalshi
def best_quotes(book):
    ob = (book or {}).get("orderbook_fp") or {}
    yes = [(float(p), float(q)) for p, q in (ob.get("yes_dollars") or [])]
    no = [(float(p), float(q)) for p, q in (ob.get("no_dollars") or [])]
    yb = max(yes) if yes else (None, None)
    nb = max(no) if no else (None, None)
    ya = (round(1 - nb[0], 4), nb[1]) if nb[0] is not None else (None, None)
    return yb[0], yb[1], ya[0], ya[1]


def kalshi_snapshot(snap_ts):
    rows, settle_rows = [], []
    now = int(time.time())
    for series in CITIES:
        data = get(f"{KALSHI}/markets?series_ticker={series}&status=open&limit=100") or {}
        mkts = data.get("markets", [])
        with ThreadPoolExecutor(6) as pool:
            books = list(pool.map(lambda m: get(f"{KALSHI}/markets/{m['ticker']}/orderbook?depth=5"), mkts))
        for m, book in zip(mkts, books):
            yb, ybs, ya, yas = best_quotes(book)
            rows.append([snap_ts, series, m.get("event_ticker"), m["ticker"], m.get("strike_type"),
                         m.get("floor_strike"), m.get("cap_strike"), yb, ya, ybs, yas,
                         m.get("volume_fp"), m.get("close_time"), m.get("yes_sub_title")])
        time.sleep(0.3)
        done = get(f"{KALSHI}/markets?series_ticker={series}&status=settled&min_close_ts={now - 3 * 86400}&limit=200") or {}
        for m in done.get("markets", []):
            settle_rows.append([m["ticker"], series, m.get("event_ticker"), m.get("strike_type"),
                                m.get("floor_strike"), m.get("cap_strike"), m.get("close_time"),
                                m.get("result"), m.get("expiration_value")])
    return rows, settle_rows


# ---------------------------------------------------------------- forecasts
def forecasts(snap_ts):
    rows = []
    for series, (station, name, lat, lon, tz) in CITIES.items():
        q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "daily": "temperature_2m_max",
                                    "temperature_unit": "fahrenheit", "timezone": tz, "forecast_days": 4})
        # 1) ensembles: every member is one possible future -> probability per bracket
        for model in ("gfs025", "ecmwf_ifs025"):
            d = get(f"https://ensemble-api.open-meteo.com/v1/ensemble?{q}&models={model}") or {}
            daily = d.get("daily") or {}
            dates = daily.get("time", [])
            for key, vals in daily.items():
                if not key.startswith("temperature_2m_max"):
                    continue
                member = key.replace("temperature_2m_max", "").strip("_") or "control"
                for date, v in zip(dates, vals):
                    if v is not None:
                        rows.append([snap_ts, series, date, f"ens_{model}", member, v])
        # 2) deterministic models (best single guesses)
        d = get(f"https://api.open-meteo.com/v1/forecast?{q}&models=best_match,gfs_seamless,ecmwf_ifs025,icon_seamless,gem_seamless") or {}
        daily = d.get("daily") or {}
        for key, vals in daily.items():
            if key.startswith("temperature_2m_max"):
                src = key.replace("temperature_2m_max", "").strip("_") or "best_match"
                for date, v in zip(daily.get("time", []), vals):
                    if v is not None:
                        rows.append([snap_ts, series, date, f"det_{src}", "", v])
        # 3) official NWS forecast high
        pt = get(f"https://api.weather.gov/points/{lat},{lon}") or {}
        url = (pt.get("properties") or {}).get("forecast")
        fc = get(url) if url else None
        for p in ((fc or {}).get("properties") or {}).get("periods", []):
            if p.get("isDaytime"):
                rows.append([snap_ts, series, p["startTime"][:10], "nws", "", p.get("temperature")])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="wdata")
    a = ap.parse_args()
    snap = datetime.now(timezone.utc)
    snap_ts = snap.strftime("%Y-%m-%dT%H:%M:%SZ")
    folder = os.path.join(a.out, snap.strftime("%Y-%m-%d"))

    mrows, srows = kalshi_snapshot(snap_ts)
    append(os.path.join(folder, "markets.csv"),
           ["snap_ts", "series", "event_ticker", "ticker", "strike_type", "floor_strike", "cap_strike",
            "yes_bid", "yes_ask", "yes_bid_size", "yes_ask_size", "volume", "close_time", "label"], mrows)
    append(os.path.join(folder, "settlements.csv"),
           ["ticker", "series", "event_ticker", "strike_type", "floor_strike", "cap_strike",
            "close_time", "result", "expiration_value"], srows)
    frows = forecasts(snap_ts)
    append(os.path.join(folder, "forecasts.csv"),
           ["snap_ts", "series", "date", "source", "member", "tmax_f"], frows)

    by_city = {s: sum(1 for r in mrows if r[1] == s) for s in CITIES}
    print(f"{snap_ts}: {len(mrows)} bracket quotes {by_city}, {len(srows)} settlement rows, {len(frows)} forecast values")
    if not mrows or not frows:
        raise SystemExit("Nothing recorded from Kalshi or forecasts - check the log above.")


if __name__ == "__main__":
    main()
