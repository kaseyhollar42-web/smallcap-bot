"""Shared helpers: config, API keys, market data, logging."""

import logging
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ET = ZoneInfo("America/New_York")
HERE = os.path.dirname(os.path.abspath(__file__))

# IEX is one exchange (~2-3% of all US volume). When we only have IEX data,
# we scale its volume up to estimate total market volume for the liquidity filter.
IEX_VOLUME_SHARE = 0.025


def load_config(path: str = None) -> dict:
    path = path or os.path.join(HERE, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["watchlist"] = list(dict.fromkeys(s.strip().upper() for s in cfg["watchlist"]))
    return cfg


def get_keys():
    load_dotenv(os.path.join(HERE, ".env"))
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret or "your_" in key:
        sys.exit("Missing API keys. On GitHub: Settings > Secrets and variables > Actions "
                 "(ALPACA_API_KEY and ALPACA_SECRET_KEY). Locally: a .env file.")
    return key, secret


def data_client() -> StockHistoricalDataClient:
    key, secret = get_keys()
    return StockHistoricalDataClient(key, secret)


def _fetch(client, symbols, start, end, feed):
    out = {}
    for i in range(0, len(symbols), 50):  # keep each request a reasonable size
        chunk = symbols[i:i + 50]
        df = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=chunk, timeframe=TimeFrame.Day, start=start, end=end,
            adjustment=Adjustment.ALL, feed=DataFeed(feed))).df
        if df.empty:
            continue
        for sym in chunk:
            if sym not in df.index.get_level_values(0):
                continue
            d = df.loc[sym].copy()
            d.index = pd.to_datetime(pd.to_datetime(d.index).tz_convert(ET).date)
            out[sym] = d[["open", "high", "low", "close", "volume"]]
    return out


def get_daily_bars(client, symbols, start: datetime, end: datetime = None, feed: str = "sip"):
    """
    Returns ({symbol: DataFrame}, feed_used). Tries the full-market SIP feed
    first (free for data older than 15 minutes) and falls back to IEX.
    """
    end = end or datetime.now(ET) - timedelta(minutes=16)
    if feed == "sip":
        try:
            return _fetch(client, symbols, start, end, "sip"), "sip"
        except APIError as e:
            logging.getLogger("bot").warning(f"SIP data not available on this plan ({e}); using IEX instead.")
    return _fetch(client, symbols, start, end, "iex"), "iex"


def to_panels(bars: dict):
    """Turn {symbol: bars} into date x symbol DataFrames."""
    if not bars:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    close = pd.DataFrame({s: b["close"] for s, b in bars.items()}).sort_index()
    opn = pd.DataFrame({s: b["open"] for s, b in bars.items()}).reindex(close.index)
    low = pd.DataFrame({s: b["low"] for s, b in bars.items()}).reindex(close.index)
    vol = pd.DataFrame({s: b["volume"] for s, b in bars.items()}).reindex(close.index)
    return close, opn, low, vol


def setup_logging(name: str = "bot") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(HERE, "bot.log"))):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log
