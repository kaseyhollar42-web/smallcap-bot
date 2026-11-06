"""
Backtest the small-cap momentum bot on past data.

    python backtest.py               # last `backtest.years` years (config.yaml)
    python backtest.py --years 3 --plot

How it simulates the live bot:
  * rankings are computed from each day's closing prices
  * trades happen at the NEXT day's opening price
  * stop losses trigger intraday using the day's low (never on the day of purchase)
  * the safety floor is checked at each day's close; if hit, it sells and stops for good
  * slippage_pct is charged on every buy and sell
  * the daily scanner is NOT included (we can't know what it would have found)

BIG CAVEAT: the watchlist was picked in November 2026 from articles about stocks
that had ALREADY done well. Testing on them is like betting on a race you've
already seen. Real results will very likely be worse than this backtest.
"""

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from common import ET, data_client, get_daily_bars, load_config, to_panels
from strategy import history_days_needed, momentum_panel

pd.set_option("display.width", 200)


def simulate(close, opn, low, vol, cfg, feed, start_date):
    p = momentum_panel(close, vol, cfg, feed)
    rank, elig = p["rank"], p["eligible"]
    cf = close.ffill()
    pos_cfg, risk, mom = cfg["positions"], cfg["risk"], cfg["momentum"]
    slip = cfg["backtest"]["slippage_pct"] / 100
    stop = risk["stop_loss_pct"] / 100
    floor = float(cfg["floor"])
    max_pos, buy_rank, hold_rank = int(pos_cfg["max_positions"]), int(mom["buy_rank"]), int(mom["hold_rank"])

    cash = float(cfg["budget"])
    pos, trades, curve = {}, [], []
    pending_sells, pending_buys = set(), []
    prev_eq, halted_on = cash, None
    dates = close.index[close.index >= start_date]

    def exit_trade(sym, px, d, why):
        nonlocal cash
        ps = pos.pop(sym)
        cash += ps["shares"] * px * (1 - slip)
        trades.append({"symbol": sym, "entry": ps["day"], "exit": d, "why": why,
                       "return_%": (px * (1 - slip) / (ps["entry"] / (1 - slip)) - 1) * 100,
                       "days": (d - ps["day"]).days})

    for d in dates:
        # 1) yesterday's decisions fill at today's open
        for sym in list(pending_sells):
            if sym in pos:
                px = opn.at[d, sym]
                exit_trade(sym, px if px == px else cf.at[d, sym], d, "rank/filter")
        pending_sells.clear()
        for sym, dollars in pending_buys:
            px = opn.at[d, sym]
            dollars = min(dollars, cash)
            if px != px or dollars < pos_cfg["min_order"]:
                continue
            pos[sym] = {"shares": dollars * (1 - slip) / px, "entry": px, "day": d}
            cash -= dollars
        pending_buys = []

        # 2) stop losses during the day (not on the day we bought)
        for sym, ps in list(pos.items()):
            if ps["day"] == d:
                continue
            stop_px = ps["entry"] * (1 - stop)
            lo = low.at[d, sym]
            if lo == lo and lo <= stop_px:
                o = opn.at[d, sym]
                exit_trade(sym, min(o, stop_px) if o == o else stop_px, d, "stop loss")

        # 3) value at the close + safety floor
        equity = cash + sum(ps["shares"] * cf.at[d, s] for s, ps in pos.items())
        if equity <= floor:
            for sym in list(pos):
                exit_trade(sym, cf.at[d, sym], d, "FLOOR")
            curve.append((d, cash))
            halted_on = d
            break
        curve.append((d, equity))

        # 4) decide tomorrow's trades
        for sym, ps in pos.items():
            if ps["day"] == d:
                continue  # never sell on the day of purchase
            if not elig.at[d, sym] or rank.at[d, sym] > hold_rank:
                pending_sells.add(sym)
        if equity / prev_eq - 1 > -risk["daily_loss_limit_pct"] / 100:
            slots = max_pos - (len(pos) - len(pending_sells))
            free = cash + sum(pos[s]["shares"] * cf.at[d, s] * (1 - slip) for s in pending_sells)
            target = min(equity / max_pos, equity * pos_cfg["max_position_pct"] / 100)
            row = rank.loc[d]
            for sym in row[row <= buy_rank].sort_values().index:
                if slots <= 0:
                    break
                if sym in pos:
                    continue
                amt = min(target, free)
                if amt < max(pos_cfg["min_order"], target * 0.5):
                    break
                pending_buys.append((sym, amt))
                free -= amt
                slots -= 1
        prev_eq = equity

    eq = pd.Series(dict(curve))
    if halted_on is not None:  # flat after the floor
        rest = close.index[close.index > halted_on]
        eq = pd.concat([eq, pd.Series(eq.iloc[-1], index=rest)])
    return eq, pd.DataFrame(trades), halted_on, pos


def stats(eq):
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    return {"final_$": eq.iloc[-1], "total_%": total * 100,
            "yearly_%": ((1 + total) ** (365.25 / days) - 1) * 100 if total > -1 else -100,
            "max_drop_%": (eq / eq.cummax() - 1).min() * 100}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=float)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    years = args.years or cfg["backtest"]["years"]
    bench = cfg["backtest"].get("benchmark", "IWM")
    now = datetime.now(ET)
    start_date = pd.Timestamp((now - timedelta(days=int(years * 365.25))).date())
    syms = cfg["watchlist"] + [bench]
    print(f"Downloading daily bars for {len(syms)} symbols ...")
    bars, feed = get_daily_bars(data_client(), syms, now - timedelta(days=int(years * 365.25 + history_days_needed() * 1.6)),
                                feed=cfg.get("data_feed", "sip"))
    bench_bars = bars.pop(bench, None)
    close, opn, low, vol = to_panels(bars)
    missing = set(cfg["watchlist"]) - set(bars)
    if missing:
        print(f"  no data for: {', '.join(sorted(missing))}")

    eq, trades, halted_on, still_held = simulate(close, opn, low, vol, cfg, feed, start_date)
    budget = float(cfg["budget"])
    rows = {"small-cap bot": stats(eq)}
    curves = {"small-cap bot": eq}
    if bench_bars is not None:
        b = bench_bars["close"]
        b = b[b.index >= start_date]
        b = b / b.iloc[0] * budget
        rows[f"{bench} buy & hold"] = stats(b)
        curves[f"{bench} buy & hold"] = b
    ew = close[close.index >= start_date]
    ew = ew.loc[:, ew.iloc[0].notna()]
    ew = (ew / ew.iloc[0]).ffill().mean(axis=1) * budget
    rows["watchlist equal-weight hold"] = stats(ew)
    curves["watchlist equal-weight hold"] = ew

    print(f"\nTest period: {start_date.date()} to today | starting ${budget:,.0f} | "
          f"floor ${cfg['floor']} | data feed {feed.upper()}\n")
    summary = pd.DataFrame(rows).T
    print(summary.to_string(float_format=lambda x: f"{x:,.1f}"))

    if len(trades):
        wins = (trades["return_%"] > 0).mean() * 100
        print(f"\nTrades: {len(trades)} | winners {wins:.0f}% | average hold {trades['days'].mean():.0f} days | "
              f"stop-loss exits {int((trades['why'] == 'stop loss').sum())}")
        print("\nBest trades:")
        print(trades.nlargest(5, "return_%")[["symbol", "entry", "exit", "return_%", "days"]].to_string(index=False))
        print("\nWorst trades:")
        print(trades.nsmallest(5, "return_%")[["symbol", "entry", "exit", "return_%", "days"]].to_string(index=False))
    if halted_on is not None:
        print(f"\n!!! The ${float(cfg['floor']):,.0f} floor was hit on {halted_on.date()}: the bot sold everything and stopped.")
    if still_held:
        print(f"\nStill holding at the end: {', '.join(still_held)}")
    print("\nREMEMBER: these stocks were chosen because they ALREADY went up. Expect worse in real life.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for n, c in curves.items():
            ax.plot(c.index, c.values, label=n, lw=1.8 if n == "small-cap bot" else 1.1,
                    ls="-" if n == "small-cap bot" else "--")
        ax.axhline(float(cfg["floor"]), color="red", lw=0.8, ls=":", label=f"${float(cfg['floor']):,.0f} floor")
        ax.set_title("Small-cap bot: account value")
        ax.set_ylabel("$")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("backtest.png", dpi=130)
        print("Chart saved to backtest.png")


if __name__ == "__main__":
    main()
