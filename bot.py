"""
Small-Cap Momentum Bot.

    python bot.py --status      # money, positions, today's rankings
    python bot.py --dry-run     # show what it WOULD do right now; places nothing
    python bot.py               # normal run (GitHub runs this every hour on market days)
    python bot.py --resume      # the "go ahead" after the safety floor stopped the bot

Each run:
  1. Checks the safety floor. If the bot's money is at/below `floor`, it sells
     everything (or just stops buying) and STAYS stopped until you resume it.
  2. Sells any stock that has fallen `stop_loss_pct` below what we paid.
  3. After 10:00 ET: ranks the watchlist + today's scanner finds, sells holdings
     that slipped out of the top `hold_rank`, and fills empty slots with the
     top-ranked stocks.
It never buys and sells the same stock on the same day (avoids day-trade rules),
except in a floor emergency.
"""

import argparse
import math
import os
import re
import sys
import uuid
from datetime import datetime, time as dtime, timedelta

from alpaca.common.exceptions import APIError
from alpaca.data.enums import MostActivesBy
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MarketMoversRequest, MostActivesRequest, NewsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import CreateWatchlistRequest, GetOrdersRequest, MarketOrderRequest

from common import ET, data_client, get_daily_bars, get_keys, load_config, setup_logging, to_panels
from strategy import history_days_needed, latest_table

log = setup_logging("bot")
HALT_NAME = "SMALLCAP-BOT-HALTED"
DEAD = {OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED}
TICKER = re.compile(r"^[A-Z]{1,5}$")


class Bot:
    def __init__(self, cfg: dict, dry_run: bool = False):
        self.cfg, self.dry_run = cfg, dry_run
        self.paper = cfg.get("paper", True) is not False
        if not self.paper and os.getenv("I_UNDERSTAND_THIS_IS_REAL_MONEY") != "yes":
            sys.exit("config.yaml has paper: false, but the real-money safety switch is off. See README.")
        key, secret = get_keys()
        self.trading = TradingClient(key, secret, paper=self.paper)
        self.data = data_client()
        self.screener = ScreenerClient(key, secret)
        self.news = NewsClient(key, secret)
        self.report, self.sold = [], set()

    # ---------- output ----------
    def say(self, msg, level="info"):
        getattr(log, level)(msg)
        self.report.append(("**" + msg + "**") if level in ("warning", "error") else msg)

    def write_summary(self):
        path = os.getenv("GITHUB_STEP_SUMMARY")
        if path and self.report:
            with open(path, "a") as f:
                f.write("## Small-Cap Bot run\n\n" + "\n\n".join(self.report) + "\n")

    @property
    def mode(self):
        return ("PAPER" if self.paper else "*** LIVE MONEY ***") + (" / DRY RUN" if self.dry_run else "")

    # ---------- money ----------
    def money(self):
        acct = self.trading.get_account()
        positions = [p for p in self.trading.get_all_positions() if float(p.qty) > 0]
        equity = float(acct.equity)
        budget, start, floor = (float(self.cfg[k]) for k in ("budget", "account_start_value", "floor"))
        if floor >= budget:
            sys.exit(f"config.yaml: floor (${floor:,.0f}) must be below budget (${budget:,.0f}).")
        if budget > start:
            sys.exit(f"config.yaml: budget (${budget:,.0f}) is more than account_start_value (${start:,.0f}). "
                     f"Set budget to what your account started with, or less.")
        reserve = start - budget
        if reserve and equity <= reserve:
            sys.exit(f"Your account has ${equity:,.2f}, but config.yaml says it started with "
                     f"${self.cfg['account_start_value']:,}. Fix `account_start_value` (see README).")
        bot_equity = equity - reserve
        last = float(acct.last_equity) - reserve
        return {
            "acct": acct, "positions": positions, "reserve": reserve,
            "equity": bot_equity,
            "cash": max(0.0, min(float(acct.cash), float(acct.buying_power)) - reserve),
            "invested": sum(float(p.market_value) for p in positions),
            "day_pct": (bot_equity / last - 1) * 100 if last > 0 else 0.0,
        }

    # ---------- halt flag (stored in Alpaca as a watchlist, so it survives between runs) ----------
    def halt_flag(self):
        return next((w for w in self.trading.get_watchlists() if w.name == HALT_NAME), None)

    def set_halt(self):
        try:
            self.trading.create_watchlist(CreateWatchlistRequest(name=HALT_NAME, symbols=[]))
        except APIError:
            self.trading.create_watchlist(CreateWatchlistRequest(name=HALT_NAME, symbols=["IWM"]))

    # ---------- orders ----------
    def sell_all(self, symbol, why):
        if self.dry_run:
            self.say(f"[DRY RUN] would SELL ALL {symbol}: {why}")
            return True
        try:
            self.trading.close_position(symbol)
            self.say(f"SELL ALL {symbol}: {why}")
            self.sold.add(symbol)
            return True
        except APIError as e:
            self.say(f"Sell of {symbol} failed: {e}", "error")
            return False

    def buy(self, symbol, dollars, price, why):
        try:
            asset = self.trading.get_asset(symbol)
        except APIError:
            self.say(f"{symbol}: not found on Alpaca, skipping", "warning")
            return False
        if not asset.tradable or str(getattr(asset.exchange, "value", asset.exchange)).upper() == "OTC":
            self.say(f"{symbol}: not tradable on Alpaca, skipping")
            return False
        qty = None
        if not asset.fractionable:
            qty = math.floor(dollars / price)
            if qty < 1:
                self.say(f"{symbol}: ${dollars:,.0f} is less than one whole share (${price:,.2f}), skipping")
                return False
        what = f"{qty} shares" if qty else f"${dollars:,.2f}"
        if self.dry_run:
            self.say(f"[DRY RUN] would BUY {what} of {symbol}: {why}")
            return True
        try:
            o = self.trading.submit_order(MarketOrderRequest(
                symbol=symbol, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                notional=None if qty else round(dollars, 2), qty=qty,
                client_order_id=f"scb-{symbol}-{uuid.uuid4().hex[:10]}"))
            self.say(f"BUY {what} of {symbol} ({o.status.value}): {why}")
            head = self.headline(symbol)
            if head:
                self.say(f"   latest {symbol} news: {head}")
            return True
        except APIError as e:
            self.say(f"Buy of {symbol} rejected: {e}", "error")
            return False

    def headline(self, symbol):
        try:
            res = self.news.get_news(NewsRequest(symbols=symbol, limit=1))
            items = res.data.get("news", []) if hasattr(res, "data") else res.get("news", [])
            if items:
                n = items[0]
                return f"{n.headline} ({n.source}, {n.created_at:%b %d})"
        except Exception:
            pass
        return None

    # ---------- research ----------
    def scanner(self):
        sc = self.cfg.get("scanner", {}) or {}
        if not sc.get("enabled"):
            return []
        found = []
        try:
            movers = self.screener.get_market_movers(MarketMoversRequest(top=int(sc.get("top_gainers", 20))))
            found += [m.symbol for m in movers.gainers if m.price >= self.cfg["filters"]["min_price"]]
            act = self.screener.get_most_actives(MostActivesRequest(top=int(sc.get("most_active", 20)),
                                                                   by=MostActivesBy.VOLUME))
            found += [a.symbol for a in act.most_actives]
        except Exception as e:
            self.say(f"Scanner unavailable today ({e}); using the watchlist only.", "warning")
        skip = set(sc.get("exclude", [])) | set(self.cfg["watchlist"])
        return list(dict.fromkeys(s for s in found if TICKER.match(s) and s not in skip))

    def rankings(self, extra=()):
        extra_syms = self.scanner()
        universe = list(dict.fromkeys(self.cfg["watchlist"] + list(extra) + extra_syms))
        today = datetime.now(ET).date()
        bars, feed = get_daily_bars(self.data, universe,
                                    start=datetime.now(ET) - timedelta(days=int(history_days_needed() * 1.6)),
                                    feed=self.cfg.get("data_feed", "sip"))
        bars = {s: b[b.index.date < today] for s, b in bars.items()}  # finished days only
        bars = {s: b for s, b in bars.items() if len(b)}
        close, _, _, vol = to_panels(bars)
        if close.empty:
            raise RuntimeError("No market data returned.")
        t = latest_table(close, vol, self.cfg, feed)
        t["source"] = ["scanner" if s in extra_syms else "watchlist" for s in t.index]
        cap = (self.cfg.get("scanner") or {}).get("max_dollar_volume")
        if cap:  # scanner finds must still be "small": skip giants and big ETFs
            too_big = (t["source"] == "scanner") & (t["dollar_vol_M"] * 1e6 > cap)
            t.loc[too_big, "eligible"] = False
            t.loc[too_big, "rank"] = float("nan")
            elig = t[t["eligible"]]
            t.loc[elig.index, "rank"] = elig["score"].rank(ascending=False, method="first")
            t = t.sort_values(["rank", "score"], ascending=[True, False], na_position="last")
        self.say(f"Ranked {len(t)} stocks ({int(t['eligible'].sum())} pass filters; "
                 f"{len(extra_syms)} from today's scanner; data: {feed.upper()}).")
        return t

    def show_top(self, t, n=10):
        top = t[t["eligible"]].head(n)
        lines = ["| # | Stock | Price | 1-mo | 3-mo | 6-mo | Source |", "|---|---|---|---|---|---|---|"]
        for sym, r in top.iterrows():
            six = "n/a" if r["r126_%"] != r["r126_%"] else f"{r['r126_%']:+.0f}%"
            lines.append(f"| {int(r['rank'])} | {sym} | ${r['price']:,.2f} | {r['r21_%']:+.0f}% | "
                         f"{r['r63_%']:+.0f}% | {six} | {r['source']} |")
        log.info("Top ranked:\n" + "\n".join(lines))
        self.report.append("**Top ranked today**\n\n" + "\n".join(lines))

    # ---------- main run ----------
    def run(self) -> int:
        clock = self.trading.get_clock()
        now = clock.timestamp.astimezone(ET)
        m = self.money()
        c = self.cfg
        self.say(f"Run {now:%a %b %d %I:%M %p} ET | {self.mode} | bot money ${m['equity']:,.2f} "
                 f"({m['day_pct']:+.1f}% today) | cash ${m['cash']:,.2f} | floor ${c['floor']:,.0f}")

        if self.halt_flag():
            self.say("Bot is STOPPED by the safety floor. Run 'resume' when you want it to start again.", "warning")
            if c.get("on_floor") == "sell_all" and m["positions"] and not self.dry_run:
                self.trading.close_all_positions(cancel_orders=True)
                self.say("Re-sent sell orders for leftover positions.")
            return 0

        if m["equity"] <= float(c["floor"]):
            if not m["positions"]:
                # cash alone can't fall below the floor, so this is a settings mix-up, not a loss
                sys.exit(f"Bot money is ${m['equity']:,.2f} with no stocks held, already below the "
                         f"${float(c['floor']):,.0f} floor. Your account is smaller than config.yaml expects: "
                         f"set budget, account_start_value and floor to match it (see README step 3).")
            return self.trigger_floor(m)

        if not clock.is_open and not self.dry_run:
            self.say(f"Market closed. Next open {clock.next_open.astimezone(ET):%a %b %d %I:%M %p} ET.")
            return 0

        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        todays = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=start_of_day, limit=500))
        todays = [o for o in todays if o.status not in DEAD]
        bought_today = {o.symbol for o in todays if o.side == OrderSide.BUY}
        sold_today = {o.symbol for o in todays if o.side == OrderSide.SELL}
        open_orders = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        pending = {o.symbol for o in open_orders}
        pending_buys = sum(1 for o in open_orders if o.side == OrderSide.BUY)

        # 1) stop losses
        stop = float(c["risk"]["stop_loss_pct"])
        for p in m["positions"]:
            pl = float(p.unrealized_plpc) * 100
            if pl <= -stop and p.symbol not in pending:
                if p.symbol in bought_today:
                    self.say(f"{p.symbol} is {pl:.1f}% (past the stop) but was bought today; "
                             f"will sell next run to avoid a day trade.", "warning")
                else:
                    self.sell_all(p.symbol, f"stop loss ({pl:.1f}%)")

        # 2) ranking & rotation (once the opening chaos settles)
        h, mi = map(int, str(c["risk"].get("trade_after", "10:00")).split(":"))
        if now.time() < dtime(h, mi) and not self.dry_run:
            self.say(f"Rankings and new buys start after {h}:{mi:02d} ET.")
            return 0

        held = {p.symbol: p for p in m["positions"] if p.symbol not in self.sold}
        t = self.rankings(extra=held.keys())
        self.show_top(t)
        hold_rank = int(c["momentum"]["hold_rank"])

        for sym in list(held):
            if sym in pending or sym in bought_today:
                continue
            if sym not in t.index:
                self.say(f"{sym}: no fresh price data, keeping it for now", "warning")
                continue
            r = t.loc[sym]
            if not r["eligible"]:
                why = "no longer up over 3 months" if r["r63_%"] <= 0 else "no longer passes the filters"
                if self.sell_all(sym, why):
                    held.pop(sym)
            elif r["rank"] > hold_rank:
                if self.sell_all(sym, f"dropped to rank #{int(r['rank'])} (sell below #{hold_rank})"):
                    held.pop(sym)

        # 3) buys
        if m["day_pct"] <= -float(c["risk"]["daily_loss_limit_pct"]):
            self.say(f"Down {m['day_pct']:.1f}% today: hit the daily loss limit, no new buys.", "warning")
            return 0
        pos = c["positions"]
        slots = int(pos["max_positions"]) - len(held) - pending_buys
        cash = m["cash"]
        target = min(m["equity"] / int(pos["max_positions"]), m["equity"] * float(pos["max_position_pct"]) / 100)
        buy_rank = int(c["momentum"]["buy_rank"])
        cands = t[t["eligible"] & (t["rank"] <= buy_rank)]
        for sym, r in cands.iterrows():
            if slots <= 0:
                break
            if sym in held or sym in pending or sym in sold_today or sym in self.sold:
                continue
            amount = min(target, cash)
            if amount < max(float(pos["min_order"]), target * 0.5):
                self.say("Not enough free cash for another position (sells free up cash by the next run).")
                break
            why = f"rank #{int(r['rank'])}, up {r['r21_%']:+.0f}% (1 mo) / {r['r63_%']:+.0f}% (3 mo)"
            if self.buy(sym, amount, float(r["price"]), why):
                cash -= amount
                slots -= 1
        if slots > 0 and not len(cands):
            self.say("Nothing passes the filters right now. Sitting in cash.")
        return 0

    def trigger_floor(self, m) -> int:
        c = self.cfg
        self.say(f"SAFETY FLOOR HIT: bot money ${m['equity']:,.2f} is at/below ${c['floor']:,.0f}.", "error")
        if self.dry_run:
            self.say("[DRY RUN] would sell everything and stop the bot.")
            return 0
        if c.get("on_floor", "sell_all") == "sell_all":
            self.trading.cancel_orders()
            self.trading.close_all_positions(cancel_orders=True)
            self.say("Sold all positions.", "error")
        self.set_halt()
        self.say("Bot is now STOPPED. It will do nothing until you run 'resume'.", "error")
        return 2  # makes the GitHub run show as failed, so GitHub emails you

    # ---------- other commands ----------
    def status(self):
        m = self.money()
        c = self.cfg
        print(f"\nMode:        {self.mode}")
        print(f"Bot money:   ${m['equity']:,.2f}  ({m['day_pct']:+.2f}% today)   budget ${c['budget']:,.0f}")
        print(f"Cash:        ${m['cash']:,.2f}   invested ${m['invested']:,.2f}")
        print(f"Floor:       ${c['floor']:,.0f}  (${m['equity'] - c['floor']:,.2f} above it)")
        print(f"State:       {'STOPPED (run resume to restart)' if self.halt_flag() else 'active'}")
        if m["positions"]:
            print("\nPositions:")
            for p in m["positions"]:
                print(f"  {p.symbol:6} ${float(p.market_value):>9,.2f}   paid ${float(p.avg_entry_price):,.2f}"
                      f"   now ${float(p.current_price):,.2f}   P/L {float(p.unrealized_plpc) * 100:+.1f}%")
        else:
            print("\nNo positions.")
        try:
            t = self.rankings(extra=[p.symbol for p in m["positions"]])
            self.show_top(t)
        except Exception as e:
            print(f"(Couldn't load rankings: {e})")
        clock = self.trading.get_clock()
        print(f"\nMarket is {'OPEN' if clock.is_open else 'closed'}; next open "
              f"{clock.next_open.astimezone(ET):%a %b %d %I:%M %p} ET\n")

    def resume(self):
        flag = self.halt_flag()
        if not flag:
            self.say("Bot wasn't stopped. Nothing to resume.")
            return 0
        self.trading.delete_watchlist_by_id(flag.id)
        m = self.money()
        self.say(f"Resumed. Bot money is ${m['equity']:,.2f}.")
        if m["equity"] <= float(self.cfg["floor"]):
            self.say(f"Heads up: that's still at/below the ${self.cfg['floor']} floor, so it will stop again "
                     f"on the next run. Lower `floor` or raise `budget` in config.yaml first.", "warning")
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="restart after the safety floor stopped the bot")
    args = ap.parse_args()
    bot = Bot(load_config(), dry_run=args.dry_run)
    code = 0
    try:
        if args.status:
            bot.status()
        elif args.resume:
            code = bot.resume()
        else:
            code = bot.run()
    finally:
        bot.write_summary()
    sys.exit(code)


if __name__ == "__main__":
    main()
