#!/usr/bin/env python3
"""
Offline back-test that runs the **full live-trading stack**:
PNG → Kimi → Impact → Micro-price → Risk-guard → Trade → PnL
(LOB & sentiment mocked; Kimi stubbed for speed)
"""

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# 1)  Remove every "src." prefix
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from brain import Decision, KimiDecisionMaker
from data_ingestion.candle_builder import CandleBuilder
from execution.impact_model import ImpactEstimate, ImpactModel
from execution.micro_price import MicroPriceEngine, SizedOrder
from performance.pnl_tracker import PnLTracker      # ← no more src.performance
from risk.portfolio_risk import PortfolioRisk       # ← no more src.risk
from risk.reg_t_guard import RegTGuard              # ← no more src.risk
from utils.config import load_config               # ← no more src.utils
from utils.logger import get_logger                # ← no more src.utils
from dotenv import load_dotenv
load_dotenv("credentials.env")   # looks in project root

cfg = load_config()
log = get_logger("LIVE_REPLAY")


class LiveStackReplay:
    """
    Runs the **exact** decision stack used by Supervisor.py
    but offline on historical bars.
    """

    def __init__(self) -> None:
        self.symbol = "INTC"
        self.technical = TechnicalAgent()
        self.brain = KimiDecisionMaker(cfg["model"])
        self.sentiment = SentimentAgent(cfg["model"]["kimi_key"])  # stub for API
        self.impact = ImpactModel(
            gamma=cfg["impact"]["gamma"], eta=cfg["impact"]["eta"]
        )
        self.micro = MicroPriceEngine()
        self.port_risk = PortfolioRisk(None)  # portfolio snapshot stub
        self.reg_t = RegTGuard(None)          # Reg-T stub

        # running state
        self.position = {"qty": 0, "avg_price": 0.0, "real_pnl": 0.0}
        self.trades: list[dict] = []

    # ---------- helpers ----------
    def _mock_nav(self) -> float:
        """Fake NAV so risk-sizing works."""
        return 100_000.0

    def _mock_margin(self) -> float:
        """Fake margin usage."""
        return 0.0

    def _mock_lob(self, mid: float):
        """Quick LOB mock like FakeLob."""
        return type(
            "MockLob",
            (),
            {
                "bid": [(mid - 0.01 * i, 100) for i in range(1, 6)],
                "ask": [(mid + 0.01 * i, 100) for i in range(1, 6)],
                "latency_us": 200,
            },
        )()

    # ---------- decision loop ----------
    async def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("timestamp").sort_index()
        log.info("Loaded %d bars", len(df))

        builder = CandleBuilder()
        for idx, (ts, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            # skip every 2nd bar to speed up
            if idx % 2:
                continue

            lookback = df.loc[:ts].tail(60)
            if len(lookback) < 20:
                continue

            # 1) PNG chart
            png = builder.render_png(lookback)

            # 2) sentiment stub (offline)
            sentiment_score = 0.0

            # 3) technical agent (optional check)
            tech_action, tech_conf = self.technical.decide(png, lookback)
            if tech_action == "HOLD":
                continue

            # 4) Kimi LLM decision
            decision: Decision = await self.brain.decide(
                png_bytes=png,
                agent=self.technical,
                position=self.position,
                headline="",
                sent_score=sentiment_score,
                memory="",
            )
            if decision.action == "HOLD":
                continue

            # 5) micro-structure & sizing
            nav = self._mock_nav()
            margin = self._mock_margin()

            sized = SizedOrder(
                action=decision.action,
                qty=100,  # fixed lot for replay
                limit=float(row["close"]),
                stop=decision.stop_loss,
                take=decision.take_profit,
            )

            # 6) impact & micro-price check
            mid = float(row["close"])
            lob = self._mock_lob(mid)
            impact: ImpactEstimate = self.impact.estimate(sized.qty, sized.action, lob)
            if impact.slippage_bps > cfg["impact"]["max_slippage_bps"]:
                log.debug("Skip – slippage too high")
                continue

            micro = self.micro.compute(lob, sized.qty)
            if micro.cost_bps > cfg["micro"]["max_cost_bps"]:
                log.debug("Skip – adverse-selection cost too high")
                continue

            # 7) risk guards (stubbed)
            if self.reg_t.snapshot()["breach"]:
                log.debug("Skip – Reg-T breach")
                continue
            if self.port_risk.snapshot()["var_95"] > cfg["risk"]["max_var_usd"]:
                log.debug("Skip – VAR breach")
                continue

            # 8) record trade
            fill_px = impact.expected_price
            if decision.action == "BUY":
                self.position["avg_price"] = (
                    self.position["avg_price"] * self.position["qty"] + fill_px * sized.qty
                ) / max(self.position["qty"] + sized.qty, 1)
                self.position["qty"] += sized.qty
            elif decision.action == "SELL":
                if self.position["qty"] > 0:
                    closed = min(self.position["qty"], sized.qty)
                    self.position["real_pnl"] += (fill_px - self.position["avg_price"]) * closed
                self.position["qty"] -= sized.qty

            self.trades.append(
                {
                    "timestamp": ts.isoformat(),
                    "action": decision.action,
                    "fill_px": fill_px,
                    "qty_after": self.position["qty"],
                    "real_pnl": self.position["real_pnl"],
                    "stop": decision.stop,
                    "take": decision.take,
                    "reasoning": decision.reasoning,
                    "slippage_bps": impact.slippage_bps,
                }
            )

        return pd.DataFrame(self.trades)

    # ---------- summary ----------
    def summary(self, trades: pd.DataFrame) -> None:
        if trades.empty:
            log.warning("No trades")
            return
        realized = trades["real_pnl"].iloc[-1] if not trades.empty else 0
        log.info("=" * 60)
        log.info("BACK-TEST SUMMARY")
        log.info("Trades      : %d", len(trades))
        log.info("Realized PnL: %.4f pts", realized)
        log.info("=" * 60)


# ---------- CLI ----------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/replay/INTC_1m.csv")
    args = parser.parse_args()

    csv_path = Path(__file__).parents[2] / args.file
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    bt = LiveStackReplay()
    trades = await bt.run(df)
    bt.summary(trades)

    out = Path(__file__).parents[2] / "data/backtest/INTC_kimi_trades.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_json(out, orient="records", lines=True)
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    asyncio.run(main())