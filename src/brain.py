# src/brain.py
"""
Ultra-context-aware Kimi LLM decision wrapper
- Adds LOB imbalance, VAR, Reg-T, impact cost, hedge delta
- Zero blind spots
"""
import base64
import os
from typing import Any, Dict

import httpx
from pydantic import BaseModel, Field

from agents.technical_agent import TechnicalAgent
from utils.config import load_config
from performance.pnl_tracker import PnLTracker          # for NAV
from risk.portfolio_risk import PortfolioRisk            # for VAR
from risk.reg_t_guard import RegTGuard                   # for SMA
from execution.impact_model import ImpactEstimate        # for impact cost
from execution.micro_price import MicroPriceEngine       # for adverse cost
from risk.hedge_engine import HedgeEngine                # for beta hedge

cfg = load_config()


class Decision(BaseModel):
    action: str = Field(pattern=r"^(BUY|SELL|HOLD)$")
    stop_loss: float = Field(ge=0)
    take_profit: float = Field(ge=0)
    reasoning: str


class KimiDecisionMaker:
    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        self.model_cfg = model_cfg
        self.url = "https://api.moonshot.cn/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {os.getenv('KIMI_API_KEY')}"}

    async def decide(
        self,
        png_bytes: bytes,
        agent: TechnicalAgent,
        position: Dict[str, Any],
        headline: str = "",
        sent_score: float = 0.0,
        memory: str = "",
        # --- NEW context sources ---
        nav: float = 0.0,
        var_95: float = 0.0,
        sma_ratio: float = 0.0,
        impact_cost_bps: float = 0.0,
        adverse_cost_bps: float = 0.0,
        hedge_delta: float = 0.0,
        lob_imbalance: float = 0.0,
    ) -> Decision:
        img_b64 = base64.b64encode(png_bytes).decode()

        pos_summary = (
            f"Current position: {position['qty']} shares "
            f"avg_price={position['avg_price']:.4f} "
            f"unreal_pnl={position['unreal_pnl']:.2f} "
            f"real_pnl={position['real_pnl']:.2f}"
            if position["qty"]
            else "FLAT – no position"
        )

        extras = []
        if headline:
            extras.append(f"Headline: {headline} (sentiment={sent_score:+.2f})")
        if memory:
            extras.append(f"Memory:\n{memory}")

        context = "\n".join([
            f"NAV: ${nav:,.0f}",
            f"Portfolio VAR-95: ${var_95:,.0f}",
            f"Reg-T SMA ratio: {sma_ratio:.2f}",
            f"Impact cost: {impact_cost_bps:.1f} bps",
            f"Adverse-selection cost: {adverse_cost_bps:.1f} bps",
            f"Hedge delta needed: {hedge_delta:+.0f}",
            f"LOB imbalance: {lob_imbalance:+.2f}",
        ])

        prompt_text = (
            "You are a professional, risk-averse trader.\n"
            f"{pos_summary}\n"
            + ("\n".join(extras) + "\n" if extras else "")
            + f"{context}\n"
            + "Output JSON: {\"action\":\"BUY|SELL|HOLD\",\"stop_loss\":float,\"take_profit\":float,\"reasoning\":str}"
        )

        payload = {
            "model": self.model_cfg["kimi_model"],
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "max_tokens": 256,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(self.url, headers=self.headers, json=payload)
            r.raise_for_status()
            data = r.json()["choices"][0]["message"]["content"]
            return Decision.model_validate_json(data)