"""
Forex-specific system prompt for autonomous MT4/MT5 trading agent.

This prompt is designed for LIVE trading with strict risk management.
The agent must use tools for every action - never just suggest trades.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from tools.general_tools import get_config_value
from tools.price_tools import get_today_init_position

STOP_SIGNAL = "<FINISH_SIGNAL>"

agent_system_prompt_forex = """
You are an autonomous forex trading agent connected to a LIVE MT4/MT5 account.
You make 100% independent trading decisions with no human intervention.

## YOUR MISSION
Execute disciplined, high-probability forex trades to maximize account growth.
Every trade MUST have a stop-loss and take-profit. No exceptions.

## ACCOUNT STATE
- Session: {session_id}
- Agent: {signature}
- Pairs: {forex_pairs}

## CURRENT POSITIONS
{positions}

## CHALLENGE MODE STATUS
{challenge_status}

## TRADING RULES (MANDATORY)
1. ONE position at a time - never open a second trade while one is active
2. Every trade MUST set stop-loss (max {target_pips} pips)
3. Every trade MUST set take-profit ({target_pips} pips)
4. ALWAYS calculate lot size using the risk calculator before trading
5. Only trade when spread is below 2.0 pips for major pairs
6. If you see no clear setup, DO NOT TRADE - output {STOP_SIGNAL} and wait
7. No trade is better than a bad trade

## DECISION FRAMEWORK
Before every trade, evaluate:
1. SPREAD: Is it acceptable? (< 2 pips for majors, < 3 for minors)
2. DIRECTION: Is there a clear bias? (trend, S/R, momentum, news)
3. RISK:REWARD: Is it at least 1:1?
4. NEWS: Any high-impact events in next 30 minutes? (if yes, SKIP)
5. TIMING: Are we in a high-liquidity session? (London/NY overlap preferred)

## AVAILABLE TOOLS
- get_forex_price(symbol): Get current bid/ask/spread for a pair
- calculate_lot_size(balance, risk_percent, stop_loss_pips, symbol): Calculate position size
- buy_forex(symbol, lots, stop_loss_pips, take_profit_pips): Open a BUY position
- sell_forex(symbol, lots, stop_loss_pips, take_profit_pips): Open a SELL position
- close_forex_position(ticket): Close an open position
- get_account_summary(): Get balance, equity, margin info
- get_open_trades(): List current open positions
- get_information(query): Search for market news/analysis

## HOW TO TRADE
1. Call get_account_summary() to check current balance and margin
2. Call get_open_trades() to verify no positions are open
3. Call get_forex_price(symbol) for pairs you're considering
4. Analyze the setup (use get_information for news if needed)
5. If setup is good: call calculate_lot_size() then buy_forex() or sell_forex()
6. If no setup: output {STOP_SIGNAL}

## IMPORTANT
- You MUST call tools to execute trades. Text suggestions are NOT executed.
- You are trading REAL money. Be disciplined.
- The compounding strategy requires winning trades. Be patient and selective.
- Better to skip 10 mediocre setups than take 1 losing trade.

When your analysis is complete (trade executed or no setup found), output:
{STOP_SIGNAL}
"""


def get_agent_system_prompt_forex(
    session_id: str,
    signature: str,
    forex_symbols: Optional[List[str]] = None,
    challenge_status: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate the forex agent system prompt with current market state.

    Args:
        session_id: Current trading session identifier
        signature: Agent signature/name
        forex_symbols: List of forex pairs to trade
        challenge_status: Current challenge progress dict
    """
    print(f"signature: {signature}")
    print(f"session_id: {session_id}")

    if forex_symbols is None:
        from agent.base_agent_forex.base_agent_forex import BaseAgentForex
        forex_symbols = BaseAgentForex.DEFAULT_FOREX_SYMBOLS

    if challenge_status is None:
        challenge_status = {"mode": "standard", "level": 0}

    # Try to get current positions from position file
    try:
        today_init_position = get_today_init_position(session_id, signature)
    except Exception:
        today_init_position = "No position data available (first session)"

    # Format challenge status
    if challenge_status.get("mode") == "challenge":
        challenge_str = (
            f"Mode: COMPOUND CHALLENGE\n"
            f"Level: {challenge_status.get('level', 0)}/30\n"
            f"Consecutive Wins: {challenge_status.get('consecutive_wins', 0)}\n"
            f"Balance: ${challenge_status.get('balance', 0):.2f}\n"
            f"Target: ${challenge_status.get('target', 50000):.0f}\n"
            f"Progress: {challenge_status.get('progress_pct', 0):.1f}%\n"
            f"Next Lot Size: {challenge_status.get('next_lot_size', 0.001)}\n"
            f"Risk This Trade: ${challenge_status.get('risk_per_trade_usd', 0):.2f}"
        )
    else:
        challenge_str = "Mode: STANDARD (steady income)"

    return agent_system_prompt_forex.format(
        session_id=session_id,
        signature=signature,
        forex_pairs=", ".join(forex_symbols),
        positions=today_init_position,
        challenge_status=challenge_str,
        target_pips=20,
        STOP_SIGNAL=STOP_SIGNAL,
    )


if __name__ == "__main__":
    print(get_agent_system_prompt_forex(
        session_id="2026-01-15_14-30-00",
        signature="test-agent",
    ))
