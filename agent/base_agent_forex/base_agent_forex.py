"""
BaseAgentForex class - Autonomous forex trading agent for MT4/MT5

Extends the AI-Trader framework to connect to live forex markets via MetaTrader 4/5.
Supports multiple connection backends: MT5 native Python, MetaApi cloud, ZeroMQ bridge.

Key differences from other agents:
- LIVE market execution (not historical replay)
- Mandatory stop-loss/take-profit on every trade
- One position at a time (challenge mode)
- Continuous loop (not daily batch)
- Leverage and margin awareness
- Compounding challenge mode support
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class DeepSeekChatOpenAI(ChatOpenAI):
    """Custom ChatOpenAI wrapper for DeepSeek API compatibility."""

    def _create_message_dicts(self, messages: list, stop: Optional[list] = None) -> list:
        message_dicts = super()._create_message_dicts(messages, stop)
        for message_dict in message_dicts:
            if "tool_calls" in message_dict:
                for tool_call in message_dict["tool_calls"]:
                    if "function" in tool_call and "arguments" in tool_call["function"]:
                        args = tool_call["function"]["arguments"]
                        if isinstance(args, str):
                            try:
                                tool_call["function"]["arguments"] = json.loads(args)
                            except json.JSONDecodeError:
                                pass
        return message_dicts

    def _generate(self, messages: list, stop: Optional[list] = None, **kwargs):
        result = super()._generate(messages, stop, **kwargs)
        for generation in result.generations:
            for gen in generation:
                if hasattr(gen, "message") and hasattr(gen.message, "additional_kwargs"):
                    tool_calls = gen.message.additional_kwargs.get("tool_calls")
                    if tool_calls:
                        for tool_call in tool_calls:
                            if "function" in tool_call and "arguments" in tool_call["function"]:
                                args = tool_call["function"]["arguments"]
                                if isinstance(args, str):
                                    try:
                                        tool_call["function"]["arguments"] = json.loads(args)
                                    except json.JSONDecodeError:
                                        pass
        return result

    async def _agenerate(self, messages: list, stop: Optional[list] = None, **kwargs):
        result = await super()._agenerate(messages, stop, **kwargs)
        for generation in result.generations:
            for gen in generation:
                if hasattr(gen, "message") and hasattr(gen.message, "additional_kwargs"):
                    tool_calls = gen.message.additional_kwargs.get("tool_calls")
                    if tool_calls:
                        for tool_call in tool_calls:
                            if "function" in tool_call and "arguments" in tool_call["function"]:
                                args = tool_call["function"]["arguments"]
                                if isinstance(args, str):
                                    try:
                                        tool_call["function"]["arguments"] = json.loads(args)
                                    except json.JSONDecodeError:
                                        pass
        return result


from prompts.agent_prompt_forex import STOP_SIGNAL, get_agent_system_prompt_forex
from tools.general_tools import (extract_conversation, extract_tool_messages,
                                 get_config_value, write_config_value)
from tools.price_tools import add_no_trade_record


load_dotenv()


class BaseAgentForex:
    """
    Autonomous forex trading agent for live MT4/MT5 markets.

    Operates in two modes:
    1. CHALLENGE MODE: Compound $20 -> $50k with 30% risk, 20-pip targets
    2. STANDARD MODE: Steady income generation with conservative risk

    Connects to MT4/MT5 via configurable backend:
    - mt5_native: Direct IPC (Windows, lowest latency)
    - metaapi: Cloud relay (any OS, ~100ms latency)
    - zeromq: ZeroMQ bridge (flexible, ~10ms latency)
    """

    # Major forex pairs (pip value = $10/standard lot for USD-quoted pairs)
    MAJOR_PAIRS = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "NZDUSD", "USDCAD",
    ]

    # Minor/cross pairs
    MINOR_PAIRS = [
        "EURGBP", "EURJPY", "EURCHF", "GBPJPY",
        "GBPCHF", "AUDJPY", "NZDJPY", "AUDNZD",
    ]

    # Default trading pairs (majors only for tightest spreads)
    DEFAULT_FOREX_SYMBOLS = MAJOR_PAIRS

    # Pip values per standard lot (1.0 lot = 100,000 units)
    # For USD-quoted pairs (EURUSD, GBPUSD, etc): 1 pip = $10
    # For JPY pairs: 1 pip = ~$6.67 (varies with USD/JPY rate)
    PIP_VALUES = {
        "EURUSD": 10.0, "GBPUSD": 10.0, "AUDUSD": 10.0,
        "NZDUSD": 10.0, "USDCHF": 10.0, "USDCAD": 10.0,
        "USDJPY": 6.67, "EURGBP": 12.50, "EURJPY": 6.67,
        "EURCHF": 10.0, "GBPJPY": 6.67, "GBPCHF": 10.0,
        "AUDJPY": 6.67, "NZDJPY": 6.67, "AUDNZD": 6.50,
    }

    def __init__(
        self,
        signature: str,
        basemodel: str,
        forex_symbols: Optional[List[str]] = None,
        mcp_config: Optional[Dict[str, Dict[str, Any]]] = None,
        log_path: Optional[str] = None,
        max_steps: int = 50,
        max_retries: int = 3,
        base_delay: float = 2.0,
        openai_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        initial_cash: float = 20.0,
        init_date: str = "2026-01-01",
        market: str = "forex",
        # Challenge mode params
        challenge_mode: bool = True,
        risk_percent: float = 30.0,
        target_pips: float = 20.0,
        target_balance: float = 50000.0,
        max_daily_loss_percent: float = 5.0,
        max_drawdown_percent: float = 10.0,
        # MT4/MT5 connection params
        mt4_mt5_config: Optional[Dict[str, Any]] = None,
        # Trading session control
        trading_sessions: Optional[List[str]] = None,
        loop_interval_seconds: int = 300,
    ):
        """
        Initialize BaseAgentForex.

        Args:
            signature: Agent identifier (e.g., "claude-opus-forex")
            basemodel: LLM model identifier (e.g., "anthropic/claude-opus-4-6")
            forex_symbols: Trading pairs (default: major pairs)
            mcp_config: MCP tool server configuration
            log_path: Path for trade logs
            max_steps: Max LLM reasoning steps per trading cycle
            max_retries: Retry attempts per cycle
            base_delay: Base delay between retries (seconds)
            openai_base_url: LLM API base URL
            openai_api_key: LLM API key
            initial_cash: Starting balance (USD)
            init_date: Start date for logging
            market: Market type (always "forex")
            challenge_mode: If True, use aggressive compounding rules
            risk_percent: Percent of balance to risk per trade
            target_pips: Pip target per trade
            target_balance: Balance target for challenge completion
            max_daily_loss_percent: Max daily loss before stopping (%)
            max_drawdown_percent: Max total drawdown before stopping (%)
            mt4_mt5_config: MT4/MT5 connection parameters
            trading_sessions: When to trade (e.g., ["london_ny_overlap"])
            loop_interval_seconds: Seconds between trading cycles
        """
        self.signature = signature
        self.basemodel = basemodel
        self.market = "forex"

        # Forex symbols
        self.forex_symbols = forex_symbols or self.DEFAULT_FOREX_SYMBOLS

        # Agent config
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.initial_cash = initial_cash
        self.init_date = init_date

        # MCP config
        self.mcp_config = mcp_config or self._get_default_mcp_config()

        # Log paths
        self.base_log_path = log_path or "./data/agent_data_forex"

        # LLM API config
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_API_BASE")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Challenge mode configuration
        self.challenge_mode = challenge_mode
        self.risk_percent = risk_percent
        self.target_pips = target_pips
        self.target_balance = target_balance
        self.max_daily_loss_percent = max_daily_loss_percent
        self.max_drawdown_percent = max_drawdown_percent

        # MT4/MT5 connection config
        self.mt4_mt5_config = mt4_mt5_config or {}

        # Trading session control
        self.trading_sessions = trading_sessions or ["london_ny_overlap"]
        self.loop_interval_seconds = loop_interval_seconds

        # Challenge state tracking
        self.challenge_level = 0
        self.consecutive_wins = 0
        self.daily_starting_balance = initial_cash
        self.peak_balance = initial_cash

        # Internal components (initialized in initialize())
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: Optional[List] = None
        self.model: Optional[ChatOpenAI] = None
        self.agent: Optional[Any] = None

        # Data paths
        self.data_path = os.path.join(self.base_log_path, self.signature)
        self.position_file = os.path.join(self.data_path, "position", "position.jsonl")

    def _get_default_mcp_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default MCP configuration for forex trading."""
        return {
            "math": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('MATH_HTTP_PORT', '8000')}/mcp",
            },
            "search": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('SEARCH_HTTP_PORT', '8001')}/mcp",
            },
            "forex_trade": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('FOREX_HTTP_PORT', '8006')}/mcp",
            },
        }

    async def initialize(self) -> None:
        """Initialize MCP client and AI model."""
        print(f"🚀 Initializing forex agent: {self.signature}")
        print(f"💱 Mode: {'CHALLENGE' if self.challenge_mode else 'STANDARD'}")
        print(f"💰 Starting balance: ${self.initial_cash}")
        if self.challenge_mode:
            print(f"🎯 Target: ${self.target_balance:,.0f}")
            print(f"📊 Risk per trade: {self.risk_percent}%")
            print(f"📏 Target per trade: {self.target_pips} pips")

        # Validate LLM API key
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not set. Configure OPENAI_API_KEY in .env or config."
            )

        # Initialize MCP client
        try:
            self.client = MultiServerMCPClient(self.mcp_config)
            self.tools = await self.client.get_tools()
            if not self.tools:
                print("Warning: No MCP tools loaded. Ensure forex MCP services are running.")
            else:
                print(f"Loaded {len(self.tools)} MCP tools")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MCP client: {e}\n"
                f"Ensure MCP services are running: python agent_tools/start_mcp_services.py"
            )

        # Initialize LLM
        try:
            if "deepseek" in self.basemodel.lower():
                self.model = DeepSeekChatOpenAI(
                    model=self.basemodel,
                    base_url=self.openai_base_url,
                    api_key=self.openai_api_key,
                    max_retries=3,
                    timeout=60,
                )
            else:
                self.model = ChatOpenAI(
                    model=self.basemodel,
                    base_url=self.openai_base_url,
                    api_key=self.openai_api_key,
                    max_retries=3,
                    timeout=60,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AI model: {e}")

        print(f"Forex Agent {self.signature} initialization completed")

    def _setup_logging(self, session_id: str) -> str:
        """Set up log file for a trading session."""
        log_path = os.path.join(self.base_log_path, self.signature, "log", session_id)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        return os.path.join(log_path, "log.jsonl")

    def _log_message(self, log_file: str, new_messages) -> None:
        """Log messages to JSONL file."""
        log_entry = {
            "signature": self.signature,
            "timestamp": datetime.utcnow().isoformat(),
            "new_messages": new_messages,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    async def _ainvoke_with_retry(self, message: List[Dict[str, str]]) -> Any:
        """Agent invocation with exponential backoff retry."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self.agent.ainvoke(
                    {"messages": message}, {"recursion_limit": 100}
                )
            except Exception as e:
                if attempt == self.max_retries:
                    raise e
                wait = self.base_delay * (2 ** (attempt - 1))
                print(f"Attempt {attempt} failed, retrying in {wait}s: {e}")
                await asyncio.sleep(wait)

    # ── Risk Management ──────────────────────────────────────────────

    def calculate_lot_size(self, balance: float, symbol: str = "EURUSD") -> float:
        """
        Calculate lot size based on challenge risk rules.

        Formula:
            risk_amount = balance * (risk_percent / 100)
            pip_value_needed = risk_amount / target_pips
            lot_size = pip_value_needed / pip_value_per_standard_lot
        """
        risk_amount = balance * (self.risk_percent / 100)
        pip_value_needed = risk_amount / self.target_pips
        pip_value_per_lot = self.PIP_VALUES.get(symbol, 10.0)
        lot_size = pip_value_needed / pip_value_per_lot
        # Round down to nearest micro lot (0.001)
        lot_size = max(0.001, round(lot_size, 3))
        return lot_size

    def check_daily_loss_limit(self, current_balance: float) -> bool:
        """Check if daily loss limit has been breached."""
        daily_loss = (self.daily_starting_balance - current_balance) / self.daily_starting_balance * 100
        return daily_loss >= self.max_daily_loss_percent

    def check_max_drawdown(self, current_balance: float) -> bool:
        """Check if maximum drawdown from peak has been breached."""
        drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100
        return drawdown >= self.max_drawdown_percent

    def update_challenge_state(self, trade_result: Dict[str, Any]) -> None:
        """Update challenge tracking after a trade."""
        pips = trade_result.get("pips", 0)
        new_balance = trade_result.get("balance", self.initial_cash)

        if pips >= self.target_pips:
            self.consecutive_wins += 1
            self.challenge_level += 1
            print(f"Level {self.challenge_level} COMPLETE | "
                  f"Streak: {self.consecutive_wins} | Balance: ${new_balance:.2f}")
        else:
            self.consecutive_wins = 0
            print(f"Loss recorded | Streak reset | Balance: ${new_balance:.2f}")

        # Update peak balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

    def should_reset_challenge(self, current_balance: float) -> bool:
        """Check if challenge target reached."""
        return current_balance >= self.target_balance

    def get_challenge_status(self, current_balance: float) -> Dict[str, Any]:
        """Get current challenge progress."""
        return {
            "mode": "challenge" if self.challenge_mode else "standard",
            "level": self.challenge_level,
            "consecutive_wins": self.consecutive_wins,
            "balance": round(current_balance, 2),
            "target": self.target_balance,
            "progress_pct": round((current_balance / self.target_balance) * 100, 2),
            "peak_balance": round(self.peak_balance, 2),
            "next_lot_size": self.calculate_lot_size(current_balance),
            "risk_per_trade_usd": round(current_balance * self.risk_percent / 100, 2),
        }

    # ── Trading Session Control ──────────────────────────────────────

    def is_valid_trading_time(self) -> bool:
        """
        Check if current UTC time is within allowed trading sessions.

        Sessions:
        - london_ny_overlap: 13:00-17:00 UTC (highest liquidity)
        - london: 08:00-16:00 UTC
        - new_york: 13:00-22:00 UTC
        - asian: 00:00-08:00 UTC
        - all: 00:00-23:59 UTC (24/5)
        """
        now = datetime.utcnow()

        # Never trade on weekends (Fri 22:00 - Sun 22:00 UTC)
        weekday = now.weekday()
        hour = now.hour
        if weekday == 5:  # Saturday
            return False
        if weekday == 6:  # Sunday
            return hour >= 22  # Market opens ~22:00 UTC Sunday
        if weekday == 4 and hour >= 22:  # Friday after market close
            return False

        if "all" in self.trading_sessions:
            return True

        session_windows = {
            "london_ny_overlap": (13, 17),
            "london": (8, 16),
            "new_york": (13, 22),
            "asian": (0, 8),
        }

        for session in self.trading_sessions:
            if session in session_windows:
                start_hour, end_hour = session_windows[session]
                if start_hour <= hour < end_hour:
                    return True

        return False

    # ── Core Trading Loop ────────────────────────────────────────────

    async def run_trading_session(self, session_id: str) -> None:
        """
        Run a single trading analysis cycle.

        The AI agent receives current state and decides whether to trade.
        """
        print(f"Starting forex trading cycle: {session_id}")

        log_file = self._setup_logging(session_id)
        write_config_value("LOG_FILE", log_file)

        # Create agent with current market state in the system prompt
        self.agent = create_agent(
            self.model,
            tools=self.tools,
            system_prompt=get_agent_system_prompt_forex(
                session_id=session_id,
                signature=self.signature,
                forex_symbols=self.forex_symbols,
                challenge_status=self.get_challenge_status(self.initial_cash),
            ),
        )

        # Initial prompt
        user_query = [{"role": "user", "content": (
            f"Analyze current forex market conditions for session {session_id}. "
            f"Check prices, spreads, and decide if there's a high-probability setup. "
            f"If yes, calculate lot size and execute. If no good setup, wait."
        )}]
        message = user_query.copy()
        self._log_message(log_file, user_query)

        # Trading reasoning loop
        current_step = 0
        while current_step < self.max_steps:
            current_step += 1
            print(f"Step {current_step}/{self.max_steps}")

            try:
                response = await self._ainvoke_with_retry(message)
                agent_response = extract_conversation(response, "final")

                if STOP_SIGNAL in agent_response:
                    print("Agent decided: no trade this cycle")
                    self._log_message(log_file, [{"role": "assistant", "content": agent_response}])
                    break

                tool_msgs = extract_tool_messages(response)
                tool_response = "\n".join([msg.content for msg in tool_msgs])

                new_messages = [
                    {"role": "assistant", "content": agent_response},
                    {"role": "user", "content": f"Tool results: {tool_response}"},
                ]
                message.extend(new_messages)
                self._log_message(log_file, new_messages[0])
                self._log_message(log_file, new_messages[1])

            except Exception as e:
                print(f"Trading cycle error: {e}")
                raise

        # Handle result
        await self._handle_trading_result(session_id)

    async def _handle_trading_result(self, session_id: str) -> None:
        """Process post-trade state."""
        if_trade = get_config_value("IF_TRADE")
        if if_trade:
            write_config_value("IF_TRADE", False)
            print(f"Trade executed in session {session_id}")
        else:
            print(f"No trade in session {session_id}")
            write_config_value("IF_TRADE", False)

    async def run_live_loop(self) -> None:
        """
        Continuous live trading loop.

        Runs indefinitely (24/5), checking for trade setups every
        loop_interval_seconds. Respects trading session windows.
        """
        print(f"Starting live forex trading loop")
        print(f"Interval: {self.loop_interval_seconds}s")
        print(f"Sessions: {self.trading_sessions}")
        print(f"Pairs: {self.forex_symbols}")

        cycle = 0
        while True:
            try:
                if not self.is_valid_trading_time():
                    print(f"Outside trading session. Sleeping {self.loop_interval_seconds}s...")
                    await asyncio.sleep(self.loop_interval_seconds)
                    continue

                cycle += 1
                session_id = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
                print(f"\n{'='*60}")
                print(f"Cycle {cycle} | Session: {session_id}")
                print(f"{'='*60}")

                # Run one trading analysis cycle
                await self.run_trading_session(session_id)

                # Check challenge completion
                # (would need to read current balance from MT4/MT5)

                # Wait for next cycle
                print(f"Sleeping {self.loop_interval_seconds}s until next cycle...")
                await asyncio.sleep(self.loop_interval_seconds)

            except KeyboardInterrupt:
                print("\nLive loop stopped by user")
                break
            except Exception as e:
                print(f"Cycle error: {e}")
                # Back off on errors
                await asyncio.sleep(self.loop_interval_seconds * 2)

    # ── Compatibility Methods (match BaseAgent interface) ────────────

    def register_agent(self) -> None:
        """Register new forex agent, create initial position file."""
        if os.path.exists(self.position_file):
            print(f"Position file already exists, skipping registration")
            return

        position_dir = os.path.join(self.data_path, "position")
        if not os.path.exists(position_dir):
            os.makedirs(position_dir)

        # Forex positions track lot sizes (0.0 = no position)
        init_position = {symbol: 0.0 for symbol in self.forex_symbols}
        init_position["CASH"] = self.initial_cash

        with open(self.position_file, "w") as f:
            f.write(json.dumps({
                "date": self.init_date,
                "id": 0,
                "positions": init_position,
                "challenge_state": self.get_challenge_status(self.initial_cash),
            }) + "\n")

        print(f"Forex Agent {self.signature} registered")
        print(f"Balance: ${self.initial_cash}")
        print(f"Pairs: {len(self.forex_symbols)}")

    def get_trading_dates(self, init_date: str, end_date: str) -> List[str]:
        """
        Generate trading dates for forex (Mon-Fri).

        For live mode, this is less relevant since we use run_live_loop().
        Kept for compatibility with the date-range backtesting interface.
        """
        dates = []
        start = datetime.strptime(init_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        current = start + timedelta(days=1)
        while current <= end:
            # Forex trades Mon-Fri (weekday 0-4)
            if current.weekday() < 5:
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        return dates

    async def run_date_range(self, init_date: str, end_date: str) -> None:
        """Run trading across a date range (backtest-compatible interface)."""
        print(f"Running forex date range: {init_date} to {end_date}")
        trading_dates = self.get_trading_dates(init_date, end_date)

        if not trading_dates:
            print("No trading days to process")
            return

        print(f"Trading days: {len(trading_dates)}")

        for date in trading_dates:
            write_config_value("TODAY_DATE", date)
            write_config_value("SIGNATURE", self.signature)
            try:
                await self.run_with_retry(date)
            except Exception as e:
                print(f"Error on {date}: {e}")
                raise

        print(f"Forex processing completed for {self.signature}")

    async def run_with_retry(self, session_id: str) -> None:
        """Run with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                await self.run_trading_session(session_id)
                return
            except Exception as e:
                if attempt == self.max_retries:
                    raise
                wait = self.base_delay * attempt
                print(f"Attempt {attempt} failed, retrying in {wait}s: {e}")
                await asyncio.sleep(wait)

    def get_position_summary(self) -> Dict[str, Any]:
        """Get current position summary."""
        if not os.path.exists(self.position_file):
            return {"error": "Position file does not exist"}

        positions = []
        with open(self.position_file, "r") as f:
            for line in f:
                positions.append(json.loads(line))

        if not positions:
            return {"error": "No position records"}

        latest = positions[-1]
        return {
            "signature": self.signature,
            "latest_date": latest.get("date"),
            "positions": latest.get("positions", {}),
            "total_records": len(positions),
            "challenge_state": latest.get("challenge_state", {}),
        }

    def __str__(self) -> str:
        mode = "CHALLENGE" if self.challenge_mode else "STANDARD"
        return (
            f"BaseAgentForex(signature='{self.signature}', "
            f"basemodel='{self.basemodel}', "
            f"pairs={len(self.forex_symbols)}, "
            f"mode={mode}, "
            f"balance=${self.initial_cash})"
        )

    def __repr__(self) -> str:
        return self.__str__()
