"""
EUR/USD LLM-Powered Trading System
Main system coordinator with enhanced risk management
"""

import os
import time
import logging
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import colorama
from colorama import Fore, Style

from analysis import MarketAnalyzer
from agent import AnalysisAgent, ReviewAgent
from executor import OrderManager
from risk import RiskManager
from utils import OandaClient, save_chart

# Initialize colorama for colored terminal output
colorama.init()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EURUSDTrader")

# System configuration
CONFIG = {
    "instrument": "EUR_USD",
    "timeframes": {
        "m15": {"granularity": "M15", "count": 96},  # 24 hours
        "h1": {"granularity": "H1", "count": 48},    # 2 days
        "h4": {"granularity": "H4", "count": 30},    # 5 days
        "d1": {"granularity": "D", "count": 20},     # 20 days
    },
    "indicators": {
        "sma": [20, 50, 200],      # Simple Moving Averages
        "ema": [8, 21, 55],        # Exponential Moving Averages
        "rsi": {"period": 14},     # Relative Strength Index
        "macd": {"fast": 12, "slow": 26, "signal": 9},  # MACD
        "bbands": {"period": 20, "std_dev": 2},       # Bollinger Bands
        "atr": {"period": 14}      # Average True Range
    },
    "cycle_minutes": 15,                   # Minutes between trading cycles
    "analysis_hours": 4,                   # Hours between deep analysis
    "max_positions": 3,                    # Maximum open positions
    "base_risk_percent": 1.0,              # Base risk percentage
    "max_risk_percent": 1.5,               # Maximum risk percentage
    "min_risk_percent": 0.5,               # Minimum risk percentage
    "max_account_risk": 4.0,               # Maximum total account risk
    "breakeven_move_atr": 1.0,             # ATR multiplier to move stop to breakeven
    "partial_profit_r": 1.5,               # R-multiple for first partial take profit
    "partial_profit_pct": 30,              # Percentage to close at first take profit
    "second_profit_r": 2.5,                # R-multiple for second partial take profit
    "second_profit_pct": 30,               # Percentage to close at second take profit
    "min_quality_score": 7.0,              # Minimum analysis quality to execute trade
    "min_risk_reward": 1.5,                # Minimum risk-reward ratio
    "max_daily_loss_pct": 5.0,             # Maximum daily loss percentage to pause trading
    "max_consecutive_losses": 3,           # Maximum consecutive losses to adjust risk
    "llm_model": "gpt-4-turbo-preview",    # Model for analysis
    "daily_budget": float(os.getenv("DAILY_LLM_BUDGET", 20.0)),
}


class TradingSystem:
    """EUR/USD LLM-Powered Trading System"""
    
    def __init__(self):
        """Initialize the trading system"""
        self.config = CONFIG
        self.oanda = OandaClient()
        self.memory = self._initialize_memory()
        
        # Initialize components
        self.market_analyzer = MarketAnalyzer(self.config)
        self.risk_manager = RiskManager(self.config)
        self.order_manager = OrderManager(self.oanda, self.config, self.risk_manager)
        
        # Initialize LLM agents
        self.analysis_agent = AnalysisAgent(
            model=self.config["llm_model"],
            budget_manager=self._track_usage
        )
        
        self.review_agent = ReviewAgent(
            model=self.config["llm_model"],
            budget_manager=self._track_usage
        )
        
        # Initialize session budget
        self.session_usage = {
            "budget": self.config["daily_budget"],
            "spent": 0.0,
            "last_reset": datetime.now(timezone.utc).date().isoformat()
        }
        
        # Initialize market regime and state
        self.market_regime = "unknown"
        self.last_deep_analysis = datetime.now(timezone.utc)
        
        # Initialize daily P&L tracking
        self.daily_pnl = {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "start_balance": 0.0,
            "current_balance": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl_pct": 0.0
        }
        
    def _initialize_memory(self):
        """Initialize or load trading memory"""
        try:
            if os.path.exists("memory.json"):
                with open("memory.json", "r") as f:
                    memory = json.load(f)
                    logger.info(f"Loaded existing memory with {len(memory.get('trades', []))} trade records")
                    return memory
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            
        # Create new memory structure
        memory = {
            "trades": [],
            "analyses": {},
            "last_cycle": "",
            "performance": {
                "win_count": 0,
                "loss_count": 0,
                "total_return_pct": 0.0,
                "largest_win_pct": 0.0,
                "largest_loss_pct": 0.0,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
                "pattern_performance": {}
            },
            "feedback": {
                "analysis": "",
                "review": ""
            },
            "learning": {
                "successful_patterns": {},
                "failed_patterns": {},
                "timeframe_effectiveness": {},
                "risk_adjustments": []
            },
            "recommendations": []
        }
        return memory
    
    def save_memory(self):
        """Save trading memory to disk"""
        try:
            with open("memory.json", "w") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            
    def _track_usage(self, cost, agent_name):
        """Track LLM API usage"""
        # Reset budget if it's a new day
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self.session_usage["last_reset"]:
            self.session_usage = {
                "budget": self.config["daily_budget"],
                "spent": 0.0,
                "last_reset": today
            }
            
        # Update spent amount
        self.session_usage["spent"] += cost
        logger.info(f"LLM usage: ${cost:.4f} for {agent_name}, total: ${self.session_usage['spent']:.2f} of ${self.session_usage['budget']:.2f}")
        
        # Return remaining budget
        return self.session_usage["budget"] - self.session_usage["spent"]
    
    def log_trade(self, trade_data):
        """Log a trade to memory and analyze performance"""
        # Add timestamp if not present
        if "timestamp" not in trade_data:
            trade_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add to trades list
        self.memory["trades"].append(trade_data)
        
        # Update performance metrics if it's a closed trade
        if trade_data.get("action_type") == "CLOSE":
            outcome = trade_data.get("outcome", "")
            
            if "WIN" in outcome or "PROFIT" in outcome:
                self.memory["performance"]["win_count"] += 1
                self.memory["performance"]["consecutive_wins"] += 1
                self.memory["performance"]["consecutive_losses"] = 0
                
                # Update largest win if applicable
                return_pct = trade_data.get("return_percent", 0.0)
                if return_pct > self.memory["performance"]["largest_win_pct"]:
                    self.memory["performance"]["largest_win_pct"] = return_pct
                    
                # Track successful patterns
                pattern = trade_data.get("pattern", "unknown")
                if pattern in self.memory["learning"]["successful_patterns"]:
                    self.memory["learning"]["successful_patterns"][pattern] += 1
                else:
                    self.memory["learning"]["successful_patterns"][pattern] = 1
                    
                # Track pattern performance
                if pattern not in self.memory["performance"]["pattern_performance"]:
                    self.memory["performance"]["pattern_performance"][pattern] = {
                        "wins": 1, "losses": 0, "profit": return_pct
                    }
                else:
                    self.memory["performance"]["pattern_performance"][pattern]["wins"] += 1
                    self.memory["performance"]["pattern_performance"][pattern]["profit"] += return_pct
                    
            elif "LOSS" in outcome or "STOPPED" in outcome:
                self.memory["performance"]["loss_count"] += 1
                self.memory["performance"]["consecutive_losses"] += 1
                self.memory["performance"]["consecutive_wins"] = 0
                
                # Update largest loss if applicable
                return_pct = abs(trade_data.get("return_percent", 0.0))
                if return_pct > self.memory["performance"]["largest_loss_pct"]:
                    self.memory["performance"]["largest_loss_pct"] = return_pct
                
                # Track failed patterns
                pattern = trade_data.get("pattern", "unknown")
                if pattern in self.memory["learning"]["failed_patterns"]:
                    self.memory["learning"]["failed_patterns"][pattern] += 1
                else:
                    self.memory["learning"]["failed_patterns"][pattern] = 1
                
                # Track pattern performance
                if pattern not in self.memory["performance"]["pattern_performance"]:
                    self.memory["performance"]["pattern_performance"][pattern] = {
                        "wins": 0, "losses": 1, "profit": -return_pct
                    }
                else:
                    self.memory["performance"]["pattern_performance"][pattern]["losses"] += 1
                    self.memory["performance"]["pattern_performance"][pattern]["profit"] -= return_pct
            
            # Update realized P&L for the day
            profit = trade_data.get("profit", 0)
            self.daily_pnl["realized_pnl"] += float(profit)
            
            # Check for risk adjustments based on consecutive losses
            if self.memory["performance"]["consecutive_losses"] >= self.config["max_consecutive_losses"]:
                adjustment = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "risk_reduction",
                    "reason": f"{self.memory['performance']['consecutive_losses']} consecutive losses",
                    "adjustment": "Reduced risk to minimum"
                }
                self.memory["learning"]["risk_adjustments"].append(adjustment)
                logger.warning(f"Risk reduced to {self.config['min_risk_percent']}% due to consecutive losses")
        
        # Save to disk
        self.save_memory()
        
        # Print trade info
        self._print_trade_info(trade_data)
    
    def _print_trade_info(self, trade_data):
        """Print formatted trade information to terminal"""
        action_type = trade_data.get("action_type", "UNKNOWN")
        
        if action_type == "OPEN":
            message = (
                f"\n{Fore.CYAN}➡️  NEW TRADE{Style.RESET_ALL}\n"
                f"  {Fore.YELLOW}{trade_data.get('direction')}{Style.RESET_ALL} {trade_data.get('instrument')} @ {trade_data.get('entry_price')}\n"
                f"  Size: {trade_data.get('size')} | Risk: {trade_data.get('risk_percent')}%\n"
                f"  Stop Loss: {trade_data.get('stop_loss')}\n"
                f"  Pattern: {trade_data.get('pattern')}\n"
                f"  Quality: {trade_data.get('quality_score', 'N/A')}/10\n"
            )
            print(message)
            
        elif action_type == "CLOSE":
            outcome = trade_data.get("outcome", "UNKNOWN")
            color = Fore.GREEN if "WIN" in outcome or "PROFIT" in outcome else Fore.RED
            
            message = (
                f"\n{color}✓  CLOSED POSITION{Style.RESET_ALL}\n"
                f"  {trade_data.get('instrument')} @ {trade_data.get('close_price')}\n"
                f"  Profit/Loss: {color}{trade_data.get('profit')}{Style.RESET_ALL}\n"
                f"  Return: {color}{trade_data.get('return_percent', 0.0):.2f}%{Style.RESET_ALL}\n"
                f"  R-Multiple: {color}{trade_data.get('r_multiple', 0.0):.2f}R{Style.RESET_ALL}\n"
                f"  Outcome: {color}{outcome}{Style.RESET_ALL}\n"
            )
            print(message)
            
        elif action_type == "UPDATE_STOP":
            message = (
                f"\n{Fore.YELLOW}⚙️  UPDATED STOP{Style.RESET_ALL}\n"
                f"  {trade_data.get('instrument')} to {trade_data.get('new_level')}\n"
                f"  Reason: {trade_data.get('reason')}\n"
            )
            print(message)
            
        elif action_type == "PARTIAL_CLOSE":
            message = (
                f"\n{Fore.GREEN}🔸  PARTIAL CLOSE{Style.RESET_ALL}\n"
                f"  {trade_data.get('instrument')} @ {trade_data.get('close_price')}\n"
                f"  Closed: {trade_data.get('close_percent')}% of position\n"
                f"  Profit: {Fore.GREEN}{trade_data.get('profit')}{Style.RESET_ALL}\n"
            )
            print(message)
    
    def get_recent_trades(self, limit=10):
        """Get recent trades from memory"""
        trades = self.memory.get("trades", [])
        
        # Sort by timestamp (newest first)
        trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return trades[:limit]
    
    def update_daily_pnl(self, account_info, positions):
        """Update daily P&L tracking"""
        today = datetime.now(timezone.utc).date().isoformat()
        
        # If a new day, reset the tracking
        if today != self.daily_pnl["date"]:
            self.daily_pnl = {
                "date": today,
                "start_balance": float(account_info.get("balance", 0)),
                "current_balance": float(account_info.get("balance", 0)),
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "total_pnl_pct": 0.0
            }
            return
            
        # If first update of the day, set the start balance
        if self.daily_pnl["start_balance"] == 0:
            self.daily_pnl["start_balance"] = float(account_info.get("balance", 0))
            
        # Update current values
        current_balance = float(account_info.get("balance", 0))
        unrealized_pnl = sum([float(position.get("profit", 0)) for position in positions])
        
        self.daily_pnl["current_balance"] = current_balance
        self.daily_pnl["unrealized_pnl"] = unrealized_pnl
        
        # Calculate total P&L percentage
        if self.daily_pnl["start_balance"] > 0:
            total_change = (current_balance - self.daily_pnl["start_balance"]) + unrealized_pnl
            self.daily_pnl["total_pnl_pct"] = (total_change / self.daily_pnl["start_balance"]) * 100
    
    def run_trading_cycle(self):
        """Run a complete trading cycle"""
        logger.info("Starting trading cycle")
        
        try:
            # Check if we have enough budget
            remaining_budget = self.session_usage["budget"] - self.session_usage["spent"]
            if remaining_budget < 1.0:
                logger.warning(f"Insufficient budget remaining (${remaining_budget:.2f}). Skipping cycle.")
                return False
            
            # Get market data
            market_data = self.market_analyzer.get_market_data(self.oanda)
            
            # Get account and position data
            account_info = self.oanda.get_account()
            positions = self.oanda.get_open_positions()
            
            # Update daily P&L tracking
            self.update_daily_pnl(account_info, positions)
            
            # Check if we've hit maximum daily loss
            if self.daily_pnl["total_pnl_pct"] <= -self.config["max_daily_loss_pct"]:
                logger.warning(f"Maximum daily loss of {self.config['max_daily_loss_pct']}% reached. Pausing trading.")
                print(f"\n{Fore.RED}⚠️  MAXIMUM DAILY LOSS REACHED ({self.daily_pnl['total_pnl_pct']:.2f}%). TRADING PAUSED.{Style.RESET_ALL}\n")
                
                # Log this event
                self.memory["learning"]["risk_adjustments"].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "trading_pause",
                    "reason": "Maximum daily loss reached",
                    "pnl_percent": self.daily_pnl["total_pnl_pct"]
                })
                self.save_memory()
                return False
            
            # Calculate market regime
            self.market_regime = self.market_analyzer.calculate_market_regime(market_data)
            logger.info(f"Current market regime: {self.market_regime}")
            
            # Process technical indicators
            indicators = self.market_analyzer.calculate_indicators(market_data)
            
            # Determine if we need a deep analysis
            hours_since_analysis = (datetime.now(timezone.utc) - self.last_deep_analysis).total_seconds() / 3600
            need_deep_analysis = hours_since_analysis >= self.config["analysis_hours"]
            
            # Run trade management first
            self.order_manager.manage_positions(positions, market_data, indicators, self.log_trade)
            
            # Get recent trades for context
            recent_trades = self.get_recent_trades()
            
            # Run LLM analysis if needed
            analysis_result = None
            
            if need_deep_analysis or len(positions) < self.config["max_positions"]:
                analysis_result = self.analysis_agent.run(
                    market_data=market_data,
                    indicators=indicators,
                    account_data=account_info,
                    positions=positions,
                    recent_trades=recent_trades,
                    config=self.config,
                    market_regime=self.market_regime,
                    memory=self.memory
                )
                
                # Store analysis feedback and update timestamp
                if analysis_result and "self_improvement" in analysis_result:
                    self.memory["feedback"]["analysis"] = analysis_result["self_improvement"]
                    self.last_deep_analysis = datetime.now(timezone.utc)
                
                # Execute trades based on analysis
                if analysis_result and "trading_opportunities" in analysis_result:
                    opportunities = analysis_result["trading_opportunities"]
                    
                    for opportunity in opportunities:
                        # Determine risk percentage based on strategy quality and past performance
                        quality_score = float(opportunity.get("quality_score", 0))
                        risk_percent = self.risk_manager.calculate_risk_percentage(
                            quality_score=quality_score,
                            consecutive_losses=self.memory["performance"]["consecutive_losses"],
                            pattern=opportunity.get("pattern", "unknown"),
                            win_rate=self.memory["performance"].get("win_count", 0) / 
                                    max(1, self.memory["performance"].get("win_count", 0) + 
                                        self.memory["performance"].get("loss_count", 0))
                        )
                        
                        # Execute the trade
                        success = self.order_manager.execute_trade(
                            opportunity=opportunity,
                            risk_percent=risk_percent,
                            account_balance=float(account_info.get("balance", 1000)),
                            log_callback=self.log_trade
                        )
                        
                        if success:
                            # Save chart for reference
                            try:
                                save_chart(
                                    instrument=self.config["instrument"],
                                    timeframe="h1",
                                    indicators=indicators.get("h1", {}),
                                    entry_price=opportunity.get("entry_price"),
                                    stop_loss=opportunity.get("stop_loss"),
                                    direction=opportunity.get("direction"),
                                    pattern=opportunity.get("pattern")
                                )
                            except Exception as e:
                                logger.error(f"Error saving chart: {e}")
            
            # Run review agent periodically
            review_cycle = len(self.memory["trades"]) % 5 == 0  # Run every 5 trade records
            
            if review_cycle and analysis_result:
                review_result = self.review_agent.run(
                    analysis_result=analysis_result,
                    market_data=market_data,
                    account_data=account_info,
                    positions=positions,
                    recent_trades=recent_trades,
                    config=self.config,
                    market_regime=self.market_regime,
                    memory=self.memory
                )
                
                # Store review feedback and recommendations
                if review_result:
                    if "feedback" in review_result:
                        self.memory["feedback"]["review"] = review_result["feedback"]
                    
                    if "recommendations" in review_result:
                        self.memory["recommendations"] = (
                            self.memory.get("recommendations", []) + 
                            review_result["recommendations"]
                        )[-10:]  # Keep only the last 10 recommendations
                    
                    if "learning_insights" in review_result:
                        for key, value in review_result["learning_insights"].items():
                            if key in self.memory["learning"]:
                                self.memory["learning"][key].update(value)
            
            # Update last cycle timestamp
            self.memory["last_cycle"] = datetime.now(timezone.utc).isoformat()
            self.save_memory()
            
            # Display system status
            self._display_status(account_info, positions, market_data)
            
            # Log budget status
            remaining = self.session_usage["budget"] - self.session_usage["spent"]
            used_pct = (self.session_usage["spent"] / self.session_usage["budget"]) * 100
            logger.info(f"Budget status: ${remaining:.2f} remaining ({used_pct:.1f}% used)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            return False
    
    def _display_status(self, account_info, positions, market_data):
        """Display current system status in the terminal"""
        print("\n" + "="*80)
        print(f"{Fore.CYAN}EUR/USD TRADING SYSTEM STATUS{Style.RESET_ALL}".center(80))
        print("="*80)
        
        # Account summary
        balance = float(account_info.get("balance", 0))
        currency = account_info.get("currency", "USD")
        
        print(f"\n{Fore.YELLOW}Account Summary:{Style.RESET_ALL}")
        print(f"  Balance: {balance:.2f} {currency}")
        print(f"  Daily P&L: {self.daily_pnl['total_pnl_pct']:.2f}% ({self.daily_pnl['realized_pnl']:.2f} realized, {self.daily_pnl['unrealized_pnl']:.2f} unrealized)")
        
        # Performance
        win_count = self.memory["performance"]["win_count"]
        loss_count = self.memory["performance"]["loss_count"]
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n{Fore.YELLOW}Performance:{Style.RESET_ALL}")
        print(f"  Win Rate: {win_rate:.1f}% ({win_count}/{total_trades} trades)")
        print(f"  Largest Win: {self.memory['performance']['largest_win_pct']:.2f}%")
        print(f"  Largest Loss: {self.memory['performance']['largest_loss_pct']:.2f}%")
        print(f"  Consecutive Wins: {self.memory['performance']['consecutive_wins']}")
        print(f"  Consecutive Losses: {self.memory['performance']['consecutive_losses']}")
        
        # Open positions
        print(f"\n{Fore.YELLOW}Open Positions ({len(positions)}):{Style.RESET_ALL}")
        if positions:
            for p in positions:
                profit_color = Fore.GREEN if float(p.get("profit", 0)) > 0 else Fore.RED
                print(f"  {p.get('direction')} {p.get('epic')} @ {p.get('level')} | Profit: {profit_color}{p.get('profit', 0)}{Style.RESET_ALL} | Stop: {p.get('stop_level', 'None')}")
        else:
            print("  No open positions")
        
        # Market regime
        regime_color = Fore.GREEN if "uptrend" in self.market_regime else Fore.RED if "downtrend" in self.market_regime else Fore.YELLOW
        print(f"\n{Fore.YELLOW}Market Regime:{Style.RESET_ALL} {regime_color}{self.market_regime}{Style.RESET_ALL}")
        
        # Current price
        if "current" in market_data:
            current = market_data["current"]
            bid = current.get("bid", 0)
            ask = current.get("offer", current.get("ask", 0))
            spread = (ask - bid) * 10000  # Convert to pips
            print(f"\n{Fore.YELLOW}Current Price:{Style.RESET_ALL} {bid}/{ask} (Spread: {spread:.1f} pips)")
        
        # Budget status
        remaining = self.session_usage["budget"] - self.session_usage["spent"]
        used_pct = (self.session_usage["spent"] / self.session_usage["budget"]) * 100
        print(f"\n{Fore.YELLOW}LLM Budget:{Style.RESET_ALL} ${remaining:.2f} remaining ({used_pct:.1f}% used)")
        
        # Recent recommendations
        if self.memory.get("recommendations"):
            print(f"\n{Fore.GREEN}Recent Recommendations:{Style.RESET_ALL}")
            for rec in self.memory["recommendations"][-3:]:
                print(f"  • {rec}")
                
        print("\n" + "="*80 + "\n")
    
    def run(self):
        """Main trading loop"""
        logger.info("Starting EUR/USD LLM-Powered Trading System")
        print("\n" + "="*80)
        print(f"{Fore.CYAN}STARTING EUR/USD LLM-POWERED TRADING SYSTEM{Style.RESET_ALL}".center(80))
        print("="*80)
        
        print(f"\n{Fore.YELLOW}Instrument:{Style.RESET_ALL} {self.config['instrument']}")
        print(f"{Fore.YELLOW}Timeframes:{Style.RESET_ALL} {', '.join(self.config['timeframes'].keys())}")
        print(f"{Fore.YELLOW}Budget:{Style.RESET_ALL} ${self.config['daily_budget']:.2f}")
        
        # Initial account info
        account = self.oanda.get_account()
        initial_balance = float(account.get("balance", 0))
        currency = account.get("currency", "USD")
        print(f"{Fore.YELLOW}Account Balance:{Style.RESET_ALL} {initial_balance:.2f} {currency}")
        
        # Set initial daily P&L tracking
        self.daily_pnl["start_balance"] = initial_balance
        self.daily_pnl["current_balance"] = initial_balance
        
        # Trading loop
        while True:
            try:
                # Run a trading cycle
                self.run_trading_cycle()
                
                # Sleep between cycles
                cycle_minutes = self.config["cycle_minutes"]
                print(f"\n{Fore.BLUE}⏱️  Cycle complete. Sleeping for {cycle_minutes} minutes.{Style.RESET_ALL}")
                time.sleep(cycle_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Trading system stopped by user.")
                print(f"\n{Fore.RED}TRADING SYSTEM STOPPED BY USER{Style.RESET_ALL}")
                break
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                print(f"\n{Fore.RED}ERROR IN MAIN LOOP: {e}{Style.RESET_ALL}")
                # Sleep shorter time on error
                time.sleep(60)

def main():
    # Check if OANDA credentials are set
    if not os.getenv("OANDA_API_TOKEN") or not os.getenv("OANDA_ACCOUNT_ID"):
        print(f"{Fore.RED}ERROR: OANDA API credentials not set. Please check your .env file.{Style.RESET_ALL}")
        print("Required variables: OANDA_API_TOKEN, OANDA_ACCOUNT_ID")
        return
        
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{Fore.RED}ERROR: OpenAI API key not set. Please check your .env file.{Style.RESET_ALL}")
        print("Required variable: OPENAI_API_KEY")
        return
        
    # Start trading system
    system = TradingSystem()
    system.run()

if __name__ == "__main__":
    main()