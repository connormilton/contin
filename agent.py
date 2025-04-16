"""
LLM Agent Module
Handles interactions with LLM for analysis and review
"""

import json
import logging
import os
from datetime import datetime, timezone
import openai

logger = logging.getLogger("EURUSDTrader")

class BaseAgent:
    """Base class for LLM agents"""
    
    def __init__(self, model, budget_manager=None):
        """Initialize the base agent with model and budget tracking"""
        self.model = model
        self.budget_manager = budget_manager
        
        # Set up API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
    
    def _call_llm(self, system_message, user_message, temperature=0.3):
        """Call LLM API with appropriate formatting"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Build parameters for API call
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=60
            )
            
            # Calculate cost (simplified approximation)
            tokens_in = response.usage.prompt_tokens
            tokens_out = response.usage.completion_tokens
            
            if "gpt-4" in self.model:
                cost = (tokens_in * 0.03 + tokens_out * 0.06) / 1000
            else:  # GPT-3.5
                cost = (tokens_in * 0.0015 + tokens_out * 0.002) / 1000
            
            # Track cost if budget manager provided
            if self.budget_manager:
                self.budget_manager(cost, self.__class__.__name__)
            
            # Parse response
            content = response.choices[0].message.content
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON if wrapped in code blocks
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                if json_match:
                    try:
                        return json.loads(json_match.group(1))
                    except:
                        pass
                
                # If still can't parse, return error
                logger.error(f"Failed to parse JSON from response: {content[:100]}...")
                return {"error": "Failed to parse response as JSON"}
                
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.error(f"LLM API error after {duration:.1f}s: {e}")
            return {"error": str(e)}
    
    def run(self, **kwargs):
        """Run the agent (must be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement run method")


class AnalysisAgent(BaseAgent):
    """Agent for analyzing market opportunities with enhanced EUR/USD analysis"""
    
    def run(self, market_data, indicators, account_data, positions, recent_trades, config, market_regime, memory):
        """Run the analysis agent to find trading opportunities"""
        logger.info("Running Analysis Agent")
        
        try:
            # Create market summary for prompt
            market_summary = self._format_market_data(market_data)
            
            # Create indicator summary
            indicator_summary = self._format_indicators(indicators)
            
            # Create position summary
            position_summary = []
            for position in positions:
                position_summary.append({
                    "instrument": position.get("epic"),
                    "direction": position.get("direction"),
                    "size": position.get("size"),
                    "entry_price": position.get("level"),
                    "current_profit": position.get("profit"),
                    "stop_level": position.get("stop_level", 0)
                })
            
            # Create trade history summary
            trade_history = []
            for trade in recent_trades:
                # Only include relevant info
                if trade.get("instrument") != config["instrument"]:
                    continue
                    
                trade_history.append({
                    "timestamp": trade.get("timestamp"),
                    "direction": trade.get("direction"),
                    "action_type": trade.get("action_type", "OPEN"),
                    "entry_price": trade.get("entry_price"),
                    "close_price": trade.get("close_price"),
                    "outcome": trade.get("outcome", ""),
                    "pattern": trade.get("pattern", ""),
                    "r_multiple": trade.get("r_multiple"),
                    "return_percent": trade.get("return_percent"),
                    "reasoning": trade.get("reasoning", "")
                })
            
            # Get pattern effectiveness from memory
            pattern_effectiveness = {
                "successful": memory.get("learning", {}).get("successful_patterns", {}),
                "failed": memory.get("learning", {}).get("failed_patterns", {})
            }
            
            # Create prompt
            system_message = self._get_system_message()
            user_message = self._get_user_message(
                market_summary=market_summary,
                indicator_summary=indicator_summary,
                position_summary=position_summary,
                trade_history=trade_history,
                account_data=account_data,
                config=config,
                market_regime=market_regime,
                pattern_effectiveness=pattern_effectiveness
            )
            
            # Call LLM API with higher temperature for creative analysis
            result = self._call_llm(system_message, user_message, temperature=0.4)
            
            if result and "trading_opportunities" in result:
                logger.info(f"Analysis found {len(result['trading_opportunities'])} trading opportunities")
                return result
            else:
                if "error" in result:
                    logger.error(f"Analysis agent error: {result['error']}")
                else:
                    logger.warning("Analysis agent produced no results")
                return None
                
        except Exception as e:
            logger.error(f"Error in analysis agent: {e}", exc_info=True)
            return None
    
    def _format_market_data(self, market_data):
        """Format market data for the prompt"""
        formatted = {}
        
        # Current price
        current = market_data.get("current", {})
        formatted["current"] = {
            "bid": current.get("bid"),
            "ask": current.get("offer", current.get("ask")),
            "timestamp": current.get("timestamp")
        }
        
        # Recent candles for each timeframe
        for tf, candles in market_data.items():
            if tf == "current" or not candles:
                continue
                
            # Add the most recent candles
            formatted[tf] = []
            for candle in candles[-10:]:  # Last 10 candles
                formatted[tf].append(candle)
        
        return formatted
    
    def _format_indicators(self, indicators):
        """Format technical indicators for the prompt"""
        formatted = {}
        
        for timeframe, data in indicators.items():
            # Only include the most recent indicators
            if data:
                formatted[timeframe] = data[-1]  # Most recent indicators
        
        return formatted
    
    def _get_system_message(self):
        """Get system message for the analysis agent"""
        return """You are an expert forex trading analyst specializing in EUR/USD pair technical analysis. Your expertise involves multi-timeframe analysis, identifying high-probability trade setups, detecting key patterns, and providing detailed trade plans with precise entry, exit, and risk management parameters.

Key responsibilities:
1. Analyze all available timeframes (M15, H1, H4, D1) to identify the strongest trading opportunities
2. Apply multiple technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, Fibonacci levels)
3. Identify key support/resistance levels and chart patterns specific to EUR/USD
4. Detect RSI divergences and other momentum indicators
5. Determine market regime (trending, ranging, volatile) and adjust strategy accordingly
6. Calculate precise entry zones, stop losses, and multiple take profit targets
7. Provide in-depth reasoning and analysis for each trade recommendation
8. Consider previous trade performance and pattern effectiveness in your analysis
9. Adapt strategy based on prevailing market conditions for EUR/USD
10. Calculate precise risk-reward ratios based on technical levels

You provide detailed explanations of your thought process, ensuring your recommendations are clear, justified by technical evidence, and include thorough risk management parameters. You focus on EUR/USD only and understand its specific behavior patterns.

Respond with JSON containing your complete market analysis and trading recommendations."""
    
    def _get_user_message(self, market_summary, indicator_summary, position_summary, trade_history, account_data, config, market_regime, pattern_effectiveness):
        """Get user message for the analysis agent"""
        # Extract recent performance metrics
        win_count = len([t for t in trade_history if t.get("outcome", "").startswith("WIN") or t.get("outcome", "").startswith("PROFIT")])
        loss_count = len([t for t in trade_history if t.get("outcome", "").startswith("LOSS") or t.get("outcome", "").startswith("STOPPED")])
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        # Format pattern effectiveness
        pattern_section = "## Pattern Effectiveness\n"
        
        if pattern_effectiveness["successful"]:
            pattern_section += "### Successful Patterns\n"
            for pattern, count in sorted(pattern_effectiveness["successful"].items(), key=lambda x: x[1], reverse=True):
                pattern_section += f"- {pattern}: {count} successful trades\n"
                
        if pattern_effectiveness["failed"]:
            pattern_section += "\n### Failed Patterns\n"
            for pattern, count in sorted(pattern_effectiveness["failed"].items(), key=lambda x: x[1], reverse=True):
                pattern_section += f"- {pattern}: {count} failed trades\n"
        
        return f"""
# EUR/USD Enhanced Market Analysis Task

## Account Status
- Balance: {account_data.get('balance')} {account_data.get('currency')}
- Win Rate: {win_rate:.1f}% ({win_count}/{total_trades} trades)
- Open Positions: {len(position_summary)}

## Trading Configuration
- Instrument: EUR/USD
- Min Quality Score: {config['min_quality_score']}/10
- Min Risk-Reward: {config['min_risk_reward']}
- Base Risk: {config['base_risk_percent']}%

## Current Market Regime
- EUR/USD: {market_regime}

## Current Positions
{json.dumps(position_summary, indent=2)}

## Recent Trades
{json.dumps(trade_history[:5], indent=2)}
{pattern_section}

## Current Market Data
{json.dumps(market_summary, indent=2)}

## Technical Indicators
{json.dumps(indicator_summary, indent=2)}

## Analysis Instructions
1. Perform multi-timeframe analysis on EUR/USD using all available timeframes
2. Apply technical indicators to identify high-probability setups, including:
   - Moving Averages (SMA and EMA)
   - RSI (including divergence analysis)
   - MACD (signal line crossovers and histogram analysis)
   - Bollinger Bands (including squeeze patterns)
   - Fibonacci retracement and extension levels
   - Support/Resistance levels across multiple timeframes
   - ADX for trend strength assessment
   - ATR for volatility measurement and stop loss calculation

3. For EUR/USD specifically, identify:
   - Overall trend direction on each timeframe
   - Key support and resistance levels
   - Notable chart patterns (triangle patterns, head and shoulders, double tops/bottoms, etc.)
   - Price action signals (engulfing candles, pin bars, inside bars)
   - Potential price traps to avoid
   - Market structure (higher highs/lows or lower highs/lows)

4. Account for market regime and adjust analysis accordingly:
   - In trending markets: focus on pullbacks to moving averages, flag patterns
   - In ranging markets: focus on support/resistance bounces and breakouts
   - In volatile markets: widen stops and prioritize higher probability setups
   - In tight ranges: prepare for potential breakouts with tight risk management

5. For each trading opportunity, provide:
   - Precise entry zone with ideal price and acceptable range
   - Multiple timeframe confirmation
   - Stop loss with clear reasoning based on market structure
   - Multiple take profit targets with specific levels
   - Quality score (1-10) based on pattern clarity and confluence
   - Risk-reward calculation with exact numbers
   - Pattern identification for tracking effectiveness
   - Detailed step-by-step reasoning explaining the opportunity
   - Explicit price conditions that would invalidate the setup

6. Focus on finding patterns that have been historically successful based on the pattern effectiveness data
7. Only recommend trades with quality scores of {config['min_quality_score']}+ and risk-reward of at least {config['min_risk_reward']}
8. If no high-quality setups exist, clearly state that no trades meet the criteria

## Response Format
Respond with a JSON object containing:
1. "market_analysis" object with detailed market analysis for each timeframe
2. "support_resistance" array with key price levels and their significance
3. "market_structure" object describing the current EUR/USD structure
4. "trading_opportunities" array with detailed analysis for each trading opportunity
5. "risk_assessment" object with evaluation of current market conditions
6. "self_improvement" object with reflections on your analysis process
"""