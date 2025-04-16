"""
Order Execution Module
Handles trade execution, order management, and position monitoring
"""

import logging
from datetime import datetime, timezone
import json
import numpy as np

logger = logging.getLogger("EURUSDTrader")

class OrderManager:
    """Order management and execution for EUR/USD trading"""
    
    def __init__(self, oanda_client, config, risk_manager):
        """Initialize the order manager"""
        self.oanda = oanda_client
        self.config = config
        self.risk_manager = risk_manager
        self.instrument = config["instrument"]
        
    def execute_trade(self, opportunity, risk_percent, account_balance, log_callback=None):
        """Execute a trade based on analysis opportunity"""
        try:
            # Extract trade details
            direction = opportunity.get("direction")
            entry_price = float(opportunity.get("entry_price", 0))
            stop_loss = float(opportunity.get("stop_loss", 0))
            take_profit = opportunity.get("take_profit_levels", [None])[0]
            if take_profit:
                take_profit = float(take_profit)
            
            # Validate required fields
            if not direction or not entry_price or not stop_loss:
                logger.warning(f"Missing required trade parameters: Direction: {direction}, Entry: {entry_price}, Stop: {stop_loss}")
                return False
            
            # Calculate position size
            units = self.risk_manager.calculate_position_size(
                account_balance=account_balance,
                risk_percent=risk_percent,
                entry_price=entry_price,
                stop_loss=stop_loss
            )
            
            # Apply direction
            if direction == "SELL":
                units = -abs(units)
            else:
                units = abs(units)
            
            # Execute the trade
            logger.info(f"Executing trade: {self.instrument} | Units: {units} | SL: {stop_loss} | TP: {take_profit}")
            success, trade_result = self.oanda.execute_trade(
                instrument=self.instrument,
                units=units,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if success:
                # Add additional trade info
                trade_result.update({
                    "risk_percent": risk_percent,
                    "pattern": opportunity.get("pattern"),
                    "reasoning": opportunity.get("reasoning"),
                    "quality_score": opportunity.get("quality_score"),
                    "risk_reward": opportunity.get("risk_reward"),
                    "take_profit_levels": opportunity.get("take_profit_levels"),
                    "stop_loss": stop_loss
                })
                
                # Log the trade if callback provided
                if log_callback:
                    log_callback(trade_result)
                
                return True
            else:
                logger.warning(f"Failed to execute trade: {trade_result.get('reason', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def manage_positions(self, positions, market_data, indicators, log_callback=None):
        """Manage existing positions - trailing stops, partial closes, etc."""
        if not positions:
            return
        
        # Get current price
        current_price = market_data.get("current", {})
        if not current_price:
            logger.warning("Unable to manage positions - current price data not available")
            return
        
        # Current bid/ask
        bid = float(current_price.get("bid", 0))
        ask = float(current_price.get("offer", current_price.get("ask", 0)))
        
        # Get most recent H1 ATR value for stop management
        h1_indicators = indicators.get("h1", [])
        atr_value = 0.001  # Default if not available
        if h1_indicators and len(h1_indicators) > 0:
            atr_value = h1_indicators[-1].get("atr", 0.001)
        
        # Process each position
        for position in positions:
            # Skip if not our instrument
            if position.get("epic") != self.instrument:
                continue
                
            # Get position details
            direction = position.get("direction")
            entry_level = float(position.get("level", 0))
            current_stop = float(position.get("stop_level", 0))
            deal_id = position.get("dealId")
            position_size = abs(int(position.get("size", 0)))
            
            # Get price based on direction
            current_level = bid if direction == "SELL" else ask
            
            # Check if we should update stop to breakeven
            if self.risk_manager.should_move_to_breakeven(entry_level, current_level, direction, atr_value):
                # Only update if current stop is worse than breakeven
                if (direction == "BUY" and (current_stop < entry_level or current_stop == 0)) or \
                   (direction == "SELL" and (current_stop > entry_level or current_stop == 0)):
                    
                    # Move to breakeven plus a small buffer
                    pip_size = 0.0001
                    buffer_pips = 5
                    
                    if direction == "BUY":
                        new_stop = entry_level + (buffer_pips * pip_size)
                    else:
                        new_stop = entry_level - (buffer_pips * pip_size)
                    
                    # Update stop
                    success, update_result = self.oanda.update_stop_loss(self.instrument, new_stop)
                    
                    if success and log_callback:
                        # Log the update
                        log_callback({
                            "action_type": "UPDATE_STOP",
                            "instrument": self.instrument,
                            "deal_id": deal_id,
                            "new_level": new_stop,
                            "reason": "Moved to breakeven"
                        })
                        
                        logger.info(f"Updated stop to breakeven: {self.instrument} @ {new_stop}")
                    
                    continue  # Done with this position
            
            # Calculate potential trailing stop
            new_stop = self.risk_manager.calculate_trailing_stop(
                entry_level, current_level, direction, atr_value, current_stop
            )
            
            # Only update if new stop is better than current
            if (direction == "BUY" and new_stop > current_stop) or \
               (direction == "SELL" and new_stop < current_stop and current_stop > 0):
                
                # Update stop
                success, update_result = self.oanda.update_stop_loss(self.instrument, new_stop)
                
                if success and log_callback:
                    # Log the update
                    log_callback({
                        "action_type": "UPDATE_STOP",
                        "instrument": self.instrument,
                        "deal_id": deal_id,
                        "new_level": new_stop,
                        "reason": "Trailing stop update"
                    })
                    
                    logger.info(f"Updated trailing stop: {self.instrument} @ {new_stop}")
            
            # Check for partial take profit (using R multiples)
            try:
                # Calculate R value (initial risk)
                if current_stop > 0:
                    r_value = abs(entry_level - current_stop)
                    
                    # Calculate current profit in R multiples
                    if direction == "BUY":
                        current_r = (current_level - entry_level) / r_value
                    else:
                        current_r = (entry_level - current_level) / r_value
                    
                    # Check for first partial take profit
                    first_tp_r = self.config["partial_profit_r"]
                    second_tp_r = self.config["second_profit_r"]
                    
                    # First partial
                    if current_r >= first_tp_r and current_r < second_tp_r:
                        # Calculate units to close
                        close_units = self.risk_manager.calculate_partial_close_amount(position_size, 1)
                        
                        # Check if we've already taken partial profit (would have smaller size)
                        partial_taken = position_size < close_units * 2  # Assuming we start with at least 2x the partial size
                        
                        if close_units > 0 and not partial_taken:
                            # Close portion of position
                            success, close_result = self.oanda.close_partial(
                                self.instrument, 
                                close_units if direction == "BUY" else -close_units
                            )
                            
                            if success and log_callback:
                                # Calculate percentage closed
                                pct_closed = (close_units / position_size) * 100
                                
                                # Log the partial close
                                log_callback({
                                    "action_type": "PARTIAL_CLOSE",
                                    "instrument": self.instrument,
                                    "deal_id": deal_id,
                                    "close_units": close_units,
                                    "close_percent": pct_closed,
                                    "close_price": current_level,
                                    "profit": close_result.get("profit"),
                                    "reason": f"First target reached at {first_tp_r}R"
                                })
                                
                                logger.info(f"Partial close at first target: {self.instrument} @ {current_level}, {pct_closed:.1f}% of position")
                    
                    # Second partial
                    elif current_r >= second_tp_r:
                        # Calculate units to close
                        close_units = self.risk_manager.calculate_partial_close_amount(position_size, 2)
                        
                        # Check if we've already taken second partial profit
                        partial_taken = position_size < close_units  # Would be smaller after first and second partials
                        
                        if close_units > 0 and not partial_taken:
                            # Close portion of position
                            success, close_result = self.oanda.close_partial(
                                self.instrument, 
                                close_units if direction == "BUY" else -close_units
                            )
                            
                            if success and log_callback:
                                # Calculate percentage closed
                                pct_closed = (close_units / position_size) * 100
                                
                                # Log the partial close
                                log_callback({
                                    "action_type": "PARTIAL_CLOSE",
                                    "instrument": self.instrument,
                                    "deal_id": deal_id,
                                    "close_units": close_units,
                                    "close_percent": pct_closed,
                                    "close_price": current_level,
                                    "profit": close_result.get("profit"),
                                    "reason": f"Second target reached at {second_tp_r}R"
                                })
                                
                                logger.info(f"Partial close at second target: {self.instrument} @ {current_level}, {pct_closed:.1f}% of position")
            except Exception as e:
                logger.error(f"Error processing partial close: {e}")
    
    def close_position(self, position, reason, log_callback=None):
        """Close an entire position"""
        try:
            # Get position details
            instrument = position.get("epic")
            deal_id = position.get("dealId")
            
            # Close position
            success, close_result = self.oanda.close_position(instrument)
            
            if success and log_callback:
                # Add reason to result
                close_result["reason"] = reason
                close_result["deal_id"] = deal_id
                
                # Log the close
                log_callback(close_result)
                
                logger.info(f"Closed position: {instrument}, Reason: {reason}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False