"""
Risk Management Module
Handles position sizing, risk calculation, and money management
"""

import logging
import numpy as np

logger = logging.getLogger("EURUSDTrader")

class RiskManager:
    """Risk management and position sizing for EUR/USD trading"""
    
    def __init__(self, config):
        """Initialize the risk manager"""
        self.config = config
        self.instrument = config["instrument"]
        self.base_risk_percent = config["base_risk_percent"]
        self.max_risk_percent = config["max_risk_percent"]
        self.min_risk_percent = config["min_risk_percent"]
        self.max_account_risk = config["max_account_risk"]
        
    def calculate_risk_percentage(self, quality_score, consecutive_losses, pattern, win_rate):
        """Calculate optimal risk percentage based on multiple factors"""
        # Base risk from config
        risk_percent = self.base_risk_percent
        
        # Adjust based on quality score (higher quality = higher risk)
        quality_factor = (quality_score - 7) / 3  # Range -0.33 to 1.0
        risk_percent += quality_factor * 0.5  # Add/subtract up to 0.5%
        
        # Adjust for consecutive losses (reduce risk after losses)
        if consecutive_losses >= self.config["max_consecutive_losses"]:
            risk_percent = self.min_risk_percent
        elif consecutive_losses > 0:
            risk_percent *= (1 - (consecutive_losses * 0.15))  # Reduce by 15% per loss
        
        # Adjust for pattern historical performance
        if win_rate > 0.7:  # High win rate
            risk_percent *= 1.1  # Increase by 10%
        elif win_rate < 0.4 and win_rate > 0:  # Low win rate
            risk_percent *= 0.9  # Decrease by 10%
        
        # Ensure within configured limits
        risk_percent = min(risk_percent, self.max_risk_percent)
        risk_percent = max(risk_percent, self.min_risk_percent)
        
        logger.info(f"Calculated risk percentage: {risk_percent:.2f}% (Quality: {quality_score}, Consecutive losses: {consecutive_losses}, Win rate: {win_rate:.2f})")
        
        return risk_percent
        
    def calculate_position_size(self, account_balance, risk_percent, entry_price, stop_loss, account_currency="USD"):
        """Calculate position size based on account balance, risk, and stop distance"""
        # Calculate risk amount
        risk_amount = account_balance * (risk_percent / 100)
        
        # Calculate distance to stop in price
        stop_distance = abs(entry_price - stop_loss)
        
        # Ensure stop distance is not too small
        min_distance = 0.0001
        if stop_distance < min_distance:
            logger.warning(f"Stop distance ({stop_distance}) too small for {self.instrument}, using minimum {min_distance}")
            stop_distance = min_distance
        
        # Calculate pip value based on instrument (always 0.0001 for EUR/USD)
        pip_size = 0.0001
        
        # Convert stop distance to pips
        stop_pips = stop_distance / pip_size
        
        # Calculate pip value in account currency
        # For EUR/USD with USD account, each pip is worth $10 per standard lot (100,000 units)
        # Simplified calculation for EUR/USD
        pip_value_per_lot = 10 if account_currency == "USD" else 10 / entry_price
        
        # Calculate position size in lots
        lots = risk_amount / (stop_pips * pip_value_per_lot)
        
        # Convert to units (1 lot = 100,000 units)
        units = lots * 100000
        
        # Apply a scaling factor to prevent margin issues
        # This reduces all position sizes by 50% for safety
        units = units * 0.5
        
        # Round down to whole units
        units = int(units)
        
        # Ensure minimum position size
        if units < 1:
            units = 1
        
        # Cap at 10,000 units maximum for additional safety
        if units > 10000:
            logger.warning(f"Position size {units} exceeds maximum of 10,000. Capping at 10,000 units.")
            units = 10000
        
        logger.info(f"Position size calculation: {risk_percent}% risk on {account_balance} = {risk_amount} {account_currency}")
        logger.info(f"Entry: {entry_price}, Stop: {stop_loss}, Distance: {stop_distance} ({stop_pips} pips)")
        logger.info(f"Calculated position size: {units} units (after scaling)")
        
        return units
        
    def calculate_total_account_risk(self, positions, account_balance):
        """Calculate total account risk from all open positions"""
        total_risk = 0
        
        for position in positions:
            # Get position details
            risk_percent = position.get("risk_percent", self.base_risk_percent)
            total_risk += risk_percent
        
        return total_risk
        
    def calculate_stop_loss_distance(self, entry_price, direction, atr_value):
        """Calculate optimal stop loss distance based on ATR"""
        # Use ATR-based stops (1.5 x ATR for normal volatility)
        atr_multiplier = 1.5
        stop_distance = atr_value * atr_multiplier
        
        # Convert to price
        if direction == "BUY":
            stop_price = entry_price - stop_distance
        else:  # SELL
            stop_price = entry_price + stop_distance
            
        # Round to 5 decimal places
        stop_price = round(stop_price, 5)
        
        return stop_price
        
    def calculate_take_profit_levels(self, entry_price, stop_loss, direction, r_multiples=[1.5, 2.5, 3.5]):
        """Calculate multiple take profit levels based on R-multiples"""
        # Calculate R value (risk in price terms)
        r_value = abs(entry_price - stop_loss)
        
        # Calculate take profit levels
        take_profit_levels = []
        
        for r in r_multiples:
            if direction == "BUY":
                tp = entry_price + (r_value * r)
            else:  # SELL
                tp = entry_price - (r_value * r)
            
            # Round to 5 decimal places
            tp = round(tp, 5)
            take_profit_levels.append(tp)
        
        return take_profit_levels
        
    def should_move_to_breakeven(self, entry_price, current_price, direction, atr_value):
        """Determine if stop loss should be moved to breakeven"""
        # Calculate price movement in ATR units
        price_movement = current_price - entry_price if direction == "BUY" else entry_price - current_price
        movement_in_atr = price_movement / atr_value
        
        # Move to breakeven after price has moved in our favor by config["breakeven_move_atr"] ATRs
        return movement_in_atr >= self.config["breakeven_move_atr"]
        
    def calculate_trailing_stop(self, entry_price, current_price, direction, atr_value, current_stop=None):
        """Calculate trailing stop level based on ATR and price movement"""
        # Default ATR multiplier for trailing stops
        atr_multiplier = 2.0
        
        # Calculate new stop level
        if direction == "BUY":
            new_stop = current_price - (atr_value * atr_multiplier)
            # Only trail upward
            if current_stop and new_stop <= current_stop:
                return current_stop
        else:  # SELL
            new_stop = current_price + (atr_value * atr_multiplier)
            # Only trail downward
            if current_stop and new_stop >= current_stop:
                return current_stop
        
        # Round to 5 decimal places
        new_stop = round(new_stop, 5)
        
        return new_stop
        
    def calculate_partial_close_amount(self, position_size, profit_level):
        """Calculate amount to close at a partial profit target"""
        if profit_level == 1:  # First target
            return int(position_size * (self.config["partial_profit_pct"] / 100))
        elif profit_level == 2:  # Second target
            return int(position_size * (self.config["second_profit_pct"] / 100))
        else:
            return 0  # Unknown profit level