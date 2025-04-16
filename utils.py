"""
Utilities Module
Provides utility functions and OANDA API connector
"""

import os
import logging
import json
import requests
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger("EURUSDTrader")

class OandaClient:
    """OANDA API client for forex trading"""
    
    def __init__(self):
        """Initialize OANDA API client"""
        # Determine if using practice or live account
        self.practice = os.getenv("OANDA_PRACTICE", "True").lower() in ["true", "1", "yes"]
        
        # Set API base URL
        if self.practice:
            self.base_url = "https://api-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
        
        # Set API token and account ID
        self.api_token = os.getenv("OANDA_API_TOKEN")
        if not self.api_token:
            raise ValueError("OANDA_API_TOKEN not set in environment variables")
            
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        if not self.account_id:
            raise ValueError("OANDA_ACCOUNT_ID not set in environment variables")
            
        # Set default headers
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to OANDA API"""
        try:
            account = self.get_account()
            if account:
                currency = account.get("currency", "Unknown")
                balance = account.get("balance", "Unknown")
                logger.info(f"Connected to OANDA account: {self.account_id}")
                logger.info(f"Account Balance: {balance} {currency}")
            else:
                logger.error("Failed to connect to OANDA API")
        except Exception as e:
            logger.error(f"OANDA connection error: {e}")
            raise
    
    def _make_request(self, method: str, endpoint: str, params=None, data=None) -> Dict:
        """Make request to OANDA API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            # Set request timeout
            timeout = int(os.getenv("OANDA_REQUEST_TIMEOUT", "60"))
            
            # Make request
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data,
                timeout=timeout
            )
            
            # Log request (debug level)
            logger.debug(f"OANDA API request: {method} {endpoint}")
            
            # Raise exception for HTTP errors
            response.raise_for_status()
            
            # Return JSON response
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            # Log error details
            try:
                error_details = response.json()
            except:
                error_details = {"text": response.text}
                
            logger.error(f"OANDA API HTTP error: {e}")
            logger.error(f"Error details: {error_details}")
            raise
            
        except Exception as e:
            logger.error(f"OANDA API request error: {e}")
            raise
    
    def get_account(self) -> Dict:
        """Get account details"""
        try:
            response = self._make_request("GET", f"/v3/accounts/{self.account_id}")
            return response.get("account", {})
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {}
    
    def get_open_positions(self) -> List[Dict]:
        """Get open positions"""
        try:
            response = self._make_request("GET", f"/v3/accounts/{self.account_id}/openPositions")
            positions = response.get("positions", [])
            
            # Convert to standardized format
            result = []
            for position in positions:
                instrument = position.get("instrument")
                
                # Get long and short units
                long_units = int(float(position.get("long", {}).get("units", 0))) if position.get("long") else 0
                short_units = int(float(position.get("short", {}).get("units", 0))) if position.get("short") else 0
                
                # Determine direction and units
                if long_units > 0:
                    direction = "BUY"
                    units = long_units
                    details = position.get("long", {})
                else:
                    direction = "SELL"
                    units = abs(short_units)
                    details = position.get("short", {})
                
                # Calculate profit/loss
                profit = float(details.get("unrealizedPL", 0))
                
                # Create position entry
                result.append({
                    "dealId": f"position_{instrument}_{direction}",
                    "epic": instrument,
                    "direction": direction,
                    "size": units,
                    "level": float(details.get("averagePrice", 0)),
                    "profit": profit,
                    "stop_level": float(details.get("stopLossOrder", {}).get("price", 0)) if details.get("stopLossOrder") else 0
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_candles(self, instrument: str, granularity: str, count: int) -> List[Dict]:
        """Get candle data for an instrument"""
        try:
            # Build parameters
            params = {
                "price": "M",  # Midpoint candles
                "granularity": granularity,
                "count": count
            }
            
            # Make request
            response = self._make_request(
                "GET",
                f"/v3/instruments/{instrument}/candles",
                params=params
            )
            
            # Return candles
            return response.get("candles", [])
            
        except Exception as e:
            logger.error(f"Error getting candles for {instrument}: {e}")
            return []
    
    def get_price(self, instrument: str) -> Dict:
        """Get current price for an instrument"""
        try:
            # Build parameters
            params = {"instruments": instrument}
            
            # Make request
            response = self._make_request(
                "GET",
                f"/v3/accounts/{self.account_id}/pricing",
                params=params
            )
            
            # Return the first price
            prices = response.get("prices", [])
            if prices:
                # Format timestamp
                if "time" in prices[0]:
                    prices[0]["timestamp"] = prices[0].pop("time")
                
                # Standardize offer/ask field name
                if "ask" in prices[0] and "offer" not in prices[0]:
                    prices[0]["offer"] = prices[0]["ask"]
                
                return prices[0]
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting price for {instrument}: {e}")
            return {}
    
    def execute_trade(self, instrument: str, units: int, stop_loss=None, take_profit=None) -> Tuple[bool, Dict]:
        """Execute a trade"""
        try:
            logger.info(f"Executing trade: {instrument} | Units: {units} | SL: {stop_loss} | TP: {take_profit}")
            
            # Ensure units is an integer
            units = int(units)
            
            # Build order data
            order_data = {
                "order": {
                    "instrument": instrument,
                    "units": str(units),
                    "type": "MARKET",
                    "positionFill": "DEFAULT",
                    "timeInForce": "FOK"  # Fill Or Kill
                }
            }
            
            # Add stop loss if provided
            if stop_loss:
                order_data["order"]["stopLossOnFill"] = {
                    "price": str(stop_loss),
                    "timeInForce": "GTC"  # Good Till Cancelled
                }
            
            # Add take profit if provided
            if take_profit:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": str(take_profit),
                    "timeInForce": "GTC"  # Good Till Cancelled
                }
            
            # Make request
            response = self._make_request(
                "POST",
                f"/v3/accounts/{self.account_id}/orders",
                data=order_data
            )
            
            # Check if order was created and filled
            if "orderFillTransaction" in response:
                # Get fill details
                fill = response["orderFillTransaction"]
                
                # Create result
                direction = "BUY" if int(float(fill.get("units", 0))) > 0 else "SELL"
                
                result = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "instrument": instrument,
                    "direction": direction,
                    "size": abs(int(float(fill.get("units", 0)))),
                    "entry_price": float(fill.get("price", 0)),
                    "action_type": "OPEN",
                    "outcome": "EXECUTED",
                    "deal_id": fill.get("id"),
                }
                
                return True, result
            else:
                # Order was not filled
                logger.warning(f"Order not filled: {response}")
                return False, {"outcome": "REJECTED", "reason": response.get("errorMessage", "Unknown error")}
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False, {"outcome": "ERROR", "reason": str(e)}
    
    def close_position(self, instrument: str) -> Tuple[bool, Dict]:
        """Close a position"""
        try:
            logger.info(f"Closing position: {instrument}")
            
            # Build close data
            close_data = {
                "longUnits": "ALL",
                "shortUnits": "ALL"
            }
            
            # Make request
            response = self._make_request(
                "PUT",
                f"/v3/accounts/{self.account_id}/positions/{instrument}/close",
                data=close_data
            )
            
            # Check if position was closed
            long_closed = "longOrderFillTransaction" in response
            short_closed = "shortOrderFillTransaction" in response
            
            if long_closed or short_closed:
                # Get close details
                close_tx = response.get("longOrderFillTransaction" if long_closed else "shortOrderFillTransaction", {})
                
                # Create result
                result = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "instrument": instrument,
                    "direction": "CLOSE",
                    "action_type": "CLOSE",
                    "size": abs(int(float(close_tx.get("units", 0)))),
                    "close_price": float(close_tx.get("price", 0)),
                    "profit": float(close_tx.get("pl", 0)),
                    "outcome": "CLOSED"
                }
                
                return True, result
            else:
                # Position was not closed
                error_msg = "No position to close"
                if "errorMessage" in response:
                    error_msg = response["errorMessage"]
                
                logger.warning(f"Position not closed: {error_msg}")
                return False, {"outcome": "REJECTED", "reason": error_msg}
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False, {"outcome": "ERROR", "reason": str(e)}
    
    def close_partial(self, instrument: str, units: int) -> Tuple[bool, Dict]:
        """Close a partial position"""
        try:
            logger.info(f"Closing partial position: {instrument} | Units: {units}")
            
            # Build close data
            if units > 0:
                close_data = {"longUnits": str(units)}
            else:
                close_data = {"shortUnits": str(abs(units))}
            
            # Make request
            response = self._make_request(
                "PUT",
                f"/v3/accounts/{self.account_id}/positions/{instrument}/close",
                data=close_data
            )
            
            # Check if position was closed
            long_closed = "longOrderFillTransaction" in response
            short_closed = "shortOrderFillTransaction" in response
            
            if long_closed or short_closed:
                # Get close details
                close_tx = response.get("longOrderFillTransaction" if long_closed else "shortOrderFillTransaction", {})
                
                # Create result
                result = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "instrument": instrument,
                    "direction": "PARTIAL_CLOSE",
                    "action_type": "PARTIAL_CLOSE",
                    "size": abs(int(float(close_tx.get("units", 0)))),
                    "close_price": float(close_tx.get("price", 0)),
                    "profit": float(close_tx.get("pl", 0)),
                    "outcome": "CLOSED_PARTIAL"
                }
                
                return True, result
            else:
                # Position was not closed
                error_msg = "No position to close"
                if "errorMessage" in response:
                    error_msg = response["errorMessage"]
                
                logger.warning(f"Partial position not closed: {error_msg}")
                return False, {"outcome": "REJECTED", "reason": error_msg}
                
        except Exception as e:
            logger.error(f"Error closing partial position: {e}")
            return False, {"outcome": "ERROR", "reason": str(e)}
    
    def update_stop_loss(self, instrument: str, stop_loss: float) -> Tuple[bool, Dict]:
        """Update stop loss for a position"""
        try:
            logger.info(f"Updating stop loss for {instrument} to {stop_loss}")
            
            # Get open trades for the instrument
            params = {
                "instrument": instrument,
                "state": "OPEN"
            }
            
            response = self._make_request(
                "GET",
                f"/v3/accounts/{self.account_id}/trades",
                params=params
            )
            
            trades = response.get("trades", [])
            
            if not trades:
                logger.warning(f"No open trades found for {instrument}")
                return False, {"outcome": "REJECTED", "reason": "No open trades found"}
            
            # Update stop loss for each trade
            success = False
            for trade in trades:
                trade_id = trade.get("id")
                
                # Build update data
                update_data = {
                    "stopLoss": {
                        "price": str(stop_loss),
                        "timeInForce": "GTC"
                    }
                }
                
                # Make request
                update_response = self._make_request(
                    "PUT",
                    f"/v3/accounts/{self.account_id}/trades/{trade_id}/orders",
                    data=update_data
                )
                
                # Check if stop loss was updated
                if "stopLossOrderTransaction" in update_response:
                    success = True
            
            if success:
                # Create result
                result = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "instrument": instrument,
                    "action_type": "UPDATE_STOP",
                    "new_level": stop_loss,
                    "outcome": "UPDATED"
                }
                
                return True, result
            else:
                # Stop loss was not updated
                logger.warning(f"Stop loss not updated for {instrument}")
                return False, {"outcome": "REJECTED", "reason": "Stop loss not updated"}
                
        except Exception as e:
            logger.error(f"Error updating stop loss: {e}")
            return False, {"outcome": "ERROR", "reason": str(e)}


def save_chart(instrument, timeframe, indicators, entry_price, stop_loss, direction, pattern=None):
    """Save a chart image with trade details"""
    try:
        # Create directory for charts if it doesn't exist
        os.makedirs("charts", exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"charts/{instrument}_{timeframe}_{timestamp}.png"
        
        # Convert indicators to DataFrame
        df = pd.DataFrame(indicators)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot candlesticks
        times = pd.to_datetime(df["time"])
        
        # Plot OHLC data
        ax.plot(times, df["close"], label="Close Price")
        
        # Plot key moving averages
        if "ema_21" in df.columns:
            ax.plot(times, df["ema_21"], label="EMA 21", color="blue", linestyle="--")
        if "sma_50" in df.columns:
            ax.plot(times, df["sma_50"], label="SMA 50", color="orange", linestyle="--")
        if "sma_200" in df.columns:
            ax.plot(times, df["sma_200"], label="SMA 200", color="red", linestyle="--")
        
        # Plot entry and stop
        if entry_price:
            plt.axhline(y=entry_price, color="g" if direction == "BUY" else "r", linestyle="-", 
                        label=f"Entry: {entry_price}")
        if stop_loss:
            plt.axhline(y=stop_loss, color="r", linestyle="--", label=f"Stop: {stop_loss}")
        
        # Set title and labels
        title = f"{instrument} {timeframe} - {direction} "
        if pattern:
            title += f"({pattern})"
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Price")
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.xticks(rotation=45)
        
        # Add legend
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Chart saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving chart: {e}")