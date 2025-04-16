"""
Market Analysis Module
Handles market data collection and technical analysis
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger("EURUSDTrader")

class MarketAnalyzer:
    """Market analysis and technical indicators for EUR/USD"""
    
    def __init__(self, config):
        """Initialize the market analyzer"""
        self.config = config
        self.instrument = config["instrument"]
        
    def get_market_data(self, oanda_client):
        """Collect market data for EUR/USD"""
        market_data = {}
        
        # Get candles for each timeframe
        for tf_name, tf_config in self.config["timeframes"].items():
            candles = oanda_client.get_candles(
                instrument=self.instrument,
                granularity=tf_config["granularity"],
                count=tf_config["count"]
            )
            
            # Format candles into DataFrame
            if candles:
                market_data[tf_name] = self._format_candles(candles)
        
        # Get current price
        price = oanda_client.get_price(self.instrument)
        market_data["current"] = price
        
        return market_data
    
    def _format_candles(self, candles):
        """Format candle data into pandas DataFrame"""
        formatted_candles = []
        
        for candle in candles:
            # Handle OANDA format
            if "mid" in candle:
                formatted_candles.append({
                    "time": candle.get("time"),
                    "open": float(candle["mid"]["o"]),
                    "high": float(candle["mid"]["h"]),
                    "low": float(candle["mid"]["l"]),
                    "close": float(candle["mid"]["c"]),
                    "volume": int(candle.get("volume", 0))
                })
            else:
                # Already in our format
                formatted_candles.append(candle)
        
        return formatted_candles
    
    def calculate_market_regime(self, market_data):
        """Determine the current market regime using advanced techniques"""
        try:
            # Get daily candles
            daily_candles = market_data.get("d1", [])
            
            if not daily_candles or len(daily_candles) < 20:
                return "unknown"
            
            # Convert to DataFrame
            df = pd.DataFrame(daily_candles)
            
            # Calculate ADX (Average Directional Index) for trend strength
            df = self._calculate_adx(df, 14)
            
            # Calculate volatility using ATR
            df = self._calculate_atr(df, 14)
            
            # Calculate directional movement
            df['direction'] = np.where(df['close'] > df['close'].shift(1), 1, 
                               np.where(df['close'] < df['close'].shift(1), -1, 0))
            
            # Calculate Bollinger Bands for range identification
            df = self._calculate_bollinger_bands(df, 20, 2)
            
            # Get most recent values
            recent_adx = df['adx'].iloc[-1]
            recent_atr = df['atr'].iloc[-1]
            avg_atr = df['atr'].rolling(window=20).mean().iloc[-1]
            band_width = (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) / df['close'].iloc[-1] * 100
            
            # Calculate directional consistency (last 10 days)
            direction_consistency = abs(df['direction'].iloc[-10:].sum()) / 10
            
            # Get EMAs for trend direction
            df = self._calculate_ema(df, 20)
            df = self._calculate_ema(df, 50)
            ema20 = df['ema_20'].iloc[-1]
            ema50 = df['ema_50'].iloc[-1]
            
            # Determine regime
            # Strong trend
            if recent_adx > 30 and direction_consistency > 0.7:
                if ema20 > ema50 and df['close'].iloc[-1] > ema20:
                    return "strong_uptrend"
                elif ema20 < ema50 and df['close'].iloc[-1] < ema20:
                    return "strong_downtrend"
            
            # Trend
            elif recent_adx > 20:
                if ema20 > ema50:
                    return "uptrend"
                else:
                    return "downtrend"
            
            # Volatile
            elif recent_atr > avg_atr * 1.5:
                return "volatile"
            
            # Tight range
            elif band_width < 2.0:
                return "tight_range"
            
            # Normal range
            elif band_width < 4.0:
                return "ranging"
            
            # Unclear
            else:
                return "unclear"
                
        except Exception as e:
            logger.error(f"Error calculating market regime: {e}")
            return "unknown"
    
    def calculate_indicators(self, market_data):
        """Calculate technical indicators for all timeframes"""
        indicators = {}
        
        for timeframe, candles in market_data.items():
            # Skip current price
            if timeframe == "current":
                continue
                
            # Convert to DataFrame
            if not candles:
                continue
                
            df = pd.DataFrame(candles)
            
            # Calculate common indicators
            df = self._calculate_sma(df, self.config["indicators"]["sma"])
            df = self._calculate_ema(df, self.config["indicators"]["ema"])
            df = self._calculate_rsi(df, self.config["indicators"]["rsi"]["period"])
            df = self._calculate_macd(df, 
                                      self.config["indicators"]["macd"]["fast"],
                                      self.config["indicators"]["macd"]["slow"],
                                      self.config["indicators"]["macd"]["signal"])
            df = self._calculate_bollinger_bands(df, 
                                                self.config["indicators"]["bbands"]["period"],
                                                self.config["indicators"]["bbands"]["std_dev"])
            df = self._calculate_atr(df, self.config["indicators"]["atr"]["period"])
            
            # Add additional EUR/USD specific indicators
            df = self._calculate_fibonacci_levels(df)
            df = self._calculate_support_resistance(df)
            df = self._detect_divergence(df)
            
            # Store in indicators dictionary
            indicators[timeframe] = df.to_dict(orient='records')
        
        return indicators
    
    def _calculate_sma(self, df, periods):
        """Calculate Simple Moving Averages"""
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    def _calculate_ema(self, df, periods):
        """Calculate Exponential Moving Averages"""
        if isinstance(periods, list):
            for period in periods:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        else:
            df[f'ema_{periods}'] = df['close'].ewm(span=periods, adjust=False).mean()
        return df
    
    def _calculate_rsi(self, df, period):
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS
        rs = gain / loss
        
        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    def _calculate_macd(self, df, fast_period, slow_period, signal_period):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        # Calculate EMA fast and slow
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df['macd'] = ema_fast - ema_slow
        
        # Calculate signal line
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _calculate_bollinger_bands(self, df, period, std_dev):
        """Calculate Bollinger Bands"""
        # Calculate middle band (SMA)
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        df['bb_std'] = df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        
        # Calculate band width (normalized)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate percentage B (position within bands)
        df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _calculate_atr(self, df, period):
        """Calculate Average True Range"""
        # Calculate true range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['true_range'].rolling(window=period).mean()
        
        # Drop temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3'], axis=1)
        
        return df
    
    def _calculate_adx(self, df, period):
        """Calculate Average Directional Index for trend strength"""
        # Calculate +DM and -DM
        df['up_move'] = df['high'].diff()
        df['down_move'] = df['low'].shift().diff(-1).abs()
        
        df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Calculate TR (True Range)
        if 'true_range' not in df.columns:
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate smoothed +DM, -DM, and TR
        df['+dm_smooth'] = df['+dm'].rolling(window=period).sum()
        df['-dm_smooth'] = df['-dm'].rolling(window=period).sum()
        df['tr_smooth'] = df['true_range'].rolling(window=period).sum()
        
        # Calculate +DI and -DI
        df['+di'] = 100 * df['+dm_smooth'] / df['tr_smooth']
        df['-di'] = 100 * df['-dm_smooth'] / df['tr_smooth']
        
        # Calculate DX (Directional Index)
        df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
        
        # Calculate ADX (Average Directional Index)
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        # Clean up temporary columns
        cols_to_drop = ['up_move', 'down_move', '+dm', '-dm', 'tr1', 'tr2', 'tr3', 
                        '+dm_smooth', '-dm_smooth', 'tr_smooth', 'dx']
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        return df
    
    def _calculate_fibonacci_levels(self, df):
        """Calculate Fibonacci retracement and extension levels"""
        # Find recent swing high and low for the last 20 candles
        window = min(20, len(df))
        recent_df = df.iloc[-window:]
        
        swing_high = recent_df['high'].max()
        swing_high_idx = recent_df['high'].idxmax()
        
        swing_low = recent_df['low'].min()
        swing_low_idx = recent_df['low'].idxmin()
        
        # Determine if we're in an uptrend or downtrend based on the order of swing points
        is_uptrend = swing_low_idx < swing_high_idx
        
        # Calculate Fibonacci levels
        if is_uptrend:
            diff = swing_high - swing_low
            df['fib_0'] = swing_low
            df['fib_23.6'] = swing_low + diff * 0.236
            df['fib_38.2'] = swing_low + diff * 0.382
            df['fib_50'] = swing_low + diff * 0.5
            df['fib_61.8'] = swing_low + diff * 0.618
            df['fib_78.6'] = swing_low + diff * 0.786
            df['fib_100'] = swing_high
            df['fib_161.8'] = swing_low + diff * 1.618
        else:
            diff = swing_high - swing_low
            df['fib_0'] = swing_high
            df['fib_23.6'] = swing_high - diff * 0.236
            df['fib_38.2'] = swing_high - diff * 0.382
            df['fib_50'] = swing_high - diff * 0.5
            df['fib_61.8'] = swing_high - diff * 0.618
            df['fib_78.6'] = swing_high - diff * 0.786
            df['fib_100'] = swing_low
            df['fib_161.8'] = swing_high - diff * 1.618
        
        # Store trend direction for reference
        df['fib_trend'] = 'up' if is_uptrend else 'down'
        
        return df
    
    def _calculate_support_resistance(self, df):
        """Identify key support and resistance levels"""
        # Use pivot points method
        # Calculate previous day's high, low, close
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        
        # Calculate pivot point
        df['pivot'] = (df['prev_high'] + df['prev_low'] + df['prev_close']) / 3
        
        # Calculate support and resistance levels
        df['r1'] = (2 * df['pivot']) - df['prev_low']
        df['s1'] = (2 * df['pivot']) - df['prev_high']
        df['r2'] = df['pivot'] + (df['prev_high'] - df['prev_low'])
        df['s2'] = df['pivot'] - (df['prev_high'] - df['prev_low'])
        df['r3'] = df['pivot'] + 2 * (df['prev_high'] - df['prev_low'])
        df['s3'] = df['pivot'] - 2 * (df['prev_high'] - df['prev_low'])
        
        # Clean up temporary columns
        df = df.drop(['prev_high', 'prev_low', 'prev_close'], axis=1)
        
        return df
    
    def _detect_divergence(self, df):
        """Detect RSI divergence"""
        # Ensure RSI is calculated
        if 'rsi' not in df.columns:
            df = self._calculate_rsi(df, 14)
        
        # Find local extremes in price (high/lows)
        df['price_higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['price_lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        # Find local extremes in RSI
        df['rsi_higher_high'] = (df['rsi'] > df['rsi'].shift(1)) & (df['rsi'] > df['rsi'].shift(-1))
        df['rsi_lower_low'] = (df['rsi'] < df['rsi'].shift(1)) & (df['rsi'] < df['rsi'].shift(-1))
        
        # Look for divergence patterns
        # Bearish divergence: Price makes higher high but RSI makes lower high
        for i in range(5, len(df)):
            if df['price_higher_high'].iloc[i]:
                # Find previous price high within last 10 bars
                for j in range(i-10, i):
                    if j >= 0 and df['price_higher_high'].iloc[j]:
                        # Check if current price is higher than previous
                        if df['high'].iloc[i] > df['high'].iloc[j]:
                            # Check if current RSI is lower than previous
                            if df['rsi'].iloc[i] < df['rsi'].iloc[j]:
                                df.loc[df.index[i], 'bearish_divergence'] = True
                                break
        
        # Bullish divergence: Price makes lower low but RSI makes higher low
        for i in range(5, len(df)):
            if df['price_lower_low'].iloc[i]:
                # Find previous price low within last 10 bars
                for j in range(i-10, i):
                    if j >= 0 and df['price_lower_low'].iloc[j]:
                        # Check if current price is lower than previous
                        if df['low'].iloc[i] < df['low'].iloc[j]:
                            # Check if current RSI is higher than previous
                            if df['rsi'].iloc[i] > df['rsi'].iloc[j]:
                                df.loc[df.index[i], 'bullish_divergence'] = True
                                break
        
        # Cleanup temporary columns
        df = df.drop(['price_higher_high', 'price_lower_low', 'rsi_higher_high', 'rsi_lower_low'], axis=1)
        
        return df