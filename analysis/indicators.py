"""
Technical indicator calculations for ETF analysis.

Computes: MA (5/20/60/120), RSI (14), MACD (12/26/9),
Bollinger Bands (20, 2σ), Volume MA (20).
"""

import logging

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA5, MA20, MA60, MA120 columns."""
    for period in [5, 20, 60, 120]:
        df[f"ma{period}"] = ta.sma(df["close"], length=period)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI column (default 14-period)."""
    df["rsi"] = ta.rsi(df["close"], length=period)
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Add MACD, MACD signal, MACD histogram columns."""
    macd_result = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if macd_result is not None and not macd_result.empty:
        df["macd"] = macd_result.iloc[:, 0]
        df["macd_signal"] = macd_result.iloc[:, 1]
        df["macd_hist"] = macd_result.iloc[:, 2]
    return df


def add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std: float = 2.0
) -> pd.DataFrame:
    """Add Bollinger Bands (upper, middle, lower) columns."""
    bb_result = ta.bbands(df["close"], length=period, std=std)
    if bb_result is not None and not bb_result.empty:
        df["bb_lower"] = bb_result.iloc[:, 0]
        df["bb_middle"] = bb_result.iloc[:, 1]
        df["bb_upper"] = bb_result.iloc[:, 2]
        df["bb_bandwidth"] = bb_result.iloc[:, 3]
        df["bb_percent"] = bb_result.iloc[:, 4]
    return df


def add_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add volume moving average and volume ratio columns."""
    df["volume_ma20"] = ta.sma(df["volume"], length=period)
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]
    return df


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators on an OHLCV DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, volume.
            Must have a DatetimeIndex.

    Returns:
        DataFrame with all indicator columns added.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to calculate_all_indicators")
        return df

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_ma(df)

    logger.debug(
        "Calculated indicators: %d rows, %d columns",
        len(df),
        len(df.columns),
    )
    return df
