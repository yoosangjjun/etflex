"""
ETF Universe Management.

Fetches the complete list of ETFs currently listed on KRX
and caches it locally.
"""

import logging
import time
from datetime import date
from typing import List, Optional

from pykrx import stock

from config.settings import DATE_FORMAT_KRX, MAX_RETRIES, RETRY_DELAY_SEC

logger = logging.getLogger(__name__)


def get_etf_ticker_list(target_date: Optional[date] = None) -> List[str]:
    """
    Fetch all ETF tickers listed on KRX for the given date.

    Args:
        target_date: The date to query. Defaults to today.

    Returns:
        List of 6-digit ticker strings.

    Raises:
        RuntimeError: If the fetch fails after all retries.
    """
    if target_date is None:
        target_date = date.today()

    date_str = target_date.strftime(DATE_FORMAT_KRX)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            tickers = stock.get_etf_ticker_list(date_str)
            if tickers:
                logger.info("Fetched %d ETF tickers for %s", len(tickers), date_str)
                return tickers
            else:
                logger.warning(
                    "Empty ticker list for %s (attempt %d/%d)",
                    date_str,
                    attempt,
                    MAX_RETRIES,
                )
        except Exception as e:
            logger.warning(
                "Failed to fetch ETF tickers (attempt %d/%d): %s",
                attempt,
                MAX_RETRIES,
                e,
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SEC)

    raise RuntimeError(
        f"Failed to fetch ETF ticker list after {MAX_RETRIES} attempts"
    )


def get_etf_name(ticker: str) -> str:
    """Get the Korean name of an ETF by ticker."""
    try:
        name = stock.get_market_ticker_name(ticker)
        return name if name else ticker
    except Exception:
        logger.warning("Could not fetch name for ticker %s", ticker)
        return ticker


def validate_tickers(
    tickers: List[str], target_date: Optional[date] = None
) -> List[str]:
    """
    Filter a list of tickers to only those currently listed on KRX.

    Returns:
        Filtered list containing only valid (currently listed) tickers.
    """
    universe = set(get_etf_ticker_list(target_date))
    valid = [t for t in tickers if t in universe]
    invalid = [t for t in tickers if t not in universe]

    if invalid:
        logger.warning("Tickers not found in KRX universe: %s", invalid)

    return valid
