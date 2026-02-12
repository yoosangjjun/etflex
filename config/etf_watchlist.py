"""
ETF watchlist organized by category.

Each entry: (ticker: str, name: str)
Tickers are KRX 6-digit codes.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ETFCategory:
    name_kr: str
    name_en: str
    etfs: List[Tuple[str, str]]


WATCHLIST: Dict[str, ETFCategory] = {
    "market": ETFCategory(
        name_kr="시장 추종",
        name_en="Market Index",
        etfs=[
            ("069500", "KODEX 200"),
            ("229200", "KODEX 코스닥150"),
            ("278530", "KODEX 200TR"),
            ("102110", "TIGER 200"),
            ("228790", "TIGER 코스닥150"),
        ],
    ),
    "sector": ETFCategory(
        name_kr="섹터",
        name_en="Sector",
        etfs=[
            ("091160", "KODEX 반도체"),
            ("091170", "KODEX 은행"),
            ("266370", "KODEX 2차전지산업"),
            ("244580", "KODEX 바이오"),
            ("140700", "KODEX 보험"),
            ("117680", "KODEX 철강"),
        ],
    ),
    "theme": ETFCategory(
        name_kr="테마",
        name_en="Theme",
        etfs=[
            ("418660", "KODEX AI반도체핵심장비"),
            ("363580", "KODEX 전기차배터리"),
            ("385590", "KODEX K-로봇액티브"),
        ],
    ),
    "asset_class": ETFCategory(
        name_kr="자산군",
        name_en="Asset Class",
        etfs=[
            ("148070", "KODEX 국고채3년"),
            ("152380", "KODEX 국고채10년"),
            ("132030", "KODEX 골드선물(H)"),
            ("271060", "KODEX 3대농산물선물(H)"),
        ],
    ),
    "overseas": ETFCategory(
        name_kr="해외",
        name_en="Overseas",
        etfs=[
            ("379810", "KODEX 미국S&P500TR"),
            ("368590", "KODEX 나스닥100TR"),
            ("251350", "KODEX 선진국MSCI World"),
            ("117460", "KODEX 일본TOPIX100"),
        ],
    ),
    "leverage_inverse": ETFCategory(
        name_kr="레버리지/인버스",
        name_en="Leverage/Inverse",
        etfs=[
            ("122630", "KODEX 레버리지"),
            ("114800", "KODEX 인버스"),
            ("252670", "KODEX 200선물인버스2X"),
            ("233740", "KODEX 코스닥150레버리지"),
        ],
    ),
}


def get_all_watchlist_tickers() -> List[str]:
    """Return a flat list of all ticker codes in the watchlist."""
    tickers = []
    for category in WATCHLIST.values():
        for ticker, _name in category.etfs:
            tickers.append(ticker)
    return tickers


def get_ticker_name_map() -> Dict[str, str]:
    """Return a dict mapping ticker -> ETF name for all watchlist ETFs."""
    mapping = {}
    for category in WATCHLIST.values():
        for ticker, name in category.etfs:
            mapping[ticker] = name
    return mapping


def get_tickers_by_category(category_key: str) -> List[Tuple[str, str]]:
    """Return list of (ticker, name) for a given category key."""
    if category_key not in WATCHLIST:
        raise ValueError(
            f"Unknown category: {category_key}. "
            f"Available: {list(WATCHLIST.keys())}"
        )
    return WATCHLIST[category_key].etfs


def get_category_for_ticker(ticker: str) -> str:
    """Return the category key for a given ticker, or 'unknown' if not found."""
    for key, category in WATCHLIST.items():
        for t, _name in category.etfs:
            if t == ticker:
                return key
    return "unknown"


def get_tickers_grouped_by_category() -> Dict[str, List[str]]:
    """Return dict mapping category_key -> list of ticker codes."""
    grouped: Dict[str, List[str]] = {}
    for key, category in WATCHLIST.items():
        grouped[key] = [ticker for ticker, _name in category.etfs]
    return grouped
