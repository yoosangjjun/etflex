"""collector.py 테스트."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.collector import ETFDataCollector


@pytest.fixture
def collector():
    watchlist = [
        {"ticker": "069500", "name": "KODEX 200"},
        {"ticker": "229200", "name": "KODEX 코스닥150"},
    ]
    return ETFDataCollector(watchlist=watchlist)


@patch("src.collector.stock.get_etf_ticker_list")
def test_get_etf_tickers(mock_tickers, collector):
    """ETF 티커 목록 조회 테스트."""
    mock_tickers.return_value = ["069500", "229200", "114800"]
    result = collector.get_etf_tickers("20260320")
    assert len(result) == 3
    mock_tickers.assert_called_once_with("20260320")


@patch("src.collector.stock.get_etf_ticker_list")
def test_get_etf_tickers_failure(mock_tickers, collector):
    """API 실패 시 빈 리스트 반환."""
    mock_tickers.side_effect = Exception("API Error")
    result = collector.get_etf_tickers()
    assert result == []


@patch("src.collector.time.sleep")
@patch("src.collector.stock.get_etf_ohlcv_by_date")
def test_fetch_ohlcv(mock_ohlcv, mock_sleep, collector):
    """OHLCV 수집 테스트."""
    mock_df = pd.DataFrame(
        {"시가": [100], "고가": [110], "저가": [90], "종가": [105], "거래량": [1000]},
        index=pd.to_datetime(["2026-03-20"]),
    )
    mock_ohlcv.return_value = mock_df

    result = collector.fetch_ohlcv("069500", days=30)
    assert not result.empty
    assert len(result) == 1
    mock_sleep.assert_called_once()


@patch("src.collector.time.sleep")
@patch("src.collector.stock.get_etf_ohlcv_by_date")
def test_fetch_ohlcv_empty(mock_ohlcv, mock_sleep, collector):
    """빈 데이터 반환 테스트."""
    mock_ohlcv.return_value = pd.DataFrame()
    result = collector.fetch_ohlcv("999999")
    assert result.empty


@patch("src.collector.USE_ALL_ETFS", False)
@patch("src.collector.time.sleep")
@patch("src.collector.stock.get_etf_ohlcv_by_date")
def test_collect_all(mock_ohlcv, mock_sleep, collector):
    """전체 수집 테스트 (watchlist 모드)."""
    mock_df = pd.DataFrame(
        {"시가": [100], "고가": [110], "저가": [90], "종가": [105], "거래량": [1000]},
        index=pd.to_datetime(["2026-03-20"]),
    )
    mock_ohlcv.return_value = mock_df

    result = collector.collect_all()
    assert len(result) == 2
    assert "069500" in result
