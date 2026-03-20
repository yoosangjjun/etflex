"""pykrx 기반 ETF 데이터 수집 모듈."""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock

from config.settings import API_CALL_DELAY, DATA_FETCH_DAYS, USE_ALL_ETFS, load_watchlist

logger = logging.getLogger(__name__)


class ETFDataCollector:
    """KRX ETF 시세 데이터를 수집하는 클래스."""

    def __init__(self, watchlist: list[dict] | None = None):
        self.watchlist = watchlist or load_watchlist()

    def get_etf_tickers(self, date: str | None = None) -> list[str]:
        """KRX 전체 ETF 티커 목록을 조회한다.

        Args:
            date: 조회 기준일 (YYYYMMDD). None이면 오늘.

        Returns:
            ETF 티커 코드 리스트.
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        try:
            tickers = stock.get_etf_ticker_list(date)
            logger.info("ETF 티커 %d개 조회 완료 (기준일: %s)", len(tickers), date)
            return tickers
        except Exception:
            logger.exception("ETF 티커 목록 조회 실패")
            return []

    def fetch_ohlcv(
        self, ticker: str, days: int = DATA_FETCH_DAYS
    ) -> pd.DataFrame:
        """특정 ETF의 OHLCV 데이터를 수집한다.

        Args:
            ticker: ETF 티커 코드.
            days: 수집할 과거 일수.

        Returns:
            날짜 인덱스, 시가/고가/저가/종가/거래량 컬럼의 DataFrame.
        """
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        try:
            df = stock.get_etf_ohlcv_by_date(start, end, ticker)
            time.sleep(API_CALL_DELAY)

            if df.empty:
                logger.warning("데이터 없음: %s (%s ~ %s)", ticker, start, end)
                return pd.DataFrame()

            df.index.name = "date"
            logger.info("OHLCV 수집 완료: %s (%d행)", ticker, len(df))
            return df
        except Exception:
            logger.exception("OHLCV 수집 실패: %s", ticker)
            return pd.DataFrame()

    def fetch_trading_value(
        self, ticker: str, days: int = DATA_FETCH_DAYS
    ) -> pd.DataFrame:
        """투자자별 거래량/거래대금을 수집한다.

        Args:
            ticker: ETF 티커 코드.
            days: 수집할 과거 일수.

        Returns:
            투자자별 거래대금 DataFrame.
        """
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        try:
            df = stock.get_etf_trading_volume_and_value(start, end, ticker)
            time.sleep(API_CALL_DELAY)
            logger.info("투자자별 거래대금 수집 완료: %s (%d행)", ticker, len(df))
            return df
        except Exception:
            logger.exception("투자자별 거래대금 수집 실패: %s", ticker)
            return pd.DataFrame()

    def collect_all(self) -> dict[str, pd.DataFrame]:
        """전체 ETF 또는 워치리스트 종목의 OHLCV를 수집한다.

        USE_ALL_ETFS=True이면 KRX 전체 ETF를 대상으로 수집한다.

        Returns:
            {ticker: ohlcv_dataframe} 딕셔너리.
        """
        result: dict[str, pd.DataFrame] = {}

        if USE_ALL_ETFS:
            tickers = self.get_etf_tickers()
            total = len(tickers)
            logger.info("전체 ETF 모드: %d개 종목 수집 시작", total)
            for i, ticker in enumerate(tickers, 1):
                logger.info("수집 중: %s (%d/%d)", ticker, i, total)
                df = self.fetch_ohlcv(ticker)
                if not df.empty:
                    result[ticker] = df
        else:
            total = len(self.watchlist)
            for etf in self.watchlist:
                ticker = etf["ticker"]
                name = etf.get("name", ticker)
                logger.info("수집 시작: %s (%s)", name, ticker)
                df = self.fetch_ohlcv(ticker)
                if not df.empty:
                    result[ticker] = df

        logger.info("전체 수집 완료: %d/%d 종목", len(result), total)
        return result
