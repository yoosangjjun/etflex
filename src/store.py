"""SQLite 데이터 저장소 모듈."""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from config.settings import DB_PATH

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    name TEXT,
    date TEXT NOT NULL,
    signal_type TEXT NOT NULL,  -- 'BUY' or 'SELL'
    composite_score REAL,
    ma_score REAL,
    rsi_score REAL,
    macd_score REAL,
    bb_score REAL,
    volume_score REAL,
    close_price REAL,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS analysis_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    tickers_analyzed INTEGER,
    signals_generated INTEGER,
    status TEXT,
    message TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);
"""


class DataStore:
    """SQLite 기반 데이터 저장소."""

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        """DB 스키마를 초기화한다."""
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.info("DB 초기화 완료: %s", self.db_path)

    def save_ohlcv(self, ticker: str, df: pd.DataFrame) -> int:
        """OHLCV 데이터를 저장한다.

        Args:
            ticker: ETF 티커 코드.
            df: pykrx에서 수집한 OHLCV DataFrame.

        Returns:
            저장된 행 수.
        """
        if df.empty:
            return 0

        records = []
        for date, row in df.iterrows():
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            records.append((
                ticker, date_str,
                row.get("시가", row.get("open")),
                row.get("고가", row.get("high")),
                row.get("저가", row.get("low")),
                row.get("종가", row.get("close")),
                row.get("거래량", row.get("volume")),
            ))

        with self._get_conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO ohlcv (ticker, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                records,
            )
        logger.info("OHLCV 저장: %s (%d행)", ticker, len(records))
        return len(records)

    def save_signal(self, signal: dict) -> None:
        """시그널을 저장한다.

        Args:
            signal: 시그널 정보 딕셔너리.
        """
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO signals "
                "(ticker, name, date, signal_type, composite_score, "
                "ma_score, rsi_score, macd_score, bb_score, volume_score, close_price) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    signal["ticker"],
                    signal.get("name"),
                    signal["date"],
                    signal["signal_type"],
                    signal["composite_score"],
                    signal.get("ma_score", 0),
                    signal.get("rsi_score", 0),
                    signal.get("macd_score", 0),
                    signal.get("bb_score", 0),
                    signal.get("volume_score", 0),
                    signal.get("close_price"),
                ),
            )
        logger.info("시그널 저장: %s %s (%.2f)", signal["ticker"], signal["signal_type"], signal["composite_score"])

    def save_analysis_log(self, run_date: str, tickers: int, signals: int, status: str, message: str = "") -> None:
        """분석 실행 로그를 저장한다."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO analysis_log (run_date, tickers_analyzed, signals_generated, status, message) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_date, tickers, signals, status, message),
            )

    def get_ohlcv(self, ticker: str, limit: int = 200) -> pd.DataFrame:
        """저장된 OHLCV 데이터를 조회한다."""
        with self._get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM ohlcv "
                "WHERE ticker = ? ORDER BY date DESC LIMIT ?",
                conn,
                params=(ticker, limit),
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        return df

    def get_recent_signals(self, days: int = 7) -> pd.DataFrame:
        """최근 시그널을 조회한다."""
        with self._get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM signals WHERE date >= date('now', ? || ' days') ORDER BY date DESC",
                conn,
                params=(f"-{days}",),
            )
        return df
