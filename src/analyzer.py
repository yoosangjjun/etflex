"""기술적 분석 지표 계산 모듈."""

import logging

import numpy as np
import pandas as pd

from config.settings import (
    BB_PERIOD,
    BB_STD,
    MA_PERIODS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    VOLUME_AVG_PERIOD,
)

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """기술적 분석 지표를 계산하는 클래스."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 날짜 인덱스, close/high/low/open/volume 컬럼이 있는 DataFrame.
                pykrx 한글 컬럼(종가, 시가 등)도 자동 변환.
        """
        self.df = self._normalize_columns(df.copy())

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """pykrx 한글 컬럼명을 영문으로 변환한다."""
        col_map = {
            "시가": "open", "고가": "high", "저가": "low",
            "종가": "close", "거래량": "volume",
        }
        df.rename(columns=col_map, inplace=True)
        return df

    def calc_ma(self) -> pd.DataFrame:
        """이동평균선을 계산한다."""
        for period in MA_PERIODS:
            self.df[f"ma_{period}"] = self.df["close"].rolling(window=period).mean()
        return self.df

    def calc_rsi(self, period: int = RSI_PERIOD) -> pd.Series:
        """RSI(Relative Strength Index)를 계산한다."""
        delta = self.df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.df["rsi"] = 100 - (100 / (1 + rs))
        return self.df["rsi"]

    def calc_macd(
        self,
        fast: int = MACD_FAST,
        slow: int = MACD_SLOW,
        signal: int = MACD_SIGNAL,
    ) -> pd.DataFrame:
        """MACD를 계산한다."""
        ema_fast = self.df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df["close"].ewm(span=slow, adjust=False).mean()
        self.df["macd"] = ema_fast - ema_slow
        self.df["macd_signal"] = self.df["macd"].ewm(span=signal, adjust=False).mean()
        self.df["macd_hist"] = self.df["macd"] - self.df["macd_signal"]
        return self.df

    def calc_bollinger(self, period: int = BB_PERIOD, std: float = BB_STD) -> pd.DataFrame:
        """볼린저밴드를 계산한다."""
        sma = self.df["close"].rolling(window=period).mean()
        std_dev = self.df["close"].rolling(window=period).std()
        self.df["bb_upper"] = sma + std * std_dev
        self.df["bb_middle"] = sma
        self.df["bb_lower"] = sma - std * std_dev
        return self.df

    def calc_volume_ratio(self, period: int = VOLUME_AVG_PERIOD) -> pd.Series:
        """거래량 비율(현재 거래량 / 평균 거래량)을 계산한다."""
        avg_vol = self.df["volume"].rolling(window=period).mean()
        self.df["volume_ratio"] = self.df["volume"] / avg_vol.replace(0, np.nan)
        return self.df["volume_ratio"]

    def analyze(self) -> pd.DataFrame:
        """모든 기술적 지표를 계산하여 반환한다."""
        self.calc_ma()
        self.calc_rsi()
        self.calc_macd()
        self.calc_bollinger()
        self.calc_volume_ratio()
        logger.info("기술적 분석 완료: %d행", len(self.df))
        return self.df
