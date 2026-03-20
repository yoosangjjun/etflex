"""복합 시그널 생성 모듈."""

import logging
from dataclasses import dataclass

import pandas as pd

from config.settings import (
    BUY_THRESHOLD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    SELL_THRESHOLD,
    SIGNAL_WEIGHTS,
    VOLUME_SURGE_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """매매 시그널 결과."""
    ticker: str
    name: str
    date: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    composite_score: float
    ma_score: float
    rsi_score: float
    macd_score: float
    bb_score: float
    volume_score: float
    close_price: float
    details: list[str]

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "date": self.date,
            "signal_type": self.signal_type,
            "composite_score": self.composite_score,
            "ma_score": self.ma_score,
            "rsi_score": self.rsi_score,
            "macd_score": self.macd_score,
            "bb_score": self.bb_score,
            "volume_score": self.volume_score,
            "close_price": self.close_price,
        }


class SignalGenerator:
    """복합 시그널 기반 매수/매도 추천 생성기."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or SIGNAL_WEIGHTS

    def _calc_ma_signal(self, row: pd.Series, prev: pd.Series) -> tuple[float, float, str]:
        """MA 크로스 시그널을 계산한다. (매수점수, 매도점수, 설명)"""
        buy, sell = 0.0, 0.0
        detail = ""

        ma5 = row.get("ma_5")
        ma20 = row.get("ma_20")
        prev_ma5 = prev.get("ma_5")
        prev_ma20 = prev.get("ma_20")

        if pd.isna(ma5) or pd.isna(ma20) or pd.isna(prev_ma5) or pd.isna(prev_ma20):
            return 0.0, 0.0, ""

        # 골든크로스: 5일선이 20일선을 상향돌파
        if prev_ma5 <= prev_ma20 and ma5 > ma20:
            buy = 1.0
            detail = "MA: 5일선이 20일선 상향돌파 (골든크로스)"
        # 데드크로스: 5일선이 20일선을 하향돌파
        elif prev_ma5 >= prev_ma20 and ma5 < ma20:
            sell = 1.0
            detail = "MA: 5일선이 20일선 하향돌파 (데드크로스)"
        # 정배열/역배열 추세
        elif ma5 > ma20:
            buy = 0.5
            detail = "MA: 단기 이평선 정배열"
        elif ma5 < ma20:
            sell = 0.5
            detail = "MA: 단기 이평선 역배열"

        return buy, sell, detail

    def _calc_rsi_signal(self, row: pd.Series) -> tuple[float, float, str]:
        """RSI 시그널을 계산한다."""
        rsi = row.get("rsi")
        if pd.isna(rsi):
            return 0.0, 0.0, ""

        if rsi < RSI_OVERSOLD:
            return 1.0, 0.0, f"RSI: {rsi:.1f} (과매도 영역)"
        elif rsi < 40:
            return 0.5, 0.0, f"RSI: {rsi:.1f} (매수 관심)"
        elif rsi > RSI_OVERBOUGHT:
            return 0.0, 1.0, f"RSI: {rsi:.1f} (과매수 영역)"
        elif rsi > 60:
            return 0.0, 0.5, f"RSI: {rsi:.1f} (매도 관심)"
        return 0.0, 0.0, ""

    def _calc_macd_signal(self, row: pd.Series, prev: pd.Series) -> tuple[float, float, str]:
        """MACD 시그널을 계산한다."""
        macd = row.get("macd")
        sig = row.get("macd_signal")
        prev_macd = prev.get("macd")
        prev_sig = prev.get("macd_signal")

        if any(pd.isna(v) for v in [macd, sig, prev_macd, prev_sig]):
            return 0.0, 0.0, ""

        if prev_macd <= prev_sig and macd > sig:
            return 1.0, 0.0, "MACD: Signal선 상향돌파"
        elif prev_macd >= prev_sig and macd < sig:
            return 0.0, 1.0, "MACD: Signal선 하향돌파"
        elif macd > sig:
            return 0.3, 0.0, "MACD: Signal선 위 유지"
        elif macd < sig:
            return 0.0, 0.3, "MACD: Signal선 아래 유지"
        return 0.0, 0.0, ""

    def _calc_bb_signal(self, row: pd.Series) -> tuple[float, float, str]:
        """볼린저밴드 시그널을 계산한다."""
        close = row.get("close")
        upper = row.get("bb_upper")
        lower = row.get("bb_lower")
        rsi = row.get("rsi")

        if any(pd.isna(v) for v in [close, upper, lower]):
            return 0.0, 0.0, ""

        if close <= lower:
            score = 1.0 if (not pd.isna(rsi) and rsi < RSI_OVERSOLD) else 0.7
            return score, 0.0, "BB: 하단밴드 접근"
        elif close >= upper:
            score = 1.0 if (not pd.isna(rsi) and rsi > RSI_OVERBOUGHT) else 0.7
            return 0.0, score, "BB: 상단밴드 이탈"
        return 0.0, 0.0, ""

    def _calc_volume_signal(self, row: pd.Series) -> tuple[float, float, str]:
        """거래량 시그널을 계산한다."""
        vol_ratio = row.get("volume_ratio")
        close = row.get("close")
        prev_close = row.get("prev_close")

        if pd.isna(vol_ratio) or pd.isna(close) or pd.isna(prev_close):
            return 0.0, 0.0, ""

        if vol_ratio >= VOLUME_SURGE_THRESHOLD:
            if close > prev_close:
                return 1.0, 0.0, f"거래량: 평균 대비 {vol_ratio:.1f}배 급증 (상승)"
            elif close < prev_close:
                return 0.0, 1.0, f"거래량: 평균 대비 {vol_ratio:.1f}배 급증 (하락)"
        return 0.0, 0.0, ""

    def generate(self, df: pd.DataFrame, ticker: str, name: str) -> Signal | None:
        """분석된 DataFrame의 최신 데이터로 시그널을 생성한다.

        Args:
            df: TechnicalAnalyzer.analyze() 결과.
            ticker: ETF 티커 코드.
            name: ETF 이름.

        Returns:
            Signal 객체. 시그널 없으면 None.
        """
        if len(df) < 2:
            return None

        df = df.copy()
        df["prev_close"] = df["close"].shift(1)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        details: list[str] = []

        # 각 지표별 시그널 계산
        ma_buy, ma_sell, d = self._calc_ma_signal(latest, prev)
        if d:
            details.append(d)

        rsi_buy, rsi_sell, d = self._calc_rsi_signal(latest)
        if d:
            details.append(d)

        macd_buy, macd_sell, d = self._calc_macd_signal(latest, prev)
        if d:
            details.append(d)

        bb_buy, bb_sell, d = self._calc_bb_signal(latest)
        if d:
            details.append(d)

        vol_buy, vol_sell, d = self._calc_volume_signal(latest)
        if d:
            details.append(d)

        # 복합 점수 계산
        w = self.weights
        buy_score = (
            w["ma_cross"] * ma_buy
            + w["rsi"] * rsi_buy
            + w["macd"] * macd_buy
            + w["bollinger"] * bb_buy
            + w["volume"] * vol_buy
        )
        sell_score = (
            w["ma_cross"] * ma_sell
            + w["rsi"] * rsi_sell
            + w["macd"] * macd_sell
            + w["bollinger"] * bb_sell
            + w["volume"] * vol_sell
        )

        # 시그널 판정
        if buy_score >= BUY_THRESHOLD:
            signal_type = "BUY"
            score = buy_score
        elif sell_score >= SELL_THRESHOLD:
            signal_type = "SELL"
            score = sell_score
        else:
            return None

        date_str = (
            latest.name.strftime("%Y-%m-%d")
            if hasattr(latest.name, "strftime")
            else str(latest.name)
        )

        signal = Signal(
            ticker=ticker,
            name=name,
            date=date_str,
            signal_type=signal_type,
            composite_score=score,
            ma_score=ma_buy if signal_type == "BUY" else ma_sell,
            rsi_score=rsi_buy if signal_type == "BUY" else rsi_sell,
            macd_score=macd_buy if signal_type == "BUY" else macd_sell,
            bb_score=bb_buy if signal_type == "BUY" else bb_sell,
            volume_score=vol_buy if signal_type == "BUY" else vol_sell,
            close_price=latest["close"],
            details=details,
        )
        logger.info("시그널 생성: %s %s (%.2f)", name, signal_type, score)
        return signal
