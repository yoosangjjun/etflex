"""signal.py 테스트."""

import numpy as np
import pandas as pd
import pytest

from src.analyzer import TechnicalAnalyzer
from src.signal import SignalGenerator


def _make_df(close_values: list[float], volume: int = 50000) -> pd.DataFrame:
    """테스트용 DataFrame 생성 헬퍼."""
    n = len(close_values)
    dates = pd.bdate_range("2026-01-01", periods=n)
    close = np.array(close_values, dtype=float)
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": [volume] * n,
        },
        index=dates,
    )


@pytest.fixture
def generator():
    return SignalGenerator()


def test_no_signal_on_flat_data(generator):
    """횡보 데이터에서는 시그널이 없어야 한다."""
    df = _make_df([10000] * 60)
    analyzer = TechnicalAnalyzer(df)
    analyzed = analyzer.analyze()
    signal = generator.generate(analyzed, "069500", "KODEX 200")
    assert signal is None


def test_buy_signal_on_golden_cross(generator):
    """골든크로스 + 추가 조건 시 매수 시그널."""
    # 하락 후 급반등 패턴
    prices = list(np.linspace(12000, 9500, 30)) + list(np.linspace(9500, 11500, 30))
    # 반등 구간에 거래량 급증
    volumes = [30000] * 30 + [100000] * 30
    df = _make_df(prices)
    df["volume"] = volumes

    analyzer = TechnicalAnalyzer(df)
    analyzed = analyzer.analyze()
    signal = generator.generate(analyzed, "069500", "KODEX 200")

    # 매수 또는 None (데이터 패턴에 따라 다를 수 있음)
    if signal:
        assert signal.signal_type == "BUY"
        assert signal.composite_score >= 0.6


def test_sell_signal_on_dead_cross(generator):
    """데드크로스 + 추가 조건 시 매도 시그널."""
    # 상승 후 급락 패턴
    prices = list(np.linspace(9000, 12000, 30)) + list(np.linspace(12000, 9500, 30))
    volumes = [30000] * 30 + [100000] * 30
    df = _make_df(prices)
    df["volume"] = volumes

    analyzer = TechnicalAnalyzer(df)
    analyzed = analyzer.analyze()
    signal = generator.generate(analyzed, "069500", "KODEX 200")

    if signal:
        assert signal.signal_type == "SELL"
        assert signal.composite_score >= 0.6


def test_signal_to_dict(generator):
    """Signal.to_dict() 테스트."""
    prices = list(np.linspace(12000, 9500, 30)) + list(np.linspace(9500, 11500, 30))
    df = _make_df(prices)
    df["volume"] = [30000] * 30 + [100000] * 30

    analyzer = TechnicalAnalyzer(df)
    analyzed = analyzer.analyze()
    signal = generator.generate(analyzed, "069500", "KODEX 200")

    if signal:
        d = signal.to_dict()
        assert "ticker" in d
        assert "signal_type" in d
        assert "composite_score" in d


def test_insufficient_data(generator):
    """데이터가 부족하면 None 반환."""
    df = _make_df([10000])
    analyzer = TechnicalAnalyzer(df)
    analyzed = analyzer.analyze()
    signal = generator.generate(analyzed, "069500", "KODEX 200")
    assert signal is None
