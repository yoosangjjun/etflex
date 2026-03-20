"""analyzer.py 테스트."""

import numpy as np
import pandas as pd
import pytest

from src.analyzer import TechnicalAnalyzer


@pytest.fixture
def sample_df():
    """60일 분량 샘플 OHLCV 데이터."""
    np.random.seed(42)
    n = 60
    dates = pd.bdate_range("2026-01-01", periods=n)
    close = 10000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame(
        {
            "open": close - np.random.rand(n) * 50,
            "high": close + np.random.rand(n) * 100,
            "low": close - np.random.rand(n) * 100,
            "close": close,
            "volume": np.random.randint(10000, 100000, n),
        },
        index=dates,
    )


def test_calc_ma(sample_df):
    analyzer = TechnicalAnalyzer(sample_df)
    analyzer.calc_ma()
    assert "ma_5" in analyzer.df.columns
    assert "ma_20" in analyzer.df.columns
    # ma_5의 처음 4개는 NaN
    assert analyzer.df["ma_5"].isna().sum() == 4


def test_calc_rsi(sample_df):
    analyzer = TechnicalAnalyzer(sample_df)
    rsi = analyzer.calc_rsi()
    # RSI는 0~100 범위
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_calc_macd(sample_df):
    analyzer = TechnicalAnalyzer(sample_df)
    analyzer.calc_macd()
    assert "macd" in analyzer.df.columns
    assert "macd_signal" in analyzer.df.columns
    assert "macd_hist" in analyzer.df.columns


def test_calc_bollinger(sample_df):
    analyzer = TechnicalAnalyzer(sample_df)
    analyzer.calc_bollinger()
    valid = analyzer.df.dropna(subset=["bb_upper", "bb_lower"])
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


def test_calc_volume_ratio(sample_df):
    analyzer = TechnicalAnalyzer(sample_df)
    ratio = analyzer.calc_volume_ratio()
    valid = ratio.dropna()
    assert (valid > 0).all()


def test_analyze_all(sample_df):
    analyzer = TechnicalAnalyzer(sample_df)
    result = analyzer.analyze()
    expected_cols = ["ma_5", "ma_20", "rsi", "macd", "bb_upper", "volume_ratio"]
    for col in expected_cols:
        assert col in result.columns


def test_korean_columns():
    """pykrx 한글 컬럼명 자동 변환 테스트."""
    df = pd.DataFrame(
        {"시가": [100], "고가": [110], "저가": [90], "종가": [105], "거래량": [1000]},
        index=pd.to_datetime(["2026-03-20"]),
    )
    analyzer = TechnicalAnalyzer(df)
    assert "close" in analyzer.df.columns
    assert "종가" not in analyzer.df.columns
