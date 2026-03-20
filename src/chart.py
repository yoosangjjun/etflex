"""차트 이미지 생성 모듈."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from config.settings import BASE_DIR

logger = logging.getLogger(__name__)

CHART_DIR = BASE_DIR / "charts"


def generate_chart(df: pd.DataFrame, ticker: str, name: str) -> str | None:
    """ETF 가격 + 기술적 지표 오버레이 차트를 생성한다.

    Args:
        df: TechnicalAnalyzer.analyze() 결과 DataFrame.
        ticker: ETF 티커 코드.
        name: ETF 이름.

    Returns:
        생성된 이미지 파일 경로. 실패 시 None.
    """
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 최근 60일만 표시
        plot_df = df.tail(60).copy()

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[3, 1, 1],
                                  sharex=True, gridspec_kw={"hspace": 0.1})

        # 1) 가격 + MA + 볼린저밴드
        ax1 = axes[0]
        ax1.plot(plot_df.index, plot_df["close"], label="종가", color="black", linewidth=1.5)
        for col in ["ma_5", "ma_20", "ma_60"]:
            if col in plot_df.columns:
                ax1.plot(plot_df.index, plot_df[col], label=col.upper(), linewidth=0.8)

        if "bb_upper" in plot_df.columns:
            ax1.fill_between(plot_df.index, plot_df["bb_lower"], plot_df["bb_upper"],
                             alpha=0.1, color="gray", label="BB")

        ax1.set_title(f"{name} ({ticker})", fontsize=14)
        ax1.legend(loc="upper left", fontsize=8)
        ax1.set_ylabel("가격")
        ax1.grid(True, alpha=0.3)

        # 2) RSI
        ax2 = axes[1]
        if "rsi" in plot_df.columns:
            ax2.plot(plot_df.index, plot_df["rsi"], color="purple", linewidth=1)
            ax2.axhline(y=70, color="red", linestyle="--", linewidth=0.7)
            ax2.axhline(y=30, color="green", linestyle="--", linewidth=0.7)
            ax2.fill_between(plot_df.index, 30, 70, alpha=0.05, color="gray")
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        # 3) MACD
        ax3 = axes[2]
        if "macd" in plot_df.columns:
            ax3.plot(plot_df.index, plot_df["macd"], label="MACD", linewidth=1)
            ax3.plot(plot_df.index, plot_df["macd_signal"], label="Signal", linewidth=1)
            colors = ["green" if v >= 0 else "red" for v in plot_df["macd_hist"]]
            ax3.bar(plot_df.index, plot_df["macd_hist"], color=colors, alpha=0.5, width=0.8)
        ax3.set_ylabel("MACD")
        ax3.legend(loc="upper left", fontsize=8)
        ax3.grid(True, alpha=0.3)

        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        fig.autofmt_xdate()

        path = str(CHART_DIR / f"{ticker}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.info("차트 생성: %s", path)
        return path

    except Exception:
        logger.exception("차트 생성 실패: %s", ticker)
        return None
