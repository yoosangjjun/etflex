"""ETFlex 전체 설정 모듈."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "etflex.db"

# 텔레그램 설정
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# pykrx 설정
# KRX 로그인 정보
KRX_USERNAME = os.getenv("KRX_USERNAME", "")
KRX_PASSWORD = os.getenv("KRX_PASSWORD", "")

API_CALL_DELAY = 1  # 초 단위 sleep
USE_ALL_ETFS = True  # True: 전체 ETF 대상, False: watchlist만

# 데이터 수집 기간 (일)
DATA_FETCH_DAYS = 200

# 기술적 분석 파라미터
MA_PERIODS = [5, 20, 60, 120]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
VOLUME_AVG_PERIOD = 20
VOLUME_SURGE_THRESHOLD = 2.0  # 평균 대비 배수

# 시그널 가중치
SIGNAL_WEIGHTS = {
    "ma_cross": 0.30,
    "rsi": 0.20,
    "macd": 0.25,
    "bollinger": 0.15,
    "volume": 0.10,
}

# 시그널 임계값
BUY_THRESHOLD = 0.6
SELL_THRESHOLD = 0.6

# RSI 임계값
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# 스케줄러
ANALYSIS_HOUR = 16
ANALYSIS_MINUTE = 30


def load_watchlist() -> list[dict]:
    """etf_watchlist.yaml에서 모니터링 대상 ETF 목록을 로드한다."""
    watchlist_path = BASE_DIR / "config" / "etf_watchlist.yaml"
    with open(watchlist_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("etfs", [])
