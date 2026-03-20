"""ETFlex 메인 진입점 - 스케줄러 및 파이프라인 실행."""

import asyncio
import logging
import sys
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config.settings import (
    ANALYSIS_HOUR, ANALYSIS_MINUTE, KRX_PASSWORD, KRX_USERNAME, USE_ALL_ETFS,
    load_watchlist,
)
from src.krx_auth import krx_login, patch_pykrx_session
from src.analyzer import TechnicalAnalyzer
from src.chart import generate_chart
from src.collector import ETFDataCollector
from src.notifier import TelegramNotifier
from src.signal import Signal, SignalGenerator
from src.store import DataStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("etflex")


async def run_pipeline() -> list[Signal]:
    """전체 분석 파이프라인을 실행한다.

    Collector → Analyzer → SignalGenerator → Notifier
    """
    logger.info("=== ETFlex 분석 파이프라인 시작 ===")

    # KRX 로그인
    if KRX_USERNAME and KRX_PASSWORD:
        try:
            session = krx_login(KRX_USERNAME, KRX_PASSWORD)
            patch_pykrx_session(session)
        except RuntimeError:
            logger.exception("KRX 로그인 실패 - 로그인 없이 진행")

    collector = ETFDataCollector()
    store = DataStore()
    generator = SignalGenerator()
    notifier = TelegramNotifier()

    ohlcv_data = collector.collect_all()
    signals: list[Signal] = []

    # 전체 ETF 모드: ticker만으로 순회 / watchlist 모드: name 포함
    if USE_ALL_ETFS:
        etf_list = [{"ticker": t, "name": t} for t in ohlcv_data.keys()]
    else:
        watchlist = load_watchlist()
        etf_list = watchlist

    for etf in etf_list:
        ticker = etf["ticker"]
        name = etf.get("name", ticker)

        df = ohlcv_data.get(ticker)
        if df is None or df.empty:
            continue

        # DB에 저장
        store.save_ohlcv(ticker, df)

        # 기술적 분석
        analyzer = TechnicalAnalyzer(df)
        analyzed = analyzer.analyze()

        # 시그널 생성
        signal = generator.generate(analyzed, ticker, name)
        if signal:
            signals.append(signal)
            store.save_signal(signal.to_dict())

            # 알림 전송
            await notifier.send_signal(signal)

            # 차트 생성 및 전송
            chart_path = generate_chart(analyzed, ticker, name)
            if chart_path:
                await notifier.send_chart(None, chart_path)

    # 분석 로그 저장
    today = datetime.now().strftime("%Y-%m-%d")
    store.save_analysis_log(today, len(ohlcv_data), len(signals), "OK")

    logger.info("=== 파이프라인 완료: %d건 시그널 ===", len(signals))
    return signals


def main() -> None:
    """스케줄러 모드로 실행한다."""
    if "--once" in sys.argv:
        asyncio.run(run_pipeline())
        return

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_pipeline,
        "cron",
        hour=ANALYSIS_HOUR,
        minute=ANALYSIS_MINUTE,
        day_of_week="mon-fri",
        id="etflex_daily",
    )
    scheduler.start()
    logger.info(
        "스케줄러 시작: 매 평일 %02d:%02d 분석 실행",
        ANALYSIS_HOUR, ANALYSIS_MINUTE,
    )

    loop = asyncio.new_event_loop()
    try:
        loop.run_forever()
    except (KeyboardInterrupt, SystemExit):
        logger.info("스케줄러 종료")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
