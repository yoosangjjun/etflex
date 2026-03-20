"""텔레그램 봇 알림 전송 모듈."""

import logging

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config.settings import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN
from src.signal import Signal

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """텔레그램 봇을 통한 시그널 알림 전송 클래스."""

    def __init__(self, token: str = TELEGRAM_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
        self.token = token
        self.chat_id = chat_id
        self.bot = Bot(token=token)

    def _format_signal_message(self, signal: Signal) -> str:
        """시그널을 텔레그램 메시지 형식으로 포맷한다."""
        emoji = "\U0001f7e2" if signal.signal_type == "BUY" else "\U0001f534"
        label = "매수 추천" if signal.signal_type == "BUY" else "매도 추천"

        lines = [
            f"{emoji} {label} | {signal.name} ({signal.ticker})",
            "\u2500" * 24,
            f"현재가: {signal.close_price:,.0f}원",
            f"복합 점수: {signal.composite_score:.2f} / 1.00",
            "",
        ]
        for detail in signal.details:
            lines.append(f"\u25b6 {detail}")

        lines.append("")
        lines.append(f"\u23f0 {signal.date} 분석 기준")

        return "\n".join(lines)

    async def send_signal(self, signal: Signal) -> bool:
        """시그널 알림을 전송한다.

        Args:
            signal: 전송할 시그널.

        Returns:
            전송 성공 여부.
        """
        message = self._format_signal_message(signal)
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
            logger.info("알림 전송 완료: %s %s", signal.name, signal.signal_type)
            return True
        except Exception:
            logger.exception("알림 전송 실패: %s", signal.name)
            return False

    async def send_chart(self, chat_id: str | None, image_path: str) -> bool:
        """차트 이미지를 전송한다."""
        cid = chat_id or self.chat_id
        try:
            with open(image_path, "rb") as f:
                await self.bot.send_photo(chat_id=cid, photo=f)
            return True
        except Exception:
            logger.exception("차트 전송 실패: %s", image_path)
            return False

    async def send_text(self, text: str) -> bool:
        """일반 텍스트 메시지를 전송한다."""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text)
            return True
        except Exception:
            logger.exception("메시지 전송 실패")
            return False


def create_bot_app(token: str = TELEGRAM_TOKEN) -> Application:
    """텔레그램 봇 Application을 생성한다. (커맨드 핸들러 포함)"""
    app = Application.builder().token(token).build()

    async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/status 명령: 시스템 상태를 응답한다."""
        await update.message.reply_text(
            "ETFlex 시스템 정상 작동 중\n"
            "다음 분석 예정: 매일 16:30"
        )

    async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/analyze [티커] 명령: 특정 종목을 즉시 분석한다."""
        if not context.args:
            await update.message.reply_text("사용법: /analyze 069500")
            return

        ticker = context.args[0]
        await update.message.reply_text(f"{ticker} 분석을 시작합니다...")

        # 분석 파이프라인 실행 (지연 임포트로 순환 참조 방지)
        try:
            from src.collector import ETFDataCollector
            from src.analyzer import TechnicalAnalyzer
            from src.signal import SignalGenerator

            collector = ETFDataCollector(watchlist=[])
            df = collector.fetch_ohlcv(ticker)
            if df.empty:
                await update.message.reply_text(f"{ticker}: 데이터를 가져올 수 없습니다.")
                return

            analyzer = TechnicalAnalyzer(df)
            analyzed = analyzer.analyze()
            gen = SignalGenerator()
            signal = gen.generate(analyzed, ticker, ticker)

            if signal:
                notifier = TelegramNotifier(token=token)
                msg = notifier._format_signal_message(signal)
                await update.message.reply_text(msg)
            else:
                await update.message.reply_text(f"{ticker}: 현재 매매 시그널 없음 (HOLD)")
        except Exception as e:
            logger.exception("분석 실패: %s", ticker)
            await update.message.reply_text(f"분석 중 오류 발생: {e}")

    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("analyze", cmd_analyze))

    return app
