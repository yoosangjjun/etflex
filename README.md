# ETFlex

KRX ETF 매매 추천 자동화 시스템. 기술적 분석 지표를 자동 계산하고, 매수/매도 시그널 발생 시 텔레그램으로 알림을 전송한다.

> **주의:** 본 시스템은 투자 참고용이며, 실제 투자 판단 및 손실에 대한 책임은 사용자에게 있습니다.

## 주요 기능

- pykrx 기반 KRX ETF 시세/거래량 자동 수집
- 기술적 분석 지표 계산 (MA, RSI, MACD, 볼린저밴드, 거래량)
- 복합 시그널 기반 매수/매도 추천 (다중 지표 교차 확인)
- 텔레그램 봇 실시간 알림 (종목명, 시그널, 근거, 차트 이미지)
- 매일 장 마감 후 자동 분석 (APScheduler)
- SQLite 분석 이력/시그널 로그 저장

## 설치

```bash
git clone <repo-url> && cd etflex

# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 환경 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 필수 정보를 입력한다:

```
TELEGRAM_TOKEN=7012345678:AAF-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
TELEGRAM_CHAT_ID=123456789
KRX_USERNAME=your_krx_id
KRX_PASSWORD=your_krx_password
```

### KRX 계정

2026-02-27부터 KRX 데이터 시스템 이용에 로그인이 필수이다.
[KRX 정보데이터시스템](https://data.krx.co.kr)에서 회원가입 후 `.env`에 계정 정보를 입력한다.

### 텔레그램 봇 생성 방법

1. 텔레그램에서 **@BotFather** 검색 → `/newbot` 명령으로 봇 생성
2. 발급받은 **Bot Token**을 `.env`의 `TELEGRAM_TOKEN`에 입력
3. 생성된 봇과 대화를 시작한 뒤, 브라우저에서 아래 URL로 Chat ID 확인:
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
4. Chat ID를 `.env`의 `TELEGRAM_CHAT_ID`에 입력

### ETF 대상 설정

기본값은 **KRX 전체 ETF** 대상 분석이다. `config/settings.py`에서 변경 가능:

```python
USE_ALL_ETFS = True   # True: 전체 ETF, False: watchlist만
```

watchlist 모드 사용 시 `config/etf_watchlist.yaml`에서 대상 ETF를 추가/제거한다:

```yaml
etfs:
  - ticker: "069500"
    name: "KODEX 200"
  - ticker: "364690"
    name: "KODEX Fn반도체"
```

### 분석 파라미터 조정

`config/settings.py`에서 지표 파라미터, 시그널 가중치, 임계값 등을 조정할 수 있다:

```python
MA_PERIODS = [5, 20, 60, 120]   # 이동평균 기간
RSI_PERIOD = 14                  # RSI 기간
BUY_THRESHOLD = 0.6              # 매수 시그널 임계값
SELL_THRESHOLD = 0.6             # 매도 시그널 임계값

SIGNAL_WEIGHTS = {
    "ma_cross": 0.30,    # MA 크로스
    "rsi": 0.20,         # RSI
    "macd": 0.25,        # MACD
    "bollinger": 0.15,   # 볼린저밴드
    "volume": 0.10,      # 거래량
}
```

## 실행

### 단발 실행

```bash
python run_analysis.py
```

### 스케줄러 실행 (매 평일 16:30 자동 분석)

```bash
python main.py
```

### 단발 실행 (--once 플래그)

```bash
python main.py --once
```

### 텔레그램 봇 커맨드

| 커맨드 | 설명 |
|--------|------|
| `/status` | 시스템 상태 확인 |
| `/analyze 069500` | 특정 종목 즉시 분석 |

## Docker 실행

```bash
# .env 파일 설정 후
docker compose up -d

# 로그 확인
docker compose logs -f

# 즉시 단발 테스트
docker compose run --rm etflex python main.py --once
```

## 테스트

```bash
python -m pytest tests/ -v
```

## 프로젝트 구조

```
etflex/
├── config/
│   ├── settings.py           # 전체 설정 (분석 파라미터, ETF 목록)
│   └── etf_watchlist.yaml    # 모니터링 대상 ETF 목록
├── src/
│   ├── collector.py          # pykrx 데이터 수집
│   ├── analyzer.py           # 기술적 분석 지표 계산
│   ├── signal.py             # 매수/매도 시그널 생성
│   ├── notifier.py           # 텔레그램 봇 알림 전송
│   ├── store.py              # SQLite 데이터 저장소
│   ├── chart.py              # 차트 이미지 생성
│   └── krx_auth.py           # KRX 로그인 및 pykrx 세션 패치
├── tests/
├── main.py                   # 진입점 (스케줄러)
├── run_analysis.py           # 단발 실행
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## 시그널 알림 예시

```
🟢 매수 추천 | KODEX 200 (069500)
────────────────────────
현재가: 35,420원
복합 점수: 0.78 / 1.00

▶ MA: 5일선이 20일선 상향돌파 (골든크로스)
▶ RSI: 32.5 (과매도 영역)
▶ MACD: Signal선 상향돌파
▶ 거래량: 평균 대비 2.3배 급증 (상승)

⏰ 2026-03-20 분석 기준
```
