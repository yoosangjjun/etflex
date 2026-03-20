"""KRX 데이터 시스템 로그인 및 pykrx 세션 주입 모듈."""

import logging

import requests

logger = logging.getLogger(__name__)

KRX_SESSION_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
KRX_LOGIN_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"


def krx_login(username: str, password: str) -> requests.Session:
    """KRX 데이터 시스템에 로그인하고 인증된 세션을 반환한다.

    Args:
        username: KRX 회원 ID.
        password: KRX 비밀번호.

    Returns:
        로그인 쿠키가 설정된 requests.Session.

    Raises:
        RuntimeError: 로그인 실패 시.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",
    })

    # 세션 쿠키 획득
    session.get(KRX_SESSION_URL)

    payload = {
        "mbrId": username,
        "pw": password,
        "mbrNm": "",
        "telNo": "",
        "di": "",
        "certType": "",
        "skipDup": "Y",
    }

    resp = session.post(KRX_LOGIN_URL, data=payload)
    resp.raise_for_status()

    result = resp.json()
    code = result.get("_error_code", result.get("result", {}).get("code", ""))
    msg = result.get("_error_message", result.get("result", {}).get("message", ""))

    if code == "CD001":
        logger.info("KRX 로그인 성공: %s", username)
        return session
    else:
        raise RuntimeError(f"KRX 로그인 실패: [{code}] {msg}")


def patch_pykrx_session(session: requests.Session) -> None:
    """pykrx의 HTTP 호출을 로그인된 세션으로 교체한다.

    pykrx는 requests.post/get을 직접 호출하므로,
    webio 모듈의 Post/Get 클래스의 read 메서드를 패치한다.
    """
    from pykrx.website.comm import webio

    original_post_read = webio.Post.read
    original_get_read = webio.Get.read

    def patched_post_read(self, **params):
        resp = session.post(self.url, headers=self.headers, data=params)
        return resp

    def patched_get_read(self, **params):
        resp = session.get(self.url, headers=self.headers, params=params)
        return resp

    webio.Post.read = patched_post_read
    webio.Get.read = patched_get_read
    logger.info("pykrx 세션 패치 완료")
