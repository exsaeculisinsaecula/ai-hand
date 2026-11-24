"""VLM 연결 (로컬 Ollama · Gemma 3 4B) — 요청 문장 이해 + 손 위 공 확인.

역할
  1. 요청 해석: 대시보드 검색창에 입력한 문장을 VLM 이 이해해 빨간/파란/초록 공 중 하나로 정한다
     (글자만, 사진 없음). 답은 JSON 스키마로 강제한다.
  2. 손 위 확인 (팀 vlm-go 방식): 손 영역에 공이 멈추면 그 사진과 대상("빨간 공")을 VLM 에 보여 주고
     {"found": true/false} 만 받는다. true → 로봇 GRIP, false → 로봇 NG.

Ollama 가 꺼져 있으면 즉시 단어 규칙(parse_request)·색 규칙으로 진행한다 (화면이 멈추지 않도록).
연결 상태는 백그라운드에서 10초마다 다시 확인하므로, 실행 중에 Ollama 를 켜도 자동으로 붙는다.

  ollama pull gemma3:4b
  python handover_app.py --mock
"""
from __future__ import annotations

import base64
from concurrent.futures import Future, ThreadPoolExecutor
import json
import re
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

from vision.parts import ASK_AGAIN, LABEL_KO, SUPPORTED, mentioned_colors, parse_request

KO_TO_LABEL = {v: k for k, v in LABEL_KO.items() if k in SUPPORTED}      # 빨간 공→red ...

PART_CUES = {
    "red": "빨간색 공 (붉은색·레드, 사과·딸기·소방차·토마토 색)",
    "blue": "파란색 공 (푸른색·블루, 하늘·바다 색)",
    "green": "초록색 공 (녹색·그린, 풀·나뭇잎·잔디 색)",
}

INTERPRET_PROMPT = """너는 공 전달 로봇의 요청 해석기다. 작업자가 입력한 문장이 아래 공 중 무엇을 원하는지 고른다.
공 목록:
{options}

요청: "{text}"

규칙
- 요청은 한국어 구어체나 영어다. 별명·비유·오타·띄어쓰기 없는 말도 해석한다 (예: "사과 색 공" → 빨간 공).
- "A 말고 B", "A 아니고 B"처럼 부정이 있으면 원하는 쪽(B)을 고른다.
- 색을 짐작할 단서가 전혀 없으면(예: "공 줘", "아무거나") 짐작하지 말고 "모름".
- 목록에 없는 색(노랑 등), 두 가지 이상을 동시에 요구, 공과 무관한 말, 어느 쪽인지 정할 수 없으면 "모름".
JSON 으로만 답한다: {{"part": "{names} / 모름 중 하나", "reason": "짧은 근거"}}"""

# 팀 vlm-go(jihustar12-design/vlm-go) 의 단호한 프롬프트 — 답은 found 하나뿐
FOUND_PROMPT = ("사용자가 '{target}'을(를) 찾고 있어. 이 사진 안에 {target}이(가) 명확하게 있으면 "
                "오직 {{\"found\": true}}라고 대답하고, 없거나 헷갈리면 {{\"found\": false}}라고만 대답해. "
                "다른 말은 절대 하지 마.")
FOUND_SCHEMA = {"type": "object", "properties": {"found": {"type": "boolean"}}, "required": ["found"]}

# 손 위 공 인식: 어떤 공인지(또는 없음)를 VLM 이 직접 고른다.
# 실제 Gemma 3 4B 비교 (2026-11-24): 이 문장 5/5, "손 위에 공이 있나→색" 3/5, 영어 4/5 (빨간 공을 '없음'으로 놓침)
BALL_PROMPT = "사진에서 가장 눈에 띄는 공의 색을 고른다. 빨간색·파란색·초록색 공이 보이지 않으면 없음."
BALL_COLORS = {"빨간색": "red", "파란색": "blue", "초록색": "green"}

# 여러 정보를 한 번에 관찰 (판정에는 쓰지 않고 화면 표시용, Gemma 3 4B 약 2~2.5초)
DESCRIBE_PROMPT = ("사진은 로봇 손 부분을 찍은 것이다. 보이는 것을 짧게 관찰해 JSON 으로만 답한다. "
                   "object: 손 위나 손가락 사이에 있는 물건이 무엇인지 (예: 빨간 고무공, 없음), "
                   "color: 그 물건의 색, count: 공 개수(숫자), position: 위치 (손바닥 가운데 / 손가락 사이 / 손 밖 등), "
                   "state: 상태 (손바닥 위에 놓임 / 손가락이 쥐고 있음 / 떨어지는 중 / 없음), "
                   "scene: 손 주변에 보이는 것 한 줄, note: 특이사항 한 줄 (없으면 없음).")
DESCRIBE_FIELDS = ("object", "color", "count", "position", "state", "scene", "note")

CHECK_PROMPT = """사진은 로봇 손바닥 위를 찍은 것이다. 손 위에 색깔 공이 하나 있다.
공은 다음 중 하나다.
{options}
보이는 색만 근거로 판단하고, 정말 보이지 않으면 "모름"이라고 한다.
JSON 으로만 답한다: {{"seen": "보이는 색 한 줄", "name": "{names} / 모름 중 하나"}}"""


def _norm(t: str) -> str:
    return re.sub(r"\s+", "", str(t or "")).lower()


FILLER = re.compile(r"(공|볼|ball|balls|줘|줘요|주세요|주십시오|줄래|주라|받아|받을게|하나|한개|한\s*개|좀|만|거|것|걸|그거|"
                    r"아무거나|아무|색|색깔|give|me|the|a|an|one|please|pls|can|i|have|want|plz|[을를이가은는도로으]|[.,!?~])",
                    re.I)


def has_clue(text: str) -> bool:
    """'공 줘' 처럼 색을 짐작할 말이 하나도 없으면 False (VLM 이 아무 색이나 고르지 않게)"""
    rest = FILLER.sub(" ", str(text).lower())
    return bool(re.sub(r"\s+", "", rest))


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except json.JSONDecodeError:
            return {}


def parse_found(raw: str) -> bool | None:
    """{"found": true/false} → bool. 형식이 깨졌으면 true/false 단어로, 그것도 없으면 None."""
    d = parse_json(raw)
    if "found" in d:
        f = d["found"]
        return f is True or str(f).strip().lower() == "true"
    if re.search(r"\btrue\b", raw or "", re.I):
        return True
    if re.search(r"\bfalse\b", raw or "", re.I):
        return False
    return None


class VlmBridge:
    def __init__(self, model: str = "gemma3:4b", url: str = "http://127.0.0.1:11434",
                 text_timeout: float = 20.0, image_timeout: float = 60.0, verify_below: float = 1.0):
        self.model = model
        self.url = url.rstrip("/").removesuffix("/v1")
        self.text_timeout = text_timeout
        self.image_timeout = image_timeout
        self.verify_below = verify_below            # 1.0 = 항상 VLM 이 손 위를 확인, 0 = 끔
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._no_think = True
        self.online = True                          # 기본은 연결된 것으로 보고, 모니터가 고친다
        self.model_ready = True
        self.status_note = ""
        self._monitor: threading.Thread | None = None
        self.stats = {"interpret_calls": 0, "interpret_changed": 0, "check_calls": 0,
                      "check_agree": 0, "check_disagree": 0, "check_fail": 0}

    # ---- 연결 상태 ----
    def probe(self, timeout: float = 1.5) -> bool:
        """Ollama 가 켜져 있고 모델이 받아져 있는지 확인 (/api/tags)."""
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=timeout) as r:
                names = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
        except Exception as exc:
            self.online, self.model_ready = False, False
            self.status_note = f"Ollama 에 연결할 수 없습니다 ({exc.__class__.__name__})"
            return False
        want = self.model if ":" in self.model else self.model + ":latest"
        self.model_ready = want in names
        self.online = self.model_ready
        self.status_note = "" if self.model_ready else f"모델 {self.model} 이 없습니다 — 'ollama pull {self.model}'"
        return self.online

    def start_monitor(self, every: float = 10.0, on_change=None) -> None:
        def _loop():
            prev = None
            while True:
                now = self.probe()
                if now != prev:
                    if on_change is not None:
                        on_change(now)
                    if now:
                        self.warmup()               # 막 연결됨 → 모델을 미리 올려 둠
                prev = now
                time.sleep(every)
        self._monitor = threading.Thread(target=_loop, daemon=True)
        self._monitor.start()

    def status(self) -> dict:
        return {"online": self.online, "model": self.model, "url": self.url, "note": self.status_note,
                "verify": self.verify_below > 0}

    # ---- Ollama 호출 ----
    def _chat(self, prompt: str, images: list[str] | None = None, timeout: float = 20.0,
              schema: dict | None = None) -> str:
        msg = {"role": "user", "content": prompt}
        if images:
            msg["images"] = images
        body = {"model": self.model, "stream": False, "messages": [msg], "options": {"temperature": 0}}
        if schema is not None:
            body["format"] = schema                     # 답 형식을 JSON 스키마로 강제 (Ollama structured outputs)
        if self._no_think:
            body["think"] = False                       # 생각 과정을 출력하는 모델이면 끄기
        req = urllib.request.Request(f"{self.url}/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["message"].get("content", "") or ""
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="ignore")
            if self._no_think and "think" in text.lower():   # 생각 기능이 없는 모델(gemma3) → 옵션 빼고 다시
                self._no_think = False
                return self._chat(prompt, images, timeout, schema)
            raise RuntimeError(f"Ollama {e.code}: {text[:120]}")
        except (urllib.error.URLError, OSError):
            self.online = False                         # 다음 모니터 확인까지는 바로 규칙으로
            raise

    def warmup(self) -> None:
        """모델을 메모리에 올려 둔다 (첫 요청이 느리지 않도록). 백그라운드."""
        def _w():
            try:
                self._chat("한 단어로 답해: 준비", timeout=120)
            except Exception:
                pass
        self._pool.submit(_w)

    # ---- 1. 요청 해석 ----
    def interpret(self, text: str) -> tuple[str | None, str, dict]:
        """→ (red/blue/green/None, 작업자에게 보여 줄 답, 정보 {method, reason, seconds})"""
        rule_t, rule_reply = parse_request(text)
        if not _norm(text):
            return None, rule_reply, {"method": "규칙", "reason": "빈 입력", "seconds": 0.0}
        if len(mentioned_colors(text)) > 1:             # "빨간 공이랑 파란 공" — VLM 이 하나를 골라 버리지 않게
            return None, "한 번에 한 가지 색만 말해 주십시오.", {"method": "규칙", "reason": "색이 두 가지 이상", "seconds": 0.0}
        if not mentioned_colors(text) and not has_clue(text):
            return None, ASK_AGAIN, {"method": "규칙", "reason": "색을 짐작할 말이 없음", "seconds": 0.0}
        if not self.online:
            return rule_t, rule_reply, {"method": "단어 규칙 (VLM 꺼짐)", "reason": self.status_note, "seconds": 0.0}

        self.stats["interpret_calls"] += 1
        names = [LABEL_KO[l] for l in SUPPORTED]
        options = "\n".join(f"- {LABEL_KO[l]}: {PART_CUES[l]}" for l in SUPPORTED)
        schema = {"type": "object", "properties": {"part": {"type": "string", "enum": names + ["모름"]},
                                                   "reason": {"type": "string"}}, "required": ["part", "reason"]}
        t0 = time.monotonic()
        try:
            raw = self._chat(INTERPRET_PROMPT.format(options=options, text=text, names=" / ".join(names)),
                             timeout=self.text_timeout, schema=schema)
            d = parse_json(raw)
        except Exception as exc:
            d, raw = {}, f"오류: {exc.__class__.__name__}"
        dt = round(time.monotonic() - t0, 2)
        got = _norm(d.get("part", ""))
        label = next((KO_TO_LABEL[n] for n in names if _norm(n) == got), None) or \
            next((KO_TO_LABEL[n] for n in names if got and _norm(n) in got), None)
        reason = str(d.get("reason", "")) if d else f"VLM 응답 없음 ({raw[:40]})"

        if not d:                                       # VLM 실패 → 단어 규칙
            return rule_t, rule_reply, {"method": "단어 규칙 (VLM 실패)", "reason": reason, "seconds": dt}
        if label is None:
            return None, ASK_AGAIN, {"method": "VLM", "reason": reason, "seconds": dt}
        if label != rule_t:
            self.stats["interpret_changed"] += 1
        return label, f"{LABEL_KO[label]}을 받겠습니다. 로봇 손에 쥐어 주십시오.", \
            {"method": "VLM", "reason": reason, "seconds": dt}

    # ---- 2. 손 위 확인 ----
    def should_check(self, conf: float) -> bool:
        return self.online and self.verify_below > 0 and (self.verify_below >= 1.0 or conf < self.verify_below)

    @staticmethod
    def crop(frame: np.ndarray, box, margin: float = 0.6, side: int = 672) -> np.ndarray | None:
        """탐지 상자(x, y, w, h) 둘레를 넉넉히 잘라 확대"""
        if frame is None or not box:
            return None
        x, y, w, h = box
        cx, cy, half = x + w / 2, y + h / 2, max(w, h) * (0.5 + margin / 2)
        H, W = frame.shape[:2]
        c = frame[int(max(0, cy - half)):int(min(H, cy + half)), int(max(0, cx - half)):int(min(W, cx + half))]
        return cv2.resize(c, (side, side), interpolation=cv2.INTER_CUBIC) if c.size else None

    def check_found(self, img: np.ndarray, target: str) -> dict:
        """팀 vlm-go 방식: 사진 + 대상 → {"found": True/False/None, "raw", "seconds", "ok"}"""
        self.stats["check_calls"] += 1
        name = LABEL_KO.get(target, target)
        t0 = time.monotonic()
        try:
            h, w = img.shape[:2]
            if max(h, w) > 1024:
                s = 1024 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            raw = self._chat(FOUND_PROMPT.format(target=name), [base64.b64encode(buf.tobytes()).decode()],
                             timeout=self.image_timeout, schema=FOUND_SCHEMA)
            found = parse_found(raw)
        except Exception as exc:
            raw, found = f"오류: {exc.__class__.__name__}", None
        return {"found": found, "raw": raw.strip()[:80], "seconds": round(time.monotonic() - t0, 1),
                "ok": found is not None}

    def check_ball(self, img: np.ndarray) -> dict:
        """손 위 사진 → {"label": red/blue/green/None, "empty": 공 없음 여부, "raw", "seconds", "ok"}"""
        self.stats["check_calls"] += 1
        schema = {"type": "object", "properties": {"ball": {"type": "string", "enum": list(BALL_COLORS) + ["없음"]}},
                  "required": ["ball"]}
        t0 = time.monotonic()
        try:
            h, w = img.shape[:2]
            if max(h, w) > 1024:
                s = 1024 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            raw = self._chat(BALL_PROMPT, [base64.b64encode(buf.tobytes()).decode()],
                             timeout=self.image_timeout, schema=schema)
            got = str(parse_json(raw).get("ball", "")).strip()
        except Exception as exc:
            raw, got = f"오류: {exc.__class__.__name__}", ""
        label = BALL_COLORS.get(got)
        empty = got == "없음"
        return {"label": label, "empty": empty, "raw": raw.strip()[:80],
                "seconds": round(time.monotonic() - t0, 1), "ok": label is not None or empty}

    def describe(self, img: np.ndarray) -> dict:
        """사진 → {"fields": {object, color, count, position, state, scene, note}, "seconds", "ok"}"""
        schema = {"type": "object",
                  "properties": {k: ({"type": "integer"} if k == "count" else {"type": "string"}) for k in DESCRIBE_FIELDS},
                  "required": list(DESCRIBE_FIELDS)}
        t0 = time.monotonic()
        try:
            h, w = img.shape[:2]
            if max(h, w) > 1024:
                s = 1024 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            d = parse_json(self._chat(DESCRIBE_PROMPT, [base64.b64encode(buf.tobytes()).decode()],
                                      timeout=self.image_timeout, schema=schema))
        except Exception:
            d = {}
        fields = {k: str(d.get(k, "")).strip()[:60] for k in DESCRIBE_FIELDS if str(d.get(k, "")).strip()}
        return {"fields": fields, "seconds": round(time.monotonic() - t0, 1), "ok": bool(fields)}

    def submit_describe(self, img: np.ndarray) -> Future:
        return self._pool.submit(self.describe, img)

    def submit_ball(self, img: np.ndarray) -> Future:
        return self._pool.submit(self.check_ball, img)

    def submit_found(self, img: np.ndarray, target: str) -> Future:
        return self._pool.submit(self.check_found, img, target)

    def check_part(self, img: np.ndarray) -> dict:
        """(vlm_eval.py 용) 사진 → {"label": red/blue/green/None, "seen", "seconds", "ok"}"""
        self.stats["check_calls"] += 1
        names = [LABEL_KO[l] for l in SUPPORTED]
        options = "\n".join(f"- {LABEL_KO[l]}: {PART_CUES[l]}" for l in SUPPORTED)
        t0 = time.monotonic()
        try:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            raw = self._chat(CHECK_PROMPT.format(options=options, names=" / ".join(names)),
                             [base64.b64encode(buf.tobytes()).decode()], timeout=self.image_timeout)
            d = parse_json(raw)
        except Exception as exc:
            d, raw = {}, f"오류: {exc.__class__.__name__}"
        got = _norm(d.get("name", ""))
        label = next((KO_TO_LABEL[n] for n in names if _norm(n) == got), None)
        return {"label": label, "seen": str(d.get("seen", ""))[:60] if d else raw[:60],
                "seconds": round(time.monotonic() - t0, 1), "ok": bool(d)}

    def submit_check(self, img: np.ndarray) -> Future:
        return self._pool.submit(self.check_part, img)
