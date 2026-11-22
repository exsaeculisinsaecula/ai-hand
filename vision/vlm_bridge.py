"""VLM 연결 (로컬 Ollama · Gemma 3 4B) — 요청 해석 보조 + 쥐기 전 교차 확인.

역할
  1. 요청 해석: 작업자가 입력한 문장을 VLM 이 해석한다 (글자만, 사진 없음, 1~3초).
     VLM 이 응답하지 않을 때만 기존 단어 규칙(parse_request)으로 대신한다.
  2. 교차 확인: YOLO 점수가 낮을 때(기본 0.70 미만)만 손 영역 사진을 VLM 에 보여 주고
     볼트/너트를 따로 판단하게 한다. 다르면 쥐지 않고 작업자 확인을 기다린다.

VLM 이 꺼져 있거나 응답이 없으면 규칙·YOLO 결과로 그대로 진행한다 (시연이 멈추지 않도록).
단, 짝 관계 표현은 VLM 없이 규칙만으로 정하면 반대 부품이 될 수 있어 다시 묻는다.

  ollama pull gemma3:4b
  python3 handover_app.py --mock --vlm
"""
from __future__ import annotations

import base64
from concurrent.futures import Future, ThreadPoolExecutor
import json
import re
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

from vision.parts import LABEL_KO, SUPPORTED, parse_request

KO_TO_LABEL = {v: k for k, v in LABEL_KO.items() if k in SUPPORTED}      # 볼트→bolt, 너트→nut

PART_CUES = {
    "bolt": "길쭉한 막대 모양, 한쪽 끝에 머리, 몸통에 나사산",
    "nut": "두껍고 바깥 테두리가 육각형, 가운데 둥근 구멍",
}

INTERPRET_PROMPT = """너는 부품 전달 로봇의 요청 해석기다. 작업자가 말한 요청이 아래 부품 중 무엇을 원하는지 고른다.
부품 목록:
{options}

요청: "{text}"

규칙
- 요청은 한국어 구어체다. 다른 이름·별명·모양 설명·오타·띄어쓰기 없는 말도 해석한다.
- 부품 목록의 모양 설명을 기준으로, 요청이 설명하는 모양이나 쓰임에 맞는 부품을 고른다.
- "A 말고 B", "A 아니고 B"처럼 부정이 있으면 원하는 쪽(B)을 고른다.
- "A에 끼우는/꽂는/돌려 끼우는 것"은 A가 아니라 A와 짝이 되는 부품이다.
- 목록에 없는 물건, 두 종류를 동시에 요구, 부품과 무관한 말, 어느 쪽인지 정할 수 없으면 "모름".
JSON 한 줄로만 답한다: {{"part": "{names} / 모름 중 하나", "reason": "짧은 근거"}}"""

CHECK_PROMPT = """사진은 로봇 손바닥 위를 위에서 찍은 것이다. 투명한 캡슐 안이나 손 위에 금속 부품이 하나 있다.
부품은 다음 중 하나다.
{options}
보이는 모양만 근거로 판단하고, 정말 보이지 않으면 "모름"이라고 한다.
JSON 한 줄로만 답한다: {{"seen": "보이는 모양 한 줄", "name": "{names} / 모름 중 하나"}}"""

PARTNER_RE = re.compile(r"(볼트|볼뜨|너트|넛트|넛|나사|bolt|nut)[가-힣a-z]{0,3}에(다|다가)?(끼우|꽂|넣|조이|돌려|체결|맞추|맞는)")


def _norm(t: str) -> str:
    return re.sub(r"\s+", "", str(t or "")).lower()


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


class VlmBridge:
    def __init__(self, model: str = "gemma3:4b", url: str = "http://localhost:11434",
                 text_timeout: float = 20.0, image_timeout: float = 60.0, verify_below: float = 0.70):
        self.model = model
        self.url = url.rstrip("/").removesuffix("/v1")
        self.text_timeout = text_timeout
        self.image_timeout = image_timeout
        self.verify_below = verify_below            # YOLO 점수가 이보다 낮을 때만 교차 확인 (0 이면 끔)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._no_think = True
        self.stats = {"interpret_calls": 0, "interpret_changed": 0, "check_calls": 0,
                      "check_agree": 0, "check_disagree": 0, "check_fail": 0}

    # ---- Ollama 호출 ----
    def _chat(self, prompt: str, images: list[str] | None = None, timeout: float = 20.0) -> str:
        msg = {"role": "user", "content": prompt}
        if images:
            msg["images"] = images
        body = {"model": self.model, "stream": False, "messages": [msg], "options": {"temperature": 0.1}}
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
                return self._chat(prompt, images, timeout)
            raise RuntimeError(f"Ollama {e.code}: {text[:120]}")

    def warmup(self) -> None:
        """처음 한 번 모델을 메모리에 올려 둔다 (첫 요청이 느리지 않도록). 백그라운드."""
        def _w():
            try:
                self._chat("한 단어로 답해: 준비", timeout=120)
            except Exception:
                pass
        self._pool.submit(_w)

    # ---- 1. 요청 해석 ----
    def interpret(self, text: str) -> tuple[str | None, str, dict]:
        """→ (bolt/nut/None, 작업자에게 보여 줄 답, 정보 {method, reason, seconds})"""
        rule_t, rule_reply = parse_request(text)
        t = _norm(text)
        partner = bool(PARTNER_RE.search(t))
        if not t:
            return None, rule_reply, {"method": "규칙", "reason": "빈 입력", "seconds": 0.0}

        self.stats["interpret_calls"] += 1
        names = [LABEL_KO[l] for l in SUPPORTED]
        options = "\n".join(f"- {LABEL_KO[l]}: {PART_CUES[l]}" for l in SUPPORTED)
        t0 = time.monotonic()
        try:
            raw = self._chat(INTERPRET_PROMPT.format(options=options, text=text, names=" / ".join(names)),
                             timeout=self.text_timeout)
            d = parse_json(raw)
        except Exception as exc:
            d, raw = {}, f"오류: {exc.__class__.__name__}"
        dt = round(time.monotonic() - t0, 2)
        got = _norm(d.get("part", ""))
        label = next((KO_TO_LABEL[n] for n in names if _norm(n) == got), None) or \
            next((KO_TO_LABEL[n] for n in names if got and _norm(n) in got), None)
        reason = str(d.get("reason", "")) if d else f"VLM 응답 없음 ({raw[:40]})"

        if not d:                                       # VLM 실패
            if partner:
                return None, "어느 부품인지 확실하지 않아요. '볼트' 또는 '너트'로 말해 주세요.", \
                    {"method": "규칙(VLM 실패)", "reason": reason, "seconds": dt}
            return rule_t, rule_reply, {"method": "규칙(VLM 실패)", "reason": reason, "seconds": dt}
        if label is None:
            msg = rule_reply if "와셔" in rule_reply else "볼트나 너트 중에서 말해 주세요."
            return None, msg, {"method": "VLM", "reason": reason, "seconds": dt}
        if label != rule_t:
            self.stats["interpret_changed"] += 1
        return label, f"{LABEL_KO[label]}을(를) 받을게요. 로봇 손에 쥐어 주세요. (VLM 해석)", \
            {"method": "VLM", "reason": reason, "seconds": dt}

    # ---- 2. 교차 확인 ----
    def should_check(self, conf: float) -> bool:
        return self.verify_below > 0 and conf < self.verify_below

    @staticmethod
    def crop(frame: np.ndarray, box, margin: float = 0.6, side: int = 672) -> np.ndarray | None:
        """YOLO 상자(x, y, w, h) 둘레를 넉넉히 잘라 확대"""
        if frame is None or not box:
            return None
        x, y, w, h = box
        cx, cy, half = x + w / 2, y + h / 2, max(w, h) * (0.5 + margin / 2)
        H, W = frame.shape[:2]
        c = frame[int(max(0, cy - half)):int(min(H, cy + half)), int(max(0, cx - half)):int(min(W, cx + half))]
        return cv2.resize(c, (side, side), interpolation=cv2.INTER_CUBIC) if c.size else None

    def check_part(self, img: np.ndarray) -> dict:
        """→ {"label": bolt/nut/None, "seen", "seconds", "ok"}"""
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
