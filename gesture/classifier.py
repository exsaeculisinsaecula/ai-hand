"""MediaPipe Hands 21점 랜드마크 기반 규칙 분류기 (1 / 2 / 3 / PALM / NONE)."""
from dataclasses import dataclass
import math
from typing import Sequence

GESTURE_1 = "1"
GESTURE_2 = "2"
GESTURE_3 = "3"
GESTURE_PALM = "PALM"
GESTURE_NONE = "NONE"

# (MCP, PIP, TIP) 인덱스
_FINGERS = (
    (5, 6, 8),    # 검지
    (9, 10, 12),  # 중지
    (13, 14, 16), # 약지
    (17, 18, 20), # 소지
)


@dataclass(frozen=True)
class GesturePrediction:
    label: str
    confidence: float
    fingers: tuple[bool, bool, bool, bool, bool] = (False, False, False, False, False)


def _xy(lm) -> tuple[float, float]:
    if hasattr(lm, "x") and hasattr(lm, "y"):
        return float(lm.x), float(lm.y)
    return float(lm[0]), float(lm[1])


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def classify_landmarks(landmarks: Sequence, hand_score: float = 1.0) -> GesturePrediction:
    """21개 손 랜드마크에서 손짓 클래스(1/2/3/PALM/NONE)와 신뢰도를 계산한다."""
    if len(landmarks) != 21:
        return GesturePrediction(label=GESTURE_NONE, confidence=0.0)

    pts = [_xy(lm) for lm in landmarks]
    wrist = pts[0]

    ratios: list[float] = []
    states: list[bool] = []
    for _, pip_idx, tip_idx in _FINGERS:
        d_pip = max(_dist(wrist, pts[pip_idx]), 1e-6)
        d_tip = _dist(wrist, pts[tip_idx])
        ratio = d_tip / d_pip
        ratios.append(ratio)
        states.append(ratio > 1.15)

    # 엄지 판별: 소지 MCP(17) 기준 거리 및 검지 MCP(5)로부터의 벌림 정도
    d_thumb_ip = max(_dist(pts[17], pts[3]), 1e-6)
    d_thumb_tip = _dist(pts[17], pts[4])
    palm_ref = max(_dist(wrist, pts[5]), 1e-6)
    thumb_spread = _dist(pts[5], pts[4]) / palm_ref
    thumb_open = (d_thumb_tip / d_thumb_ip) > 1.12 and thumb_spread > 0.45

    index_on, middle_on, ring_on, pinky_on = states
    fingers_tuple = (thumb_open, index_on, middle_on, ring_on, pinky_on)

    if (index_on, middle_on, ring_on, pinky_on) == (True, False, False, False):
        label = GESTURE_1
    elif (index_on, middle_on, ring_on, pinky_on) == (True, True, False, False):
        label = GESTURE_2
    elif (index_on, middle_on, ring_on, pinky_on) in {(True, True, True, False), (False, True, True, True)}:
        label = GESTURE_3
    elif thumb_open and index_on and middle_on and ring_on and pinky_on:
        label = GESTURE_PALM
    else:
        return GesturePrediction(label=GESTURE_NONE, confidence=round(0.35 * hand_score, 3), fingers=fingers_tuple)

    # 임계값(1.15)으로부터의 여유폭으로 규칙 확신도 산출
    margins = [min(abs(r - 1.15) / 0.35, 1.0) for r in ratios]
    rule_conf = 0.70 + 0.30 * (sum(margins) / len(margins))
    confidence = round(max(0.0, min(1.0, rule_conf * hand_score)), 3)
    return GesturePrediction(label=label, confidence=confidence, fingers=fingers_tuple)
