"""로컬 부품 판별기 — 빨간 공 / 파란 공 / 초록 공 (인터넷·API 불필요).

VLM 이 켜져 있으면 색 이름은 팀원 ask_vlm 이 정한다 (vision/vlm_bridge.py ask_team_color).
여기의 색 규칙은 VLM 이 꺼졌거나 답이 없을 때, 그리고 공 위치를 찾을 때만 쓴다.
1. 색 규칙: 손 영역에서 채도가 높은 가장 큰 덩어리의 색상(Hue)으로 빨강·파랑·초록을 고른다.
2. 학습: 대시보드에서 공 색별 사진을 찍으면 data/parts/<label>/*.jpg 에 저장되고,
   같은 특징 벡터(색상·채도·밝기)로 k-최근접 분류기를 만든다. 학습 샘플이 있으면 kNN을 우선 사용.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time

import cv2
import numpy as np

LABELS = ("red", "blue", "green")            # 받을 공 3종
YOLO_LABELS = LABELS                         # (공 YOLO 모델이 생기면) 이 이름으로 학습할 것
SUPPORTED = LABELS                           # 요청 가능한 공
LABEL_KO = {"red": "빨간 공", "blue": "파란 공", "green": "초록 공", "object": "물체", "none": "없음"}
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "parts"


@dataclass
class PartResult:
    label: str               # red | blue | green | none
    confidence: float
    reason: str
    box: tuple[int, int, int, int] | None = None
    features: list[float] | None = None

    def to_dict(self) -> dict:
        return {"label": self.label, "label_ko": LABEL_KO[self.label], "confidence": round(self.confidence, 2),
                "reason": self.reason, "box": list(self.box) if self.box else None}


WORK_WIDTH = 640  # 판별은 이 폭으로 줄여서 (1080p 웹캠도 가볍게)
SAT_MIN = 110     # 이보다 채도가 낮으면 공이 아닌 배경·피부로 봄 (OpenCV S 0~255)
VAL_MIN = 50


def _prep(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = WORK_WIDTH / w if w > WORK_WIDTH else 1.0
    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def _color_blob(frame: np.ndarray, roi=None, background: np.ndarray | None = None):
    """채도가 높은 가장 큰 덩어리 → (윤곽, 마스크, 원점)"""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h)
    if roi:
        x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] >= SAT_MIN) & (hsv[..., 2] >= VAL_MIN)).astype(np.uint8) * 255
    if background is not None and background.shape == frame.shape:
        # 배경 차이: 빈 화면과 달라진 부분만 공 후보
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(background[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        mask &= ((cv2.absdiff(g, bg) > 28).astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = 0.004 * crop.shape[0] * crop.shape[1]
    best = max(contours, key=cv2.contourArea, default=None)
    if best is None or cv2.contourArea(best) < area_min:
        return None
    blob = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(blob, [best], -1, 255, -1)
    blob &= mask
    return best, hsv, blob, (x0, y0)


def extract_features(frame: np.ndarray, roi=None, background: np.ndarray | None = None):
    """→ ([cos색상, sin색상, 채도, 밝기] (0~1), 상자) 또는 (None, None)"""
    frame, scale = _prep(frame)
    if background is not None:
        background, _ = _prep(background)
    found = _color_blob(frame, roi, background)
    if found is None:
        return None, None
    outer, hsv, blob, (ox, oy) = found
    sel = blob > 0
    hue = hsv[..., 0][sel].astype(np.float32) * (2 * math.pi / 180.0)     # OpenCV H 0~179 → 라디안
    c, s = float(np.cos(hue).mean()), float(np.sin(hue).mean())
    feats = [(c + 1) / 2, (s + 1) / 2, float(hsv[..., 1][sel].mean()) / 255, float(hsv[..., 2][sel].mean()) / 255]
    x, y, w, h = cv2.boundingRect(outer)
    inv = 1.0 / scale
    return feats, (int((x + ox) * inv), int((y + oy) * inv), int(w * inv), int(h * inv))


def hue_deg(f: list[float]) -> float:
    return math.degrees(math.atan2(f[1] * 2 - 1, f[0] * 2 - 1)) % 360


def rule_classify(f: list[float]) -> tuple[str, float, str]:
    h, sat = hue_deg(f), f[2]
    conf = min(0.9, 0.5 + sat * 0.4)
    if h < 25 or h >= 320:
        return "red", conf, f"빨간색 (색상 {h:.0f}°)"
    if 75 <= h < 170:
        return "green", conf, f"초록색 (색상 {h:.0f}°)"
    if 180 <= h < 270:
        return "blue", conf, f"파란색 (색상 {h:.0f}°)"
    # 경계 색: 가장 가까운 색으로, 낮은 신뢰도
    centers = {"red": 0, "green": 120, "blue": 220}
    lbl = min(centers, key=lambda k: min(abs(h - centers[k]), 360 - abs(h - centers[k])))
    return lbl, 0.4, f"색이 애매함 (색상 {h:.0f}°)"


def find_balls(frame: np.ndarray, roi=None, exclude=None, min_area: float = 0.0015,
               max_balls: int = 12) -> list[dict]:
    """쟁반용: 화면(또는 roi) 안의 색 공을 모두 찾는다 → [{"label", "conf", "box"(x,y,w,h 원본 좌표)}]

    채도 높은 덩어리마다 평균 색상으로 빨강/파랑/초록을 고른다 (rule_classify 와 같은 기준).
    exclude: 빼고 볼 영역(정규화 x0,y0,x1,y1) — 손 영역 안의 공은 쟁반 공이 아님.
    너무 길쭉하거나 속이 빈 덩어리(공이 아닌 띠·테두리)는 버린다.
    """
    img, scale = _prep(frame)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h)
    if roi:
        x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] >= SAT_MIN) & (hsv[..., 2] >= VAL_MIN)).astype(np.uint8) * 255
    if exclude:
        ex = (int(exclude[0] * w) - x0, int(exclude[1] * h) - y0, int(exclude[2] * w) - x0, int(exclude[3] * h) - y0)
        mask[max(0, ex[1]):max(0, ex[3]), max(0, ex[0]):max(0, ex[2])] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = min_area * w * h
    inv = 1.0 / scale
    out = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[: max_balls * 3]:
        area = cv2.contourArea(c)
        if area < area_min:
            break
        bx, by, bw, bh = cv2.boundingRect(c)
        if max(bw, bh) / max(1, min(bw, bh)) > 2.2 or area / max(1, bw * bh) < 0.45:
            continue                                    # 길쭉한 띠·속 빈 테두리 → 공 아님
        blob = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(blob, [c], -1, 255, -1)
        sel = (blob > 0) & (mask > 0)
        hue = hsv[..., 0][sel].astype(np.float32) * (2 * math.pi / 180.0)
        cs, sn = float(np.cos(hue).mean()), float(np.sin(hue).mean())
        f = [(cs + 1) / 2, (sn + 1) / 2, float(hsv[..., 1][sel].mean()) / 255, float(hsv[..., 2][sel].mean()) / 255]
        label, conf, _why = rule_classify(f)
        out.append({"label": label, "conf": round(conf, 2),
                    "box": (int((bx + x0) * inv), int((by + y0) * inv), int(bw * inv), int(bh * inv))})
        if len(out) >= max_balls:
            break
    return out


def hand_zone_from_motion(f_open: np.ndarray, f_grip: np.ndarray, f_open2: np.ndarray, thr: int = 30):
    """로봇 손을 폈을 때·쥐었을 때 사진의 차이 = 손가락이 움직인 자리 → 그 가운데를 '공을 올려 놓을 자리'로.
    다시 폈을 때 사진과도 비교해 그 사이 화면이 바뀐(사람이 지나간) 곳은 뺀다.
    → ((x0, y0, x1, y1) 정규화, (bx, by, bw, bh) 움직인 범위 픽셀) 또는 None"""
    if f_open is None or f_grip is None or f_open2 is None or f_open.shape != f_grip.shape or f_open.shape != f_open2.shape:
        return None
    a, b, c = (cv2.GaussianBlur(x, (5, 5), 0) for x in (f_open, f_grip, f_open2))
    moved = cv2.absdiff(a, b).max(axis=2) > thr
    still = cv2.absdiff(a, c).max(axis=2) <= thr
    mask = (moved & still).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    if n <= 1:
        return None
    i = max(range(1, n), key=lambda k: stats[k, cv2.CC_STAT_AREA])
    bx, by, bw, bh, area = stats[i]
    if area < 0.0015 * w * h:
        return None
    ys, xs = np.nonzero(lab == i)
    cx, cy = float(xs.mean()), float(ys.mean())           # 손가락이 모이는 쪽 = 움직임의 무게중심
    zw, zh = max(bw * 0.7, w * 0.08), max(bh * 0.7, h * 0.08)
    x0, y0 = max(0.0, (cx - zw / 2) / w), max(0.0, (cy - zh / 2) / h)
    x1, y1 = min(1.0, (cx + zw / 2) / w), min(1.0, (cy + zh / 2) / h)
    return (round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)), (int(bx), int(by), int(bw), int(bh))


def detect_change(frame: np.ndarray, roi, background: np.ndarray, min_frac: float = 0.025) -> PartResult:
    """색과 상관없이 '빈 손 배경과 달라진 덩어리'가 있는지만 본다 (VLM 이 판정하기 전 단계).
    → PartResult("object", 비율 기반 점수, box=원본 좌표) 또는 PartResult("none")"""
    img, scale = _prep(frame)
    bg, _ = _prep(background)
    if bg.shape != img.shape:
        return PartResult("none", 0.0, "배경 없음")
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)) if roi else (0, 0, w, h)
    a = cv2.GaussianBlur(img[y0:y1, x0:x1], (5, 5), 0)
    b = cv2.GaussianBlur(bg[y0:y1, x0:x1], (5, 5), 0)
    if a.size == 0:
        return PartResult("none", 0.0, "손 영역 없음")
    diff = cv2.absdiff(a, b).max(axis=2)                       # 밝기·색 어느 쪽이 달라져도
    mask = (diff > 32).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = max(contours, key=cv2.contourArea, default=None)
    area = (x1 - x0) * (y1 - y0)
    frac = cv2.contourArea(best) / max(1, area) if best is not None else 0.0
    if frac < min_frac:
        return PartResult("none", 0.0, "손 영역에 놓인 것이 없습니다")
    bx, by, bw, bh = cv2.boundingRect(best)
    inv = 1.0 / scale
    return PartResult("object", min(0.95, 0.5 + frac * 2), f"손 영역에 무언가 놓임 ({frac:.0%})",
                      (int((bx + x0) * inv), int((by + y0) * inv), int(bw * inv), int(bh * inv)))


class PartClassifier:
    def __init__(self, data_dir: Path = DATA_DIR, k: int = 3):
        self.data_dir = Path(data_dir)
        self.k = k
        self.samples: list[tuple[list[float], str]] = []
        self.reload()

    def reload(self) -> None:
        self.samples = []
        for label in LABELS:
            for p in sorted((self.data_dir / label).glob("*.jpg")):
                fj = p.with_suffix(".json")
                if fj.exists():
                    f = json.loads(fj.read_text(encoding="utf-8"))
                else:
                    img = cv2.imread(str(p))
                    f = extract_features(img)[0] if img is not None else None
                if f:
                    self.samples.append((f, label))

    def counts(self) -> dict[str, int]:
        return {l: sum(1 for _, s in self.samples if s == l) for l in LABELS}

    def add_sample(self, frame: np.ndarray, label: str, background: np.ndarray | None = None) -> bool:
        if label not in LABELS:
            raise ValueError(label)
        f, _ = extract_features(frame, background=background)
        if not f:
            return False
        d = self.data_dir / label
        d.mkdir(parents=True, exist_ok=True)
        stem = d / f"{int(time.time() * 1000)}"
        cv2.imwrite(str(stem.with_suffix(".jpg")), frame)
        stem.with_suffix(".json").write_text(json.dumps(f), encoding="utf-8")
        self.samples.append((f, label))
        return True

    def classify(self, frame: np.ndarray, roi=None, background: np.ndarray | None = None) -> PartResult:
        f, box = extract_features(frame, roi, background)
        if not f:
            return PartResult("none", 0.0, "공이 보이지 않습니다")
        counts = self.counts()
        if all(counts[l] >= 2 for l in LABELS):
            dists = sorted((math.dist(f, sf), sl) for sf, sl in self.samples)[: self.k]
            votes: dict[str, float] = {}
            for d, l in dists:
                votes[l] = votes.get(l, 0.0) + 1.0 / (d + 1e-3)
            label = max(votes, key=votes.get)
            conf = votes[label] / sum(votes.values())
            return PartResult(label, conf, f"학습 샘플 {len(self.samples)}장 기준 최근접 {self.k}개", box, f)
        label, conf, why = rule_classify(f)
        return PartResult(label, conf, why + " (학습 전: 색 규칙)", box, f)


class YoloParts:
    """YOLO 로 손 영역 안의 공을 판별 (클래스 이름이 red/blue/green 인 모델만 쓸모 있음)."""

    def __init__(self, weights: Path, imgsz: int = 640, conf: float = 0.35):
        from ultralytics import YOLO          # 무거운 의존성: 실제로 쓸 때만 import
        self.weights = Path(weights)
        self.model = YOLO(str(self.weights))
        self.names = {int(k): str(v).lower() for k, v in dict(self.model.names).items()}
        self.imgsz, self.conf = imgsz, conf
        self.last_ms = 0.0

    def detect(self, frame: np.ndarray) -> list[dict]:
        t0 = time.perf_counter()
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=0.45,
                               agnostic_nms=True, verbose=False)[0]
        self.last_ms = (time.perf_counter() - t0) * 1000
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append({"label": self.names.get(int(b.cls), str(int(b.cls))), "conf": float(b.conf),
                        "box": (x1, y1, x2 - x1, y2 - y1)})
        return out

    def classify(self, frame: np.ndarray, roi=None, dets: list[dict] | None = None) -> PartResult:
        """roi(정규화) 안에 중심이 있는 탐지 중 신뢰도가 가장 높은 것."""
        dets = self.detect(frame) if dets is None else dets
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = roi if roi else (0, 0, 1, 1)
        inside = []
        for d in dets:
            bx, by, bw, bh = d["box"]
            cx, cy = (bx + bw / 2) / w, (by + bh / 2) / h
            if x0 <= cx <= x1 and y0 <= cy <= y1 and d["label"] in YOLO_LABELS:
                inside.append(d)
        if not inside:
            return PartResult("none", 0.0, "손 영역에 공이 없습니다" if roi else "공이 보이지 않습니다")
        best = max(inside, key=lambda d: d["conf"])
        extra = f" · 같은 영역에 {len(inside)}개" if len(inside) > 1 else ""
        return PartResult(best["label"], best["conf"], f"YOLO {self.weights.name}{extra}", best["box"])


# ---- 타이핑 해석 (로컬 규칙) ----
_WORDS = {
    "red": ("빨간", "빨강", "빨갛", "붉은", "적색", "레드", "red"),
    "blue": ("파란", "파랑", "파랗", "푸른", "청색", "블루", "blue"),
    "green": ("초록", "녹색", "그린", "green"),
}
ASK_AGAIN = "빨간 공 / 파란 공 / 초록 공 중에서 말해 주십시오."


def mentioned_colors(text: str) -> list[str]:
    """부정("말고/빼고") 뒤쪽에 직접 나온 색 단어들 → ["red", "blue"] 처럼"""
    t = re.sub(r"\s+", "", str(text).lower())
    neg = re.search(r"(말고|빼고|아니고|아닌거|제외)", t)
    tail = t[neg.end():] if neg else t
    return [l for l, ws in _WORDS.items() if any(w in tail for w in ws)]


def parse_request(text: str) -> tuple[str | None, str]:
    t = re.sub(r"\s+", "", str(text).lower())
    if not t:
        return None, "받을 공을 입력해 주십시오 (빨간 공 / 파란 공 / 초록 공)."
    neg = re.search(r"(말고|빼고|아니고|아닌거|제외)", t)
    head, tail = (t[: neg.start()], t[neg.end():]) if neg else ("", t)
    hit = [l for l, ws in _WORDS.items() if any(w in tail for w in ws)]
    excluded = [l for l, ws in _WORDS.items() if any(w in head for w in ws)]
    hit = [l for l in hit if l not in excluded]
    if len(hit) == 1:
        return hit[0], f"{LABEL_KO[hit[0]]}을 받겠습니다. 로봇 손에 쥐어 주십시오."
    if len(hit) > 1:
        return None, "한 번에 한 가지 색만 말해 주십시오."
    if excluded:
        rest = [l for l in SUPPORTED if l not in excluded]
        if len(rest) == 1:
            return rest[0], f"{LABEL_KO[rest[0]]}을 받겠습니다. 로봇 손에 쥐어 주십시오."
        return None, f"{' / '.join(LABEL_KO[l] for l in rest)} 중 무엇인지 알려 주십시오."
    return None, ASK_AGAIN
