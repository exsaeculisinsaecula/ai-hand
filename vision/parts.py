"""로컬 부품 판별기 — 볼트 / 너트 / 와셔 (인터넷·API 불필요).

1. 윤곽선 모양 특징: 길쭉함(볼트), 구멍 + 육각 꼭짓점(너트), 구멍 + 원형(와셔).
2. 학습: 대시보드에서 부품별 사진을 찍으면 data/parts/<label>/*.jpg 에 저장되고,
   같은 특징 벡터로 k-최근접 분류기를 만든다. 학습 샘플이 있으면 kNN을 우선 사용.
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

LABELS = ("bolt", "nut", "washer")          # 로컬 모양 규칙이 아는 종류
YOLO_LABELS = ("bolt", "nut")                # 팀 YOLO(best_two.pt)가 아는 종류
SUPPORTED = YOLO_LABELS                      # 요청 가능한 부품 (와셔는 시연에서 제외)
LABEL_KO = {"bolt": "볼트", "nut": "너트", "washer": "와셔", "none": "없음"}
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "parts"


@dataclass
class PartResult:
    label: str               # bolt | nut | washer | none
    confidence: float
    reason: str
    box: tuple[int, int, int, int] | None = None
    features: list[float] | None = None

    def to_dict(self) -> dict:
        return {"label": self.label, "label_ko": LABEL_KO[self.label], "confidence": round(self.confidence, 2),
                "reason": self.reason, "box": list(self.box) if self.box else None}


WORK_WIDTH = 640  # 판별은 이 폭으로 줄여서 (1080p 웹캠도 가볍게)


def _prep(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = WORK_WIDTH / w if w > WORK_WIDTH else 1.0
    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def _main_contour(frame: np.ndarray, roi=None, background: np.ndarray | None = None):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h)
    if roi:
        x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    crop = frame[y0:y1, x0:x1]
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    if background is not None and background.shape == frame.shape:
        # 배경 차이: 빈 화면과 달라진 부분만 부품 후보 (구멍은 배경이 비쳐 구멍으로 남음)
        bg = cv2.GaussianBlur(cv2.cvtColor(background[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mask = (cv2.absdiff(gray, bg) > 28).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        source = mask
    else:
        edges = cv2.Canny(gray, 40, 120)
        source = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, hier = cv2.findContours(source, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    area_min = 0.004 * crop.shape[0] * crop.shape[1]
    best = None
    for i, c in enumerate(contours):
        if hier[0][i][3] != -1:
            continue  # 바깥 윤곽만
        a = cv2.contourArea(c)
        if a > area_min and (best is None or a > best[1]):
            best = (i, a)
    if best is None:
        return None
    idx, area = best
    outer = contours[idx]
    hole = 0.0
    child = hier[0][idx][2]
    while child != -1:
        hole = max(hole, cv2.contourArea(contours[child]))
        child = hier[0][child][0]
    return outer, area, hole, (x0, y0)


def extract_features(frame: np.ndarray, roi=None, background: np.ndarray | None = None):
    frame, scale = _prep(frame)
    if background is not None:
        background, _ = _prep(background)
    found = _main_contour(frame, roi, background)
    if found is None:
        return None, None
    outer, area, hole, (ox, oy) = found
    (cw, ch) = cv2.minAreaRect(outer)[1]
    elong = max(cw, ch) / max(1.0, min(cw, ch))
    peri = cv2.arcLength(outer, True)
    circ = 4 * math.pi * area / max(1.0, peri * peri)
    hull = cv2.convexHull(outer)
    solidity = area / max(1.0, cv2.contourArea(hull))
    verts = len(cv2.approxPolyDP(hull, 0.035 * cv2.arcLength(hull, True), True))
    hole_ratio = hole / max(1.0, area)
    x, y, w, h = cv2.boundingRect(outer)
    feats = [min(elong, 8.0) / 8.0, circ, solidity, min(verts, 12) / 12.0, min(hole_ratio * 3, 1.0)]
    inv = 1.0 / scale
    return feats, (int((x + ox) * inv), int((y + oy) * inv), int(w * inv), int(h * inv))


def rule_classify(f: list[float]) -> tuple[str, float, str]:
    elong, circ, solidity, verts, hole = f[0] * 8, f[1], f[2], f[3] * 12, f[4] / 3
    if elong >= 2.2:
        return "bolt", min(0.95, 0.6 + (elong - 2.2) * 0.1), f"길쭉한 모양 (가로세로비 {elong:.1f})"
    if hole >= 0.04:
        if 5 <= verts <= 7 and circ < 0.86:
            return "nut", 0.8, f"가운데 구멍 + 육각 꼭짓점 {verts:.0f}개"
        return "washer", 0.8, f"가운데 구멍 + 둥근 외곽 (원형도 {circ:.2f})"
    if 5 <= verts <= 7:
        return "nut", 0.55, "육각 외곽 (구멍은 안 보임)"
    if circ > 0.8:
        return "washer", 0.5, "둥근 외곽 (구멍은 안 보임)"
    return "bolt", 0.45, "모양이 애매함"


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
            return PartResult("none", 0.0, "부품이 보이지 않아요")
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
        return PartResult(label, conf, why + " (학습 전: 모양 규칙)", box, f)


class YoloParts:
    """팀이 학습한 YOLO(best_two.pt: 0=bolt, 1=nut)로 손 영역 안의 부품을 판별."""

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
            return PartResult("none", 0.0, "손 영역에 부품이 없어요" if roi else "부품이 보이지 않아요")
        best = max(inside, key=lambda d: d["conf"])
        extra = f" · 같은 영역에 {len(inside)}개" if len(inside) > 1 else ""
        return PartResult(best["label"], best["conf"], f"YOLO {self.weights.name}{extra}", best["box"])


# ---- 타이핑 해석 (로컬 규칙) ----
_WORDS = {
    "bolt": ("볼트", "bolt", "나사", "스크류", "screw"),
    "nut": ("너트", "nut", "넛"),
    "washer": ("와셔", "washer", "워셔", "와샤", "링"),
}


def parse_request(text: str) -> tuple[str | None, str]:
    t = re.sub(r"\s+", "", str(text).lower())
    if not t:
        return None, "받을 부품을 입력해 주세요 (볼트 / 너트)."
    neg = re.search(r"(말고|빼고|아니고|아닌거|제외)", t)
    head, tail = (t[: neg.start()], t[neg.end():]) if neg else ("", t)
    hit = [l for l, ws in _WORDS.items() if any(w in tail for w in ws)]
    excluded = [l for l, ws in _WORDS.items() if any(w in head for w in ws)]
    hit = [l for l in hit if l not in excluded]
    if hit == ["washer"]:
        return None, "와셔는 지금 모델(볼트·너트)이 인식하지 못해요. 볼트나 너트를 요청해 주세요."
    hit = [l for l in hit if l in SUPPORTED]
    excluded = [l for l in excluded if l in SUPPORTED]
    if len(hit) == 1:
        return hit[0], f"{LABEL_KO[hit[0]]}을(를) 받을게요. 로봇 손에 쥐어 주세요."
    if len(hit) > 1:
        return None, "한 번에 한 종류만 말해 주세요."
    if excluded and not hit:
        rest = [l for l in SUPPORTED if l not in excluded]
        if len(rest) == 1:
            return rest[0], f"{LABEL_KO[rest[0]]}을(를) 받을게요. 로봇 손에 쥐어 주세요."
        return None, f"{' / '.join(LABEL_KO[l] for l in rest)} 중 무엇인지 알려 주세요."
    return None, "볼트나 너트 중에서 말해 주세요."
