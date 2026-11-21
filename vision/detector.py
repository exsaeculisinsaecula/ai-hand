"""YOLO(ultralytics) 추론 래퍼 — 검출 결과를 부품/손짓 판정으로 요약한다.

ultralytics 는 선택 의존성이다. 설치되지 않았거나 모델 파일이 없으면
`YoloDetector.available == False` 이고 `error` 에 사유가 담기며, 런타임은
기존 MediaPipe 손짓 인식으로 자동 대체된다.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
import threading

import numpy as np

from config import YOLO_CLASSES_PATH, YOLO_MODEL_PATH
from gesture.classifier import GESTURE_NONE, GESTURE_PALM
from .classes import KIND_GESTURE, KIND_PART, ClassMap

PART_COLORS_BGR = {1: (60, 76, 231), 2: (219, 152, 52), 3: (113, 204, 46)}
GESTURE_COLOR_BGR = (0, 165, 255)
OTHER_COLOR_BGR = (160, 160, 160)


@dataclass(frozen=True)
class Detection:
    name: str
    kind: str
    value: int | str | None
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "value": self.value,
            "conf": round(self.conf, 3),
            "box": list(self.box),
        }


@dataclass
class VisionSummary:
    gesture: str = GESTURE_NONE
    gesture_conf: float = 0.0
    part_counts: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    part_best_conf: dict[int, float] = field(default_factory=dict)

    @property
    def parts_visible(self) -> int:
        return sum(self.part_counts.values())


def summarize(detections: list[Detection], min_conf: float = 0.0) -> VisionSummary:
    """검출 목록 → 대표 손짓(최고 신뢰도) + 부품별 개수."""
    out = VisionSummary()
    for det in detections:
        if det.conf < min_conf:
            continue
        if det.kind == KIND_GESTURE and det.conf > out.gesture_conf:
            out.gesture = str(det.value)
            out.gesture_conf = det.conf
        elif det.kind == KIND_PART and det.value in (1, 2, 3):
            pid = int(det.value)
            out.part_counts[pid] = out.part_counts.get(pid, 0) + 1
            out.part_best_conf[pid] = max(out.part_best_conf.get(pid, 0.0), det.conf)
    return out


def describe_gesture(summary: VisionSummary, min_conf: float) -> tuple[bool, str]:
    """MediaPipe 판정과 같은 형식의 (성공 여부, 한국어 사유)."""
    if summary.gesture == GESTURE_NONE:
        if summary.parts_visible:
            return False, f"손짓 미검출 · 부품 {summary.parts_visible}개 검출"
        return False, "손짓 미검출 (YOLO)"
    if summary.gesture_conf < min_conf:
        return False, f"신뢰도 부족 ({summary.gesture_conf:.2f} < {min_conf:.2f})"
    label_ko = "손바닥(취소)" if summary.gesture == GESTURE_PALM else f"손짓 {summary.gesture}"
    return True, f"YOLO 인식 성공: {label_ko} ({summary.gesture_conf:.2f})"


def draw_detections(frame: np.ndarray, detections: list[Detection], banner: str = "", passed: bool = False) -> None:
    import cv2

    for det in detections:
        x1, y1, x2, y2 = det.box
        if det.kind == KIND_PART:
            color = PART_COLORS_BGR.get(int(det.value), OTHER_COLOR_BGR)
            label = f"#{det.value} {det.name} {det.conf:.2f}"
        elif det.kind == KIND_GESTURE:
            color = GESTURE_COLOR_BGR
            label = f"G:{det.value} {det.conf:.2f}"
        else:
            color = OTHER_COLOR_BGR
            label = f"{det.name} {det.conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 6)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(frame, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if banner:
        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, 36), (34, 30, 27), -1)
        color = (90, 201, 134) if passed else (58, 88, 226)
        cv2.putText(frame, banner, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


class YoloDetector:
    """ultralytics YOLO 모델 로더 + 단일 프레임 추론."""

    def __init__(
        self,
        model_path: Path = YOLO_MODEL_PATH,
        classes_path: Path | None = YOLO_CLASSES_PATH,
        conf: float = 0.5,
        imgsz: int = 640,
    ):
        self.model_path = Path(model_path)
        self.classes_path = Path(classes_path) if classes_path else None
        self.conf = conf
        self.imgsz = imgsz
        self.error = ""
        self.class_map = ClassMap()
        self.names: dict[int, str] = {}
        self._model = None
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True
        try:
            self.class_map = ClassMap.load(self.classes_path)
        except (ValueError, OSError) as exc:
            self.error = f"클래스 매핑 오류: {exc}"
            return False
        if not self.model_path.exists():
            self.error = f"YOLO 모델 파일이 없습니다: {self.model_path}"
            return False
        try:
            from ultralytics import YOLO
        except ImportError:
            self.error = "ultralytics 미설치 (pip install ultralytics)"
            return False
        try:
            self._model = YOLO(str(self.model_path))
            self.names = {int(k): str(v) for k, v in dict(self._model.names).items()}
        except Exception as exc:  # 손상된 가중치 등
            self._model = None
            self.error = f"YOLO 모델 로드 실패: {exc}"
            return False
        self.error = ""
        return True

    @property
    def available(self) -> bool:
        return self.load()

    @property
    def classes(self) -> list[dict]:
        return self.class_map.describe(self.names)

    @property
    def has_gesture_classes(self) -> bool:
        return any(row["kind"] == KIND_GESTURE for row in self.classes)

    @property
    def has_part_classes(self) -> bool:
        return any(row["kind"] == KIND_PART for row in self.classes)

    def info(self) -> dict:
        ok = self.available
        return {
            "available": ok,
            "error": self.error,
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            "class_map_source": self.class_map.source,
            "classes": self.classes,
            "has_gestures": ok and self.has_gesture_classes,
            "has_parts": ok and self.has_part_classes,
            "conf": self.conf,
        }

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if not self.available:
            return []
        with self._lock:
            results = self._model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        dets: list[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                name = self.names.get(int(k), str(int(k)))
                info = self.class_map.resolve(name)
                dets.append(Detection(name, info.kind, info.value, float(c), (int(x1), int(y1), int(x2), int(y2))))
        return dets


def analyze_image_bytes_yolo(raw_bytes: bytes, detector: YoloDetector, min_confidence: float = 0.70) -> dict:
    """정지 이미지 YOLO 판정 — gesture.webcam.analyze_image_bytes 와 같은 응답 형식 + 검출 목록."""
    import cv2

    if not detector.available:
        return {"ok": False, "error": detector.error}
    frame = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"ok": False, "error": "이미지를 디코딩할 수 없습니다."}

    dets = detector.detect(frame)
    summary = summarize(dets)
    passed, reason = describe_gesture(summary, min_confidence)
    banner = f"YOLO | {summary.gesture} ({summary.gesture_conf:.2f})" if summary.gesture != GESTURE_NONE else (
        f"YOLO | parts 1:{summary.part_counts[1]} 2:{summary.part_counts[2]} 3:{summary.part_counts[3]}"
    )
    draw_detections(frame, dets, banner=banner, passed=passed)
    ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok_jpg else ""
    return {
        "ok": True,
        "backend": "yolo",
        "hand_detected": summary.gesture != GESTURE_NONE,
        "passed": passed,
        "label": summary.gesture,
        "confidence": summary.gesture_conf,
        "fingers": [False] * 5,
        "reason": reason,
        "detections": [d.to_dict() for d in dets],
        "part_counts": summary.part_counts,
        "image_data_url": f"data:image/jpeg;base64,{jpeg_b64}" if jpeg_b64 else "",
    }
