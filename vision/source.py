"""YOLO 실시간 카메라 소스 — WebcamGestureSource 와 같은 인터페이스.

- 손짓: YOLO 모델에 손짓 클래스가 있으면 YOLO 결과를, 없으면 같은 프레임에서
  MediaPipe 21점 규칙 분류기를 함께 돌려 손짓을 낸다.
- 부품: 최근 `window_sec` 동안의 프레임별 부품 개수 최댓값으로 깜빡임을 줄이고,
  상태 머신의 PICK 전 대조(`check_part`)에 사용한다.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import threading
import time

from config import MODEL_PATH
from gesture.classifier import GESTURE_NONE, GesturePrediction
from .detector import Detection, YoloDetector, describe_gesture, draw_detections, summarize


class YoloCameraSource:
    backend = "yolo"

    def __init__(
        self,
        detector: YoloDetector,
        camera_index: int = 0,
        min_confidence: float = 0.70,
        hand_model_path: Path = MODEL_PATH,
        flip: bool = True,
        window_sec: float = 1.0,
    ):
        self.detector = detector
        self.camera_index = camera_index
        self.min_confidence = min_confidence
        self.hand_model_path = Path(hand_model_path)
        self.flip = flip
        self.window_sec = window_sec

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest = GesturePrediction(label=GESTURE_NONE, confidence=0.0)
        self._gesture_source = "yolo"
        self._passed = False
        self._reason = "YOLO 카메라 대기 중"
        self._detections: list[Detection] = []
        self._history: deque[tuple[float, dict[int, int]]] = deque(maxlen=120)
        self._latest_jpeg: bytes | None = None
        self._fps = 0.0
        self._error = ""

    # ---- WebcamGestureSource 호환 인터페이스 ----
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> str:
        return self._error

    def start(self) -> tuple[bool, str]:
        if self._running:
            return True, "이미 실행 중"
        if not self.detector.available:
            self._error = self.detector.error
            return False, self._error
        self._error = ""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True, "YOLO 카메라 인식 시작"

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._latest = GesturePrediction(label=GESTURE_NONE, confidence=0.0)
            self._passed = False
            self._reason = "YOLO 카메라 중지됨"
            self._detections = []
            self._history.clear()
            self._fps = 0.0

    def get_latest(self) -> tuple[GesturePrediction, float]:
        with self._lock:
            return self._latest, self._fps

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_status(self) -> dict:
        with self._lock:
            return {
                "backend": self.backend,
                "gesture_source": self._gesture_source,
                "running": self._running,
                "fps": self._fps,
                "error": self._error,
                "hand_detected": self._latest.label != GESTURE_NONE,
                "passed": self._passed,
                "label": self._latest.label,
                "confidence": self._latest.confidence,
                "fingers": list(self._latest.fingers),
                "reason": self._reason,
                "detections": [d.to_dict() for d in self._detections],
                "part_counts": self._recent_counts_locked(),
            }

    # ---- 부품 대조 ----
    def _recent_counts_locked(self) -> dict[int, int]:
        cutoff = time.monotonic() - self.window_sec
        counts = {1: 0, 2: 0, 3: 0}
        for t, frame_counts in self._history:
            if t < cutoff:
                continue
            for pid, n in frame_counts.items():
                counts[pid] = max(counts[pid], n)
        return counts

    def recent_part_counts(self) -> dict[int, int]:
        with self._lock:
            return self._recent_counts_locked()

    def check_part(self, part: int) -> tuple[bool | None, str]:
        """True=트레이에서 목표 부품 확인, False=트레이는 보이나 목표 없음, None=판단 불가."""
        if not self._running or not self.detector.has_part_classes:
            return None, "YOLO 부품 대조 비활성"
        counts = self.recent_part_counts()
        if counts.get(part, 0) > 0:
            return True, f"YOLO 트레이 확인: 부품 {part} {counts[part]}개 검출"
        if sum(counts.values()) == 0:
            return None, "카메라에 트레이 부품이 보이지 않아 대조를 건너뜀"
        seen = ", ".join(f"{p}번×{n}" for p, n in counts.items() if n)
        return False, f"YOLO 대조 실패: 부품 {part} 미검출 (보이는 부품: {seen})"

    # ---- 캡처 루프 ----
    def _make_landmarker(self):
        if self.detector.has_gesture_classes or not self.hand_model_path.exists():
            return None
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError:
            return None
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.hand_model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    def _loop(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            with self._lock:
                self._error = f"카메라({self.camera_index})를 열 수 없습니다."
                self._reason = self._error
            self._running = False
            return

        landmarker = self._make_landmarker()
        self._gesture_source = "yolo" if landmarker is None else "mediapipe"
        prev_t = time.perf_counter()
        fps_smooth = 0.0
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                if self.flip:
                    frame = cv2.flip(frame, 1)

                dets = self.detector.detect(frame)
                summary = summarize(dets)

                if landmarker is not None:
                    pred, passed, reason = self._mediapipe_gesture(landmarker, frame)
                else:
                    pred = GesturePrediction(label=summary.gesture, confidence=summary.gesture_conf)
                    passed, reason = describe_gesture(summary, self.min_confidence)

                counts = summary.part_counts
                banner = f"YOLO | {pred.label} ({pred.confidence:.2f}) | 1:{counts[1]} 2:{counts[2]} 3:{counts[3]}"
                draw_detections(frame, dets, banner=banner, passed=passed)

                now_t = time.perf_counter()
                dt = max(now_t - prev_t, 1e-4)
                prev_t = now_t
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt) if fps_smooth else (1.0 / dt)

                ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self._lock:
                    self._latest = pred
                    self._passed = passed
                    self._reason = reason
                    self._detections = dets
                    self._history.append((time.monotonic(), dict(counts)))
                    self._fps = round(fps_smooth, 1)
                    if ok_jpg:
                        self._latest_jpeg = buf.tobytes()
        finally:
            if landmarker is not None:
                landmarker.close()
            cap.release()
            self._running = False

    def _mediapipe_gesture(self, landmarker, frame) -> tuple[GesturePrediction, bool, str]:
        import cv2
        import mediapipe as mp

        from gesture.classifier import classify_landmarks
        from gesture.webcam import evaluate_recognition

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time.monotonic() * 1000)
        )
        hand = bool(result.hand_landmarks)
        pred = GesturePrediction(label=GESTURE_NONE, confidence=0.0)
        if hand:
            score = result.handedness[0][0].score if result.handedness else 1.0
            pred = classify_landmarks(result.hand_landmarks[0], hand_score=score)
        passed, reason = evaluate_recognition(pred, hand, min_confidence=self.min_confidence)
        return pred, passed, reason
