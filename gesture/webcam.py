"""MediaPipe HandLandmarker 기반 실시간 웹캠 및 정지 이미지 손짓 인식기."""
import base64
from pathlib import Path
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from config import MODEL_PATH
from .classifier import GESTURE_NONE, GesturePrediction, classify_landmarks

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def evaluate_recognition(
    pred: GesturePrediction,
    hand_detected: bool,
    min_confidence: float = 0.70,
) -> tuple[bool, str]:
    """현재 프레임의 인식 성공/실패 여부와 한국어 사유를 반환한다."""
    if not hand_detected:
        return False, "손 미감지 (카메라 화면에 손이 보이지 않음)"
    if pred.label == GESTURE_NONE:
        return False, "미분류 손 모양 (1·2·3 또는 손바닥이 아님)"
    if pred.confidence < min_confidence:
        return False, f"신뢰도 부족 ({pred.confidence:.2f} < {min_confidence:.2f})"
    label_ko = {"1": "손짓 1", "2": "손짓 2", "3": "손짓 3", "PALM": "손바닥(취소)"}.get(pred.label, pred.label)
    return True, f"인식 성공: {label_ko} ({pred.confidence:.2f})"


def draw_recognition_overlay(
    frame: np.ndarray,
    landmarks,
    pred: GesturePrediction,
    hand_detected: bool,
    passed: bool,
) -> None:
    """21점 스켈레톤과 상단 PASS/FAIL 배너를 프레임에 그린다."""
    h, w = frame.shape[:2]
    if hand_detected and landmarks is not None:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (220, 226, 228), 2)
        dot_bgr = (90, 201, 134) if passed else (58, 88, 226)
        for p in pts:
            cv2.circle(frame, p, 4, dot_bgr, -1)

    cv2.rectangle(frame, (0, 0), (w, 36), (34, 30, 27), -1)
    if passed:
        banner = f"PASS | {pred.label} ({pred.confidence:.2f})"
        color_bgr = (90, 201, 134)
    elif not hand_detected:
        banner = "FAIL | NO HAND DETECTED"
        color_bgr = (58, 88, 226)
    else:
        banner = f"FAIL | {pred.label} ({pred.confidence:.2f})"
        color_bgr = (58, 88, 226)
    cv2.putText(frame, banner, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_bgr, 2, cv2.LINE_AA)


def analyze_image_bytes(
    raw_bytes: bytes,
    model_path: Path = MODEL_PATH,
    min_confidence: float = 0.70,
) -> dict:
    """업로드된 이미지 바이트에서 손 21점을 검출해 인식 성공/실패와 주석 이미지를 반환한다."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"ok": False, "error": "이미지를 디코딩할 수 없습니다."}
    if not Path(model_path).exists():
        return {"ok": False, "error": f"모델 파일이 없습니다: {model_path}"}

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
    )
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

    hand_detected = bool(result.hand_landmarks)
    landmarks = result.hand_landmarks[0] if hand_detected else None
    pred = GesturePrediction(label=GESTURE_NONE, confidence=0.0)
    if hand_detected:
        hand_score = result.handedness[0][0].score if result.handedness else 1.0
        pred = classify_landmarks(landmarks, hand_score=hand_score)

    passed, reason = evaluate_recognition(pred, hand_detected, min_confidence=min_confidence)
    draw_recognition_overlay(frame, landmarks, pred, hand_detected, passed)

    ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok_jpg else ""
    return {
        "ok": True,
        "hand_detected": hand_detected,
        "passed": passed,
        "label": pred.label,
        "confidence": pred.confidence,
        "fingers": list(pred.fingers),
        "reason": reason,
        "image_data_url": f"data:image/jpeg;base64,{jpeg_b64}" if jpeg_b64 else "",
    }


class WebcamGestureSource:
    """백그라운드 스레드에서 웹캠 프레임을 읽고 최신 손짓 예측·판정·JPEG 프레임을 제공한다."""

    def __init__(self, camera_index: int = 0, model_path: Path = MODEL_PATH, min_confidence: float = 0.70):
        self.camera_index = camera_index
        self.model_path = Path(model_path)
        self.min_confidence = min_confidence
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest = GesturePrediction(label=GESTURE_NONE, confidence=0.0)
        self._hand_detected: bool = False
        self._passed: bool = False
        self._reason: str = "웹캠 대기 중"
        self._latest_jpeg: bytes | None = None
        self._fps: float = 0.0
        self._error: str = ""

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> str:
        return self._error

    def start(self) -> tuple[bool, str]:
        if self._running:
            return True, "이미 실행 중"
        if not self.model_path.exists():
            self._error = f"모델 파일이 없습니다: {self.model_path}"
            return False, self._error

        self._error = ""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True, "웹캠 인식 시작"

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        with self._lock:
            self._latest = GesturePrediction(label=GESTURE_NONE, confidence=0.0)
            self._hand_detected = False
            self._passed = False
            self._reason = "웹캠 중지됨"
            self._fps = 0.0

    def get_latest(self) -> tuple[GesturePrediction, float]:
        with self._lock:
            return self._latest, self._fps

    def get_status(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "fps": self._fps,
                "error": self._error,
                "hand_detected": self._hand_detected,
                "passed": self._passed,
                "label": self._latest.label,
                "confidence": self._latest.confidence,
                "fingers": list(self._latest.fingers),
                "reason": self._reason,
            }

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            with self._lock:
                self._error = f"카메라({self.camera_index})를 열 수 없습니다."
                self._reason = self._error
            self._running = False
            return

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)

        prev_t = time.perf_counter()
        fps_smooth = 0.0
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int(time.monotonic() * 1000)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                hand_detected = bool(result.hand_landmarks)
                landmarks = result.hand_landmarks[0] if hand_detected else None
                pred = GesturePrediction(label=GESTURE_NONE, confidence=0.0)
                if hand_detected:
                    hand_score = result.handedness[0][0].score if result.handedness else 1.0
                    pred = classify_landmarks(landmarks, hand_score=hand_score)

                passed, reason = evaluate_recognition(pred, hand_detected, min_confidence=self.min_confidence)
                draw_recognition_overlay(frame, landmarks, pred, hand_detected, passed)

                now_t = time.perf_counter()
                dt = max(now_t - prev_t, 1e-4)
                prev_t = now_t
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt) if fps_smooth else (1.0 / dt)

                ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self._lock:
                    self._latest = pred
                    self._hand_detected = hand_detected
                    self._passed = passed
                    self._reason = reason
                    self._fps = round(fps_smooth, 1)
                    if ok_jpg:
                        self._latest_jpeg = buf.tobytes()
        finally:
            landmarker.close()
            cap.release()
            self._running = False
