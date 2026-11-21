"""로컬 대시보드 웹 서버 & 파이프라인 런타임."""
import functools
import http.server
import json
from pathlib import Path
import socket
import threading
import time

from config import LOG_DIR, MODEL_PATH, PipelineConfig
from controller.state_machine import DeliveryStateMachine, State
from gesture.classifier import GESTURE_NONE, GESTURE_PALM
from gesture.webcam import WebcamGestureSource, analyze_image_bytes
from link.mock_microbit import MockMicrobitLink
from link.protocol import ERR_EMPTY, ERR_GRIP, RobotLink
from logging_metrics.csv_logger import CsvEventLogger
from logging_metrics.metrics import compute_metrics_from_csv


class PipelineRuntime:
    """웹캠 인식·상태 머신·통신·CSV 로깅을 통합 실행하는 스레드 안전 런타임."""

    def __init__(
        self,
        link: RobotLink,
        config: PipelineConfig = PipelineConfig(),
        log_path: Path | None = None,
    ):
        self.cfg = config
        self.link = link
        self.log_path = log_path or (LOG_DIR / "pipeline_events.csv")
        self.logger = CsvEventLogger(self.log_path)
        self.sm = DeliveryStateMachine(link=link, config=config, logger=self.logger)
        self.webcam = WebcamGestureSource(
            camera_index=config.camera_index,
            model_path=MODEL_PATH,
            min_confidence=config.min_confidence,
        )

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._manual_gesture: str = GESTURE_NONE
        self._manual_conf: float = 0.0
        self._webcam_fps: float = 0.0

    def start(self, start_webcam: bool = False) -> None:
        if start_webcam:
            self.webcam.start()
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.webcam.stop()
        self.link.close()

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                self._tick_locked()
            time.sleep(0.05)

    def _tick_locked(self) -> None:
        gesture = GESTURE_NONE
        conf = 0.0

        if self._manual_gesture != GESTURE_NONE:
            gesture = self._manual_gesture
            conf = self._manual_conf
            # 손짓 1/2/3 홀드는 CONFIRM 진입 시 자동 해제, PALM은 1회 소비 후 해제
            if gesture == GESTURE_PALM:
                self._manual_gesture = GESTURE_NONE
                self._manual_conf = 0.0
        elif self.webcam.is_running:
            pred, fps = self.webcam.get_latest()
            gesture = pred.label
            conf = pred.confidence
            self._webcam_fps = fps

        prev_state = self.sm.state
        self.sm.step(gesture=gesture, confidence=conf)
        if prev_state == State.REQUEST and self.sm.state != State.REQUEST:
            self._manual_gesture = GESTURE_NONE
            self._manual_conf = 0.0

    def dispatch_action(self, action: str, part: int | None = None) -> dict:
        with self._lock:
            if action == "gesture" and part in (1, 2, 3):
                self._manual_gesture = str(part)
                self._manual_conf = 0.98
                self._tick_locked()
            elif action == "cancel":
                self._manual_gesture = GESTURE_PALM
                self._manual_conf = 0.98
                self._tick_locked()
            elif action == "hand":
                if isinstance(self.link, MockMicrobitLink):
                    self.link.trigger_hand_event()
                self._tick_locked()
            elif action == "proximity_pick" and part in (1, 2, 3):
                self._manual_gesture = GESTURE_NONE
                self._manual_conf = 0.0
                self.sm.trigger_proximity_pick(part)
            elif action == "return":
                self.sm.trigger_return()
            elif action == "err_empty":
                if isinstance(self.link, MockMicrobitLink):
                    self.link.inject_error(ERR_EMPTY)
                if self.sm.state == State.PICK:
                    self.link.poll()
                    self.sm._handle_rx(ERR_EMPTY)
            elif action == "err_grip":
                if isinstance(self.link, MockMicrobitLink):
                    self.link.inject_error(ERR_GRIP)
                if self.sm.state == State.PICK:
                    self.sm._handle_rx(ERR_GRIP)
            elif action == "stop":
                self._manual_gesture = GESTURE_NONE
                self.sm.emergency_stop()
            elif action == "reset_all":
                self._manual_gesture = GESTURE_NONE
                self._manual_conf = 0.0
                self.sm.reset_all()
            elif action == "false_trigger":
                self.sm.mark_false_trigger()
            elif action == "webcam_toggle":
                if self.webcam.is_running:
                    self.webcam.stop()
                    self._webcam_fps = 0.0
                else:
                    ok, msg = self.webcam.start()
                    if not ok:
                        return {"ok": False, "error": msg}
            return self.get_state_locked()

    def get_state(self) -> dict:
        with self._lock:
            return self.get_state_locked()

    def get_state_locked(self) -> dict:
        snap = self.sm.snapshot()
        metrics = compute_metrics_from_csv(self.log_path).to_dict()
        snap["metrics"] = metrics
        snap["webcam"] = self.webcam.get_status()
        return snap

    def test_image(self, raw_bytes: bytes) -> dict:
        return analyze_image_bytes(
            raw_bytes,
            model_path=MODEL_PATH,
            min_confidence=self.cfg.min_confidence,
        )


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """정적 파일과 /api/* 엔드포인트를 함께 처리한다."""

    runtime: PipelineRuntime

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/state":
            self._send_json(self.runtime.get_state())
            return
        if self.path == "/api/video":
            self._serve_mjpeg()
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/test_image":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            if not raw:
                self._send_json({"ok": False, "error": "빈 이미지입니다."}, status=400)
                return
            self._send_json(self.runtime.test_image(raw))
            return
        if self.path == "/api/action":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"ok": False, "error": "invalid json"}, status=400)
                return
            state = self.runtime.dispatch_action(
                action=str(payload.get("action", "")),
                part=payload.get("part"),
            )
            self._send_json(state)
            return
        self._send_json({"error": "not found"}, status=404)

    def _serve_mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while self.runtime.webcam.is_running:
                jpg = self.runtime.webcam.get_jpeg()
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass


def find_free_port(start: int = 8080) -> int:
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("사용 가능한 포트를 찾을 수 없습니다.")


def create_server(runtime: PipelineRuntime, root_dir: Path, port: int = 0) -> tuple[http.server.ThreadingHTTPServer, int]:
    resolved_port = port or find_free_port()

    class BoundHandler(DashboardHandler):
        pass

    BoundHandler.runtime = runtime
    handler = functools.partial(BoundHandler, directory=str(root_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", resolved_port), handler)
    return server, resolved_port
