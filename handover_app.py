"""AiHand 받기 작업 실행 — 기존 대시보드(index.html)에 실물 과정을 그대로 보여준다.

  .venv/bin/python handover_app.py --port /dev/cu.usbmodem145102 --camera 1
  .venv/bin/python handover_app.py --mock            # 로봇 없이 시험 (모의 micro:bit)

과정: 입력(볼트/너트/와셔) → 손 펴기 → 웹캠 로컬 인식 → 쥐기 → 왼쪽(180°) 회전·내려놓기 → 복귀
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
import functools
import http.server
import json
from pathlib import Path
import sys
import threading
import time
import webbrowser

import cv2

from controller.handover import HandoverTask
from vision.parts import LABEL_KO, LABELS, PartClassifier, PartResult, YoloParts

ROOT = Path(__file__).resolve().parent


FIRMWARE = ROOT / "microbit" / "handover.py"


def firmware_version() -> str:
    import re
    m = re.search(r'FW_VERSION\s*=\s*"([^"]+)"', FIRMWARE.read_text(encoding="utf-8"))
    return m.group(1) if m else "?"


class RobotLink:
    """micro:bit 자동 연결 관리: 포트 탐색 · 끊김 감지/재연결 · 2초 PING · 펌웨어 버전 확인/자동 설치.

    status: disconnected(USB 없음) | connecting | updating(펌웨어 설치 중) | connected | no_response
    """
    is_mock = False
    HEARTBEAT_S = 2.0
    NO_RESPONSE_S = 5.0

    def __init__(self, port_hint: str = "", baud: int = 115200, on_connect=None, log=None):
        self.port_hint = port_hint
        self.baud = baud
        self.on_connect = on_connect
        self.log = log or (lambda d, t: None)
        self.ser = None
        self.port = ""
        self.status = "disconnected"
        self.fw = ""
        self.fw_warning = ""
        self._install_tries = 0
        self._install_block_until = 0.0
        self.last_rx = 0.0
        self._hb_sent = 0.0
        self._hb_pending = False
        self.rx: deque[str] = deque()
        self._buf = b""
        self._lock = threading.Lock()
        self.running = True
        self._thread = None

    def start(self) -> None:
        """로그·연결 콜백을 붙인 뒤에 시작 (먼저 시작하면 첫 연결 로그와 HOME 이 빠짐)."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    # ---- public ----
    @property
    def connected(self) -> bool:
        return self.status in ("connected", "no_response") and self.ser is not None

    def send(self, cmd: str) -> bool:
        with self._lock:
            if self.ser is None or self.status in ("updating", "connecting"):
                return False
            try:
                self.ser.write((cmd.strip() + "\n").encode())
                return True
            except Exception:
                self._drop("쓰기 실패")
                return False

    def poll(self) -> list[str]:
        out = []
        while self.rx:
            out.append(self.rx.popleft())
        return out

    def info(self) -> dict:
        ago = time.monotonic() - self.last_rx if self.last_rx else None
        return {"status": self.status, "port": self.port, "fw": self.fw, "fw_expected": firmware_version(),
                "fw_warning": self.fw_warning,
                "last_rx_s": round(ago, 1) if ago is not None else None}

    def close(self) -> None:
        self.running = False
        with self._lock:
            if self.ser is not None:
                self.ser.close()
                self.ser = None

    # ---- internals ----
    def _drop(self, why: str) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        if self.status != "disconnected":
            self.log("ERR", f"로봇 연결 끊김 ({why})")
        self.status = "disconnected"
        self.port = ""
        if why == "USB 분리":
            self._install_tries = 0
            self._install_block_until = 0.0

    def _find_port(self) -> str:
        import glob
        ports = sorted(glob.glob("/dev/cu.usbmodem*"))
        if self.port_hint in ports:
            return self.port_hint
        return ports[0] if ports else ""

    def _open(self, port: str):
        import serial
        ser = serial.Serial(port, self.baud, timeout=0.05)
        time.sleep(0.3)
        ser.reset_input_buffer()
        return ser

    def _query_version(self, ser) -> str:
        ser.write(b"\nVER\n")
        deadline = time.monotonic() + 1.5
        buf = b""
        while time.monotonic() < deadline:
            buf += ser.read(256)
            for line in buf.split(b"\n"):
                t = line.decode("utf-8", "replace").strip()
                if t.startswith("VER,"):
                    return t.split(",", 1)[1]
        return ""

    def _install_firmware(self, port: str) -> bool:
        """raw REPL로 main.py 를 직접 써 넣는다 (외부 ufs 보다 타이밍에 강함)."""
        try:
            ser = self._open(port)
        except Exception as exc:
            self.log("ERR", f"펌웨어 설치: 포트 열기 실패 ({exc})")
            return False
        try:
            # 1) 실행 중인 프로그램 중단 → 일반 REPL 프롬프트 확인
            got = b""
            for _ in range(8):
                ser.write(b"\r\x03\x03")
                time.sleep(0.25)
                got += ser.read(2048)
                if b">>>" in got:
                    break
            if b">>>" not in got:
                self.log("ERR", "펌웨어 설치: 프로그램을 멈추지 못했어요 (REPL 프롬프트 없음)")
                return False
            # 2) raw REPL 진입
            ser.reset_input_buffer()
            ser.write(b"\x01")
            time.sleep(0.2)
            if b"raw REPL" not in ser.read(512):
                self.log("ERR", "펌웨어 설치: raw REPL 진입 실패")
                return False
            # 3) main.py 쓰기 (작은 조각으로 나눠 전송)
            code = FIRMWARE.read_bytes()
            ser.write(b"f=open('main.py','wb')\n")
            for k in range(0, len(code), 128):
                ser.write(b"f.write(" + repr(code[k:k + 128]).encode() + b")\n")
                time.sleep(0.01)
            ser.write(b"f.close()\n\x04")
            deadline = time.monotonic() + 10
            resp = b""
            while time.monotonic() < deadline and b"\x04>" not in resp:
                resp += ser.read(512)
            if not resp.startswith(b"OK") or b"Error" in resp:
                self.log("ERR", f"펌웨어 설치: 쓰기 실패 {resp[:80]!r}")
                return False
            # 4) raw REPL 종료 + 소프트 리셋 → 새 main.py 실행
            ser.write(b"\x02")
            time.sleep(0.1)
            ser.write(b"\x04")
            time.sleep(2.5)
            return True
        except Exception as exc:
            self.log("ERR", f"펌웨어 설치 중 오류: {exc}")
            return False
        finally:
            try:
                ser.close()
            except Exception:
                pass

    def _connect(self, port: str) -> None:
        self.status = "connecting"
        self.port = port
        try:
            ser = self._open(port)
            ver = self._query_version(ser)
            want = firmware_version()
            now = time.monotonic()
            if ver != want and self._install_tries < 2 and now >= self._install_block_until:
                ser.close()
                self._install_tries += 1
                self.status = "updating"
                self.log("SYS", f"펌웨어 {'v' + ver if ver else '확인 불가'} → v{want} 자동 설치 ({self._install_tries}/2회)")
                ok = self._install_firmware(port)
                ser = self._open(port)
                ver = self._query_version(ser)
                if ok and ver == want:
                    self.log("SYS", f"펌웨어 v{want} 설치 완료")
                elif self._install_tries >= 2:
                    self._install_block_until = time.monotonic() + 60
                    self.log("ERR", "펌웨어 자동 설치 실패 — 60초 뒤 다시 시도해요 (기존 펌웨어로 계속 연결)")
            if ver != want:
                self.fw_warning = f"펌웨어 v{ver or '?'} (최신 v{want} 아님)"
            else:
                self.fw_warning = ""
                self._install_tries = 0
            with self._lock:
                self.ser = ser
            self.fw = ver
            self.last_rx = time.monotonic()
            self.status = "connected"
            self.log("SYS", f"로봇 연결됨 ({port}, 펌웨어 v{ver})")
            if self.on_connect:
                self.on_connect()
        except Exception as exc:
            self.log("ERR", f"로봇 연결 실패: {exc}")
            self._drop("연결 실패")

    def _loop(self) -> None:
        while self.running:
            if self.ser is None:
                port = self._find_port()
                if port:
                    time.sleep(1.0)          # 꽂은 직후 안정화
                    self._connect(port)
                else:
                    self.status = "disconnected"
                    time.sleep(1.0)
                continue
            # read
            try:
                chunk = self.ser.read(512)
            except Exception:
                self._drop("USB 분리")
                continue
            if chunk:
                self._buf += chunk
                while b"\n" in self._buf:
                    line, self._buf = self._buf.split(b"\n", 1)
                    t = line.decode("utf-8", "replace").strip()
                    if not t:
                        continue
                    self.last_rx = time.monotonic()
                    if self.status == "no_response":
                        self.status = "connected"
                        self.log("SYS", "로봇 응답 복구")
                    if t == "ACK" and self._hb_pending:
                        self._hb_pending = False       # 생존 확인 응답은 로그로 넘기지 않음
                        continue
                    self.rx.append(t)
            now = time.monotonic()
            if now - self.last_rx > self.HEARTBEAT_S and not self._hb_pending:
                if self.send("PING"):
                    self._hb_pending = True
                    self._hb_sent = now
            if self._hb_pending and now - self._hb_sent > 3.0:
                self._hb_pending = False
            if now - self.last_rx > self.NO_RESPONSE_S and self.status == "connected":
                self.status = "no_response"
                self.log("ERR", "로봇이 응답하지 않아요 (전원 · 케이블 확인)")


class MockLink:
    """모의 micro:bit: 명령마다 짧은 지연 뒤 DONE 응답."""
    is_mock = True
    port = "mock"
    DELAY = {"OPEN": 0.6, "GRIP": 0.7, "PLACE": 2.4, "HOME": 0.8, "PAN": 1.0, "SERVO": 0.4}

    def __init__(self):
        self.pending: list[tuple[float, str]] = []

    def send(self, cmd: str) -> None:
        head = cmd.split(",")[0]
        if head in ("PING", "STOP"):
            self.pending.append((time.monotonic(), "ACK"))
        else:
            self.pending.append((time.monotonic() + self.DELAY.get(head, 0.5), f"DONE,{head}"))

    def poll(self) -> list[str]:
        now = time.monotonic()
        due = [m for t, m in self.pending if t <= now]
        self.pending = [(t, m) for t, m in self.pending if t > now]
        return due

    def close(self) -> None:
        pass

    connected = True
    status = "connected"

    def info(self) -> dict:
        return {"status": "connected", "port": "mock", "fw": "mock", "fw_expected": "mock", "last_rx_s": 0.0}


TRAY_MIN_CONF = 0.45          # 쟁반 안내에 쓸 YOLO 최소 점수 (겹친 오탐 0.3 대 제외)


class Runtime:
    ROI_FILE = ROOT / "data" / "palm_roi.json"
    SETTINGS_FILE = ROOT / "data" / "handover_settings.json"
    DEFAULT_ROI = (0.30, 0.35, 0.65, 0.75)       # 팀 zones.json 의 palm 영역

    def __init__(self, link, camera_index: int, model_path: Path | None = None, conf: float = 0.35, vlm=None,
                 tray_roi: tuple | None = None):
        self.tray_roi = tray_roi                    # 쟁반 영역(정규화). 없으면 '손 영역 밖 전체'를 쟁반으로 봄
        self.link = link
        self.task = HandoverTask(link, vlm=vlm)
        self.task.frame_source = lambda: self.frame           # VLM 교차 확인용 현재 프레임
        self.clf = PartClassifier()                 # YOLO 를 못 쓸 때의 대체 판별기
        self.yolo = None
        self.detector_info = "모양 규칙 (YOLO 없음)"
        if model_path and Path(model_path).exists():
            try:
                self.yolo = YoloParts(model_path, conf=conf)
                self.detector_info = f"YOLO {Path(model_path).name} · {', '.join(self.yolo.names.values())}"
            except Exception as exc:                # ultralytics/torch 미설치 등
                self.detector_info = f"모양 규칙 (YOLO 불러오기 실패: {exc.__class__.__name__})"
        self.task._log("SYS", f"인식기: {self.detector_info}")
        if vlm is not None:
            self.task._log("SYS", f"VLM: {vlm.model} — 요청 해석 보조 + YOLO 점수 {vlm.verify_below:.2f} 미만이면 교차 확인")
            vlm.warmup()
        self.camera_index = camera_index
        self._cam_switch = False
        self.cam_disabled: set[int] = set(int(i) for i in self.read_settings().get("camera_disabled", []))
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.jpeg: bytes | None = None
        self.result: dict | None = None
        self.overlay: dict = {"dets": [], "best": None}
        self.cam_error = ""
        self.fps = 0.0
        self.infer_ms = 0.0
        self.palm_roi = self._load_roi()
        self._load_settings()
        self.recv_bg = None
        self.manual_bg = None
        self.running = True
        self._cam_gen = 0                 # 카메라 스레드 세대: 멈춘 스레드를 버리고 새로 시작할 때 증가
        self._cam_opened_t = 0.0
        self._last_frame_t = 0.0
        threading.Thread(target=self._camera_loop, args=(0,), daemon=True).start()
        threading.Thread(target=self._infer_loop, daemon=True).start()
        threading.Thread(target=self._task_loop, daemon=True).start()

    # ---- settings ----
    def _load_roi(self):
        try:
            return tuple(float(x) for x in json.loads(self.ROI_FILE.read_text()))
        except Exception:
            return self.DEFAULT_ROI

    @classmethod
    def read_settings(cls) -> dict:
        try:
            return json.loads(cls.SETTINGS_FILE.read_text())
        except Exception:
            return {}

    def _load_settings(self) -> None:
        st = self.read_settings()
        try:
            self.task.place_angle = int(st.get("place_angle", self.task.place_angle))
        except (TypeError, ValueError):
            pass

    def _save_settings(self, **extra) -> None:
        st = self.read_settings()
        st["place_angle"] = self.task.place_angle
        st.update(extra)
        self.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.SETTINGS_FILE.write_text(json.dumps(st, ensure_ascii=False))

    def set_roi(self, x0, y0, x1, y1) -> bool:
        x0, x1 = sorted((max(0.0, min(1.0, float(x0))), max(0.0, min(1.0, float(x1)))))
        y0, y1 = sorted((max(0.0, min(1.0, float(y0))), max(0.0, min(1.0, float(y1)))))
        if x1 - x0 < 0.05 or y1 - y0 < 0.05:
            return False
        self.palm_roi = (round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))
        self.ROI_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.ROI_FILE.write_text(json.dumps(self.palm_roi))
        return True

    # ---- drawing ----
    def _draw(self, frame):
        vis = frame.copy()
        h, w = vis.shape[:2]
        k = max(1.0, w / 640)
        th = max(2, int(2 * k))
        receiving = self.task.state in ("RECEIVE", "VERIFY")
        best = self.overlay.get("best")
        touching = receiving and best is not None and best.get("label") not in (None, "none")
        # 손 영역
        x0, y0, x1, y1 = (int(self.palm_roi[0] * w), int(self.palm_roi[1] * h),
                          int(self.palm_roi[2] * w), int(self.palm_roi[3] * h))
        col = (90, 170, 90) if touching else (200, 150, 90) if receiving else (170, 170, 170)
        L = int(min(x1 - x0, y1 - y0) * 0.22)
        for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(vis, (cx, cy), (cx + dx * L, cy), col, th + 1)
            cv2.line(vis, (cx, cy), (cx, cy + dy * L), col, th + 1)
        if receiving:
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 1)
        cv2.putText(vis, "TOUCH" if touching else "HAND ZONE", (x0 + 6, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, col, th, cv2.LINE_AA)
        # 쟁반: YOLO 가 고른 부품 (요청 중일 때)
        tray = self.task.tray
        if self.tray_roi:
            tx0, ty0, tx1, ty1 = (int(self.tray_roi[0] * w), int(self.tray_roi[1] * h),
                                  int(self.tray_roi[2] * w), int(self.tray_roi[3] * h))
            cv2.rectangle(vis, (tx0, ty0), (tx1, ty1), (170, 170, 170), 1)
            cv2.putText(vis, "TRAY", (tx0 + 6, ty0 + int(24 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, (170, 170, 170), th)
        for d in tray["items"]:
            bx, by, bw, bh = d["box"]
            picked = tray["pick"] == d["n"]
            c = (40, 200, 40) if picked else (150, 150, 150)
            cv2.putText(vis, f"#{d['n']}", (bx, by + bh + int(28 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * k, c, th + 1, cv2.LINE_AA)
            if picked:
                cv2.rectangle(vis, (bx - 6, by - 6), (bx + bw + 6, by + bh + 6), c, th + 3)
                cv2.putText(vis, "PICK THIS", (bx, max(20, by - int(40 * k))), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9 * k, c, th + 1, cv2.LINE_AA)
        # 탐지 박스
        for d in self.overlay.get("dets", []):
            bx, by, bw, bh = d["box"]
            is_best = best is not None and tuple(best.get("box") or ()) == tuple(d["box"])
            good = self.task.target and d["label"] == self.task.target
            c = (90, 160, 90) if (is_best and good) else (70, 110, 200) if is_best else (150, 150, 150)
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), c, th + (1 if is_best else 0))
            text = f"{d['label'].upper()} {d['conf']:.2f}"
            (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, th)
            ty = max(tht + 10, by - 8)
            cv2.rectangle(vis, (bx, ty - tht - 8), (bx + tw + 12, ty + 6), c, -1)
            cv2.putText(vis, text, (bx + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (255, 255, 255), th, cv2.LINE_AA)
        return vis

    # ---- camera: capture + stream only ----
    def camera_off(self) -> bool:
        """지금 고른 카메라가 꺼짐 목록에 있으면 열지 않는다."""
        return self.camera_index in self.cam_disabled

    def _open_camera(self):
        self._cam_switch = False
        while self.running:
            if self.camera_off():                   # 꺼 둔 카메라: 열지 않고 켤 때까지 대기
                self.cam_error = ""
                self.jpeg = None
                self.fps = 0.0
                time.sleep(0.3)
                if self._cam_switch:
                    self._cam_switch = False
                continue
            cap = cv2.VideoCapture(self.camera_index)
            if cap.isOpened():
                self.cam_error = ""
                self._cam_opened_t = time.monotonic()
                self.task._log("SYS", f"USB 웹캠({self.camera_index}번) 연결됨")
                return cap
            cap.release()
            self.cam_error = f"USB 웹캠({self.camera_index}번)이 연결되지 않았어요 — 꽂으면 자동으로 연결돼요"
            self.jpeg = None
            time.sleep(2.0)
        return None

    def _camera_loop(self, gen: int = 0) -> None:
        cap = self._open_camera()
        fails = 0
        last = time.perf_counter()
        while self.running and cap is not None:
            if gen != self._cam_gen:                # 감시 장치가 새 스레드를 띄움 → 이 스레드는 정리하고 끝냄
                cap.release()
                return
            if self._cam_switch or self.camera_off():   # 다른 카메라를 고름 / 이 카메라를 끔
                cap.release()
                self.jpeg = None
                self.fps = 0.0
                cap = self._open_camera()
                fails = 0
                continue
            ok, frame = cap.read()
            if gen != self._cam_gen:
                cap.release()
                return
            if not ok:
                fails += 1
                if fails > 40:                      # 약 2초 동안 프레임 없음 = 분리
                    cap.release()
                    self.task._log("SYS", "USB 웹캠 연결이 끊겼어요 — 다시 연결을 기다려요")
                    self.jpeg = None
                    cap = self._open_camera()
                    fails = 0
                time.sleep(0.05)
                continue
            fails = 0
            self._last_frame_t = time.monotonic()
            with self.lock:
                self.frame = frame
                self.frame_id += 1
            vis = self._draw(frame)
            ok2, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok2:
                self.jpeg = buf.tobytes()
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
            last = now
        if cap is not None:
            cap.release()

    # ---- inference: latest frame → YOLO (or shape rule) + palm motion ----
    def _palm_gray(self, frame):
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = (int(self.palm_roi[0] * w), int(self.palm_roi[1] * h),
                          int(self.palm_roi[2] * w), int(self.palm_roi[3] * h))
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(cv2.resize(g, (160, max(1, int(160 * g.shape[0] / max(1, g.shape[1]))))), (5, 5), 0)

    def _infer_loop(self) -> None:
        last_id = -1
        prev_palm = None
        prev_state = "IDLE"
        while self.running:
            with self.lock:
                frame, fid = self.frame, self.frame_id
            if frame is None or fid == last_id:
                time.sleep(0.02)
                continue
            last_id = fid
            st = self.task.state
            receiving = st in ("RECEIVE", "VERIFY")
            if st == "RECEIVE" and prev_state != "RECEIVE":
                self.recv_bg = frame.copy()          # 손을 편 직후 = 빈 손 배경
            if st not in ("RECEIVE", "VERIFY", "HOME"):
                self.recv_bg = None
            prev_state = st
            # 손 영역 움직임 (사람 손이 아직 있는지)
            pg = self._palm_gray(frame)
            motion = 0.0
            if pg is not None and prev_palm is not None and prev_palm.shape == pg.shape:
                motion = float(cv2.absdiff(pg, prev_palm).mean())
            prev_palm = pg
            t0 = time.perf_counter()
            roi = self.palm_roi if (receiving or st == "HOME") else None
            if self.yolo is not None:
                try:
                    dets = self.yolo.detect(frame)
                    res = self.yolo.classify(frame, roi=roi, dets=dets)
                except Exception as exc:
                    self.task._log("ERR", f"YOLO 추론 오류: {exc}")
                    dets, res = [], PartResult("none", 0.0, "추론 오류")
            else:
                bg = self.recv_bg if self.recv_bg is not None else self.manual_bg
                res = self.clf.classify(frame, roi=roi, background=bg)
                dets = [{"label": res.label, "conf": res.confidence, "box": res.box}] if res.box else []
            self.infer_ms = 0.8 * self.infer_ms + 0.2 * (time.perf_counter() - t0) * 1000
            d = res.to_dict()
            d["motion"] = round(motion, 2)
            tray_items = self._tray_items(frame, dets)
            with self.lock:
                self.overlay = {"dets": dets, "best": d if res.label != "none" else None}
                self.result = d
                self.task.update_tray(tray_items)

    def _tray_items(self, frame, dets) -> list[dict]:
        """손 영역 밖(또는 지정한 쟁반 영역 안)의 볼트·너트 탐지"""
        if self.yolo is None:
            return []
        h, w = frame.shape[:2]
        px0, py0, px1, py1 = self.palm_roi
        out = []
        for d in dets:
            bx, by, bw, bh = d["box"]
            cx, cy = (bx + bw / 2) / w, (by + bh / 2) / h
            in_palm = px0 <= cx <= px1 and py0 <= cy <= py1
            in_tray = (self.tray_roi[0] <= cx <= self.tray_roi[2] and self.tray_roi[1] <= cy <= self.tray_roi[3]) \
                if self.tray_roi else not in_palm
            if in_tray and not in_palm and d["label"] in ("bolt", "nut") and d["conf"] >= TRAY_MIN_CONF:
                tx0, tx1 = (self.tray_roi[0], self.tray_roi[2]) if self.tray_roi else (0.0, 1.0)
                xn = round(max(0.0, min(1.0, (cx - tx0) / max(1e-6, tx1 - tx0))), 3)   # 쟁반 안 가로 위치 0=왼쪽 끝
                out.append({"label": d["label"], "conf": round(float(d["conf"]), 2), "box": tuple(d["box"]), "xn": xn})
        return out

    def _task_loop(self) -> None:
        was_connected = self.link.connected
        while self.running:
            with self.lock:
                res = self.result
                self.result = None
                now_connected = self.link.connected
                if was_connected and not now_connected:
                    self.task.link_lost()
                was_connected = now_connected
                self.task.step(res)
            self._camera_watchdog()
            time.sleep(0.05)

    STALL_S = 5.0

    def _camera_watchdog(self) -> None:
        """카메라가 열렸는데 영상이 5초 넘게 안 오면 (읽기가 멈춤) 새 스레드로 다시 연다."""
        if self.camera_off() or not self._cam_opened_t:
            return
        since = time.monotonic() - max(self._cam_opened_t, self._last_frame_t)
        if since < self.STALL_S:
            return
        self._cam_gen += 1
        self._cam_opened_t = 0.0
        self.jpeg = None
        self.fps = 0.0
        self.cam_error = f"카메라 {self.camera_index}번이 영상을 보내지 않아 다시 여는 중이에요 (USB 연결 · 허브 전원 확인)"
        self.task._log("ERR", f"카메라 {self.camera_index}번 영상 멈춤 {since:.0f}초 — 다시 엶")
        threading.Thread(target=self._camera_loop, args=(self._cam_gen,), daemon=True).start()

    def on_robot_connect(self) -> None:
        with self.lock:
            self.task._send("HOME")

    # ---- 카메라 선택 ----
    MAX_CAMERAS = 10

    @staticmethod
    def device_names() -> list[str]:
        """운영체제가 알려 주는 카메라 이름 목록 (번호와의 대응은 보장되지 않아 참고용)."""
        import platform
        import subprocess
        try:
            if platform.system() == "Darwin":
                r = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True, timeout=8)
                return [d.get("_name", "") for d in json.loads(r.stdout).get("SPCameraDataType", []) if d.get("_name")]
            if platform.system() == "Windows":
                r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                    "Get-PnpDevice -Class Camera,Image -Status OK | Select-Object -ExpandProperty FriendlyName"],
                                   capture_output=True, text=True, timeout=10)
                return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
            if platform.system() == "Linux":
                from glob import glob
                names = []
                for dev in sorted(glob("/sys/class/video4linux/video*/name")):
                    names.append(Path(dev).read_text().strip())
                return names
        except Exception:
            pass
        return []

    def cameras_in_use(self) -> dict:
        """{번호: 역할} — 지금 이 프로그램이 쓰는 카메라 (다시 열지 않고 최근 프레임으로 미리보기)"""
        return {self.camera_index: "hand"}

    def _frame_for_role(self, role: str):
        return self.frame

    @staticmethod
    def _thumb(frame) -> str:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, max(1, int(320 * h / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""

    def scan_cameras(self) -> list[dict]:
        """0~5번 카메라를 차례로 열어 미리보기 한 장씩. 쓰는 중인 카메라는 다시 열지 않는다."""
        in_use = self.cameras_in_use()
        out = []
        misses = 0
        for idx in range(self.MAX_CAMERAS):
            if idx in self.cam_disabled:            # 꺼 둔 카메라는 열지 않음 (불도 안 켜짐)
                out.append({"index": idx, "ok": False, "off": True, "role": in_use.get(idx)})
                misses = 0
                continue
            if idx in in_use:
                fr = self._frame_for_role(in_use[idx])
                item = {"index": idx, "ok": fr is not None, "role": in_use[idx]}
                if fr is not None:
                    item.update(width=fr.shape[1], height=fr.shape[0], thumb=self._thumb(fr))
                out.append(item)
                misses = 0
                continue
            cap = cv2.VideoCapture(idx)
            frame = None
            if cap.isOpened():
                for _ in range(12):                 # 자동 노출이 잡힐 때까지 몇 장 버림
                    ok, f = cap.read()
                    if ok:
                        frame = f
                    time.sleep(0.03)
            cap.release()
            if frame is None:
                misses += 1
                if misses >= 3 and idx >= 2:        # 연속 3번 없으면 그만 찾음
                    break
                continue
            misses = 0
            out.append({"index": idx, "ok": True, "role": None, "width": frame.shape[1],
                        "height": frame.shape[0], "thumb": self._thumb(frame)})
        found = ", ".join(str(c["index"]) + "번" for c in out) or "없음"
        self.task._log("SYS", f"카메라 찾기: {found}")
        return out

    def set_camera_power(self, index: int, on: bool) -> dict:
        """카메라 켜기/끄기 (번호별, 저장됨). 끈 카메라는 찾기·선택 때도 열지 않는다."""
        if on:
            self.cam_disabled.discard(index)
            self._cam_switch = True                 # 쓰는 카메라였다면 다시 엶
            msg = f"{index}번 카메라를 켰어요"
        else:
            self.cam_disabled.add(index)
            msg = f"{index}번 카메라를 껐어요"
        self._save_settings(camera_disabled=sorted(self.cam_disabled))
        self.task._log("SYS", msg)
        return {"ok": True, "reply": msg}

    def select_camera(self, role: str, index: int) -> dict:
        if role != "hand":
            return {"ok": False, "reply": "이 모드에서는 손 카메라만 고를 수 있어요"}
        if index in self.cam_disabled:
            return {"ok": False, "reply": f"{index}번은 꺼 둔 카메라예요 — 먼저 켜 주세요"}
        if index == self.camera_index:
            return {"ok": True, "reply": f"이미 {index}번 카메라를 쓰고 있어요"}
        self.camera_index = index
        self._cam_switch = True
        self._save_settings(camera_index=index)
        self.task._log("SYS", f"카메라를 {index}번으로 바꿨어요")
        return {"ok": True, "reply": f"{index}번 카메라로 바꾸는 중"}

    def _camera_ok(self) -> bool:
        return self.jpeg is not None and not self.cam_error

    def state(self) -> dict:
        with self.lock:
            snap = self.task.snapshot()
            snap["is_mock"] = self.link.is_mock
            snap["port"] = self.link.port
            snap["robot"] = self.link.info()
            snap["camera_roles"] = ["hand"]
            snap["camera_chosen"] = "camera_index" in self.read_settings()   # 아직 한 번도 고르지 않았으면 선택 창을 띄움
            snap["camera_disabled"] = sorted(self.cam_disabled)
            snap["camera"] = {"index": self.camera_index, "off": self.camera_off(), "error": self.cam_error, "fps": round(self.fps, 1),
                              "running": self._camera_ok(), "infer_ms": round(self.infer_ms)}
            snap["detector"] = {"name": self.detector_info, "yolo": self.yolo is not None,
                                "labels": list(self.yolo.names.values()) if self.yolo else list(LABELS)}
            snap["samples"] = self._capture_counts() if self.yolo else self.clf.counts()
            snap["palm_roi"] = list(self.palm_roi)
            snap["background"] = "receive" if self.recv_bg is not None else ("manual" if self.manual_bg is not None else "none")
            return snap

    CAPTURE_DIR = ROOT / "data" / "captures"

    def _capture_counts(self) -> dict:
        return {l: len(list((self.CAPTURE_DIR / l).glob("*.jpg"))) for l in ("bolt", "nut", "washer")}

    def action(self, p: dict) -> dict:
        a = p.get("action")
        if a == "camera_scan":                      # 카메라를 여는 데 시간이 걸려 잠금 밖에서
            cams = self.scan_cameras()
            st = self.state()
            st["result"] = {"ok": True, "cameras": cams, "names": self.device_names()}
            return st
        if a == "camera_power":
            try:
                idx, on = int(p["index"]), bool(p.get("on"))
            except (KeyError, TypeError, ValueError):
                st = self.state()
                st["result"] = {"ok": False, "reply": "카메라 번호가 잘못됐어요"}
                return st
            with self.lock:
                out = self.set_camera_power(idx, on)
            st = self.state()
            st["result"] = out
            return st
        if a == "camera_select":
            try:
                role, idx = str(p.get("role", "hand")), int(p["index"])
            except (KeyError, TypeError, ValueError):
                st = self.state()
                st["result"] = {"ok": False, "reply": "카메라 번호가 잘못됐어요"}
                return st
            with self.lock:
                out = self.select_camera(role, idx)
            st = self.state()
            st["result"] = out
            return st
        pre = None
        if a == "request" and self.task.vlm is not None and self.link.connected:
            pre = self.task.vlm.interpret(str(p.get("text", "")))   # VLM 은 느릴 수 있어 잠금 밖에서 먼저
        with self.lock:
            out: dict = {}
            if (a in ("request", "raw", "confirm", "hand")
                    or (a == "manual" and p.get("step") not in ("done", "fail", "idle", "placed", "skip_open", "receive"))) \
                    and not self.link.connected:
                out = {"ok": False, "reply": "로봇이 연결되지 않았어요 — USB · 전원을 확인하세요"}
            elif a == "manual":
                out = self.task.manual(str(p.get("step", "")))
            elif a == "request":
                out = self.task.request(str(p.get("text", "")), pre=pre)
            elif a in ("confirm", "hand"):
                if self._camera_ok():
                    self.task.manual_confirm()
                else:
                    self.task.confirm_without_camera()
            elif a == "cancel":
                self.task.cancel()
            elif a == "stop":
                self.task.stop()
            elif a == "resume":
                self.task.resume()
            elif a in ("learn", "bg", "roi") and not self._camera_ok():
                out = {"ok": False, "reply": "카메라가 연결되지 않았어요"}
            elif a == "learn" and p.get("label") in ("bolt", "nut", "washer"):
                lbl = p["label"]
                if self.yolo is not None:
                    # YOLO 사용 중: 재학습용 사진 저장 (라벨링은 별도)
                    d = self.CAPTURE_DIR / lbl
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"{int(time.time() * 1000)}.jpg"), self.frame)
                    out = {"ok": True, "reply": f"{LABEL_KO[lbl]} 사진 저장 (재학습용)"}
                else:
                    ok = self.frame is not None and self.clf.add_sample(self.frame, lbl, background=self.manual_bg)
                    out = {"ok": bool(ok), "reply": f"{LABEL_KO[lbl]} 샘플 저장" if ok else "부품이 안 보여요"}
                self.task._log("SYS", f"[데이터] {out['reply']}")
            elif a == "roi":
                try:
                    ok = self.set_roi(p["x0"], p["y0"], p["x1"], p["y1"])
                except (KeyError, TypeError, ValueError):
                    ok = False
                out = {"ok": ok, "reply": "손 영역을 저장했어요" if ok else "영역이 너무 작아요"}
                if ok:
                    self.task._log("SYS", f"손 영역 지정: {self.palm_roi}")
            elif a == "bg_clear":
                self.manual_bg = None
                out = {"ok": True, "reply": "빈 화면 배경을 지웠어요"}
                self.task._log("SYS", "빈 화면(배경) 지움")
            elif a == "roi_reset":
                self.palm_roi = self.DEFAULT_ROI
                self.ROI_FILE.unlink(missing_ok=True)
                out = {"ok": True, "reply": "손 영역을 기본값으로 되돌렸어요"}
            elif a == "place_angle":
                try:
                    ang = max(0, min(180, int(float(p.get("value")))))
                    self.task.place_angle = ang
                    self._save_settings()
                    out = {"ok": True, "reply": f"내려놓기 각도 {ang}°"}
                    self.task._log("SYS", f"내려놓기 손목 각도 → {ang}°")
                except (TypeError, ValueError):
                    out = {"ok": False, "reply": "각도 형식이 잘못됐어요"}
            elif a == "bg":
                if self.frame is not None:
                    self.manual_bg = self.frame.copy()
                    out = {"ok": True, "reply": "빈 화면을 배경으로 저장했어요"}
                    self.task._log("SYS", "빈 화면(배경) 저장")
            elif a == "raw" and str(p.get("cmd", "")).split(",")[0] in ("OPEN", "GRIP", "PLACE", "HOME", "PAN", "NG"):
                head = str(p["cmd"]).split(",")[0]
                cmd = head
                if head in ("PAN", "PLACE"):
                    try:
                        arg = str(p["cmd"]).split(",")[1] if "," in str(p["cmd"]) else str(self.task.place_angle)
                        cmd = f"{head},{max(0, min(180, int(float(arg))))}"
                    except (IndexError, ValueError):
                        cmd = None
                if cmd and self.task.state == "IDLE" and not self.task.stopped and not self.task._awaiting:
                    self.task._send(cmd)
                    out = {"ok": True}
                elif not cmd:
                    out = {"ok": False, "reply": "각도 형식이 잘못됐어요"}
                else:
                    out = {"ok": False, "reply": "작업 중이거나 정지 상태예요"}
        st = self.state()
        st["result"] = out
        return st


def make_handler(rt: Runtime):
    class H(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *a):
            pass

        def _json(self, d, code=200):
            b = json.dumps(d, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/api/state":
                return self._json(rt.state())
            if self.path.startswith("/api/video"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while rt.running:
                        jpg = rt.jpeg
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.07)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            return super().do_GET()

        def do_POST(self):
            if self.path == "/api/action":
                n = int(self.headers.get("Content-Length", "0"))
                try:
                    p = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    return self._json({"error": "invalid json"}, 400)
                return self._json(rt.action(p))
            self._json({"error": "not found"}, 404)

    return functools.partial(H, directory=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="AiHand 받기 작업 (로컬 인식)")
    ap.add_argument("--port", default="", help="micro:bit 시리얼 포트 (생략하면 자동 탐색 · 재연결)")
    ap.add_argument("--mock", action="store_true", help="로봇 없이 모의 micro:bit로 실행")
    ap.add_argument("--camera", type=int, default=None,
                    help="웹캠 번호 (생략하면 대시보드에서 마지막으로 고른 카메라, 처음엔 1). 대시보드 [카메라 선택]으로 바꿀 수 있음")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "neuralgrip_best_two.pt",
                    help="YOLO 가중치 (팀 best_two.pt: 볼트·너트). 없거나 불러오지 못하면 모양 규칙으로 동작")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO 탐지 최소 신뢰도 (쥐기 판단은 0.6 이상)")
    ap.add_argument("--vlm", action="store_true", help="VLM(Ollama) 연결: 요청 해석 보조 + 쥐기 전 교차 확인")
    ap.add_argument("--vlm-model", default="gemma3:4b")
    ap.add_argument("--vlm-url", default="http://localhost:11434")
    ap.add_argument("--vlm-verify-below", type=float, default=0.70,
                    help="YOLO 점수가 이보다 낮을 때만 VLM 교차 확인 (0 = 교차 확인 끔, 1 = 항상)")
    ap.add_argument("--tray-roi", default="", help="쟁반 영역 x0,y0,x1,y1 (0~1). 생략하면 손 영역 밖 전체를 쟁반으로 봄")
    ap.add_argument("--point", action="store_true", help="쟁반에서 고른 부품 쪽으로 손목을 돌려 알려 주기")
    ap.add_argument("--point-left", type=float, default=135, help="쟁반 왼쪽 끝을 향하는 손목 각도 (보정)")
    ap.add_argument("--point-right", type=float, default=45, help="쟁반 오른쪽 끝을 향하는 손목 각도 (보정)")
    ap.add_argument("--point-hold", type=float, default=1.5, help="가리킨 채 멈추는 시간 (초)")
    ap.add_argument("--http-port", type=int, default=8090)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    link = MockLink() if a.mock else RobotLink(port_hint=a.port)
    vlm = None
    if a.vlm:
        from vision.vlm_bridge import VlmBridge
        vlm = VlmBridge(a.vlm_model, a.vlm_url, verify_below=a.vlm_verify_below)
    tray_roi = tuple(float(v) for v in a.tray_roi.split(",")) if a.tray_roi else None
    if a.camera is None:
        a.camera = int(Runtime.read_settings().get("camera_index", 1))
    rt = Runtime(link, a.camera, model_path=a.model, conf=a.conf, vlm=vlm, tray_roi=tray_roi)
    if a.point:
        rt.task.set_pointing(a.point_left, a.point_right, a.point_hold)
        rt.task._log("SYS", f"가리키기 켜짐 — 쟁반 왼쪽 끝 {a.point_left:.0f}° · 오른쪽 끝 {a.point_right:.0f}°"
                     + ("" if tray_roi else " (쟁반 영역 미지정: 화면 전체 가로 기준)"))
    if not link.is_mock:
        link.log = rt.task._log
        link.on_connect = rt.on_robot_connect
        link.start()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.http_port), make_handler(rt))
    url = f"http://127.0.0.1:{a.http_port}/index.html"
    print(f"[AI-HAND] 로봇: {'모의' if link.is_mock else '자동 연결'} · 카메라 {a.camera}번 · 대시보드 {url}")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rt.running = False
        link.close()


if __name__ == "__main__":
    main()
