"""AiHand 받기 작업 실행 — 기존 대시보드(index.html)에 실물 과정을 그대로 보여준다.

  .venv/bin/python handover_app.py --port /dev/cu.usbmodem145102 --camera 1
  .venv/bin/python handover_app.py --mock            # 로봇 없이 시험 (모의 micro:bit)

과정: 입력(볼트/너트/와셔) → 손 펴기 → 웹캠 로컬 인식 → 쥐기 → 왼쪽(180°) 회전·내려놓기 → 복귀
"""
from __future__ import annotations

import argparse
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


class Runtime:
    ROI_FILE = ROOT / "data" / "palm_roi.json"
    SETTINGS_FILE = ROOT / "data" / "handover_settings.json"
    DEFAULT_ROI = (0.30, 0.35, 0.65, 0.75)       # 팀 zones.json 의 palm 영역

    def __init__(self, link, camera_index: int, model_path: Path | None = None, conf: float = 0.35):
        self.link = link
        self.task = HandoverTask(link)
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
        self.camera_index = camera_index
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
        threading.Thread(target=self._camera_loop, daemon=True).start()
        threading.Thread(target=self._infer_loop, daemon=True).start()
        threading.Thread(target=self._task_loop, daemon=True).start()

    # ---- settings ----
    def _load_roi(self):
        try:
            return tuple(float(x) for x in json.loads(self.ROI_FILE.read_text()))
        except Exception:
            return self.DEFAULT_ROI

    def _load_settings(self) -> None:
        try:
            st = json.loads(self.SETTINGS_FILE.read_text())
            self.task.place_angle = int(st.get("place_angle", self.task.place_angle))
        except Exception:
            pass

    def _save_settings(self) -> None:
        self.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.SETTINGS_FILE.write_text(json.dumps({"place_angle": self.task.place_angle}))

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
    def _open_camera(self):
        while self.running:
            cap = cv2.VideoCapture(self.camera_index)
            if cap.isOpened():
                self.cam_error = ""
                self.task._log("SYS", f"USB 웹캠({self.camera_index}번) 연결됨")
                return cap
            cap.release()
            self.cam_error = f"USB 웹캠({self.camera_index}번)이 연결되지 않았어요 — 꽂으면 자동으로 연결돼요"
            self.jpeg = None
            time.sleep(2.0)
        return None

    def _camera_loop(self) -> None:
        cap = self._open_camera()
        fails = 0
        last = time.perf_counter()
        while self.running and cap is not None:
            ok, frame = cap.read()
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
            with self.lock:
                self.overlay = {"dets": dets, "best": d if res.label != "none" else None}
                self.result = d

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
            time.sleep(0.05)

    def on_robot_connect(self) -> None:
        with self.lock:
            self.task._send("HOME")

    def _camera_ok(self) -> bool:
        return self.jpeg is not None and not self.cam_error

    def state(self) -> dict:
        with self.lock:
            snap = self.task.snapshot()
            snap["is_mock"] = self.link.is_mock
            snap["port"] = self.link.port
            snap["robot"] = self.link.info()
            snap["camera"] = {"index": self.camera_index, "error": self.cam_error, "fps": round(self.fps, 1),
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
        with self.lock:
            out: dict = {}
            if a in ("request", "raw", "confirm", "hand") and not self.link.connected:
                out = {"ok": False, "reply": "로봇이 연결되지 않았어요 — USB · 전원을 확인하세요"}
            elif a == "request":
                out = self.task.request(str(p.get("text", "")))
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
    ap.add_argument("--camera", type=int, default=1, help="웹캠 번호 (이 Mac: 0=내장 FaceTime, 1=추가 USB 웹캠)")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "neuralgrip_best_two.pt",
                    help="YOLO 가중치 (팀 best_two.pt: 볼트·너트). 없거나 불러오지 못하면 모양 규칙으로 동작")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO 탐지 최소 신뢰도 (쥐기 판단은 0.6 이상)")
    ap.add_argument("--http-port", type=int, default=8090)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    link = MockLink() if a.mock else RobotLink(port_hint=a.port)
    rt = Runtime(link, a.camera, model_path=a.model, conf=a.conf)
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
