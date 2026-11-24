"""AiHand 받기 작업 실행 — 기존 대시보드(index.html)에 실물 과정을 그대로 보여준다.

  .venv/bin/python handover_app.py                (macOS/Linux)
  .venv\\Scripts\\python.exe handover_app.py        (Windows, run_windows.bat)
  .venv/bin/python handover_app.py --mock            # 로봇 없이 시험 (모의 micro:bit)

과정: 입력(빨간 공/파란 공/초록 공) → 손 펴기 → 웹캠 로컬 인식 → 쥐기 → 왼쪽(180°) 회전·내려놓기 → 복귀
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
import functools
import http.server
import json
import re
from pathlib import Path
import sys
import threading
import time
import webbrowser

import cv2
import numpy as np

from controller.handover import HandoverTask
from vision.parts import LABEL_KO, LABELS, PartClassifier, PartResult, YoloParts, detect_change, find_balls, \
    hand_zone_from_motion

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

    MICROBIT_VID = 0x0D28                    # BBC micro:bit (DAPLink / CMSIS-DAP)

    @classmethod
    def list_microbit_ports(cls) -> list[str]:
        """Windows(COM5) · macOS(/dev/cu.usbmodem…) · Linux(/dev/ttyACM0) 공통 micro:bit 포트 찾기."""
        from serial.tools import list_ports
        found = []
        for p in list_ports.comports():
            desc = f"{p.description} {p.manufacturer or ''} {p.product or ''}".lower()
            if p.vid == cls.MICROBIT_VID or "micro:bit" in desc or "microbit" in desc or "mbed" in desc or "daplink" in desc:
                found.append(p.device)
        if not found:                         # 드라이버가 이름을 안 알려 줄 때: macOS/Linux 장치 이름으로
            found = [p.device for p in list_ports.comports()
                     if "usbmodem" in p.device or p.device.startswith("/dev/ttyACM")]
        return sorted(found)

    def _find_port(self) -> str:
        ports = self.list_microbit_ports()
        if self.port_hint and self.port_hint in ports:
            return self.port_hint
        if self.port_hint and not ports:
            from serial.tools import list_ports
            if any(p.device == self.port_hint for p in list_ports.comports()):
                return self.port_hint           # 직접 지정한 포트 (예: --port COM5)
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
                self.log("ERR", "펌웨어 설치: 프로그램을 멈추지 못했습니다 (REPL 프롬프트 없음)")
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
                    self.log("ERR", "펌웨어 자동 설치 실패 — 60초 뒤 다시 시도합니다 (기존 펌웨어로 계속 연결)")
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
                self.log("ERR", "로봇이 응답하지 않습니다 (전원 · 케이블 확인)")


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


BALL_BGR = {"red": (106, 113, 192), "blue": (181, 143, 111), "green": (122, 160, 121)}   # 대시보드 물빠진 공 색
TRAY_MIN_CONF = 0.45          # 쟁반 안내에 쓸 최소 점수 (색 규칙: 경계 색 0.4 는 제외)


MAX_VIEWS = 4                 # 카메라 칸 분할: 1~4 (1번 칸 = 손 카메라)
VIEW_SIZE = (640, 480)        # 추가 카메라 해상도: 웹캠 여러 대를 한 USB 허브에 꽂아도 대역폭이 남도록 작게 + MJPG


def draw_tray_items(vis, tray: dict, k: float, th: int) -> None:
    """쟁반에서 찾은 공: 공마다 그 색 상자 + 번호, 요청한 공은 굵게 + PICK THIS"""
    for d in tray.get("items", []):
        bx, by, bw, bh = d["box"]
        picked = tray.get("pick") == d["n"]
        c = BALL_BGR.get(d["label"], (150, 150, 150))
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), c, th + (3 if picked else 0))
        cv2.putText(vis, f"#{d['n']} {d['label'].upper()}", (bx, by + bh + int(28 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * k, c, th + 1, cv2.LINE_AA)
        if picked:
            ty = by - int(14 * k) if by > int(50 * k) else by + bh + int(64 * k)
            cv2.putText(vis, "PICK THIS", (bx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * k, c, th + 1, cv2.LINE_AA)


class ViewCam:
    """분할 화면의 추가 카메라 한 대 (웹캠 번호 또는 시험용 사진·영상 경로). 자기 스레드에서 읽고 JPEG 로 보낸다."""

    def __init__(self, rt: "Runtime", slot: int, src):
        self.rt, self.slot, self.src = rt, slot, src
        self.frame = None
        self.jpeg: bytes | None = None
        self.error = ""
        self.fps = 0.0
        self.alive = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self.alive = False

    def _open(self):
        src = self.src
        if isinstance(src, str) and not src.isdigit():
            img = cv2.imread(src)
            if img is not None:
                return "image", img
            cap = cv2.VideoCapture(src)                      # 영상 파일
            return ("cap", cap) if cap.isOpened() else (None, None)
        idx = int(src)
        if idx in self.rt.cam_disabled:
            return None, None
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # 웹캠 여러 대: 압축(MJPG) + 작은 해상도로 USB 대역폭을 나눠 씀 (지원 안 하는 카메라는 무시하고 기본값)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIEW_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIEW_SIZE[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return "cap", cap
        cap.release()
        return None, None

    def _loop(self) -> None:
        kind, src = None, None
        last = time.perf_counter()
        fails = 0
        while self.alive and self.rt.running:
            if self.src == self.rt.camera_index:             # 손 카메라와 같은 번호 → 다시 열지 않고 그 영상을 씀
                self.frame, self.jpeg, self.error, self.fps = self.rt.frame, self.rt.jpeg, self.rt.cam_error, self.rt.fps
                time.sleep(0.07)
                continue
            if kind == "cap" and self.off():               # 대시보드에서 이 카메라를 끔 → 바로 놓아 줌
                src.release()
                kind, src = None, None
            if kind is None:
                kind, src = self._open()
                if kind is None:
                    self.error = "꺼 둔 카메라입니다" if self.off() else f"카메라 {self.src}번이 연결되지 않았습니다"
                    self.frame = self.jpeg = None
                    time.sleep(2.0)
                    continue
                self.error = ""
                fails = 0
            if kind == "image":
                frame = src.copy()
                time.sleep(0.1)
            else:
                ok, frame = src.read()
                if not ok:
                    if not str(self.src).isdigit():          # 영상 파일 끝 → 처음부터
                        src.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    fails += 1
                    if fails > 40:
                        src.release()
                        kind, src = None, None
                    time.sleep(0.05)
                    continue
                fails = 0
            self.frame = frame
            vis = frame.copy()
            h, w = vis.shape[:2]
            k = max(1.0, w / 640)
            th = max(2, int(2 * k))
            if self.rt.tray_slot == self.slot and self.rt.DRAW_IN_VIDEO:
                draw_tray_items(vis, self.rt.task.tray, k, th)
            ok2, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok2:
                self.jpeg = buf.tobytes()
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
            last = now
            if kind == "cap":
                time.sleep(0.01)
        if kind == "cap" and src is not None:
            src.release()

    def off(self) -> bool:
        return str(self.src).isdigit() and int(self.src) in self.rt.cam_disabled

    def info(self) -> dict:
        return {"slot": self.slot, "src": self.src, "ok": self.jpeg is not None and not self.error, "off": self.off(),
                "error": self.error, "fps": round(self.fps, 1),
                "shared": self.src == self.rt.camera_index,
                "size": [self.frame.shape[1], self.frame.shape[0]] if self.frame is not None else None}


class Runtime:
    ROI_FILE = ROOT / "data" / "palm_roi.json"
    SETTINGS_FILE = ROOT / "data" / "handover_settings.json"
    DEFAULT_ROI = (0.30, 0.35, 0.65, 0.75)       # 팀 zones.json 의 palm 영역

    def __init__(self, link, camera_index: int, model_path: Path | None = None, conf: float = 0.35, vlm=None,
                 tray_roi: tuple | None = None):
        self.tray_roi = tray_roi                    # 쟁반 영역(정규화). 없으면 '손 영역 밖 전체'를 쟁반으로 봄
        self.link = link
        self.task = HandoverTask(link, vlm=vlm)
        self.task.frame_source = lambda: self.frame           # VLM 확인용 현재 프레임
        self.task.palm_crop = self._palm_crop                  # VLM 에 보여 줄 손 영역 사진
        self.task.on_grip_approved = lambda img: self.start_describe(img, "손 위 (쥐기 승인 직후)")
        self.vlm_detail: dict | None = None                    # VLM 관찰 (여러 정보) 마지막 결과
        self._locating = False
        self.locate_status: dict | None = None                 # [손 자동 찾기] 진행·결과
        self._describe_busy = False
        self.clf = PartClassifier()                 # YOLO 를 못 쓸 때의 대체 판별기
        self.yolo = None
        self.detector_info = "색 규칙 (공 감지 · VLM 이 꺼져 있을 때 판정)"
        if model_path and Path(model_path).exists():
            try:
                self.yolo = YoloParts(model_path, conf=conf)
                self.detector_info = f"YOLO {Path(model_path).name} · {', '.join(self.yolo.names.values())}"
            except Exception as exc:                # ultralytics/torch 미설치 등
                self.detector_info = f"색 규칙 (YOLO 불러오기 실패: {exc.__class__.__name__})"
        self.task._log("SYS", f"인식기: {self.detector_info}")
        if vlm is not None:
            seen_online = [False]

            def _vlm_changed(online: bool) -> None:
                if online:
                    seen_online[0] = True
                    self.task._log("SYS", f"VLM 연결됨: {vlm.model} ({vlm.url}) — 검색창 문장 이해 + 손 위 공 확인")
                else:                               # 처음부터 꺼져 있으면 조용히, 쓰다가 끊기면 알람
                    self.task._log("ERR" if seen_online[0] else "SYS",
                                   f"VLM 꺼짐: {vlm.status_note} — 단어 규칙·색 규칙으로 동작")
            vlm.start_monitor(on_change=_vlm_changed)
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
        # 분할 화면: 2~4번 칸에 추가 카메라 (설정에 저장), 그중 한 칸을 쟁반으로
        st = self.read_settings()
        self.split = max(1, min(MAX_VIEWS, int(st.get("split", 1))))
        self.views: dict[int, ViewCam] = {}
        self.tray_slot: int | None = st.get("tray_slot")
        self._tray_view_items: list[dict] = []
        # 화면 배치: CAM 1~4 자리에 어떤 칸을 둘지 ("hand" = 손 카메라, "2"~"4" = 추가 카메라)
        order = [str(x) for x in (st.get("view_order") or [])]
        self.view_order = order if sorted(order) == ["2", "3", "4", "hand"] else ["hand", "2", "3", "4"]
        threading.Thread(target=self._camera_loop, args=(0,), daemon=True).start()
        threading.Thread(target=self._infer_loop, daemon=True).start()
        threading.Thread(target=self._task_loop, daemon=True).start()
        threading.Thread(target=self._tray_view_loop, daemon=True).start()
        self.auto_cams = bool(st.get("auto_cams", True))   # 카메라가 새로 잡히면 빈 CAM 칸에 자동으로 띄움
        self._probe_lock = threading.Lock()
        threading.Thread(target=self._discovery_loop, daemon=True).start()
        for k, src in (st.get("view_sources") or {}).items():
            if src is not None and 2 <= int(k) <= MAX_VIEWS:
                self.views[int(k)] = ViewCam(self, int(k), src)

    # ---- 분할 화면 ----
    def set_view(self, slot: int, src) -> dict:
        if not 2 <= slot <= MAX_VIEWS:
            return {"ok": False, "reply": "칸 번호가 잘못됐습니다"}
        old = self.views.pop(slot, None)
        if old is not None:
            old.stop()
        if src in (None, "", "none"):
            if self.tray_slot == slot:
                self.tray_slot = None
            msg = f"{slot}번 칸을 비웠습니다"
        else:
            src = int(src) if str(src).isdigit() else str(src)
            self.views[slot] = ViewCam(self, slot, src)
            msg = f"{slot}번 칸: 카메라 {src}" + (" (손 카메라와 같은 영상)" if src == self.camera_index else "")
        self._save_views()
        self.task._log("SYS", msg)
        return {"ok": True, "reply": msg}

    def visible_slots(self) -> list[str]:
        return self.view_order[: self.split]

    def set_split(self, n: int) -> dict:
        self.split = max(1, min(MAX_VIEWS, int(n)))
        shown = self.visible_slots()
        for k in [k for k in self.views if str(k) not in shown]:   # 화면에서 빠진 칸의 카메라는 닫음 (USB 대역폭 · 불빛)
            self.views.pop(k).stop()
            if self.tray_slot == k:
                self.tray_slot = None
        self._save_views()
        return {"ok": True, "reply": f"카메라 {self.split}분할"}

    def set_tray_slot(self, slot) -> dict:
        slot = int(slot) if slot not in (None, "", "none") else None
        if slot is not None and slot not in self.views:
            return {"ok": False, "reply": "먼저 그 칸에 카메라를 고르십시오"}
        self.tray_slot = None if slot == self.tray_slot else slot          # 같은 칸을 다시 누르면 해제
        self._save_views()
        msg = f"{self.tray_slot}번 칸을 쟁반 카메라로 씁니다 (색 규칙)" if self.tray_slot else "쟁반 카메라 해제 — 손 카메라 화면에서 찾습니다"
        self.task._log("SYS", msg)
        return {"ok": True, "reply": msg}

    def _tray_view(self):
        """쟁반으로 지정한 칸 (손 카메라와 다른 카메라일 때만)"""
        tv = self.views.get(self.tray_slot) if self.tray_slot else None
        return tv if tv is not None and tv.src != self.camera_index else None

    def _tray_view_loop(self) -> None:
        """쟁반 칸: 손 카메라가 없어도 따로 돈다. 요청 중일 때만 초당 5번 색 규칙으로 공을 찾음."""
        while self.running:
            time.sleep(0.2)
            tv = self._tray_view()
            if tv is None or self.yolo is not None:
                continue
            active = self.task.target is not None and self.task.state in ("REQUEST", "RECEIVE")
            tf = tv.frame
            if not active or tf is None:
                self._tray_view_items = []
            else:
                fw = tf.shape[1]
                self._tray_view_items = [{"label": d["label"], "conf": d["conf"], "box": d["box"],
                                          "xn": round((d["box"][0] + d["box"][2] / 2) / fw, 3)}
                                         for d in find_balls(tf) if d["conf"] >= TRAY_MIN_CONF]
            with self.lock:
                self.task.update_tray(list(self._tray_view_items))

    def set_view_order(self, order) -> dict:
        order = [str(x) for x in (order or [])]
        if sorted(order) != ["2", "3", "4", "hand"]:
            return {"ok": False, "reply": "배치가 잘못됐습니다"}
        self.view_order = order
        self._save_views()
        return {"ok": True, "reply": "카메라 배치를 바꿨습니다"}

    # ---- 카메라 자동 감지: 운영체제가 아는 카메라 수가 바뀌면 한 번만 열어 보고 빈 칸에 배치 ----
    @staticmethod
    def os_camera_count() -> int | None:
        """카메라를 열지 않고 (불이 켜지지 않게) 연결된 카메라 수만 셈. 모르면 None."""
        import platform
        import subprocess
        try:
            if platform.system() == "Windows":
                r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                    "@(Get-PnpDevice -PresentOnly -Class Camera,Image -ErrorAction SilentlyContinue).Count"],
                                   capture_output=True, text=True, timeout=8,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                return int(r.stdout.strip() or 0)
            if platform.system() == "Linux":
                from glob import glob
                return len(glob("/sys/class/video4linux/video*"))
            return len(Runtime.device_names())
        except Exception:
            return None

    def probe_cameras(self) -> list[int]:
        """지금 열리는 카메라 번호들 (쓰는 중인 카메라는 다시 열지 않고 그대로 포함)."""
        with self._probe_lock:
            in_use = self.cameras_in_use()
            found, misses = [], 0
            for idx in range(self.MAX_CAMERAS):
                if idx in self.cam_disabled:
                    continue
                if idx in in_use:                       # 쓰는 중인 카메라는 다시 열지 않음 (윈도우는 한 프로그램만 열 수 있음)
                    role = in_use[idx]
                    v = self.views.get(int(role[4:])) if role.startswith("view") else None
                    alive = (self.jpeg is not None or not self.cam_error) if role == "hand" else                         (v is not None and (v.jpeg is not None or not v.error))
                    if alive:
                        found.append(idx)
                        misses = 0
                    else:
                        misses += 1
                    continue
                cap = cv2.VideoCapture(idx)
                ok = cap.isOpened() and cap.read()[0]
                cap.release()
                if ok:
                    found.append(idx)
                    misses = 0
                else:
                    misses += 1
                    if misses >= 3 and idx >= 2:
                        break
            return found

    def auto_place(self, found: list[int]) -> list[str]:
        """찾은 카메라를 CAM 칸에: 손 카메라는 지금 번호가 살아 있으면 그대로, 아니면 첫 번째 카메라.
        나머지는 보이는 순서대로 빈 칸에, 모자라면 분할을 늘림 (최대 4)."""
        notes = []
        if not found:
            return notes
        if self.camera_index not in found and not self.camera_off():   # 일부러 끈 손 카메라는 바꾸지 않음
            new_hand = found[0]
            if new_hand != self.camera_index:
                self.select_camera("hand", new_hand)
                notes.append(f"손 카메라 → {new_hand}번")
        shown = {v.src for v in self.views.values()}
        extra = [i for i in found if i != self.camera_index and i not in shown]
        if not extra:
            return notes
        want = min(MAX_VIEWS, 1 + len(self.views) + len(extra))
        if want > self.split:
            self.set_split(want)
        for idx in extra:
            free = [k for k in self.visible_slots() if k != "hand" and int(k) not in self.views]
            if not free:
                notes.append(f"{idx}번은 칸이 모자라 띄우지 못함 (최대 {MAX_VIEWS}칸)")
                continue
            self.set_view(int(free[0]), idx)
            notes.append(f"{idx}번 → CAM {self.view_order.index(free[0]) + 1}")
        return notes

    def _discovery_loop(self) -> None:
        last = None
        time.sleep(3.0)                                  # 손 카메라가 먼저 열리도록
        while self.running:
            n = self.os_camera_count()
            if self.auto_cams and (n is None and last is None or n is not None and n != last):
                try:
                    found = self.probe_cameras()
                    if n:                                # 카메라 한 대가 번호 두 개로 열리는 경우가 있어 운영체제가 센 수까지만
                        keep = [i for i in found if i == self.camera_index][:1]
                        found = keep + [i for i in found if i not in keep][: max(0, n - len(keep))]
                    notes = self.auto_place(found)
                    if notes:
                        self.task._log("SYS", "[카메라 자동] " + " · ".join(notes))
                    elif last is not None and n is not None and n < last:
                        self.task._log("SYS", "[카메라 자동] 카메라가 빠졌습니다 — 다시 꽂으면 그 칸에 다시 연결됩니다")
                except Exception as exc:
                    self.task._log("ERR", f"[카메라 자동] 찾기 오류: {exc.__class__.__name__}")
            last = n if n is not None else (last if last is not None else -1)
            time.sleep(4.0 if n is not None else 15.0)

    def set_auto_cams(self, on: bool) -> dict:
        self.auto_cams = bool(on)
        self._save_settings(auto_cams=self.auto_cams)
        notes = []
        if self.auto_cams:
            notes = self.auto_place(self.probe_cameras())
        msg = ("카메라 자동 배치 켬" + (" — " + " · ".join(notes) if notes else "")) if self.auto_cams else "카메라 자동 배치 끔"
        self.task._log("SYS", msg)
        return {"ok": True, "reply": msg}

    def _save_views(self) -> None:
        self._save_settings(split=self.split, tray_slot=self.tray_slot, view_order=self.view_order,
                            view_sources={str(k): v.src for k, v in self.views.items()})

    def view_jpeg(self, slot: int):
        v = self.views.get(slot)
        return v.jpeg if v is not None else None

    # ---- settings ----
    def _load_roi(self):
        try:
            return tuple(float(x) for x in json.loads(self.ROI_FILE.read_text(encoding="utf-8")))
        except Exception:
            return self.DEFAULT_ROI

    @classmethod
    def read_settings(cls) -> dict:
        try:
            return json.loads(cls.SETTINGS_FILE.read_text(encoding="utf-8"))
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
        self.SETTINGS_FILE.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")

    def start_describe(self, img, source: str) -> dict:
        """VLM 관찰: 무엇인지 · 색 · 개수 · 위치 · 상태 · 주변 · 메모 를 한 번에 (뒤에서, 약 2~3초)"""
        vlm = self.task.vlm
        if vlm is None or not vlm.online:
            return {"ok": False, "reply": "VLM 이 꺼져 있어 관찰할 수 없습니다"}
        if img is None:
            return {"ok": False, "reply": "카메라 영상이 없습니다"}
        if self._describe_busy:
            return {"ok": False, "reply": "VLM 이 이미 관찰하는 중입니다"}
        self._describe_busy = True
        self.vlm_detail = {"source": source, "busy": True, "fields": (self.vlm_detail or {}).get("fields", {})}

        def done(fut):
            try:
                r = fut.result()
            except Exception:
                r = {"fields": {}, "seconds": 0, "ok": False}
            self.vlm_detail = {"source": source, "busy": False, "time": time.strftime("%H:%M:%S"), **r}
            self._describe_busy = False
            if r["ok"]:
                f = r["fields"]
                self.task._log("SYS", f"[VLM 관찰] {source}: {f.get('object', '-')} · {f.get('state', '-')} · "
                                      f"{f.get('position', '-')} ({r['seconds']}s)")
        vlm.submit_describe(img.copy()).add_done_callback(done)
        return {"ok": True, "reply": f"VLM이 {source} 화면을 관찰하는 중입니다"}

    # ---- [손 자동 찾기]: 손을 쥐었다 펴서 움직인 자리 = 로봇 손 → VLM 이 확인 → 손 영역으로 ----
    def locate_hand(self) -> dict:
        if self._locating:
            return {"ok": False, "reply": "이미 손을 찾는 중입니다"}
        if self.task.state != "IDLE" or self.task.stopped or self.task._awaiting:
            return {"ok": False, "reply": "대기 중일 때만 찾을 수 있습니다 (로봇이 움직입니다)"}
        if not self.link.connected:
            return {"ok": False, "reply": "로봇이 연결되지 않았습니다"}
        if not self._camera_ok():
            return {"ok": False, "reply": "손 카메라 영상이 없습니다"}
        self._locating = True
        self.locate_status = {"phase": "moving", "msg": "손을 쥐었다 펴서 위치를 찾는 중"}
        threading.Thread(target=self._locate_hand_run, daemon=True).start()
        return {"ok": True, "reply": "손을 한 번 쥐었다 펴서 위치를 찾습니다 — 손 주변을 비워 주십시오"}

    def _move_and_wait(self, cmd: str, timeout: float = 6.0) -> bool:
        with self.lock:
            if not self.task._send(cmd):
                return False
        t0 = time.monotonic()
        while self.task._awaiting and time.monotonic() - t0 < timeout:
            time.sleep(0.05)
        time.sleep(0.4)                                   # 흔들림이 멎을 때까지
        return not self.task._awaiting

    def _still_frame(self, n: int = 5):
        frames = []
        for _ in range(n):
            if self.frame is not None:
                frames.append(self.frame.copy())
            time.sleep(0.06)
        return np.median(np.stack(frames), axis=0).astype(np.uint8) if frames else None

    def _locate_hand_run(self) -> None:
        log = self.task._log
        try:
            ok = self._move_and_wait("OPEN")
            f_open = self._still_frame()
            ok = ok and self._move_and_wait("GRIP")
            f_grip = self._still_frame()
            ok = ok and self._move_and_wait("OPEN")
            f_open2 = self._still_frame()
            if not ok:
                self.locate_status = {"phase": "fail", "msg": "로봇이 응답하지 않아 찾지 못했습니다"}
                log("ERR", "[손 자동 찾기] 로봇 응답 없음")
                return
            found = hand_zone_from_motion(f_open, f_grip, f_open2)
            if found is None:
                self.locate_status = {"phase": "fail", "msg": "손가락 움직임이 화면에 보이지 않습니다 — 카메라가 손을 비추는지 확인하십시오"}
                log("SYS", "[손 자동 찾기] 움직임을 찾지 못함 — 손 영역은 그대로")
                return
            zone, (bx, by, bw, bh) = found
            vlm = self.task.vlm
            verdict = ""
            if vlm is not None and vlm.online:            # 찾은 자리가 정말 로봇 손인지 VLM 이 확인 (예/아니오)
                self.locate_status = {"phase": "vlm", "msg": "VLM 이 로봇 손인지 확인하는 중", "zone": zone}
                h, w = f_open.shape[:2]
                gx, gy = bw * 0.35, bh * 0.35
                crop = f_open[int(max(0, by - gy)):int(min(h, by + bh + gy)), int(max(0, bx - gx)):int(min(w, bx + bw + gx))]
                v = vlm.check_found(crop, "로봇 손") if crop.size else {"found": None, "seconds": 0}
                if v["found"] is False:
                    self.locate_status = {"phase": "fail", "msg": "VLM: 찾은 자리가 로봇 손이 아닙니다 — 적용하지 않음", "zone": zone}
                    log("SYS", f"[손 자동 찾기] 움직임 자리 {zone} — VLM 이 로봇 손이 아니라고 판단 ({v['seconds']}s), 적용 안 함")
                    return
                verdict = f" · VLM 확인 {v['seconds']}s" if v["found"] else " · VLM 응답 없음"
            self.set_roi(*zone)
            self.locate_status = {"phase": "done", "msg": f"로봇 손을 찾아 손 영역으로 정했습니다{verdict}", "zone": zone,
                                  "t": time.monotonic()}
            log("SYS", f"[손 자동 찾기] 손 영역 = ({', '.join(f'{float(z):.2f}' for z in zone)}){verdict}")
        except Exception as exc:
            self.locate_status = {"phase": "fail", "msg": f"찾기 오류: {exc.__class__.__name__}"}
            log("ERR", f"[손 자동 찾기] 오류: {exc}")
        finally:
            self._locating = False

    def _palm_crop(self, margin: float = 0.12):
        """손 영역(+여유)을 잘라 VLM 에 보여 줄 사진"""
        f = self.frame
        if f is None:
            return None
        h, w = f.shape[:2]
        x0, y0, x1, y1 = self.palm_roi
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        c = f[int(max(0, y0 - my) * h):int(min(1, y1 + my) * h), int(max(0, x0 - mx) * w):int(min(1, x1 + mx) * w)]
        return c.copy() if c.size else None

    def set_roi(self, x0, y0, x1, y1) -> bool:
        x0, x1 = sorted((max(0.0, min(1.0, float(x0))), max(0.0, min(1.0, float(x1)))))
        y0, y1 = sorted((max(0.0, min(1.0, float(y0))), max(0.0, min(1.0, float(y1)))))
        if x1 - x0 < 0.05 or y1 - y0 < 0.05:
            return False
        self.palm_roi = (round(float(x0), 3), round(float(y0), 3), round(float(x1), 3), round(float(y1), 3))
        if self.task.state in ("RECEIVE", "VERIFY"):
            self.task.rebaseline = True                   # 손 영역을 옮겼으면 빈 손 기준도 다시
        self.ROI_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.ROI_FILE.write_text(json.dumps(self.palm_roi), encoding="utf-8")
        return True

    # ---- drawing ----
    DRAW_IN_VIDEO = False         # True: 예전처럼 영상 안에 상자를 그림 / False: 대시보드가 그림 (한글·같은 디자인)

    def _draw(self, frame):
        if not self.DRAW_IN_VIDEO:
            return frame
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
        # 쟁반: 요청한 공 (요청 중일 때, 색 규칙 또는 YOLO)
        tray = self.task.tray
        if self.tray_roi:
            tx0, ty0, tx1, ty1 = (int(self.tray_roi[0] * w), int(self.tray_roi[1] * h),
                                  int(self.tray_roi[2] * w), int(self.tray_roi[3] * h))
            cv2.rectangle(vis, (tx0, ty0), (tx1, ty1), (170, 170, 170), 1)
            cv2.putText(vis, "TRAY", (tx0 + 6, ty0 + int(24 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, (170, 170, 170), th)
        if not (self.tray_slot and self.tray_slot in self.views):
            draw_tray_items(vis, tray, k, th)
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

    def _open_camera(self, gen: int = 0):
        self._cam_switch = False
        while self.running:
            if gen != self._cam_gen:                # 새 카메라 스레드가 대신함 → 이 스레드는 열지 않고 끝냄 (같은 카메라를 두 번 여는 것 방지)
                return None
            if self.camera_off():                   # 꺼 둔 카메라: 열지 않고 켤 때까지 대기
                self.cam_error = ""
                self.jpeg = None
                self.fps = 0.0
                time.sleep(0.3)
                if self._cam_switch:
                    self._cam_switch = False
                continue
            idx = self.camera_index
            cap = cv2.VideoCapture(idx)
            if cap.isOpened() and (gen != self._cam_gen or idx != self.camera_index):
                cap.release()                       # 여는 사이 다른 스레드가 대신했거나 번호가 바뀜 → 다시
                if gen != self._cam_gen:
                    return None
                continue
            if cap.isOpened():
                self.cam_error = ""
                self._cam_opened_t = time.monotonic()
                self._opened_index = idx
                self._cam_switch = False            # 지금 고른 번호를 열었으니 바꾸기 요청은 처리됨
                self.task._log("SYS", f"USB 웹캠({idx}번) 연결됨")
                return cap
            cap.release()
            self.cam_error = f"USB 웹캠({self.camera_index}번)이 연결되지 않았습니다 — 꽂으면 자동으로 연결됩니다"
            self.jpeg = None
            time.sleep(2.0)
        return None

    def _camera_loop(self, gen: int = 0) -> None:
        cap = self._open_camera(gen)
        fails = 0
        last = time.perf_counter()
        while self.running and cap is not None:
            if gen != self._cam_gen:                # 감시 장치가 새 스레드를 띄움 → 이 스레드는 정리하고 끝냄
                cap.release()
                return
            if self._cam_switch and getattr(self, "_opened_index", None) == self.camera_index and not self.camera_off():
                self._cam_switch = False            # 이미 그 번호를 열어 둠 → 다시 열지 않음
            if self._cam_switch or self.camera_off():   # 다른 카메라를 고름 / 이 카메라를 끔
                cap.release()
                self.jpeg = None
                self.fps = 0.0
                cap = self._open_camera(gen)
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
                    self.task._log("SYS", "USB 웹캠 연결이 끊겼습니다 — 다시 연결을 기다립니다")
                    self.jpeg = None
                    cap = self._open_camera(gen)
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
                if self.task.rebaseline:                 # VLM: '공 없음' → 지금 화면을 빈 손 배경으로
                    self.task.rebaseline = False
                    self.recv_bg = frame.copy()
                bg = self.recv_bg if self.recv_bg is not None else self.manual_bg
                rule = self.clf.classify(frame, roi=roi, background=bg)
                if self.task.vlm_active() and roi is not None and bg is not None:
                    res = detect_change(frame, roi, bg)  # 색과 무관하게 '무언가 놓였는지'만 → 판정은 VLM
                else:
                    res = rule
                dets = [{"label": res.label, "conf": res.confidence, "box": res.box}] if res.box else []
            self.infer_ms = 0.8 * self.infer_ms + 0.2 * (time.perf_counter() - t0) * 1000
            d = res.to_dict()
            d["motion"] = round(motion, 2)
            if self.yolo is None:
                d["rule_label"] = rule.label            # 색 규칙: VLM 교차 확인 · VLM 이 답하지 못할 때 대신
                d["rule_conf"] = round(float(rule.confidence), 2)
            tray_items = self._tray_items(frame, dets)
            px0, py0, px1, py1 = self.palm_roi
            mx, my = (px1 - px0) * 0.15, (py1 - py0) * 0.15
            near_palm = (max(0.0, px0 - mx), max(0.0, py0 - my), min(1.0, px1 + mx), min(1.0, py1 + my))
            balls = find_balls(frame, max_balls=6) if self.yolo is None else                 [x for x in dets if x.get("label") in LABELS]
            with self.lock:
                self.hand_balls = {"size": [frame.shape[1], frame.shape[0]], "balls": balls}
                self.overlay = {"dets": dets, "best": d if res.label != "none" else None}
                self.result = d
                self.task.update_tray(tray_items)

    def _tray_items(self, frame, dets) -> list[dict]:
        """손 영역 밖(또는 지정한 쟁반 영역 안)의 공 탐지. YOLO 가 없으면 색 규칙(find_balls)."""
        if self.yolo is None:
            if self.task.target is None or self.task.state not in ("REQUEST", "RECEIVE"):
                return []                           # 요청 중일 때만 찾음
            if self._tray_view() is not None:
                return list(self._tray_view_items)   # 쟁반 칸은 자기 스레드(_tray_view_loop)가 찾음
            dets = find_balls(frame, roi=self.tray_roi, exclude=self.palm_roi)
        h, w = frame.shape[:2]
        px0, py0, px1, py1 = self.palm_roi
        out = []
        for d in dets:
            bx, by, bw, bh = d["box"]
            cx, cy = (bx + bw / 2) / w, (by + bh / 2) / h
            in_palm = px0 <= cx <= px1 and py0 <= cy <= py1
            in_tray = (self.tray_roi[0] <= cx <= self.tray_roi[2] and self.tray_roi[1] <= cy <= self.tray_roi[3]) \
                if self.tray_roi else not in_palm
            if in_tray and not in_palm and d["label"] in LABELS and d["conf"] >= TRAY_MIN_CONF:
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
        self.cam_error = f"카메라 {self.camera_index}번이 영상을 보내지 않아 다시 여는 중입니다 (USB 연결 · 허브 전원 확인)"
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
                    names.append(Path(dev).read_text(encoding="utf-8", errors="replace").strip())
                return names
        except Exception:
            pass
        return []

    def cameras_in_use(self) -> dict:
        """{번호: 역할} — 지금 이 프로그램이 쓰는 카메라 (다시 열지 않고 최근 프레임으로 미리보기)"""
        used = {self.camera_index: "hand"}
        for k, v in self.views.items():
            if isinstance(v.src, int) and v.src not in used:
                used[v.src] = f"view{k}"
        return used

    def _frame_for_role(self, role: str):
        if role.startswith("view"):
            v = self.views.get(int(role[4:]))
            return v.frame if v is not None else None
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
            msg = f"{index}번 카메라를 켰습니다"
        else:
            self.cam_disabled.add(index)
            msg = f"{index}번 카메라를 껐습니다"
        self._save_settings(camera_disabled=sorted(self.cam_disabled))
        self.task._log("SYS", msg)
        return {"ok": True, "reply": msg}

    def select_camera(self, role: str, index: int) -> dict:
        if role.startswith("view"):                  # 카메라 선택 창에서 분할 칸 고르기
            if index in self.cam_disabled:
                return {"ok": False, "reply": f"{index}번은 꺼 둔 카메라입니다 — 먼저 켜 주십시오"}
            return self.set_view(int(role[4:]), index)
        if role != "hand":
            return {"ok": False, "reply": "고를 수 없는 칸입니다"}
        if index in self.cam_disabled:
            return {"ok": False, "reply": f"{index}번은 꺼 둔 카메라입니다 — 먼저 켜 주십시오"}
        if index == self.camera_index:
            return {"ok": True, "reply": f"이미 {index}번 카메라를 쓰고 있습니다"}
        self.camera_index = index
        self._cam_switch = True
        self._save_settings(camera_index=index)
        self.task._log("SYS", f"카메라를 {index}번으로 바꿨습니다")
        return {"ok": True, "reply": f"{index}번 카메라로 바꾸는 중"}

    def _camera_ok(self) -> bool:
        return self.jpeg is not None and not self.cam_error

    def state(self) -> dict:
        with self.lock:
            snap = self.task.snapshot()
            snap["is_mock"] = self.link.is_mock
            snap["port"] = self.link.port
            snap["robot"] = self.link.info()
            snap["camera_roles"] = [("hand" if k == "hand" else f"view{k}") for k in self.visible_slots()]
            snap["view_order"] = list(self.view_order)
            snap["auto_cams"] = self.auto_cams
            snap["vlm_detail"] = self.vlm_detail
            ls = self.locate_status
            snap["locate"] = ({k: v for k, v in ls.items() if k != "t"} | {"busy": self._locating,
                              "fresh": ls.get("phase") != "done" or time.monotonic() - ls.get("t", 0) < 5}) if ls else None
            hb = getattr(self, "hand_balls", None) or {"size": None, "balls": []}
            snap["hand_overlay"] = {"size": hb["size"], "palm": list(self.palm_roi),
                                    "balls": [{"label": b["label"], "conf": round(float(b["conf"]), 2), "box": list(b["box"])}
                                              for b in hb["balls"]],
                                    "tray_here": not (self.tray_slot and self.tray_slot in self.views)}
            snap["camera_chosen"] = "camera_index" in self.read_settings()   # 아직 한 번도 고르지 않았으면 선택 창을 띄움
            snap["camera_disabled"] = sorted(self.cam_disabled)
            snap["camera"] = {"index": self.camera_index, "off": self.camera_off(), "error": self.cam_error, "fps": round(self.fps, 1),
                              "running": self._camera_ok(), "infer_ms": round(self.infer_ms)}
            vlm_on = self.task.vlm_active()
            snap["detector"] = {"name": (f"VLM {self.task.vlm.model} 이 손 위 공을 판정 · 놓임은 화면 변화로 감지" if vlm_on
                                         else self.detector_info), "yolo": self.yolo is not None,
                                "labels": list(self.yolo.names.values()) if self.yolo else list(LABELS)}
            snap["samples"] = self._capture_counts() if self.yolo else self.clf.counts()
            snap["vlm"] = self.task.vlm.status() if self.task.vlm is not None else {"online": False, "note": "--no-vlm"}
            snap["split"] = self.split
            snap["views"] = [v.info() for v in sorted(self.views.values(), key=lambda v: v.slot)]
            snap["tray_slot"] = self.tray_slot if self.tray_slot in self.views else None
            snap["palm_roi"] = list(self.palm_roi)
            snap["background"] = "receive" if self.recv_bg is not None else ("manual" if self.manual_bg is not None else "none")
            return snap

    CAPTURE_DIR = ROOT / "data" / "captures"

    def _capture_counts(self) -> dict:
        return {l: len(list((self.CAPTURE_DIR / l).glob("*.jpg"))) for l in LABELS}

    def action(self, p: dict) -> dict:
        a = p.get("action")
        if a == "locate_hand":                            # [손 자동 찾기]
            out = self.locate_hand()
            st = self.state()
            st["result"] = out
            return st
        if a == "describe":                               # 칸의 [관찰] 버튼
            slot = str(p.get("slot", "hand"))
            order = self.view_order
            pos = order.index(slot) + 1 if slot in order else 1
            if slot == "hand":
                img = self.frame
            else:
                v = self.views.get(int(slot)) if slot.isdigit() else None
                img = v.frame if v is not None else None
            out = self.start_describe(img, f"CAM {pos}")
            st = self.state()
            st["result"] = out
            return st
        if a == "view_auto":
            out = self.set_auto_cams(bool(p.get("on", not self.auto_cams)))
            st = self.state()
            st["result"] = out
            return st
        if a in ("view_set", "view_split", "view_tray", "view_order"):
            try:
                out = (self.set_view(int(p.get("slot", 0)), p.get("src")) if a == "view_set"
                       else self.set_split(int(p.get("n", 1))) if a == "view_split"
                       else self.set_view_order(p.get("order")) if a == "view_order"
                       else self.set_tray_slot(p.get("slot")))
            except (TypeError, ValueError):
                out = {"ok": False, "reply": "값이 잘못됐습니다"}
            st = self.state()
            st["result"] = out
            return st
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
                st["result"] = {"ok": False, "reply": "카메라 번호가 잘못됐습니다"}
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
                st["result"] = {"ok": False, "reply": "카메라 번호가 잘못됐습니다"}
                return st
            with self.lock:
                out = self.select_camera(role, idx)
            st = self.state()
            st["result"] = out
            return st
        pre = None
        if a == "request" and self.task.vlm is not None and self.link.connected and not self.task.stopped                 and self.task.state == "IDLE":
            pre = self.task.vlm.interpret(str(p.get("text", "")))   # VLM 은 느릴 수 있어 잠금 밖에서 먼저
        with self.lock:
            out: dict = {}
            if (a in ("request", "raw", "confirm", "hand")
                    or (a == "manual" and p.get("step") not in ("done", "fail", "idle", "placed", "skip_open", "receive"))) \
                    and not self.link.connected:
                out = {"ok": False, "reply": "로봇이 연결되지 않았습니다 — USB · 전원을 확인하십시오"}
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
                out = {"ok": False, "reply": "카메라가 연결되지 않았습니다"}
            elif a == "learn" and p.get("label") in LABELS:
                lbl = p["label"]
                if self.yolo is not None:
                    # YOLO 사용 중: 재학습용 사진 저장 (라벨링은 별도)
                    d = self.CAPTURE_DIR / lbl
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"{int(time.time() * 1000)}.jpg"), self.frame)
                    out = {"ok": True, "reply": f"{LABEL_KO[lbl]} 사진 저장 (재학습용)"}
                else:
                    ok = self.frame is not None and self.clf.add_sample(self.frame, lbl, background=self.manual_bg)
                    out = {"ok": bool(ok), "reply": f"{LABEL_KO[lbl]} 샘플 저장" if ok else "공이 안 보입니다"}
                self.task._log("SYS", f"[데이터] {out['reply']}")
            elif a == "roi":
                try:
                    ok = self.set_roi(p["x0"], p["y0"], p["x1"], p["y1"])
                except (KeyError, TypeError, ValueError):
                    ok = False
                out = {"ok": ok, "reply": "손 영역을 저장했습니다" if ok else "영역이 너무 작습니다"}
                if ok:
                    self.task._log("SYS", f"손 영역 지정: {self.palm_roi}")
            elif a == "bg_clear":
                self.manual_bg = None
                out = {"ok": True, "reply": "빈 화면 배경을 지웠습니다"}
                self.task._log("SYS", "빈 화면(배경) 지움")
            elif a == "roi_reset":
                self.palm_roi = self.DEFAULT_ROI
                self.ROI_FILE.unlink(missing_ok=True)
                out = {"ok": True, "reply": "손 영역을 기본값으로 되돌렸습니다"}
            elif a == "place_angle":
                try:
                    ang = max(0, min(180, int(float(p.get("value")))))
                    self.task.place_angle = ang
                    self._save_settings()
                    out = {"ok": True, "reply": f"내려놓기 각도 {ang}°"}
                    self.task._log("SYS", f"내려놓기 손목 각도 → {ang}°")
                except (TypeError, ValueError):
                    out = {"ok": False, "reply": "각도 형식이 잘못됐습니다"}
            elif a == "bg":
                if self.frame is not None:
                    self.manual_bg = self.frame.copy()
                    out = {"ok": True, "reply": "빈 화면을 배경으로 저장했습니다"}
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
                    out = {"ok": False, "reply": "각도 형식이 잘못됐습니다"}
                else:
                    out = {"ok": False, "reply": "작업 중이거나 정지 상태입니다"}
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
                m = re.search(r"[?&]view=(\d+)", self.path)
                slot = int(m.group(1)) if m else 0
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while rt.running:
                        jpg = rt.view_jpeg(slot) if slot >= 2 else rt.jpeg
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
    ap.add_argument("--model", type=Path, default=None,
                    help="YOLO 가중치 (클래스 red/blue/green 인 공 모델만). 생략하면 색 규칙으로 동작. "
                         "팀 best_two.pt 는 볼트·너트 전용이라 공에는 쓰지 않음")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO 탐지 최소 신뢰도 (쥐기 판단은 0.6 이상)")
    ap.add_argument("--vlm", action="store_true", help="(기본으로 켜져 있음 — 예전 실행 명령 호환용)")
    ap.add_argument("--no-vlm", action="store_true", help="VLM(Ollama) 끄기: 단어 규칙·색 규칙만 사용")
    ap.add_argument("--vlm-model", default="gemma3:4b")
    ap.add_argument("--vlm-url", default="http://127.0.0.1:11434")
    ap.add_argument("--vlm-verify-below", type=float, default=1.0,
                    help="손 위 공 VLM 확인: 1 = 항상(기본), 0 = 끔(색 규칙만), 그 사이 = 색 규칙 점수가 이보다 낮을 때만")
    ap.add_argument("--tray-roi", default="", help="쟁반 영역 x0,y0,x1,y1 (0~1). 생략하면 손 영역 밖 전체를 쟁반으로 봄")
    ap.add_argument("--point", action="store_true", help="쟁반에서 고른 공 쪽으로 손목을 돌려 알려 주기")
    ap.add_argument("--point-left", type=float, default=135, help="쟁반 왼쪽 끝을 향하는 손목 각도 (보정)")
    ap.add_argument("--point-right", type=float, default=45, help="쟁반 오른쪽 끝을 향하는 손목 각도 (보정)")
    ap.add_argument("--point-hold", type=float, default=1.5, help="가리킨 채 멈추는 시간 (초)")
    ap.add_argument("--view", action="append", default=[], metavar="SRC",
                    help="분할 화면 2·3·4번 칸 카메라 (웹캠 번호 또는 사진·영상 경로, 여러 번 가능)")
    ap.add_argument("--tray-view", type=int, default=None, help="쟁반으로 쓸 칸 번호 (2~4)")
    ap.add_argument("--http-port", type=int, default=8090)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    link = MockLink() if a.mock else RobotLink(port_hint=a.port)
    vlm = None
    if not a.no_vlm:
        from vision.vlm_bridge import VlmBridge
        vlm = VlmBridge(a.vlm_model, a.vlm_url, verify_below=a.vlm_verify_below)
    tray_roi = tuple(float(v) for v in a.tray_roi.split(",")) if a.tray_roi else None
    if a.camera is None:
        a.camera = int(Runtime.read_settings().get("camera_index", 1))
    rt = Runtime(link, a.camera, model_path=a.model, conf=a.conf, vlm=vlm, tray_roi=tray_roi)
    if a.view:
        rt.view_order = ["hand", "2", "3", "4"]
        rt.set_split(max(rt.split, len(a.view[: MAX_VIEWS - 1]) + 1))
    for i, src in enumerate(a.view[: MAX_VIEWS - 1], start=2):
        rt.set_view(i, src)
    if a.tray_view is not None:
        rt.tray_slot = None
        rt.set_tray_slot(a.tray_view)
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
