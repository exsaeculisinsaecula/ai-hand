"""비상정지(E-STOP) — 물리 버튼 감시 · 정지 잠금 · 로봇 명령 차단 · 스위치 등록.

3중 구조
  1) 하드웨어 (프로그램과 무관, 권장): 버튼의 NC 접점 ① 을 서보 전원(+) 선에 직렬로 넣는다.
     누르면 서보 전원이 바로 끊긴다. PC · micro:bit 가 멈춰 있어도 동작한다.
  2) 감지 (이 모듈): 버튼의 NC 접점 ② 를 USB-시리얼 어댑터(CH340 · CP2102 · FT232 등)의
     출력선(RTS 또는 DTR)과 입력선(CTS · DSR · DCD · RI) 사이에 연결한다.
     프로그램이 출력선을 무작위로 켰다 껐다 하고, 입력선이 그대로 따라오는지 1초에 수십 번 확인한다(능동 검사).
       따라옴      = 회로가 살아 있고 버튼이 풀려 있음
       안 따라옴   = 버튼 눌림 · 전선 끊김 · 접점 고장 · 선이 한쪽 값에 붙음  → 정지
       어댑터 없음 = 분리 · 고장                                            → 정지
     정상일 때 닫혀 있는 NC 접점을 쓰므로, 배선이 끊겨도 '눌림'과 똑같이 정지한다(fail-safe).
  3) 소프트웨어 잠금: 정지가 걸리면
       - 로봇에 STOP 을 확인 응답(ACK)이 올 때까지 반복해서 보낸다 (펌웨어는 이동 중 USB 를 잠깐씩만 본다)
       - 잠금이 풀릴 때까지 STOP · PING · VER 외의 모든 로봇 명령을 한 곳(GuardedLink)에서 막는다
       - micro:bit 이 재부팅되거나 다시 연결되면 STOP 을 다시 보낸다 (펌웨어의 정지 상태는 재부팅하면 사라짐)
     해제 조건: 버튼이 풀리고 회로가 0.5초 동안 계속 정상 + 작업자가 화면에서 해제를 누름.
     버튼을 풀기만 해서는 로봇이 절대 움직이지 않는다.

스위치 등록(pair): 어느 어댑터의 어느 선이 비상정지 스위치인지 프로그램이 직접 찾는다.
  ① 찾기   — 버튼이 풀린 상태에서 모든 어댑터 · 선 조합을 검사해 신호가 되돌아오는 회로를 찾는다
  ② 누르기 — 작업자가 버튼을 누르면 그 회로가 열리는지 확인한다 (점퍼선 · 다른 장치와 구별)
  ③ 풀기   — 버튼을 돌려 풀면 다시 닫히는지 확인하고, 응답 시간을 재서 검사 간격을 맞춘다
  어댑터의 USB 정보(제조사 · 일련번호 · 꽂은 위치)와 선 조합을 저장하고, 다음 실행부터 자동으로 찾는다.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Callable

OUT_LINES = ("rts", "dtr")
IN_LINES = ("cts", "dsr", "cd", "ri")
LINE_PAIRS = [(o, i) for o in OUT_LINES for i in IN_LINES]
LINE_KO = {"rts": "RTS", "dtr": "DTR", "cts": "CTS", "dsr": "DSR", "cd": "DCD", "ri": "RI"}
ALLOWED_WHILE_STOPPED = ("STOP", "PING", "VER")
MICROBIT_VID = 0x0D28
INPUT_KO = {"unpaired": "미등록", "absent": "연결 안 됨", "closed": "정상", "open": "눌림", "fault": "회로 이상",
            "pairing": "등록 중", "checking": "확인 중"}


# ---------------------------------------------------------------- 장치
class SerialLoopDevice:
    """USB-시리얼 어댑터의 모뎀 제어선만 쓴다 (데이터는 주고받지 않음)."""

    def __init__(self, port: str):
        import serial
        self.port = port
        self.ser = serial.Serial(port, 9600, timeout=0, write_timeout=0)

    def set_out(self, line: str, value: bool) -> None:
        setattr(self.ser, line, bool(value))

    def read_in(self, line: str) -> bool:
        return bool(getattr(self.ser, line))

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


def list_serial_ports() -> list[dict]:
    from serial.tools import list_ports
    return [{"port": p.device, "vid": p.vid, "pid": p.pid, "serial": p.serial_number or "",
             "location": p.location or "", "desc": p.description or ""} for p in list_ports.comports()]


def match_port(saved: dict, ports: list[dict]) -> dict | None:
    """저장한 어댑터를 지금 포트 목록에서 찾는다 (COM 번호가 바뀌어도 USB 정보로)."""
    if saved.get("vid") is None:
        return next((p for p in ports if p["port"] == saved.get("port")), None)
    same = [p for p in ports if p.get("vid") == saved.get("vid") and p.get("pid") == saved.get("pid")]
    if saved.get("serial"):                          # 일련번호가 있는 어댑터: 같은 번호만 인정
        return next((p for p in same if p.get("serial") == saved["serial"]), None)
    for key in ("location", "port"):
        hit = next((p for p in same if saved.get(key) and p.get(key) == saved[key]), None)
        if hit:
            return hit
    return same[0] if len(same) == 1 else None      # 다른 USB 구멍에 꽂음 — 회로 검사가 진짜인지 다시 확인한다


# ---------------------------------------------------------------- 능동 회로 검사
class LoopChecker:
    """한 번의 검사 = 출력선을 켜고 · 끄는 두 반쪽(순서는 무작위). 두 반쪽 모두 입력선이 따라와야 통과.

    반쪽 통과 = 입력선이 바뀐 값에 도달한 뒤 CONFIRM_READS 번 더 읽어도 그대로일 것 (잡음이 우연히 맞는 것을 거름).
    입력선이 한쪽 값에 붙어 있으면 반드시 한 반쪽이 실패하고, 떠 있는(잡음) 입력은 거의 통과하지 못한다.
    """
    CONFIRM_READS = 4
    POLL_S = 0.002

    def __init__(self, dev, out_line: str, in_line: str, settle_s: float = 0.04,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep, rng=None):
        self.dev, self.out_line, self.in_line = dev, out_line, in_line
        self.settle_s = settle_s
        self.clock, self.sleep = clock, sleep
        self.rng = rng or random.Random()

    def _half(self, bit: bool) -> float | None:
        self.dev.set_out(self.out_line, bit)
        t0 = self.clock()
        while True:
            if self.dev.read_in(self.in_line) == bit:
                reached = self.clock() - t0
                for _ in range(self.CONFIRM_READS):          # 도달한 뒤에도 같은 값을 유지해야 진짜 회로
                    self.sleep(self.POLL_S)
                    if self.dev.read_in(self.in_line) != bit:
                        return None
                return reached
            if self.clock() - t0 >= self.settle_s:
                return None
            self.sleep(self.POLL_S)

    def check(self) -> tuple[bool, float]:
        first = self.rng.random() < 0.5
        a = self._half(first)
        b = self._half(not first)
        return (a is not None and b is not None), max(a or 0.0, b or 0.0)


# ---------------------------------------------------------------- --mock 용 가상 회로
class MockSwitchWorld:
    """가상 비상정지 회로: RTS → (NC 버튼) → CTS. pressed · plugged 를 화면에서 바꿔 시험한다."""
    PORT = "MOCK-ESTOP"

    def __init__(self):
        self.pressed = False
        self.plugged = True
        self.out = {"rts": False, "dtr": False}

    def lister(self) -> list[dict]:
        return [{"port": self.PORT, "vid": 0x1A86, "pid": 0x7523, "serial": "", "location": "mock",
                 "desc": "가상 비상정지 어댑터 (--mock)"}] if self.plugged else []

    def opener(self, port: str):
        if not self.plugged or port != self.PORT:
            raise OSError("가상 어댑터 없음")
        return _MockLoopDevice(self)


class _MockLoopDevice:
    def __init__(self, world: MockSwitchWorld):
        self.w = world

    def set_out(self, line, value):
        if not self.w.plugged:
            raise OSError("분리됨")
        self.w.out[line] = bool(value)

    def read_in(self, line):
        if not self.w.plugged:
            raise OSError("분리됨")
        return self.w.out["rts"] if (line == "cts" and not self.w.pressed) else False

    def close(self):
        pass


# ---------------------------------------------------------------- 비상정지 본체
class EStop:
    TRIP_AFTER_FAILS = 2       # 연속 실패 검사 수 → 정지 (USB 지연 한 번에 멈추지 않게, 약 0.1초)
    CLOSE_AFTER_OKS = 5        # 정상으로 인정하려면 연속 통과 검사 수 (시작 · 다시 꽂음 · 버튼 풀림)
    RESET_STABLE_S = 0.5       # 해제하려면 회로가 이 시간 동안 계속 정상이어야 함
    WATCHDOG_S = 0.8           # 감시 스레드가 이 시간 넘게 멈추면 정지
    STOP_FAST_S = 0.05         # 로봇 확인 응답 전: STOP 재전송 간격
    STOP_FAST_FOR = 3.0        #   … 이 시간 동안 (그 뒤에는 STOP_KEEP_S 간격)
    STOP_KEEP_S = 1.0          # 정지 중 유지 전송 (micro:bit 재부팅 대비)
    REOPEN_S = 1.0             # 어댑터를 다시 찾는 간격
    PAIR_CHECKS = 12           # 등록 ①: 이만큼 연속 통과해야 회로로 인정
    PAIR_OPEN_FAILS = 3        # 등록 ②: 이만큼 연속 실패하면 '버튼으로 열림'
    PAIR_RELEASE_S = 1.0       # 등록 ③: 이 시간 동안 계속 정상이어야 완료
    PAIR_STEP_S = 30.0         # 등록 ②·③ 제한 시간
    SCAN_SETTLE_S = 0.08       # 등록 중 검사 대기 (넉넉하게)
    CYCLE_S = 0.02             # 감시 주기 (검사 1회 + 쉬는 시간) — USB 를 과하게 쓰지 않게

    def __init__(self, link, cfg: dict | None = None, log: Callable[[str, str], None] | None = None,
                 lister: Callable[[], list[dict]] = list_serial_ports, opener: Callable[[str], object] = SerialLoopDevice,
                 exclude_ports: Callable[[], set] = lambda: set(), on_save: Callable[[dict | None], None] | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep, rng=None,
                 mock_world: MockSwitchWorld | None = None):
        self.link = link                      # 원래 링크 (STOP 은 차단 없이 바로 보냄)
        self.cfg = dict(cfg) if cfg else None
        self.log = log or (lambda d, t: None)
        self.lister, self.opener = lister, opener
        self.exclude_ports = exclude_ports
        self.on_save = on_save or (lambda c: None)
        self.clock, self.sleep = clock, sleep
        self.rng = rng or random.Random()
        self.mock_world = mock_world
        self._lock = threading.RLock()

        self.latched = False
        self.sources: set[str] = set()
        self.reason = ""
        self.tripped_at = 0.0
        self.robot_ack = False
        self._stop_tx = 0.0

        self.input = "absent" if self.cfg else "unpaired"
        self.input_detail = "" if self.cfg else "비상정지 스위치가 등록되지 않았습니다"
        self.device = None
        self.checker: LoopChecker | None = None
        self.port_now = ""
        self._fails = 0
        self._oks = 0
        self._ok_since: float | None = None
        self._last_open_try = -1e9
        self._last_cycle = clock()
        self.cycles = 0
        self.latency_ms: float | None = None
        self.blocked = 0
        self._last_block_log = -1e9

        self.pairing: dict | None = None
        self._pair_cmd: str | None = None
        self._pair_devs: dict[str, object] = {}
        self._pair_queue: list = []
        self._pair_cands: list[dict] = []

        self.running = True
        self._thread: threading.Thread | None = None

    # ---- 수명 ----
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="estop")
            self._thread.start()

    def close(self) -> None:
        self.running = False
        self._drop_device()
        self._pair_close_all()

    def _loop(self) -> None:
        while self.running:
            try:
                self.tick()
            except Exception as exc:              # 감시 코드 자체의 오류도 정지로
                self._drop_device()
                self._set_input("fault", f"감시 오류: {exc.__class__.__name__}")
                self.trip("fault", f"비상정지 감시 오류 ({exc.__class__.__name__}) — 안전을 위해 정지")
                self.sleep(0.2)

    # ---- 정지 잠금 ----
    def trip(self, source: str, detail: str) -> None:
        with self._lock:
            new_source = source not in self.sources
            self.sources.add(source)
            if not self.latched:
                self.latched = True
                self.reason = detail
                self.tripped_at = self.clock()
                self.robot_ack = False
                self._send_stop()                 # 기록보다 먼저 보낸다
                self.log("ERR", f"[비상정지] {detail} — 로봇 동작을 모두 막았습니다")
            elif new_source:
                self._send_stop()

    def _send_stop(self) -> None:
        self._stop_tx = self.clock()
        try:
            self.link.send("STOP")
        except Exception:
            pass

    def reset_check(self) -> tuple[bool, str]:
        with self._lock:
            if not self.latched:
                return True, ""
            if self.pairing and self.pairing["step"] in ("starting", "scan", "press", "release"):
                return False, "스위치 등록이 끝난 뒤 해제할 수 있습니다"
            if self.cfg:
                if self.input == "open":
                    return False, "비상정지 버튼이 눌려 있습니다 — 버튼을 돌려서 푸십시오"
                if self.input == "absent":
                    return False, f"비상정지 스위치가 연결되지 않았습니다 — {self.input_detail or '어댑터 USB 를 확인하십시오'}"
                if self.input == "fault":
                    return False, f"비상정지 회로 이상 — {self.input_detail}"
                if self.input != "closed" or self._ok_since is None \
                        or self.clock() - self._ok_since < self.RESET_STABLE_S:
                    return False, "버튼 회로를 확인하는 중입니다 — 잠시 뒤 다시 누르십시오"
            return True, ""

    def reset(self) -> tuple[bool, str]:
        """작업자의 해제. 로봇은 움직이지 않는다 (움직임은 따로 [대기 복귀] 명령)."""
        with self._lock:
            ok, why = self.reset_check()
            if ok and self.latched:
                self.latched = False
                self.sources.clear()
                self.reason = ""
                self.log("SYS", "[비상정지] 해제 — 작업자가 안전을 확인했습니다")
            return ok, why

    def allow(self, cmd: str) -> bool:
        head = cmd.strip().split(",")[0].upper()
        if head in ALLOWED_WHILE_STOPPED:
            return True
        with self._lock:
            if self.latched:
                return False
            if self.cfg and self.input != "closed":    # 등록된 스위치를 아직 확인하지 못함 (시작 직후 등)
                return False
            return True

    def block_reason(self) -> str:
        if self.latched:
            return f"비상정지 중 ({self.reason or '정지'})"
        if self.cfg and self.input != "closed":
            return "비상정지 스위치 확인 전"
        return ""

    def note_blocked(self, cmd: str) -> None:
        self.blocked += 1
        now = self.clock()
        if now - self._last_block_log > 2.0:
            self._last_block_log = now
            self.log("ERR", f"[비상정지] {cmd.split(',')[0]} 명령을 막았습니다 — {self.block_reason()}")

    def note_rx(self, line: str) -> bool:
        """로봇 응답을 본다. True 면 작업 기록에 넘기지 않는다 (정지 확인용 ACK)."""
        with self._lock:
            if line == "READY":                    # micro:bit 재부팅 → 펌웨어의 정지 상태가 사라짐
                if self.latched:
                    self.robot_ack = False
                    self._send_stop()
                    self.log("SYS", "[비상정지] 로봇이 다시 켜져 STOP 을 다시 보냈습니다")
                return False
            if not self.latched and line == "ERR,STOPPED":   # 보드의 A+B 비상정지 등 펌웨어가 스스로 정지한 상태
                self.trip("robot", "micro:bit 가 정지 상태입니다 (보드 A+B 비상정지)")
                return False
            if self.latched and line in ("ACK", "ERR,STOPPED", "ERR,ABORTED"):
                if not self.robot_ack:
                    self.robot_ack = True
                    self.log("SYS", "[비상정지] 로봇이 정지 명령을 받았습니다")
                return line == "ACK"
            return False

    def robot_connected(self) -> None:
        """로봇 USB 가 (다시) 연결됨 — 정지 중이면 STOP 부터."""
        with self._lock:
            if self.latched:
                self.robot_ack = False
                self._send_stop()
                self.log("SYS", "[비상정지] 로봇이 연결돼 STOP 을 먼저 보냈습니다 (정지 유지)")

    def watchdog(self) -> None:
        """작업 루프가 주기적으로 부름: 감시 스레드가 멈췄으면 정지."""
        if self.cfg and self._thread is not None and self.clock() - self._last_cycle > self.WATCHDOG_S:
            self._set_input("fault", "감시가 멈췄습니다")
            self.trip("watchdog", "비상정지 감시가 응답하지 않습니다 — 안전을 위해 정지")

    # ---- 한 주기 ----
    def tick(self) -> None:
        t0 = self.clock()
        cmd, self._pair_cmd = self._pair_cmd, None
        if cmd == "start":
            self._pair_begin()
        elif cmd == "cancel":
            self._pair_end("cancelled", "등록을 취소했습니다")
        paced = True
        if self.pairing and self.pairing["step"] in ("scan", "press", "release"):
            paced = self.pairing["step"] != "scan"          # 찾기는 쉬지 않고, 누르기·풀기 확인은 주기에 맞춤
            self._pair_tick()
        elif self.cfg:
            self._monitor_tick()
        if paced:
            left = self.CYCLE_S - (self.clock() - t0)
            if left > 0:
                self.sleep(left)
        self._stop_keepalive()
        self._last_cycle = self.clock()

    def _stop_keepalive(self) -> None:
        with self._lock:
            if not self.latched:
                return
            now = self.clock()
            fast = not self.robot_ack and now - self.tripped_at < self.STOP_FAST_FOR
            if now - self._stop_tx >= (self.STOP_FAST_S if fast else self.STOP_KEEP_S):
                self._send_stop()

    def _set_input(self, state: str, detail: str) -> None:
        if state != self.input:
            self.input = state
            if state != "closed":
                self._ok_since = None
        self.input_detail = detail

    def _drop_device(self) -> None:
        dev, self.device, self.checker = self.device, None, None
        if dev is not None:
            try:
                dev.close()
            except Exception:
                pass

    def _monitor_tick(self) -> None:
        now = self.clock()
        if self.checker is None:
            if now - self._last_open_try < self.REOPEN_S:
                self.sleep(0.02)
                return
            self._last_open_try = now
            p = match_port(self.cfg, self._ports())
            if p is None:
                self._set_input("absent", f"어댑터({self.cfg.get('desc') or self.cfg.get('port')})를 찾을 수 없습니다")
                self.trip("device", "비상정지 스위치 어댑터가 연결되지 않았습니다")
                return
            try:
                dev = self.opener(p["port"])
            except Exception as exc:
                self._set_input("absent", f"{p['port']} 를 열 수 없습니다 ({exc.__class__.__name__})")
                self.trip("device", "비상정지 스위치 어댑터를 열 수 없습니다")
                return
            self.device, self.port_now = dev, p["port"]
            self.checker = LoopChecker(dev, self.cfg["out"], self.cfg["in"], float(self.cfg.get("settle_s", 0.04)),
                                       clock=self.clock, sleep=self.sleep, rng=self.rng)
            self._fails = self._oks = 0
            self._set_input("checking", "회로 확인 중")
            self.log("SYS", f"[비상정지] 스위치 어댑터 연결 ({p['port']} · {LINE_KO[self.cfg['out']]}→{LINE_KO[self.cfg['in']]})")
        try:
            ok, lat = self.checker.check()
        except Exception as exc:
            self._drop_device()
            self._set_input("absent", f"어댑터 분리 ({exc.__class__.__name__})")
            self.trip("device", "비상정지 스위치 어댑터가 분리됐습니다")
            return
        self.cycles += 1
        with self._lock:
            if ok:
                self._fails = 0
                self._oks += 1
                self.latency_ms = round(lat * 1000, 1)
                if self.input != "closed" and self._oks >= self.CLOSE_AFTER_OKS:
                    self._set_input("closed", "버튼 풀림 · 회로 정상")
                if self.input == "closed" and self._ok_since is None:
                    self._ok_since = self.clock()
            else:
                self._fails += 1
                self._oks = 0
                self._ok_since = None
                if self._fails >= self.TRIP_AFTER_FAILS:
                    self._set_input("open", "회로 열림 — 버튼 눌림 또는 배선 끊김")
                    self.trip("button", "비상정지 버튼이 눌렸습니다 (또는 배선 끊김)")

    def _ports(self) -> list[dict]:
        excl = set(self.exclude_ports() or ())
        return [p for p in self.lister() if p.get("vid") != MICROBIT_VID and p["port"] not in excl]

    # ---- 스위치 등록 ----
    def pair_start(self) -> tuple[bool, str]:
        with self._lock:
            if self.pairing and self.pairing["step"] in ("starting", "scan", "press", "release"):
                return False, "이미 등록 중입니다"
            self.trip("pairing", "비상정지 스위치 등록 중 — 로봇 정지")
            self.pairing = {"step": "starting", "msg": "어댑터를 찾는 중입니다", "found": 0, "port": "", "lines": ""}
            self._pair_cmd = "start"
            return True, "등록을 시작합니다 — 버튼이 풀린 상태인지 확인하십시오"

    def pair_cancel(self) -> tuple[bool, str]:
        with self._lock:
            if not self.pairing or self.pairing["step"] not in ("starting", "scan", "press", "release"):
                return False, "등록 중이 아닙니다"
            self._pair_cmd = "cancel"
            return True, "등록을 취소합니다"

    def unpair(self) -> tuple[bool, str]:
        with self._lock:
            if not self.latched:
                return False, "로봇이 정지된 상태에서만 등록을 지울 수 있습니다 — 먼저 비상정지를 누르십시오"
            self.cfg = None
            self._drop_device()
            self._set_input("unpaired", "비상정지 스위치가 등록되지 않았습니다")
            self.port_now = ""
            self.on_save(None)
            self.log("ERR", "[비상정지] 스위치 등록을 지웠습니다 — 물리 버튼 감시가 꺼졌습니다")
            return True, "스위치 등록을 지웠습니다"

    def _pair_begin(self) -> None:
        self._drop_device()
        self._pair_close_all()
        ports = self._ports()
        self._pair_queue = [(p, pair) for p in ports for pair in LINE_PAIRS]
        self._pair_cands = []
        self.pairing = {"step": "scan", "msg": f"어댑터 {len(ports)}개의 선 조합을 검사하는 중입니다 (버튼은 풀어 두십시오)",
                        "found": 0, "port": "", "lines": "", "ports": [p["port"] for p in ports]}
        self.log("SYS", f"[비상정지 등록] 시작 — 시리얼 포트 {len(ports)}개 검사")
        if not ports:
            self._pair_end("failed", "USB-시리얼 어댑터를 찾지 못했습니다 — 어댑터를 PC 에 꽂으십시오")

    def _pair_dev(self, port: str):
        if port not in self._pair_devs:
            self._pair_devs[port] = self.opener(port)
        return self._pair_devs[port]

    def _pair_tick(self) -> None:
        step = self.pairing["step"]
        now = self.clock()
        if step == "scan":
            if not self._pair_queue:
                self._pair_scan_done()
                return
            p, (o, i) = self._pair_queue.pop(0)
            try:
                dev = self._pair_dev(p["port"])
            except Exception:                               # 다른 프로그램이 쓰는 포트 등 — 건너뜀
                self._pair_queue = [(q, pr) for q, pr in self._pair_queue if q["port"] != p["port"]]
                return
            chk = LoopChecker(dev, o, i, self.SCAN_SETTLE_S, clock=self.clock, sleep=self.sleep, rng=self.rng)
            lat_max = 0.0
            try:
                for k in range(self.PAIR_CHECKS):
                    ok, lat = chk.check()
                    if not ok:
                        return
                    lat_max = max(lat_max, lat)
            except Exception:
                return
            self._pair_cands.append({"port": p, "out": o, "in": i, "checker": chk, "fails": 0, "lat": lat_max})
            self.pairing["found"] = len(self._pair_cands)
        elif step == "press":
            if now > self.pairing["deadline"]:
                self._pair_end("failed", "버튼을 누르지 않았거나, 찾은 회로가 버튼과 연결되지 않았습니다 (점퍼선일 수 있습니다)")
                return
            opened = []
            for c in self._pair_cands:
                try:
                    ok, lat = c["checker"].check()
                except Exception:
                    ok, lat = False, 0.0
                c["fails"] = 0 if ok else c["fails"] + 1
                if ok:
                    c["lat"] = max(c["lat"], lat)
                if c["fails"] >= self.PAIR_OPEN_FAILS:
                    opened.append(c)
            if len(opened) > 1:
                self._pair_end("failed", "여러 회로가 함께 열렸습니다 — 비상정지 버튼만 어댑터에 연결하십시오")
            elif opened:
                c = opened[0]
                self._pair_cands = [c]
                self.pairing.update(step="release", deadline=now + self.PAIR_STEP_S, ok_since=None,
                                    port=c["port"]["port"], lines=f"{LINE_KO[c['out']]}→{LINE_KO[c['in']]}",
                                    msg="버튼이 눌린 것을 확인했습니다 — 이제 버튼을 돌려서 푸십시오")
                self.log("SYS", f"[비상정지 등록] ② 눌림 확인 ({c['port']['port']} {self.pairing['lines']})")
        elif step == "release":
            if now > self.pairing["deadline"]:
                self._pair_end("failed", "버튼을 풀지 않았거나 회로가 다시 닫히지 않았습니다 — 배선을 확인하십시오")
                return
            c = self._pair_cands[0]
            try:
                ok, lat = c["checker"].check()
            except Exception:
                self._pair_end("failed", "어댑터가 분리됐습니다")
                return
            if not ok:
                self.pairing["ok_since"] = None
                return
            c["lat"] = max(c["lat"], lat)
            if self.pairing["ok_since"] is None:
                self.pairing["ok_since"] = now
            elif now - self.pairing["ok_since"] >= self.PAIR_RELEASE_S:
                self._pair_finish(c)

    def _pair_scan_done(self) -> None:
        n = len(self._pair_cands)
        keep = {c["port"]["port"] for c in self._pair_cands}
        for port in list(self._pair_devs):
            if port not in keep:
                self._close_pair_dev(port)
        if not n:
            self._pair_end("failed", "신호가 되돌아오는 회로가 없습니다 — 버튼이 풀려 있는지, NC 접점(보통 빨간 표시)에 "
                                     "연결했는지, 어댑터의 출력선(RTS/DTR)과 입력선(CTS/DSR/DCD)에 연결했는지 확인하십시오")
            return
        self.pairing.update(step="press", deadline=self.clock() + self.PAIR_STEP_S, found=n,
                            msg=f"회로 {n}개를 찾았습니다 — 이제 비상정지 버튼을 누르십시오")
        self.log("SYS", f"[비상정지 등록] ① 회로 {n}개 찾음 — 버튼 누르기를 기다림")

    def _pair_finish(self, c: dict) -> None:
        p = c["port"]
        settle = round(min(0.08, max(0.02, c["lat"] * 3 + 0.01)), 3)
        cfg = {"port": p["port"], "vid": p.get("vid"), "pid": p.get("pid"), "serial": p.get("serial", ""),
               "location": p.get("location", ""), "desc": p.get("desc", ""), "out": c["out"], "in": c["in"],
               "settle_s": settle}
        dev = self._pair_devs.pop(p["port"])
        self._pair_close_all()
        self.cfg = cfg
        self.device, self.port_now = dev, p["port"]
        self.checker = LoopChecker(dev, c["out"], c["in"], settle, clock=self.clock, sleep=self.sleep, rng=self.rng)
        self._fails, self._oks = 0, self.CLOSE_AFTER_OKS
        self._set_input("closed", "버튼 풀림 · 회로 정상")
        self._ok_since = self.clock()
        self.on_save(cfg)
        lines = f"{LINE_KO[c['out']]}→{LINE_KO[c['in']]}"
        self.pairing.update(step="done", port=p["port"], lines=lines,
                            msg=f"등록 완료 — {p['port']} {lines} (응답 {c['lat'] * 1000:.0f}ms, 검사 대기 {settle * 1000:.0f}ms)")
        self.log("SYS", f"[비상정지 등록] ③ 완료 — {p['port']} {lines}, 앞으로 이 회로를 계속 감시합니다")

    def _pair_end(self, step: str, msg: str) -> None:
        self._pair_close_all()
        self._pair_queue, self._pair_cands = [], []
        if self.pairing is None:
            self.pairing = {}
        self.pairing.update(step=step, msg=msg)
        self.log("ERR" if step == "failed" else "SYS", f"[비상정지 등록] {msg}")
        self._last_open_try = -1e9                       # 등록돼 있던 스위치는 바로 다시 감시

    def _close_pair_dev(self, port: str) -> None:
        dev = self._pair_devs.pop(port, None)
        if dev is not None:
            try:
                dev.close()
            except Exception:
                pass

    def _pair_close_all(self) -> None:
        for port in list(self._pair_devs):
            self._close_pair_dev(port)

    # ---- 화면용 ----
    def status(self) -> dict:
        with self._lock:
            ok, why = self.reset_check()
            pairing = None
            if self.pairing:
                pairing = {k: v for k, v in self.pairing.items() if k in ("step", "msg", "found", "port", "lines")}
                if "deadline" in self.pairing and self.pairing["step"] in ("press", "release"):
                    pairing["left_s"] = max(0, round(self.pairing["deadline"] - self.clock()))
            cfg = self.cfg or {}
            return {
                "paired": bool(self.cfg),
                "input": self.input,
                "input_ko": INPUT_KO.get(self.input, self.input),
                "detail": self.input_detail,
                "port": self.port_now or cfg.get("port", ""),
                "lines": f"{LINE_KO[cfg['out']]}→{LINE_KO[cfg['in']]}" if cfg else "",
                "latched": self.latched,
                "reason": self.reason,
                "sources": sorted(self.sources),
                "since_s": round(self.clock() - self.tripped_at, 1) if self.latched else None,
                "robot_ack": self.robot_ack,
                "reset_ready": ok,
                "reset_why": why,
                "cycles": self.cycles,
                "latency_ms": self.latency_ms,
                "blocked": self.blocked,
                "pairing": pairing,
                "mock": self.mock_world is not None,
                "mock_pressed": self.mock_world.pressed if self.mock_world else None,
                "mock_plugged": self.mock_world.plugged if self.mock_world else None,
            }


class GuardedLink:
    """로봇 링크 앞의 단 하나의 관문: 비상정지 중에는 STOP · PING · VER 만 통과시킨다."""

    def __init__(self, link, estop: EStop):
        self._link = link
        self._estop = estop
        self.block_reason = ""

    def send(self, cmd: str):
        if not self._estop.allow(cmd):
            self.block_reason = self._estop.block_reason() or "비상정지"
            self._estop.note_blocked(cmd)
            return False
        self.block_reason = ""
        return self._link.send(cmd)

    def poll(self) -> list[str]:
        return [line for line in self._link.poll() if not self._estop.note_rx(line.strip())]

    def __getattr__(self, name):
        return getattr(self._link, name)
