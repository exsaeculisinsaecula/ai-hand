"""micro:bit V2 MicroPython 펌웨어 — VLM 신호(GO) 매크로 버전.

팀 펌웨어(microbit/handover.py)를 그대로 두고 'GO' 명령 하나만 더했다.
  GO 또는 1      → DONE,GO        손가락 접기 → 3초 대기 → 손가락 펴기 (1회)
노트북의 vlm_go.py 가 VLM 판단이 true 일 때만 GO 를 보낸다.
(handover_app.py 를 다시 실행하면 팀 펌웨어로 자동 재설치된다)

--- 이하 팀 펌웨어 설명 ---

PC(USB 시리얼 115200) 명령 → 응답:
  PING            → ACK
  VER             → VER,<n>        펌웨어 버전 (PC가 자동 갱신 판단)
  OPEN            → DONE,OPEN      손 펴기 (받을 준비)
  GRIP            → DONE,GRIP      손 쥐기
  PAN,<0-180>     → DONE,PAN       회전 (90=정면, 180=왼쪽, 0=오른쪽)
  SERVO,<1-6>,<0-180>,<ms> → DONE,SERVO   단일 서보 시험
  PLACE[,각도]    → DONE,PLACE     손목을 돌려(기본 110°) 손을 펴서 내려놓고 정면 복귀
  NG              → DONE,NG        잘못된 부품: 손목 좌우로 흔들기
  HOME            → DONE,HOME      정면 + 손 펴기
  STOP            → ACK            정지 (이후 HOME 전까지 동작 명령 거부)
이벤트: EVT,BTN_A (버튼 A = '쥐어줬어요' 수동 확인)
비상 정지: 버튼 A+B 동시.

서보 컨트롤러 프로토콜(Hiwonder StartbitV2와 동일):
  P12(TX)/P8(RX) 115200 UART, 패킷 55 55 len 03 count durL durH [id posL posH]...
  pos = 500 + angle/180*2000 (µs)
micro:bit에는 UART가 하나뿐이라, 서보 패킷을 보낼 때만 잠깐 P12/P8로 바꿨다가 USB로 되돌린다.
"""
from microbit import button_a, button_b, display, pin8, pin12, sleep, uart, Image
import math

# 서보 1=엄지(반대 방향), 2~5=검지~새끼, 6=회전
# 팀 보정값 (NeuralGrip CALIBRATION.md, 2026-11-21 실물 시험)
OPEN_ANGLES = {1: 50, 2: 130, 3: 130, 4: 130, 5: 130}
GRIP_ANGLES = {1: 130, 2: 50, 3: 50, 4: 50, 5: 50}
PAN_FRONT = 90       # 손목 정면
PAN_PLACE = 110      # 손목 내려놓기 방향 (PLACE,<각도> 로 바꿀 수 있음)
GRIP_MS = 1500       # 쥐기 (ms)
OPEN_MS = 700        # 펴기 (ms)
PAN_MS = 1500        # 손목 회전 (ms)
STEP_MS = 60         # 보간 한 단계 (ms)
FW_VERSION = "5-go"  # PC가 VER 로 확인 → 다르면 자동 재설치
GO_HOLD_MS = 3000    # GO: 접은 채 기다리는 시간 (ms)

stopped = False
aborted = False      # 이동 도중 STOP 수신


def with_pan(angles, pan):
    d = dict(angles)
    d[6] = pan
    return d


def _pos(angle):
    a = max(0, min(180, int(angle)))
    return 500 + a * 2000 // 180


def _packet(sid, angle, duration_ms):
    p = _pos(angle)
    # Hiwonder StartbitV2 setPwmServo 와 동일한 10바이트 단일 서보 패킷
    return bytearray([0x55, 0x55, 0x08, 0x03, 0x01,
                      duration_ms & 0xFF, (duration_ms >> 8) & 0xFF,
                      sid, p & 0xFF, (p >> 8) & 0xFF])


def _to_servo_bus():
    uart.init(baudrate=115200, tx=pin12, rx=pin8)
    sleep(3)            # TX 라인이 idle(High)로 안정될 시간


def _to_usb():
    sleep(3)            # 마지막 바이트가 다 나갈 시간 (10바이트 ≈ 0.9ms)
    uart.init(baudrate=115200)
    # 서보 보드가 P8로 보낸 데이터(전압 보고 등)가 버퍼에 남으면 다음 명령이 깨지므로 비운다
    while uart.any():
        uart.read()


def send_servos(targets, duration_ms):
    """targets: {servo_id: angle}. 서보마다 단일 패킷(count=1)으로 보낸다 (공식 확장과 동일)."""
    _to_servo_bus()
    for sid, ang in targets.items():
        uart.write(_packet(sid, ang, duration_ms))
        sleep(2)
    _to_usb()


def diag():
    """MakeCode 초기화와 같은 순서: P12/P8 고정 → 2초 대기 → 회전 서보 크게 왕복 3회."""
    uart.init(baudrate=115200, tx=pin12, rx=pin8)
    sleep(2000)
    for _ in range(3):
        uart.write(_packet(6, 60, 500))
        sleep(700)
        uart.write(_packet(6, 120, 500))
        sleep(700)
    for sid in (1, 2, 3, 4, 5):
        uart.write(_packet(sid, 90, 400))
        sleep(5)
    sleep(500)
    uart.write(_packet(6, 90, 500))
    sleep(700)
    _to_usb()


# 현재 각도 추적 (부드러운 보간의 출발점)
cur = {1: 50, 2: 130, 3: 130, 4: 130, 5: 130, 6: 90}


def _check_stop():
    """USB 로 돌아온 짧은 순간 들어온 바이트에서 STOP 을 찾는다 (서보 보드 잡음은 버림)."""
    global aborted, stopped
    if uart.any():
        data = uart.read() or b""
        if b"STOP" in data:
            aborted = True
            stopped = True
    return aborted


def smooth(targets, total_ms):
    """코사인 이징(천천히 출발 → 가속 → 천천히 정지). 단계마다 STOP 확인 → 이동 중 중단 가능."""
    global aborted
    aborted = False
    steps = max(4, total_ms // STEP_MS)
    start = {sid: cur.get(sid, ang) for sid, ang in targets.items()}
    for k in range(1, steps + 1):
        e = (1 - math.cos(math.pi * k / steps)) / 2
        _to_servo_bus()
        for sid, ang in targets.items():
            a = start[sid] + (ang - start[sid]) * e
            uart.write(_packet(sid, a, STEP_MS + 20))
            cur[sid] = a
            sleep(1)
        sleep(2)
        uart.init(baudrate=115200)
        if _check_stop():
            display.show(Image.NO)
            return False
        sleep(max(1, STEP_MS - 10 - len(targets)))
    for sid, ang in targets.items():
        cur[sid] = ang
    return True


def reply(text):
    uart.write(text + "\n")


def move(targets, ms):
    ok = smooth(targets, ms)
    sleep(80)
    return ok


def do_place(angle):
    if not move({6: angle}, PAN_MS):
        return False
    if not move(OPEN_ANGLES, OPEN_MS):
        return False
    sleep(300)
    return move({6: PAN_FRONT}, PAN_MS)


def do_ng():
    """잘못된 부품: 손목을 좌우로 흔들어 '아니에요' 신호."""
    for _ in range(2):
        if not move({6: PAN_FRONT - 20}, 260):
            return False
        if not move({6: PAN_FRONT + 20}, 260):
            return False
    return move({6: PAN_FRONT}, 260)


def do_go():
    """VLM 신호 매크로: 손가락 접기 → 3초 대기 → 손가락 펴기. 대기 중에도 STOP 확인."""
    if not move(GRIP_ANGLES, GRIP_MS):
        return False
    waited = 0
    while waited < GO_HOLD_MS:
        sleep(100)
        waited += 100
        if _check_stop():
            return False
    return move(OPEN_ANGLES, OPEN_MS)


def handle(line):
    global stopped
    cmd = "".join(ch for ch in line if 32 <= ord(ch) < 127).strip().upper()
    if not cmd:
        return
    if cmd == "PING":
        reply("ACK")
        return
    if cmd == "VER":
        reply("VER," + FW_VERSION)
        return
    if cmd == "STOP":
        stopped = True
        display.show(Image.NO)
        reply("ACK")
        return
    if cmd == "HOME":
        stopped = False
        move(with_pan(OPEN_ANGLES, PAN_FRONT), OPEN_MS)
        display.show(Image.HAPPY)
        reply("DONE,HOME")
        return
    if stopped:
        reply("ERR,STOPPED")
        return
    try:
        if cmd == "OPEN":
            if not move(OPEN_ANGLES, OPEN_MS):
                reply("ERR,ABORTED")
                return
            display.show(Image.ARROW_S)
            reply("DONE,OPEN")
        elif cmd == "GRIP":
            if not move(GRIP_ANGLES, GRIP_MS):
                reply("ERR,ABORTED")
                return
            display.show(Image.SQUARE)
            reply("DONE,GRIP")
        elif cmd.startswith("PAN,"):
            move({6: int(cmd.split(",")[1])}, PAN_MS)
            reply("DONE,PAN")
        elif cmd.startswith("SERVO,"):
            _, sid, ang, ms = cmd.split(",")
            sid = int(sid)
            if sid < 1 or sid > 6:
                reply("ERR,ARG")
                return
            move({sid: int(ang)}, max(100, min(3000, int(ms))))
            reply("DONE,SERVO")
        elif cmd == "DIAG":
            display.show(Image.DIAMOND)
            diag()
            reply("DONE,DIAG")
        elif cmd == "PLACE" or cmd.startswith("PLACE,"):
            angle = PAN_PLACE
            if "," in cmd:
                angle = max(0, min(180, int(cmd.split(",")[1])))
            display.show(Image.ARROW_W)
            if do_place(angle):
                display.show(Image.YES)
                reply("DONE,PLACE")
            else:
                reply("ERR,ABORTED")
        elif cmd == "GO" or cmd == "1":
            display.show(Image.SQUARE)
            if do_go():
                display.show(Image.YES)
                reply("DONE,GO")
            else:
                reply("ERR,ABORTED")
        elif cmd == "NG":
            display.show(Image.NO)
            if do_ng():
                reply("DONE,NG")
            else:
                reply("ERR,ABORTED")
        else:
            reply("ERR,UNKNOWN")
    except Exception:
        reply("ERR,ARG")


def main():
    uart.init(baudrate=115200)
    sleep(300)
    display.show(Image.HAPPY)
    reply("READY")
    buf = ""
    a_was = False
    while True:
        a = button_a.is_pressed()
        if a and button_b.is_pressed():
            handle("STOP")
            sleep(400)
        elif a and not a_was:
            reply("EVT,BTN_A")
        a_was = a
        if uart.any():
            data = uart.read()
            if data:
                buf += "".join(chr(b) for b in data if b == 10 or 32 <= b < 127)
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    handle(line)
        sleep(15)


main()
