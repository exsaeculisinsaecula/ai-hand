"""micro:bit V2 MicroPython 펌웨어 초안 (단일 Hiwonder AiHand + 슈트 구성).

실행 환경: BBC micro:bit V2 (MicroPython)
역할:
  1. USB 시리얼(115200bps)로 PC/Pi의 한 줄 명령 수신 (SHOW,N / PICK,N / DELIVER / RELEASE / RETURN,N / HOME / STOP)
  2. 자세 프리셋 이동 + 모든 서보 각도 한계 클램프(과부하 예방)
  3. 초음파 거리 임계값 + 연속 N회 감지 필터로 작업자 수령(EVT,HAND) 송신
  4. 버튼 A+B 동시 입력 시 즉시 비상 정지(STOP)
"""
from microbit import button_a, button_b, display, i2c, pin1, pin2, sleep, uart
import music

# 통합 테스트 시 실측 후 확정할 서보 각도 프리셋 (pan, tilt, grip)
# 각도 한계 클램프: pan [0, 180], tilt [20, 160], grip [10, 120]
ANGLE_LIMITS = {
    "pan": (0, 180),
    "tilt": (20, 160),
    "grip": (10, 120),
}

PRESETS = {
    "HOME": (90, 90, 20),
    "TRAY_1": (135, 130, 20),
    "TRAY_2": (150, 130, 20),
    "TRAY_3": (165, 130, 20),
    "CHUTE": (40, 110, 100),   # 경사로(슈트) 상단 위치
    "GRIP_CLOSE": 100,
    "GRIP_OPEN": 20,
}

# 초음파 손 근접 감지 필터 설정
HAND_DIST_CM = 15.0
HAND_CONSECUTIVE_N = 3

current_angles = list(PRESETS["HOME"])
emergency_stopped = False
handover_armed = False
hand_hit_count = 0


def clamp(value, low, high):
    return max(low, min(high, value))


def write_servo(channel, angle):
    """Hiwonder AiHand 서보 보드(I2C/PWM) 각도 출력 자리 (하드웨어 보드 주소에 맞게 연결)."""
    # TODO: 통합 테스트 시 AiHand 서보 드라이버 I2C 프로토콜 또는 PCA9685 레지스터 쓰기 연결
    _ = (channel, angle)


def move_pose(pan, tilt, grip):
    global current_angles
    if emergency_stopped:
        return
    p = clamp(pan, *ANGLE_LIMITS["pan"])
    t = clamp(tilt, *ANGLE_LIMITS["tilt"])
    g = clamp(grip, *ANGLE_LIMITS["grip"])
    current_angles = [p, t, g]
    write_servo(0, p)
    write_servo(1, t)
    write_servo(2, g)


def read_ultrasonic_cm():
    """초음파 센서 거리 측정 자리 (실물 핀/모듈에 맞게 교체)."""
    # TODO: 트리거/에코 핀(pin1/pin2) 펄스 측정 구현
    return 999.0


def is_tray_empty(part_num):
    """WonderCam 또는 잡기 후 손가락 각도로 빈 트레이 여부를 판별한다."""
    _ = part_num
    return False


def send_line(text):
    uart.write(text + "\n")


def handle_command(line):
    global emergency_stopped, handover_armed, hand_hit_count
    cmd = line.strip()
    if not cmd:
        return

    if cmd == "STOP":
        emergency_stopped = True
        handover_armed = False
        display.show("X")
        send_line("ACK")
        return

    if cmd == "HOME":
        emergency_stopped = False
        handover_armed = False
        move_pose(*PRESETS["HOME"])
        sleep(300)
        display.clear()
        send_line("DONE,HOME")
        return

    if emergency_stopped:
        send_line("ERR,STOPPED")
        return

    if cmd.startswith("SHOW,"):
        n = cmd.split(",", 1)[1]
        display.show(n)
        music.pitch(880, 100, wait=False)
        send_line("ACK")

    elif cmd.startswith("PICK,"):
        n = cmd.split(",", 1)[1]
        key = "TRAY_" + n
        if key not in PRESETS:
            send_line("ERR,ARG")
            return
        if is_tray_empty(n):
            send_line("ERR,EMPTY")
            return
        pan, tilt, _ = PRESETS[key]
        move_pose(pan, tilt, PRESETS["GRIP_OPEN"])
        sleep(400)
        move_pose(pan, tilt, PRESETS["GRIP_CLOSE"])
        sleep(400)
        send_line("DONE,PICK")

    elif cmd == "DELIVER":
        pan, tilt, _ = PRESETS["CHUTE"]
        move_pose(pan, tilt, PRESETS["GRIP_CLOSE"])
        sleep(500)
        send_line("DONE,DELIVER")

    elif cmd == "RELEASE":
        pan, tilt, _ = PRESETS["CHUTE"]
        move_pose(pan, tilt, PRESETS["GRIP_OPEN"])
        sleep(300)
        handover_armed = True
        hand_hit_count = 0
        send_line("DONE,RELEASE")

    elif cmd.startswith("RETURN,"):
        n = cmd.split(",", 1)[1]
        key = "TRAY_" + n
        if key in PRESETS:
            pan, tilt, _ = PRESETS[key]
            move_pose(pan, tilt, PRESETS["GRIP_OPEN"])
            sleep(400)
        handover_armed = False
        send_line("DONE,RETURN")


def main():
    global hand_hit_count, handover_armed
    uart.init(baudrate=115200)
    move_pose(*PRESETS["HOME"])
    rx_buf = ""

    while True:
        # 비상 정지: 버튼 A+B 동시 입력
        if button_a.is_pressed() and button_b.is_pressed():
            handle_command("STOP")
            sleep(300)

        if uart.any():
            chunk = uart.read()
            if chunk:
                rx_buf += chunk.decode("utf-8", "ignore")
                while "\n" in rx_buf:
                    line, rx_buf = rx_buf.split("\n", 1)
                    handle_command(line)

        if handover_armed and not emergency_stopped:
            if read_ultrasonic_cm() <= HAND_DIST_CM:
                hand_hit_count += 1
                if hand_hit_count >= HAND_CONSECUTIVE_N:
                    handover_armed = False
                    hand_hit_count = 0
                    send_line("EVT,HAND")
            else:
                hand_hit_count = 0

        sleep(20)


if __name__ == "__main__":
    main()
