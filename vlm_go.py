"""초간단 VLM → 로봇 신호: "영상에 빨간 공이 있으면 인식해줘"

[1] 사용자 입력 + 웹캠 사진 한 장
[2] Gemma(로컬 Ollama)에 사진 + 단호한 프롬프트 → {"found": true/false} 만 받음
[3] true 면 micro:bit 로 'GO' 한 줄 전송, false 면 아무것도 안 함
[4] micro:bit(microbit_go.py)가 GO 를 받으면 손가락 접기 → 3초 → 펴기 1회

  python3 vlm_go.py --install            # (처음 한 번) micro:bit 에 microbit_go.py 설치
  python3 vlm_go.py                      # 문장 입력 → 한 번 확인
  python3 vlm_go.py --watch 3            # 대상 한 번 입력 → 3초마다 자동 확인 (보이면 GO, 사라졌다 다시 보이면 또 GO)
  python3 vlm_go.py --dry                # micro:bit 없이 판단만 시험
  옵션: --camera 1  --model gemma3:4b  --port /dev/cu.usbmodemXXXX

결과 사진·기록: runs/vlm_go/ (사진에 판단 결과가 적혀 저장됨), runs/vlm_go/log.csv

출처: 팀 vlm-go (https://github.com/jihustar12-design/vlm-go) — 이 저장소에 맞춰
  · 펌웨어는 팀 펌웨어(microbit/handover.py, GO 명령 포함 v6)를 그대로 씀 (대시보드와 번갈아 써도 재설치 없음)
  · Windows COM 포트 자동 탐색, 결과 사진 열기(--show) Windows 지원
  · 대시보드(handover_app.py)는 같은 VLM 판단을 vision/vlm_bridge.py 로 씀
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime
import glob
import os
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

import cv2

HERE = Path(__file__).resolve().parent
FIRMWARE = HERE / "microbit" / "handover.py"     # 팀 펌웨어 (GO 명령 포함)
GO_FIRMWARE = "6"
OUT = HERE / "runs" / "vlm_go"
LAST: dict = {}

PROMPT = ("사용자가 '{target}'을(를) 찾고 있어. 이 사진 안에 {target}이(가) 명확하게 있으면 "
          "오직 {{\"found\": true}}라고 대답하고, 없거나 헷갈리면 {{\"found\": false}}라고만 대답해. "
          "다른 말은 절대 하지 마.")
SCHEMA = {"type": "object", "properties": {"found": {"type": "boolean"}}, "required": ["found"]}


# ---------------------------------------------------------------- [1] 입력 · 사진
def extract_target(text: str) -> str:
    """'영상에 빨간 공이 있으면 인식해줘' → '빨간 공'"""
    t = text.strip().strip(".!?\"'")
    t = re.sub(r"^(지금\s*)?(영상|화면|카메라|사진|웹캠)(에|에서|속에|안에)\s*", "", t)
    t = re.sub(r"(이|가)?\s*(있으면|보이면|있는지|보이는지|나오면).*$", "", t)
    t = re.sub(r"\s*(을|를)?\s*(찾아|인식해|알려|확인해).*$", "", t)
    return t.strip() or text.strip()


class Camera:
    def __init__(self, src):
        self.src = src
        self.still = None
        if isinstance(src, str) and not src.isdigit():          # 시험용 사진 파일
            self.still = cv2.imread(src)
            if self.still is None:
                sys.exit(f"사진을 읽을 수 없어요: {src}")
            self.cap = None
            return
        self.cap = cv2.VideoCapture(int(src))
        if not self.cap.isOpened():
            sys.exit(f"웹캠 {src}번을 열 수 없어요. --camera 번호를 바꿔 보세요 (0=맥 내장, 보통 1=USB)")

    def snap(self):
        if self.still is not None:
            return self.still.copy()
        for _ in range(5):                                       # 버퍼에 남은 옛 화면 버리기
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            sys.exit("웹캠에서 화면을 못 읽었어요")
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------- [2] VLM
def ask_vlm(frame, target: str, model: str, url: str, timeout: float = 120) -> tuple[bool, str, float]:
    """→ (found, 원문 답, 걸린 초). 형식이 이상하면 false (안전한 쪽)"""
    h, w = frame.shape[:2]
    if max(h, w) > 1024:
        s = 1024 / max(h, w)
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    body = {"model": model, "stream": False, "format": SCHEMA, "options": {"temperature": 0},
            "messages": [{"role": "user", "content": PROMPT.format(target=target),
                          "images": [base64.b64encode(buf.tobytes()).decode()]}]}
    t0 = time.monotonic()
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = json.loads(r.read())["message"].get("content", "") or ""
    except urllib.error.URLError as e:
        sys.exit(f"Ollama에 연결할 수 없어요 ({e}). Ollama 앱을 켜고 'ollama pull {model}' 했는지 확인하세요.")
    dt = time.monotonic() - t0
    return parse_found(raw), raw.strip(), dt


def parse_found(raw: str) -> bool:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if m:
        try:
            f = json.loads(m.group(0)).get("found")
            return f is True or str(f).strip().lower() == "true"
        except json.JSONDecodeError:
            pass
    return bool(re.search(r"\btrue\b", raw or "", re.I))        # 형식이 깨져도 'true' 단어가 있으면 인정


# ---------------------------------------------------------------- [3] micro:bit
def find_port() -> str:
    """micro:bit 포트 자동 탐색 (Windows COM / macOS / Linux)"""
    try:
        from serial.tools import list_ports
        for p in list_ports.comports():
            if p.vid == 0x0D28 or "mbed" in (p.description or "").lower() or "micro:bit" in (p.description or "").lower():
                return p.device
    except Exception:
        pass
    return next(iter(sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*"))), "")


class MicrobitLink:
    def __init__(self, port: str = ""):
        import serial
        port = port or find_port()
        if not port:
            sys.exit("micro:bit 를 찾을 수 없어요. USB 를 꽂았는지 확인하세요 (micro:bit 없이 시험: --dry)")
        self.port = port
        self.ser = serial.Serial(port, 115200, timeout=0.1)
        time.sleep(1.0)                               # 포트 연 직후 안정될 시간
        self.ser.reset_input_buffer()

    def send(self, cmd: str, wait: str | None = None, timeout: float = 15.0, retries: int = 1) -> str:
        """명령 한 줄 보내고 응답 기다리기. 응답이 없으면 한 번 더 보냄."""
        for attempt in range(retries + 1):
            self.ser.reset_input_buffer()             # 예전에 남은 응답 버리기
            self.ser.write(f"\n{cmd}\n".encode())
            self.ser.flush()
            if not wait:
                return ""
            deadline, buf, first = time.monotonic() + timeout, b"", 3.0
            while time.monotonic() < deadline:
                buf += self.ser.read(256)
                for line in buf.decode("utf-8", "replace").splitlines():
                    line = line.strip()
                    if line.startswith(wait) or line.startswith("ERR"):
                        return line
                if not buf and time.monotonic() > deadline - timeout + first:
                    break                             # 3초 동안 아무 응답 없음 → 다시 보내기
            self.last_raw = buf
        return "TIMEOUT (응답: %r)" % getattr(self, "last_raw", b"")[:60]

    def version(self) -> str:
        r = self.send("VER", "VER,", 2.0)
        return r.split(",", 1)[1] if r.startswith("VER,") else ""

    def close(self):
        self.ser.close()


def _firmware_code() -> bytes:
    """설명·주석을 빼서 작게 만든 펌웨어 (micro:bit 메모리에 맞게, 동작은 같음)"""
    import ast
    tree = ast.parse(FIRMWARE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return (ast.unparse(tree) + "\n").encode("utf-8")


def _raw_exec(ser, code: bytes, timeout: float = 5.0) -> bytes:
    """raw REPL 에서 짧은 코드 한 덩어리 실행 → 끝날 때까지 기다림 (한꺼번에 보내면 글자가 빠짐)"""
    ser.write(code + b"\x04")
    deadline, resp = time.monotonic() + timeout, b""
    while time.monotonic() < deadline and resp.count(b"\x04") < 2:
        resp += ser.read(256)
    if not resp.startswith(b"OK") or b"Error" in resp or resp.count(b"\x04") < 2:
        raise RuntimeError(resp[:120])
    return resp


def install_firmware(port: str = "") -> None:
    """raw REPL 로 microbit_go.py 를 main.py 로 써 넣기.
    micro:bit 에 MicroPython 이 한 번은 올라가 있어야 함 (팀 펌웨어를 쓴 적 있으면 OK)."""
    import serial
    port = port or find_port()
    if not port:
        sys.exit("micro:bit 를 찾을 수 없어요")
    ser = serial.Serial(port, 115200, timeout=0.05)
    time.sleep(0.3)
    got = b""
    for _ in range(8):
        ser.write(b"\r\x03\x03")
        time.sleep(0.25)
        got += ser.read(2048)
        if b">>>" in got:
            break
    if b">>>" not in got:
        sys.exit("설치 실패: micro:bit 프로그램을 멈추지 못했어요. MicroPython 이 아니면 python.microbit.org 에서 "
                 "microbit_go.py 를 붙여 넣고 'Send to micro:bit' 로 올리세요.")
    ser.reset_input_buffer()
    ser.write(b"\x01")
    time.sleep(0.3)
    if b"raw REPL" not in ser.read(512):
        sys.exit("설치 실패: raw REPL 진입 실패")
    code = _firmware_code()
    try:
        _raw_exec(ser, b"f=open('main.py','wb')")
        n = (len(code) + 63) // 64
        for i, k in enumerate(range(0, len(code), 64)):
            _raw_exec(ser, b"f.write(" + repr(code[k:k + 64]).encode() + b")")
            print(f"\r  설치 중 {i + 1}/{n}", end="", flush=True)
        _raw_exec(ser, b"f.close()")
    except RuntimeError as exc:
        print()
        sys.exit(f"설치 실패: {exc}")
    print()
    ser.write(b"\x02")
    time.sleep(0.1)
    ser.write(b"\x04")
    time.sleep(2.5)
    ser.close()
    print(f"설치 완료 ({port}) — micro:bit 화면에 웃는 얼굴이 나오면 준비된 거예요 ({len(code)} bytes)")


# ---------------------------------------------------------------- 기록
def save(frame, target, found, raw, dt, sent):
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")
    vis = frame.copy()
    col = (40, 180, 40) if found else (60, 60, 220)
    k = max(1.0, vis.shape[1] / 900)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], int(60 * k)), (255, 255, 255), -1)
    cv2.putText(vis, f"found={str(found).lower()}  {dt:.1f}s  {'GO sent' if sent else 'wait'}", (12, int(42 * k)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1 * k, col, max(2, int(2 * k)), cv2.LINE_AA)
    path = OUT / f"{stamp}_{'found' if found else 'none'}.jpg"
    cv2.imwrite(str(path), vis)
    LAST["path"] = path
    log = OUT / "log.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "target", "found", "vlm_raw", "seconds", "go_sent", "photo"])
        w.writerow([datetime.now().strftime("%H:%M:%S"), target, found, raw, round(dt, 2), sent, path.name])
    return vis


def check_once(cam, link, target, a) -> bool:
    frame = cam.snap()
    print(f"  [사진] 찰칵 → [VLM] '{target}' 있나요? ...", end="", flush=True)
    found, raw, dt = ask_vlm(frame, target, a.model, a.url)
    print(f" {raw}  ({dt:.1f}s)")
    sent = False
    if found:
        if link is None:
            print("  [신호] true → ※ 시험 모드(--dry)라 로봇에 보내지 않음. 로봇을 움직이려면 --dry 를 빼고 실행하세요")
        else:
            print("  [신호] true → micro:bit 로 GO 전송 ... ", end="", flush=True)
            r = link.send("GO", "DONE,GO", timeout=15)
            print(r)
            sent = r.startswith("DONE")
            if not sent:
                print("  ! 손이 안 움직였어요. 펌웨어가 v6(GO 포함) 인지(맨 위 [micro:bit] 줄), 서보 전원이 켜졌는지 확인하세요")
    else:
        print("  [신호] false → 아무것도 안 하고 대기")
    vis = save(frame, target, found, raw, dt, sent)
    if a.show:                                   # 맥 '미리보기'로 판단한 사진 열기
        if sys.platform.startswith("win"):
            os.startfile(str(LAST["path"]))
        else:
            import subprocess
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(LAST["path"])], check=False)
    return found


def main():
    for stream in (sys.stdout, sys.stderr):           # Windows 콘솔(cp949)에서 '—' 등으로 멈추지 않게
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="VLM이 보면 GO — 초간단 파이프라인")
    ap.add_argument("--camera", default="1", help="웹캠 번호 (0=내장, 보통 1=USB) 또는 시험용 사진 경로")
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--port", default="", help="micro:bit 포트 (생략하면 자동)")
    ap.add_argument("--dry", action="store_true", help="micro:bit 없이 판단만")
    ap.add_argument("--watch", type=float, default=0, help="N초마다 자동 확인 (0이면 입력할 때마다 한 번)")
    ap.add_argument("--target", default="", help="찾을 대상 (생략하면 문장을 입력받음)")
    ap.add_argument("--show", action="store_true", help="판단할 때마다 그 사진을 미리보기로 열기")
    ap.add_argument("--install", action="store_true", help="micro:bit 에 팀 펌웨어(GO 포함) 설치하고 끝")
    a = ap.parse_args()

    if a.install:
        install_firmware(a.port)
        return
    link = None
    if not a.dry:
        link = MicrobitLink(a.port)
        ver = link.version()
        print(f"[micro:bit] {link.port} · 펌웨어 {ver or '응답 없음'}")
        if ver != GO_FIRMWARE:
            print(f"  ! GO 명령이 있는 펌웨어(v{GO_FIRMWARE})가 아니에요. 먼저: python vlm_go.py --install "
                  "(또는 handover_app.py 를 한 번 실행하면 자동 설치)")
    cam = Camera(a.camera)
    print(f"[준비] 웹캠 {a.camera} · VLM {a.model} · {'시험(micro:bit 없음)' if a.dry else 'micro:bit 연결'}")
    try:
        if a.watch:
            text = a.target or input("찾을 것을 말해 주세요 (예: 영상에 빨간 공이 있으면 인식해줘): ")
            target = extract_target(text)
            print(f"[대상] {target} — {a.watch:g}초마다 확인, Ctrl+C 로 종료")
            armed = True                                  # 보였다가 사라져야 다시 GO (같은 물체로 반복 동작 방지)
            while True:
                if armed:
                    if check_once(cam, link, target, a):
                        armed = False
                else:
                    frame = cam.snap()
                    found, raw, dt = ask_vlm(frame, target, a.model, a.url)
                    print(f"  (이미 동작함 · 사라지면 다시 대기) {raw} ({dt:.1f}s)")
                    armed = not found
                time.sleep(a.watch)
        while True:
            text = a.target or input("\n찾을 것을 말해 주세요 (q=종료): ").strip()
            if text.lower() == "q":
                break
            if not text:
                continue
            target = extract_target(text)
            print(f"[대상] {target}")
            check_once(cam, link, target, a)
            if a.target:
                break
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        cam.close()
        if link is not None:
            link.close()


if __name__ == "__main__":
    main()
