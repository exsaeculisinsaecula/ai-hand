"""실물 micro:bit V2 연결용 USB 시리얼 (115200bps) 링크."""
from collections import deque
import threading
import time

import serial
from serial.tools import list_ports


def find_microbit_port() -> str | None:
    """연결된 micro:bit 또는 USB 시리얼 포트를 자동 탐색한다."""
    ports = list(list_ports.comports())
    for p in ports:
        desc = f"{p.description} {p.manufacturer}".lower()
        if "microbit" in desc or "mbed" in desc or "daplink" in desc:
            return p.device
    return ports[0].device if ports else None


class SerialMicrobitLink:
    """pyserial 기반 비동기 수신 큐를 갖춘 줄 단위 시리얼 링크."""

    is_mock: bool = False

    def __init__(self, port: str | None = None, baudrate: int = 115200):
        resolved = port or find_microbit_port()
        if not resolved:
            raise RuntimeError("연결된 micro:bit 시리얼 포트를 찾을 수 없습니다.")
        self.port = resolved
        self._ser = serial.Serial(resolved, baudrate=baudrate, timeout=0.05)
        self._rx_queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while self._running:
            try:
                raw = self._ser.readline()
                if raw:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line:
                        with self._lock:
                            self._rx_queue.append(line)
            except serial.SerialException:
                time.sleep(0.05)

    def send(self, command: str) -> None:
        payload = (command.strip() + "\n").encode("utf-8")
        with self._lock:
            self._ser.write(payload)
            self._ser.flush()

    def poll(self) -> list[str]:
        with self._lock:
            items = list(self._rx_queue)
            self._rx_queue.clear()
        return items

    def close(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._ser.is_open:
            self._ser.close()
