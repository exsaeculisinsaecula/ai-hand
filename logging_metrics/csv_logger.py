"""UI·로깅 모듈 — 상태 전이, 명령/응답, 지표 계산용 타임스탬프 CSV 로거."""
import csv
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Callable


class CsvEventLogger:
    """상태 머신의 모든 이벤트를 CSV 파일로 기록한다."""

    HEADER = ["iso_time", "mono_sec", "event_type", "state", "part", "detail"]

    def __init__(self, log_path: Path, clock: Callable[[], float] = time.monotonic):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()

        write_header = not self.log_path.exists() or self.log_path.stat().st_size == 0
        if write_header:
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADER)

    def log(self, event_type: str, state: str, part: int | None = None, detail: str = "") -> None:
        iso_now = datetime.now().isoformat(timespec="milliseconds")
        mono_sec = f"{self._clock():.4f}"
        row = [iso_now, mono_sec, event_type, state, "" if part is None else str(part), detail]
        with self._lock:
            with self.log_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    def clear(self) -> None:
        """누적 CSV 이벤트 로그를 헤더만 남기고 초기화한다."""
        with self._lock:
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADER)
