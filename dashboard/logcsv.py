"""작업 이벤트 CSV 및 통계 집계."""
import csv
from datetime import datetime
from pathlib import Path
import threading

HEADER = ["iso_time", "state", "part", "event", "match", "pick", "deliver", "detail"]
DEFAULT_LOG = Path("logs/dashboard_events.csv")

_lock = threading.Lock()


class CsvLog:
    def __init__(self, path: Path = DEFAULT_LOG):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(HEADER)

    def write(
        self,
        state: str,
        part: int | None,
        event: str,
        match: str,
        pick: str,
        deliver: str,
        detail: str,
    ) -> None:
        row = [
            datetime.now().isoformat(timespec="milliseconds"),
            state,
            "" if part is None else str(part),
            event,
            match,
            pick,
            deliver,
            detail,
        ]
        with _lock:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)


def read_all(path: Path = DEFAULT_LOG) -> list[list[str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    return rows[1:] if rows[0] == HEADER else rows


def read_tail(path: Path = DEFAULT_LOG, n: int = 8) -> list[list[str]]:
    body = read_all(path)
    return body[-n:] if n > 0 else []


def clear_log(path: Path = DEFAULT_LOG) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)


def summarize_log(path: Path = DEFAULT_LOG) -> dict:
    rows = read_all(path)
    total_runs = 0
    done_runs = 0
    fail_runs = 0
    part_counts = {"1": 0, "2": 0, "3": 0}
    fail_reasons: dict[str, int] = {}

    for row in rows:
        if len(row) < 8:
            continue
        state, part, detail = row[1], row[2], row[7]
        if state == "DONE":
            total_runs += 1
            done_runs += 1
            if part in part_counts:
                part_counts[part] += 1
        elif state == "FAIL":
            total_runs += 1
            fail_runs += 1
            if part in part_counts:
                part_counts[part] += 1
            reason = detail or "기타 실패"
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    success_rate = round((done_runs / total_runs) * 100.0, 1) if total_runs else 0.0
    return {
        "total_runs": total_runs,
        "done_runs": done_runs,
        "fail_runs": fail_runs,
        "success_rate": success_rate,
        "part_counts": part_counts,
        "fail_reasons": fail_reasons,
    }
