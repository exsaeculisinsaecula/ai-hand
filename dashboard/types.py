"""대시보드 상태 스냅샷."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Snapshot:
    state: str
    tray: np.ndarray
    hand: np.ndarray
    part: int | None
    match: str
    pick: str
    deliver: str
    detail: str
    note: str = ""
    hand_status: str = "IDLE"
    stock_after: dict[int, int] | None = None
    protocol_lines: tuple[str, ...] = field(default_factory=tuple)
