"""합성 쟁반·핸드 프레임."""
import numpy as np

from dashboard.draw import hand, tray


def tray_frame(
    selected: int | None,
    *,
    stock: dict[int, int] | None = None,
    fail_part: int | None = None,
    status_label: str = "",
) -> np.ndarray:
    return tray(selected, stock=stock, fail_part=fail_part, status_label=status_label)


def hand_frame(
    held: int | None,
    *,
    status: str = "IDLE",
    detail: str = "",
) -> np.ndarray:
    return hand(held, status=status, detail=detail)
