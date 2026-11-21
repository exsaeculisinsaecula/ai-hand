"""쟁반 위 부품 셋. box는 x, y, x2, y2."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Part:
    id: int
    name: str
    box: tuple[int, int, int, int]
    code: str = ""
    spec: str = ""
    color_rgb: tuple[int, int, int] = (154, 52, 18)
    default_stock: int = 3


PARTS: tuple[Part, ...] = (
    Part(
        1,
        "캡A",
        (80, 150, 180, 250),
        code="CAP-A",
        spec="표준 실린더 캡 · 테라코타",
        color_rgb=(180, 83, 9),
        default_stock=3,
    ),
    Part(
        2,
        "캡B",
        (260, 150, 360, 250),
        code="CAP-B",
        spec="중형 육각 캡 · 슬레이트",
        color_rgb=(37, 99, 235),
        default_stock=3,
    ),
    Part(
        3,
        "캡C",
        (440, 150, 540, 250),
        code="CAP-C",
        spec="락킹 엔드 캡 · 포레스트",
        color_rgb=(21, 128, 61),
        default_stock=3,
    ),
)

FRAME_W = 640
FRAME_H = 480


def by_id(part_id: int) -> Part:
    for part in PARTS:
        if part.id == part_id:
            return part
    raise KeyError(part_id)


def default_inventory() -> dict[int, int]:
    return {part.id: part.default_stock for part in PARTS}
