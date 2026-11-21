"""OpenCV + Pillow 기반 쟁반·5지 관절 AiHand 로봇 핸드 프레임 렌더러 (RGB)."""
from functools import lru_cache
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dashboard.catalog import FRAME_H, FRAME_W, PARTS, Part, by_id

BG = (235, 231, 221)
GRID = (222, 217, 205)
TRAY_DECK = (226, 221, 210)
FILL = (212, 207, 195)
FILL_SELECTED = (250, 246, 238)
FILL_FAIL = (246, 224, 220)
INK = (28, 28, 28)
MUTED = (110, 105, 98)
ACCENT = (154, 52, 18)
PASS_COLOR = (21, 128, 61)
FAIL_COLOR = (185, 28, 28)
CHUTE_COLOR = (198, 192, 179)

# 로봇 핸드 메카니컬 팔레트
METAL_DARK = (48, 52, 58)
METAL_MID = (78, 84, 92)
METAL_LIGHT = (132, 138, 148)
METAL_EDGE = (22, 24, 27)
BRASS = (196, 146, 88)
PAD_RUBBER = (36, 38, 42)

_KR_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/malgunbd.ttf"),
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
)
_MONO_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/consola.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
)


@lru_cache(maxsize=16)
def _load_font(size: int, mono: bool = False) -> ImageFont.ImageFont:
    candidates = _MONO_FONT_CANDIDATES if mono else _KR_FONT_CANDIDATES
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_texts(
    img: np.ndarray,
    items: list[tuple[tuple[int, int], str, int, tuple[int, int, int], bool]],
) -> np.ndarray:
    """Pillow로 한글/모노스페이스 텍스트를 RGB 이미지 위에 선명하게 합성한다."""
    if not items:
        return img
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    for pos, text, size, color, mono in items:
        font = _load_font(size, mono=mono)
        draw.text(pos, text, font=font, fill=color)
    return np.asarray(pil_img, dtype=np.uint8)


def blank(width: int = FRAME_W, height: int = FRAME_H, step: int = 32) -> np.ndarray:
    img = np.empty((height, width, 3), np.uint8)
    img[:] = BG
    for x in range(0, width, step):
        cv2.line(img, (x, 0), (x, height), GRID, 1)
    for y in range(0, height, step):
        cv2.line(img, (0, y), (width, y), GRID, 1)
    return img


def _draw_hud_header(
    img: np.ndarray,
    texts: list,
    left_label: str,
    right_label: str,
    right_color: tuple[int, int, int] = MUTED,
) -> None:
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 34), (28, 28, 28), -1)
    texts.append(((14, 9), left_label, 13, (243, 241, 236), True))
    texts.append(((w - 196, 9), right_label, 13, right_color, True))


def _draw_capsule(
    img: np.ndarray,
    cx: int,
    cy: int,
    color: tuple[int, int, int],
    *,
    radius: int = 24,
    empty: bool = False,
) -> None:
    if empty:
        cv2.circle(img, (cx, cy), max(8, radius - 2), MUTED, 1, cv2.LINE_AA)
        d = max(6, int(radius * 0.6))
        cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), MUTED, 1, cv2.LINE_AA)
        return
    cv2.circle(img, (cx + max(2, radius // 8), cy + max(2, radius // 6)), radius, (176, 170, 158), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius, METAL_EDGE, 1, cv2.LINE_AA)
    highlight = tuple(min(255, c + 50) for c in color)
    r_hi = max(4, int(radius * 0.45))
    off = max(2, radius // 5)
    cv2.circle(img, (cx - off, cy - off), r_hi, highlight, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(3, radius // 3), (245, 243, 238), 1, cv2.LINE_AA)


def paint_part(
    img: np.ndarray,
    part: Part,
    selected: bool,
    *,
    stock: int | None = None,
    failed: bool = False,
) -> list[tuple[tuple[int, int], str, int, tuple[int, int, int], bool]]:
    x, y, x2, y2 = part.box
    bg_fill = FILL_FAIL if failed else (FILL_SELECTED if selected else FILL)
    cv2.rectangle(img, (x, y), (x2, y2), bg_fill, -1)

    cx = (x + x2) // 2
    cy = (y + y2) // 2 + 4
    empty = stock is not None and stock <= 0
    _draw_capsule(img, cx, cy, part.color_rgb, empty=empty)

    # 테스트 계약: 선택 시 part.box 경계(예: (260, 150))는 정확히 ACCENT(154, 52, 18)여야 함
    if selected:
        cv2.rectangle(img, (x, y), (x2, y2), ACCENT, 2)
    elif failed:
        cv2.rectangle(img, (x, y), (x2, y2), FAIL_COLOR, 2)
    else:
        cv2.rectangle(img, (x, y), (x2, y2), INK, 1)

    texts: list[tuple[tuple[int, int], str, int, tuple[int, int, int], bool]] = [
        ((x, y - 24), f"#{part.id} {part.name}", 14, ACCENT if selected else INK, False),
    ]
    if stock is not None:
        stock_txt = "품절" if stock <= 0 else f"x{stock}"
        stock_col = FAIL_COLOR if stock <= 0 else MUTED
        texts.append(((x2 - 28, y - 22), stock_txt, 12, stock_col, False))
    if part.code:
        texts.append(((x + 8, y2 + 8), part.code, 12, MUTED, True))
    if selected:
        texts.append(((x + 8, y2 + 24), "TARGET LOCK", 11, ACCENT, True))
    return texts


def _rot_pt(lx: float, ly: float, cx: float, cy: float, angle_deg: float) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return cx + lx * c - ly * s, cy + lx * s + ly * c


def _draw_tapered_link(
    img: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    w1: float,
    w2: float,
    body_color: tuple[int, int, int],
    highlight_color: tuple[int, int, int],
) -> None:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = max(1e-5, math.hypot(dx, dy))
    nx, ny = -dy / length, dx / length
    pts = np.array(
        [
            [p1[0] + nx * (w1 * 0.5), p1[1] + ny * (w1 * 0.5)],
            [p2[0] + nx * (w2 * 0.5), p2[1] + ny * (w2 * 0.5)],
            [p2[0] - nx * (w2 * 0.5), p2[1] - ny * (w2 * 0.5)],
            [p1[0] - nx * (w1 * 0.5), p1[1] - ny * (w1 * 0.5)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [pts], body_color, cv2.LINE_AA)
    # 금속 하이라이트 스트립
    hi_pts = np.array(
        [
            [p1[0] + nx * (w1 * 0.22), p1[1] + ny * (w1 * 0.22)],
            [p2[0] + nx * (w2 * 0.22), p2[1] + ny * (w2 * 0.22)],
            [p2[0] - nx * (w2 * 0.04), p2[1] - ny * (w2 * 0.04)],
            [p1[0] - nx * (w1 * 0.04), p1[1] - ny * (w1 * 0.04)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [hi_pts], highlight_color, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, METAL_EDGE, 2, cv2.LINE_AA)


def _draw_joint(img: np.ndarray, pt: tuple[float, float], radius: int, ring_col: tuple[int, int, int] = BRASS) -> None:
    ix, iy = int(round(pt[0])), int(round(pt[1]))
    cv2.circle(img, (ix, iy), radius, METAL_DARK, -1, cv2.LINE_AA)
    cv2.circle(img, (ix, iy), radius, METAL_EDGE, 2, cv2.LINE_AA)
    cv2.circle(img, (ix, iy), max(3, int(radius * 0.52)), ring_col, -1, cv2.LINE_AA)
    cv2.circle(img, (ix, iy), max(2, int(radius * 0.25)), METAL_EDGE, -1, cv2.LINE_AA)


def _draw_finger(
    img: np.ndarray,
    base_pt: tuple[float, float],
    base_angle_deg: float,
    lengths: tuple[float, float, float],
    widths: tuple[float, float, float, float],
    curls_deg: tuple[float, float, float],
    *,
    joint_ring: tuple[int, int, int] = BRASS,
    pad_accent: tuple[int, int, int] = ACCENT,
) -> tuple[float, float]:
    """3마디 기계식 로봇 손가락을 전방향 기구학으로 그린다 (0도 = 위쪽 -Y)."""
    p0 = base_pt
    pts = [p0]
    ang = base_angle_deg
    for seg_len, d_ang in zip(lengths, curls_deg):
        ang += d_ang
        rad = math.radians(ang)
        # 0도 -> (0, -1) 위쪽 방향, +각도 -> 시계 방향(오른쪽/안쪽)
        nx = pts[-1][0] + seg_len * math.sin(rad)
        ny = pts[-1][1] - seg_len * math.cos(rad)
        pts.append((nx, ny))

    for i in range(3):
        _draw_tapered_link(
            img,
            pts[i],
            pts[i + 1],
            widths[i],
            widths[i + 1],
            METAL_MID if i < 2 else METAL_LIGHT,
            METAL_LIGHT if i < 2 else (172, 178, 186),
        )
        _draw_joint(img, pts[i], max(6, int(widths[i] * 0.44)), ring_col=joint_ring)

    # 끝단 고무 그립 패드 + 팁 캡
    tx, ty = int(round(pts[-1][0])), int(round(pts[-1][1]))
    tip_r = max(6, int(widths[-1] * 0.55))
    cv2.circle(img, (tx, ty), tip_r, PAD_RUBBER, -1, cv2.LINE_AA)
    cv2.circle(img, (tx, ty), tip_r, METAL_EDGE, 2, cv2.LINE_AA)
    cv2.circle(img, (tx, ty), max(2, tip_r // 3), pad_accent, -1, cv2.LINE_AA)
    return pts[-1]


def _paint_robotic_hand(
    img: np.ndarray,
    *,
    held: int | None,
    status: str,
) -> None:
    """2배 해상도(1280x960) 캔버스 위에 5지 관절 Hiwonder AiHand를 정밀 렌더링한다."""
    failed = status == "FAIL"
    gripping = held is not None and status in {"GRIP", "CARRY"}
    delivered = status == "DELIVERED"
    approach = status == "APPROACH"

    if failed:
        palm_angle = -10.0
        led_col = FAIL_COLOR
        joint_ring = FAIL_COLOR
        pad_col = FAIL_COLOR
    elif delivered:
        palm_angle = 26.0
        led_col = PASS_COLOR
        joint_ring = BRASS
        pad_col = PASS_COLOR
    elif gripping:
        palm_angle = 8.0
        led_col = ACCENT
        joint_ring = BRASS
        pad_col = ACCENT
    elif approach:
        palm_angle = 6.0
        led_col = ACCENT
        joint_ring = BRASS
        pad_col = ACCENT
    else:
        palm_angle = -4.0
        led_col = (120, 175, 130)
        joint_ring = BRASS
        pad_col = BRASS

    # 1. 팔뚝 베이스 & 팬틸트 서보 터렛 (고정 하우징)
    wrist_cx, wrist_cy = 396.0, 646.0
    # 하단 베이스 플레이트
    cv2.rectangle(img, (268, 804), (524, 846), METAL_DARK, -1)
    cv2.rectangle(img, (268, 804), (524, 846), METAL_EDGE, 3)
    for bx in (290, 502):
        cv2.circle(img, (bx, 825), 7, METAL_LIGHT, -1, cv2.LINE_AA)
        cv2.circle(img, (bx, 825), 7, METAL_EDGE, 2, cv2.LINE_AA)

    # 팔뚝 서보 기둥 & 케이블 배관
    forearm_poly = np.array([[326, 804], [344, 672], [448, 672], [466, 804]], dtype=np.int32)
    cv2.fillPoly(img, [forearm_poly], METAL_MID, cv2.LINE_AA)
    cv2.polylines(img, [forearm_poly], True, METAL_EDGE, 3, cv2.LINE_AA)
    for ry in (708, 734, 760):
        cv2.line(img, (356, ry), (436, ry), METAL_DARK, 5, cv2.LINE_AA)
    # 측면 전선 하네스 루프
    cable_pts = np.array([[334, 760], [284, 734], [282, 680], [336, 650]], dtype=np.int32)
    cv2.polylines(img, [cable_pts], False, METAL_EDGE, 9, cv2.LINE_AA)
    cv2.polylines(img, [cable_pts], False, ACCENT if gripping else (92, 96, 102), 4, cv2.LINE_AA)

    # 손목 팬/틸트 짐벌 서보
    cv2.circle(img, (int(wrist_cx), int(wrist_cy)), 42, METAL_DARK, -1, cv2.LINE_AA)
    cv2.circle(img, (int(wrist_cx), int(wrist_cy)), 42, METAL_EDGE, 3, cv2.LINE_AA)
    cv2.circle(img, (int(wrist_cx), int(wrist_cy)), 26, joint_ring, -1, cv2.LINE_AA)
    cv2.circle(img, (int(wrist_cx), int(wrist_cy)), 14, METAL_DARK, -1, cv2.LINE_AA)

    # 손바닥 중심 (손목에서 약간 위)
    palm_cx, palm_cy = _rot_pt(0.0, -148.0, wrist_cx, wrist_cy, palm_angle)

    # 손목-손바닥 연결 브래킷
    _draw_tapered_link(img, (wrist_cx, wrist_cy - 10), (palm_cx, palm_cy + 70), 66, 92, METAL_DARK, METAL_MID)

    # 2. 파지 중인 캡슐 좌표 (손가락 안쪽 · 손바닥 상단에 위치)
    cap_pt = _rot_pt(16.0, -178.0, palm_cx, palm_cy, palm_angle)

    # 3. 5지 기구학 파라미터 (손바닥 로컬 좌표, -Y가 위)
    if gripping:
        # 캡슐을 5지로 감싸 쥐는 포즈 (검지·엄지는 양옆, 소지·약지·중지는 상단에서 감쌈)
        finger_specs = (
            ((-78, -76), -12, (54, 42, 30), (22, 18, 15, 12), (32, 52, 46)),   # 소지
            ((-32, -96), -4,  (68, 50, 36), (25, 20, 17, 13), (24, 54, 48)),   # 약지
            ((16, -104),  4,  (74, 54, 38), (26, 21, 17, 13), (12, 54, 48)),   # 중지
            ((64, -92),  22,  (68, 50, 36), (25, 20, 17, 13), (-24, -46, -42)), # 검지 (오른쪽에서 감쌈)
        )
        thumb_spec = ((88, 22), 54, (58, 44, 34), (27, 22, 18, 14), (-38, -46, -40))
    elif approach:
        finger_specs = (
            ((-78, -76), -22, (56, 44, 32), (22, 18, 15, 12), (6, 26, 28)),
            ((-32, -96), -9,  (70, 52, 38), (25, 20, 17, 13), (5, 24, 26)),
            ((16, -104),  3,  (76, 56, 40), (26, 21, 17, 13), (4, 24, 26)),
            ((64, -92),  16,  (70, 52, 38), (25, 20, 17, 13), (2, 24, 26)),
        )
        thumb_spec = ((88, 22), 68, (58, 44, 34), (27, 22, 18, 14), (-16, -24, -22))
    elif failed:
        finger_specs = (
            ((-78, -76), -20, (56, 44, 32), (22, 18, 15, 12), (10, 44, 38)),
            ((-32, -96), -8,  (70, 52, 38), (25, 20, 17, 13), (8, 42, 38)),
            ((16, -104),  4,  (76, 56, 40), (26, 21, 17, 13), (8, 42, 38)),
            ((64, -92),  15,  (70, 52, 38), (25, 20, 17, 13), (6, 40, 36)),
        )
        thumb_spec = ((88, 22), 60, (58, 44, 34), (27, 22, 18, 14), (-18, -32, -26))
    else:
        # IDLE / DELIVERED: 자연스럽게 펴진 5지 로봇 핸드
        spread = 4 if delivered else 0
        finger_specs = (
            ((-78, -76), -26 - spread, (56, 44, 32), (22, 18, 15, 12), (4, 14, 14)),
            ((-32, -96), -10 - spread // 2, (70, 52, 38), (25, 20, 17, 13), (3, 12, 12)),
            ((16, -104),  2,           (76, 56, 40), (26, 21, 17, 13), (2, 12, 12)),
            ((64, -92),  18 + spread,  (70, 52, 38), (25, 20, 17, 13), (2, 14, 14)),
        )
        thumb_spec = ((88, 22), 78, (58, 44, 34), (27, 22, 18, 14), (-12, -16, -14))

    # 4. 기계식 손바닥 섀시 (Palm Chassis)
    palm_local = (
        (-86, -86),
        (-36, -108),
        (20, -114),
        (78, -98),
        (104, -28),
        (96, 58),
        (48, 96),
        (-48, 96),
        (-92, 46),
    )
    palm_pts = np.array([_rot_pt(px, py, palm_cx, palm_cy, palm_angle) for px, py in palm_local], dtype=np.int32)
    cv2.fillPoly(img, [palm_pts], METAL_DARK, cv2.LINE_AA)
    cv2.polylines(img, [palm_pts], True, METAL_EDGE, 4, cv2.LINE_AA)

    # 내부 액추에이터 윈도우 & 서보 텐던 로드 4줄
    win_local = ((-64, -58), (64, -66), (70, 48), (-60, 48))
    win_pts = np.array([_rot_pt(px, py, palm_cx, palm_cy, palm_angle) for px, py in win_local], dtype=np.int32)
    cv2.fillPoly(img, [win_pts], (34, 37, 42), cv2.LINE_AA)
    cv2.polylines(img, [win_pts], True, METAL_LIGHT, 2, cv2.LINE_AA)

    for rx in (-40.0, -14.0, 12.0, 38.0):
        rp1 = _rot_pt(rx, -52.0, palm_cx, palm_cy, palm_angle)
        rp2 = _rot_pt(rx * 0.75, 42.0, palm_cx, palm_cy, palm_angle)
        cv2.line(
            img,
            (int(rp1[0]), int(rp1[1])),
            (int(rp2[0]), int(rp2[1])),
            METAL_LIGHT,
            4,
            cv2.LINE_AA,
        )

    # 손바닥 4코너 볼트 & 상태 LED
    for bx, by in ((-72, -68), (66, -74), (72, 66), (-68, 62)):
        bp = _rot_pt(bx, by, palm_cx, palm_cy, palm_angle)
        cv2.circle(img, (int(bp[0]), int(bp[1])), 6, METAL_LIGHT, -1, cv2.LINE_AA)
        cv2.circle(img, (int(bp[0]), int(bp[1])), 6, METAL_EDGE, 2, cv2.LINE_AA)

    led_pt = _rot_pt(0.0, 68.0, palm_cx, palm_cy, palm_angle)
    cv2.circle(img, (int(led_pt[0]), int(led_pt[1])), 12, led_col, -1, cv2.LINE_AA)
    cv2.circle(img, (int(led_pt[0]), int(led_pt[1])), 12, METAL_EDGE, 2, cv2.LINE_AA)

    # 5. 손바닥과 손가락 사이에 캡슐이 위치하도록 캡슐을 먼저 그린 뒤 5지를 앞쪽으로 감싸 렌더링
    if gripping and held is not None:
        part = by_id(held)
        _draw_capsule(img, int(round(cap_pt[0])), int(round(cap_pt[1])), part.color_rgb, radius=48)

    for local_base, rel_ang, lens, wids, curls in finger_specs:
        kb = _rot_pt(local_base[0], local_base[1], palm_cx, palm_cy, palm_angle)
        _draw_finger(
            img,
            kb,
            palm_angle + rel_ang,
            lens,
            wids,
            curls,
            joint_ring=joint_ring,
            pad_accent=pad_col,
        )

    # 엄지 (대립 관절)
    t_base, t_ang, t_lens, t_wids, t_curls = thumb_spec
    tb = _rot_pt(t_base[0], t_base[1], palm_cx, palm_cy, palm_angle)
    _draw_finger(
        img,
        tb,
        palm_angle + t_ang,
        t_lens,
        t_wids,
        t_curls,
        joint_ring=joint_ring,
        pad_accent=pad_col,
    )


def _paint_chute_2x(
    img: np.ndarray,
    *,
    held: int | None,
    delivered: bool,
) -> None:
    """2배 해상도(1280x960) 우측 경사 슈트 및 초음파 수령 트레이."""
    # 경사 슈트 지주 & 램프
    cv2.rectangle(img, (690, 668), (718, 846), METAL_MID, -1)
    cv2.rectangle(img, (690, 668), (718, 846), METAL_EDGE, 2)
    cv2.rectangle(img, (932, 742), (956, 846), METAL_MID, -1)
    cv2.rectangle(img, (932, 742), (956, 846), METAL_EDGE, 2)

    ramp_pts = np.array([[648, 636], [974, 738], [974, 772], [648, 670]], dtype=np.int32)
    cv2.fillPoly(img, [ramp_pts], CHUTE_COLOR, cv2.LINE_AA)
    cv2.polylines(img, [ramp_pts], True, METAL_EDGE, 3, cv2.LINE_AA)

    # 수령 트레이 박스
    bay_fill = FILL_SELECTED if delivered else FILL
    bay_border = PASS_COLOR if delivered else METAL_EDGE
    cv2.rectangle(img, (974, 694), (1186, 846), bay_fill, -1)
    cv2.rectangle(img, (974, 694), (1186, 846), bay_border, 4 if delivered else 3)
    # 초음파 센서 모듈 (수령 트레이 좌상단)
    cv2.rectangle(img, (996, 682), (1056, 700), METAL_DARK, -1)
    cv2.circle(img, (1012, 691), 5, PASS_COLOR if delivered else BRASS, -1, cv2.LINE_AA)
    cv2.circle(img, (1040, 691), 5, PASS_COLOR if delivered else BRASS, -1, cv2.LINE_AA)

    if delivered and held is not None:
        part = by_id(held)
        _draw_capsule(img, 1080, 770, part.color_rgb, radius=46)


def tray(
    selected: int | None,
    *,
    stock: dict[int, int] | None = None,
    fail_part: int | None = None,
    status_label: str = "",
) -> np.ndarray:
    img = blank()
    cv2.rectangle(img, (44, 100), (596, 320), TRAY_DECK, -1)
    cv2.rectangle(img, (44, 100), (596, 320), INK, 1)

    texts: list[tuple[tuple[int, int], str, int, tuple[int, int, int], bool]] = []
    hud_right = status_label or (f"TARGET #{selected}" if selected else "STANDBY")
    hud_col = ACCENT if selected else (180, 190, 185)
    _draw_hud_header(img, texts, "CAM-01 // TRAY VIEW", hud_right, hud_col)

    for part in PARTS:
        part_stock = stock.get(part.id) if stock is not None else None
        texts.extend(
            paint_part(
                img,
                part,
                part.id == selected,
                stock=part_stock,
                failed=(fail_part == part.id),
            )
        )

    cv2.line(img, (44, 434), (596, 434), GRID, 2)
    texts.append(((48, 444), "SLOT-1 [80..180]     SLOT-2 [260..360]     SLOT-3 [440..540]", 12, MUTED, True))
    return _draw_texts(img, texts)


def hand(
    held: int | None,
    *,
    status: str = "IDLE",
    detail: str = "",
) -> np.ndarray:
    # 2배 해상도(1280x960)로 5지 로봇 손 모델링 렌더링 후 640x480으로 다운샘플
    hi = blank(FRAME_W * 2, FRAME_H * 2, step=64)
    cv2.line(hi, (60, 846), (1220, 846), METAL_DARK, 4, cv2.LINE_AA)

    delivered = status == "DELIVERED"
    failed = status == "FAIL"
    gripping = held is not None and status in {"GRIP", "CARRY"}

    _paint_chute_2x(hi, held=held, delivered=delivered)
    _paint_robotic_hand(hi, held=held, status=status)

    img = cv2.resize(hi, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

    texts: list[tuple[tuple[int, int], str, int, tuple[int, int, int], bool]] = []
    if failed:
        hud_right, hud_col = "FAULT / STOP", (252, 165, 165)
        pose_txt = "PAN 090°  TILT 090°  GRIP FAULT"
    elif delivered:
        hud_right, hud_col = "CHUTE DELIVERED", (134, 239, 172)
        pose_txt = "PAN 040°  TILT 110°  GRIP 020° (RELEASE)"
    elif gripping:
        hud_right, hud_col = f"GRIPPING #{held}", (253, 186, 116)
        pose_txt = f"PAN {120 + held * 15:03d}°  TILT 130°  GRIP 100° (HOLD)"
    elif status == "APPROACH":
        hud_right, hud_col = "PRE-GRASP POSE", (253, 186, 116)
        pose_txt = "PAN 150°  TILT 130°  GRIP 020° (APPROACH)"
    else:
        hud_right, hud_col = "5-DOF AI-HAND", (180, 190, 185)
        pose_txt = "PAN 090°  TILT 090°  GRIP 020° (HOME)"

    _draw_hud_header(img, texts, "CAM-02 // HIWONDER AI-HAND", hud_right, hud_col)
    texts.append(((16, 42), pose_txt, 11, MUTED, True))
    texts.append(((336, 304), "경사 슈트 (CHUTE)", 11, MUTED, False))
    texts.append(((494, 428), "수령 트레이", 11, PASS_COLOR if delivered else MUTED, False))

    if gripping and held is not None:
        p = by_id(held)
        texts.append(((16, 60), f"파지 대상: #{p.id} {p.name} ({p.code})", 12, ACCENT, False))
    elif delivered and held is not None:
        p = by_id(held)
        texts.append(((490, 316), f"#{p.id} {p.name} 도착", 12, PASS_COLOR, False))

    footer = detail or ("5지 파지 유지 중" if gripping else ("슈트 투하 완료" if delivered else "대기 자세 (5-Finger Open)"))
    footer_col = FAIL_COLOR if failed else (PASS_COLOR if delivered else MUTED)
    texts.append(((16, 450), f"STATUS: {footer}", 12, footer_col, False))
    return _draw_texts(img, texts)
