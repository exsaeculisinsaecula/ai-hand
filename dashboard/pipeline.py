"""텍스트·빠른 선택·손짓 입력으로 인식부터 전달까지 진행하는 상태 기계."""
from pathlib import Path

from dashboard.cameras import hand_frame, tray_frame
from dashboard.catalog import by_id
from dashboard.hand import FakeHand, Hand
from dashboard.logcsv import DEFAULT_LOG, CsvLog
from dashboard.types import Snapshot
from dashboard.vlm import FakeVlm


def idle_snapshot(inventory: dict[int, int] | None = None) -> Snapshot:
    stock = dict(inventory) if inventory is not None else None
    return Snapshot(
        state="IDLE",
        tray=tray_frame(None, stock=stock, status_label="STANDBY"),
        hand=hand_frame(None, status="IDLE", detail="명령 또는 손짓 입력 대기 중"),
        part=None,
        match="",
        pick="",
        deliver="",
        detail="",
        note="대기 상태 — 부품을 선택하거나 명령을 입력하세요.",
        hand_status="IDLE",
        stock_after=stock,
    )


class Pipeline:
    def __init__(self, log_path: Path = DEFAULT_LOG):
        self.log = CsvLog(log_path)
        self.hand: Hand | None = None

    def idle(self, inventory: dict[int, int] | None = None) -> Snapshot:
        return idle_snapshot(inventory)

    def run(
        self,
        text: str,
        *,
        force_wrong: bool = False,
        force_pick_fail: bool = False,
        force_deliver_fail: bool = False,
        inventory: dict[int, int] | None = None,
        hand: Hand | None = None,
        source: str = "text",
    ) -> list[Snapshot]:
        vlm = FakeVlm()
        active_hand: Hand = hand or FakeHand(force_wrong, force_pick_fail, force_deliver_fail)
        self.hand = active_hand
        snaps: list[Snapshot] = []
        stock = dict(inventory) if inventory is not None else None

        def trace() -> tuple[str, ...]:
            raw = getattr(active_hand, "protocol_trace", None)
            return tuple(raw) if raw else ()

        def push(
            state: str,
            part: int | None,
            match: str,
            pick: str,
            deliver: str,
            detail: str,
            selected: int | None,
            held: int | None,
            *,
            hand_status: str = "IDLE",
            note: str = "",
            fail_part: int | None = None,
            cur_stock: dict[int, int] | None = None,
        ) -> None:
            stock_snap = dict(cur_stock) if cur_stock is not None else (dict(stock) if stock is not None else None)
            tray_status = f"{state} · #{selected}" if selected else state
            # YOLO 대조를 거친 뒤에는 실제 카메라 주석 프레임을 쟁반 뷰로 사용
            vision_frame = getattr(active_hand, "vision_frame", None)
            snaps.append(
                Snapshot(
                    state=state,
                    tray=vision_frame if vision_frame is not None else tray_frame(
                        selected,
                        stock=stock_snap,
                        fail_part=fail_part,
                        status_label=tray_status,
                    ),
                    hand=hand_frame(held, status=hand_status, detail=note or detail),
                    part=part,
                    match=match,
                    pick=pick,
                    deliver=deliver,
                    detail=detail,
                    note=note,
                    hand_status=hand_status,
                    stock_after=stock_snap,
                    protocol_lines=trace(),
                )
            )
            self.log.write(state, part, state, match, pick, deliver, detail)

        src_label = {"text": "명령 해석", "quick": "빠른 선택", "gesture": "손짓 인식"}.get(source, "입력")
        push("LOOK", None, "", "", "", "", None, None, hand_status="IDLE", note=f"{src_label}: '{text.strip()}' 분석 중")

        part, reason = vlm.explain(text)
        if part is None:
            push("FAIL", None, "", "", "", "인식 실패", None, None, hand_status="FAIL", note=reason)
            return snaps

        part_info = by_id(part)
        push(
            "SELECT",
            part,
            "",
            "",
            "",
            "",
            part,
            None,
            hand_status="APPROACH",
            note=f"{reason} → #{part} {part_info.name} ({part_info.code})",
        )

        if stock is not None and stock.get(part, 0) <= 0:
            push(
                "FAIL",
                part,
                "",
                "",
                "",
                "재고 없음",
                part,
                None,
                hand_status="FAIL",
                note=f"#{part} {part_info.name} 슬롯 재고가 0개입니다. 보충이 필요합니다.",
                fail_part=part,
            )
            return snaps

        seen = active_hand.observe(part)
        match = "yes" if seen == part else "no"
        vision_note = getattr(active_hand, "vision_note", "")
        push(
            "VERIFY",
            part,
            match,
            "",
            "",
            "",
            part,
            None,
            hand_status="APPROACH" if match == "yes" else "FAIL",
            note=vision_note or (
                f"카메라 대조 일치 (#{part} {part_info.name})"
                if match == "yes"
                else f"대조 불일치: 요청 #{part} ≠ 감지 #{seen}"
            ),
            fail_part=None if match == "yes" else seen,
        )
        if match == "no":
            push(
                "FAIL",
                part,
                "no",
                "",
                "",
                "불일치",
                part,
                None,
                hand_status="FAIL",
                note=(
                    f"트레이에서 #{part} {part_info.name} 미검출 — 안전 중단"
                    if seen is None
                    else f"요청(#{part})과 다른 부품(#{seen}) 감지 — 안전 중단"
                ),
                fail_part=seen,
            )
            return snaps

        if not active_hand.pick(part):
            push(
                "PICK",
                part,
                "yes",
                "no",
                "",
                "",
                part,
                None,
                hand_status="FAIL",
                note=f"#{part} {part_info.name} 파지 시도 실패",
                fail_part=part,
            )
            push(
                "FAIL",
                part,
                "yes",
                "no",
                "",
                "집기 실패",
                part,
                None,
                hand_status="FAIL",
                note="그리퍼 파지 실패 — 홈 위치 복귀",
                fail_part=part,
            )
            return snaps

        push(
            "PICK",
            part,
            "yes",
            "yes",
            "",
            "",
            part,
            part,
            hand_status="GRIP",
            note=f"#{part} {part_info.name} 파지 성공 — 슈트 이동 준비",
        )
        if not active_hand.deliver():
            push(
                "DELIVER",
                part,
                "yes",
                "yes",
                "no",
                "",
                part,
                part,
                hand_status="FAIL",
                note="경사 슈트 이송 중 타임아웃/오류",
                fail_part=part,
            )
            push(
                "FAIL",
                part,
                "yes",
                "yes",
                "no",
                "전달 실패",
                part,
                part,
                hand_status="FAIL",
                note="전달 실패 — 비상 정지(STOP) 처리",
                fail_part=part,
            )
            return snaps

        updated_stock = dict(stock) if stock is not None else None
        if updated_stock is not None and part in updated_stock:
            updated_stock[part] = max(0, updated_stock[part] - 1)

        push(
            "DELIVER",
            part,
            "yes",
            "yes",
            "yes",
            "",
            part,
            part,
            hand_status="DELIVERED",
            note=f"경사 슈트 투하 완료 — #{part} {part_info.name} 수령 대기",
            cur_stock=updated_stock,
        )
        push(
            "DONE",
            part,
            "yes",
            "yes",
            "yes",
            "",
            part,
            part,
            hand_status="DELIVERED",
            note=f"#{part} {part_info.name} 전달 완료",
            cur_stock=updated_stock,
        )
        return snaps
