"""가짜 micro:bit (파이썬 모의 객체) — 단일 AiHand + 슈트 동작 및 예외 주입 지원."""
from collections import deque
from dataclasses import dataclass
import time
from typing import Callable

from .protocol import (
    ACK,
    CMD_DELIVER,
    CMD_HOME,
    CMD_PICK,
    CMD_RELEASE,
    CMD_RETURN,
    CMD_SHOW,
    CMD_STOP,
    ERR_EMPTY,
    ERR_GRIP,
    EVT_HAND,
)


@dataclass
class _ScheduledReply:
    due_at: float
    line: str


class MockMicrobitLink:
    """로봇 없이도 판단·UI 모듈을 동일 프로토콜로 테스트할 수 있는 모의 객체."""

    is_mock: bool = True

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        auto_handover_delay: float | None = None,
    ):
        self._clock = clock
        self._pending: list[_ScheduledReply] = []
        self._rx_queue: deque[str] = deque()
        self.sent_commands: list[str] = []

        # 테스트/데모용 결함 주입 플래그
        self.auto_handover_delay = auto_handover_delay
        self.next_pick_error: str | None = None
        self.drop_next_response: bool = False

        self.delays = {
            CMD_SHOW: 0.05,
            CMD_PICK: 1.0,
            CMD_DELIVER: 0.9,
            CMD_RELEASE: 0.5,
            CMD_RETURN: 0.9,
            CMD_HOME: 0.4,
            CMD_STOP: 0.02,
        }

    def send(self, command: str) -> None:
        cmd = command.strip()
        self.sent_commands.append(cmd)
        now = self._clock()

        if cmd == CMD_STOP:
            # 비상 정지 시 진행 중이던 동작 응답은 취소하고 즉시 ACK만 보낸다.
            self._pending.clear()
            self._pending.append(_ScheduledReply(now + self.delays[CMD_STOP], ACK))
            return

        if self.drop_next_response:
            self.drop_next_response = False
            return

        op = cmd.split(",", 1)[0]
        if op == CMD_SHOW:
            self._pending.append(_ScheduledReply(now + self.delays[CMD_SHOW], ACK))
        elif op == CMD_PICK:
            if self.next_pick_error in {ERR_EMPTY, ERR_GRIP}:
                err = self.next_pick_error
                self.next_pick_error = None
                self._pending.append(_ScheduledReply(now + 0.35, err))
            else:
                self._pending.append(_ScheduledReply(now + self.delays[CMD_PICK], "DONE,PICK"))
        elif op == CMD_DELIVER:
            self._pending.append(_ScheduledReply(now + self.delays[CMD_DELIVER], "DONE,DELIVER"))
        elif op == CMD_RELEASE:
            self._pending.append(_ScheduledReply(now + self.delays[CMD_RELEASE], "DONE,RELEASE"))
            if self.auto_handover_delay is not None:
                due = now + self.delays[CMD_RELEASE] + self.auto_handover_delay
                self._pending.append(_ScheduledReply(due, EVT_HAND))
        elif op == CMD_RETURN:
            self._pending.append(_ScheduledReply(now + self.delays[CMD_RETURN], "DONE,RETURN"))
        elif op == CMD_HOME:
            self._pending.append(_ScheduledReply(now + self.delays[CMD_HOME], "DONE,HOME"))

    def trigger_hand_event(self) -> None:
        """작업자 수령(초음파 손 근접) 이벤트를 즉시 발생시킨다."""
        self._rx_queue.append(EVT_HAND)

    def inject_error(self, err_code: str) -> None:
        """다음 PICK 명령에서 오류 응답(ERR,EMPTY 또는 ERR,GRIP)을 발생시키거나 즉시 전달한다."""
        self.next_pick_error = err_code

    def poll(self) -> list[str]:
        now = self._clock()
        remaining: list[_ScheduledReply] = []
        for item in self._pending:
            if item.due_at <= now:
                self._rx_queue.append(item.line)
            else:
                remaining.append(item)
        self._pending = remaining

        items = list(self._rx_queue)
        self._rx_queue.clear()
        return items

    def reset(self) -> None:
        """예약된 응답·수신 큐·결함 주입 상태를 초기 상태로 되돌린다."""
        self._pending.clear()
        self._rx_queue.clear()
        self.next_pick_error = None
        self.drop_next_response = False

    def close(self) -> None:
        self._pending.clear()
        self._rx_queue.clear()
