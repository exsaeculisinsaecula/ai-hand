"""집기·전달 추상화 — 가짜 핸드 및 micro:bit 프로토콜 어댑터."""
import time

from link.mock_microbit import MockMicrobitLink
from link.protocol import (
    CMD_DELIVER,
    CMD_HOME,
    CMD_PICK,
    CMD_RELEASE,
    CMD_SHOW,
    CMD_STOP,
    ERR_EMPTY,
    ERR_GRIP,
    RobotLink,
    format_command,
    parse_response,
)


class Hand:
    def observe(self, part_id: int) -> int:
        raise NotImplementedError

    def pick(self, part_id: int) -> bool:
        raise NotImplementedError

    def deliver(self) -> bool:
        raise NotImplementedError


class FakeHand(Hand):
    def __init__(
        self,
        force_wrong: bool = False,
        force_pick_fail: bool = False,
        force_deliver_fail: bool = False,
    ):
        self.force_wrong = force_wrong
        self.force_pick_fail = force_pick_fail
        self.force_deliver_fail = force_deliver_fail
        self.pick_calls = 0
        self.deliver_calls = 0
        self.held: int | None = None
        self.protocol_trace: list[str] = []
        self.last_error: str = ""

    def observe(self, part_id: int) -> int:
        if self.force_wrong:
            return 1 if part_id != 1 else 2
        return part_id

    def pick(self, part_id: int) -> bool:
        self.pick_calls += 1
        if self.force_pick_fail:
            self.held = None
            self.last_error = "ERR,GRIP"
            return False
        self.held = part_id
        return True

    def deliver(self) -> bool:
        self.deliver_calls += 1
        if self.force_deliver_fail:
            self.last_error = "TIMEOUT,DELIVER"
            return False
        return True


class MicrobitHand(Hand):
    """실물 SerialMicrobitLink 또는 MockMicrobitLink를 구동하는 프로토콜 기반 핸드."""

    def __init__(
        self,
        link: RobotLink,
        *,
        force_wrong: bool = False,
        force_pick_fail: bool = False,
        force_deliver_fail: bool = False,
        timeout_sec: float = 4.0,
    ):
        self.link = link
        self.force_wrong = force_wrong
        self.force_pick_fail = force_pick_fail
        self.force_deliver_fail = force_deliver_fail
        self.timeout_sec = timeout_sec

        self.pick_calls = 0
        self.deliver_calls = 0
        self.held: int | None = None
        self.protocol_trace: list[str] = []
        self.last_error: str = ""

        # 모의 micro:bit UI 응답성을 위해 지연을 짧게 정규화
        if isinstance(self.link, MockMicrobitLink):
            self.link.delays = {
                CMD_SHOW: 0.02,
                CMD_PICK: 0.06,
                CMD_DELIVER: 0.06,
                CMD_RELEASE: 0.04,
                CMD_HOME: 0.03,
                CMD_STOP: 0.01,
            }

    def _send(self, cmd: str) -> None:
        self.link.send(cmd)
        self.protocol_trace.append(f"TX {cmd}")

    def _wait_for(self, expected_kinds: set[str], expected_arg: str = "") -> tuple[bool, str]:
        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            for raw in self.link.poll():
                self.protocol_trace.append(f"RX {raw}")
                msg = parse_response(raw)
                if msg.kind == "ERR":
                    self.last_error = msg.raw
                    return False, msg.raw
                if msg.kind in expected_kinds:
                    if not expected_arg or msg.arg == expected_arg:
                        return True, msg.raw
            time.sleep(0.01)
        self.last_error = "TIMEOUT"
        return False, "TIMEOUT"

    def observe(self, part_id: int) -> int:
        self._send(format_command(CMD_SHOW, part_id))
        self._wait_for({"ACK"})
        if self.force_wrong:
            return 1 if part_id != 1 else 2
        return part_id

    def pick(self, part_id: int) -> bool:
        self.pick_calls += 1
        if self.force_pick_fail and isinstance(self.link, MockMicrobitLink):
            self.link.inject_error(ERR_GRIP)
        self._send(format_command(CMD_PICK, part_id))
        ok, raw = self._wait_for({"DONE"}, "PICK")
        if not ok or (self.force_pick_fail and not isinstance(self.link, MockMicrobitLink)):
            self.held = None
            if not self.last_error:
                self.last_error = raw or ERR_GRIP
            self._send(CMD_HOME)
            self._wait_for({"DONE"}, "HOME")
            return False
        self.held = part_id
        return True

    def deliver(self) -> bool:
        self.deliver_calls += 1
        if self.force_deliver_fail:
            if isinstance(self.link, MockMicrobitLink):
                self.link.drop_next_response = True
            self._send(CMD_DELIVER)
            self.protocol_trace.append("ERR TIMEOUT,DELIVER")
            self._send(CMD_STOP)
            self._wait_for({"ACK"})
            self.last_error = "DELIVER_FAIL"
            return False

        self._send(CMD_DELIVER)
        ok, _ = self._wait_for({"DONE"}, "DELIVER")
        if not ok:
            self._send(CMD_STOP)
            return False

        self._send(CMD_RELEASE)
        ok, _ = self._wait_for({"DONE"}, "RELEASE")
        if not ok:
            self._send(CMD_STOP)
            return False

        self._send(CMD_HOME)
        self._wait_for({"DONE"}, "HOME")
        return True
