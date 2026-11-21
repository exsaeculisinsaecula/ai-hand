"""PC ↔ micro:bit V2 줄 단위 USB 시리얼 통신 프로토콜 정의."""
from dataclasses import dataclass
from typing import Protocol

# PC -> micro:bit 명령
CMD_SHOW = "SHOW"
CMD_PICK = "PICK"
CMD_DELIVER = "DELIVER"
CMD_RELEASE = "RELEASE"
CMD_RETURN = "RETURN"
CMD_HOME = "HOME"
CMD_STOP = "STOP"

# micro:bit -> PC 응답/이벤트
ACK = "ACK"
EVT_HAND = "EVT,HAND"
ERR_EMPTY = "ERR,EMPTY"
ERR_GRIP = "ERR,GRIP"


@dataclass(frozen=True)
class ParsedMessage:
    kind: str  # "ACK" | "DONE" | "ERR" | "EVT" | "UNKNOWN"
    arg: str = ""
    raw: str = ""


def format_command(cmd: str, part: int | None = None) -> str:
    """명령과 선택적 부품 번호(1~3)를 프로토콜 문자열로 직렬화한다."""
    if part is None:
        return cmd
    return f"{cmd},{part}"


def parse_response(line: str) -> ParsedMessage:
    """수신한 한 줄 문자열을 구조화된 응답 객체로 파싱한다."""
    clean = line.strip()
    if not clean:
        return ParsedMessage(kind="UNKNOWN", raw=line)
    if clean == ACK:
        return ParsedMessage(kind="ACK", raw=clean)
    if "," in clean:
        prefix, arg = clean.split(",", 1)
        if prefix in {"DONE", "ERR", "EVT"}:
            return ParsedMessage(kind=prefix, arg=arg.strip(), raw=clean)
    return ParsedMessage(kind="UNKNOWN", raw=clean)


class RobotLink(Protocol):
    """실물 시리얼 링크와 모의 객체가 공통으로 구현하는 통신 인터페이스."""

    is_mock: bool

    def send(self, command: str) -> None: ...

    def poll(self) -> list[str]: ...

    def close(self) -> None: ...
