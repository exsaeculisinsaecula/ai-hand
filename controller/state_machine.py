"""판단(controller) 모듈 — 7개 상태 머신, 홀드 타이머, 타임아웃, 예외 복구."""
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time
from typing import Callable

from config import PipelineConfig
from gesture.classifier import GESTURE_NONE, GESTURE_PALM
from link.protocol import (
    CMD_DELIVER,
    CMD_HOME,
    CMD_PICK,
    CMD_RELEASE,
    CMD_RETURN,
    CMD_SHOW,
    CMD_STOP,
    ERR_EMPTY,
    ERR_GRIP,
    RobotLink,
    format_command,
    parse_response,
)
from logging_metrics.csv_logger import CsvEventLogger


class State(str, Enum):
    IDLE = "IDLE"
    REQUEST = "REQUEST"
    CONFIRM = "CONFIRM"
    PICK = "PICK"
    DELIVER = "DELIVER"
    HANDOVER = "HANDOVER"
    RETURN = "RETURN"


@dataclass
class LogEntry:
    time: str
    direction: str  # "TX" | "RX" | "SYS" | "ERR"
    text: str


class DeliveryStateMachine:
    """단일 AiHand(1대) + 슈트 전달 파이프라인용 7-상태 제어기."""

    VALID_PARTS = {"1", "2", "3"}

    def __init__(
        self,
        link: RobotLink,
        config: PipelineConfig = PipelineConfig(),
        logger: CsvEventLogger | None = None,
        clock: Callable[[], float] = time.monotonic,
        part_verifier: Callable[[int], tuple[bool | None, str]] | None = None,
    ):
        self.link = link
        # PICK 직전 비전 대조 (True=확인, False=불일치→중단, None=판단 불가→진행)
        self.part_verifier = part_verifier
        self.cfg = config
        self.logger = logger
        self._clock = clock

        self.state: State = State.IDLE
        self.target_part: int | None = None
        self.state_entered_at: float = self._clock()
        self.phase_duration: float = 0.0
        self.tone: str = "pass"  # "pass" | "accent" | "warn"
        self.message: str = "대기 중 — 손짓(1·2·3)을 인식하면 요청을 시작합니다."

        self._awaiting_command: str | None = None
        self._cmd_sent_at: float = 0.0
        self._cmd_timeout: float = 0.0
        self._pick_retries: int = 0
        self._deliver_substep: str = ""  # "DELIVER" | "RELEASE"
        self._return_substep: str = ""   # "RETURN" | "HOME"
        self._handover_substep: str = "" # "WAIT" | "HOME"

        self.last_gesture: str = GESTURE_NONE
        self.last_confidence: float = 0.0
        self.logs: deque[LogEntry] = deque(maxlen=120)
        self._add_log("SYS", f"상태 머신 초기화 ({'모의 micro:bit' if link.is_mock else '실물 USB 시리얼'})")

    def _add_log(self, direction: str, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.logs.append(LogEntry(time=ts, direction=direction, text=text))

    def _csv(self, event_type: str, detail: str = "") -> None:
        if self.logger is not None:
            self.logger.log(event_type, self.state.value, self.target_part, detail)

    def _send(self, cmd: str, timeout: float | None = None) -> None:
        self.link.send(cmd)
        self._add_log("TX", cmd)
        self._csv("CMD_TX", cmd)
        if timeout is not None:
            self._awaiting_command = cmd.split(",", 1)[0]
            self._cmd_sent_at = self._clock()
            self._cmd_timeout = timeout

    def _enter(self, new_state: State, message: str, tone: str = "pass", duration: float = 0.0) -> None:
        self.state = new_state
        self.state_entered_at = self._clock()
        self.phase_duration = duration
        self.tone = tone
        self.message = message
        self._csv("STATE_ENTER", message)

    def _go_idle(self, message: str, tone: str = "pass", send_home: bool = False) -> None:
        if send_home:
            self._send(CMD_HOME)
        self._awaiting_command = None
        self._deliver_substep = ""
        self._return_substep = ""
        self._handover_substep = ""
        self._pick_retries = 0
        self._enter(State.IDLE, message, tone=tone, duration=0.0)
        self.target_part = None

    def emergency_stop(self, reason: str = "비상 정지(STOP) 트리거됨") -> None:
        """어느 상태에서나 즉시 STOP 명령을 전송하고 안전하게 대기 상태로 전환한다."""
        self._add_log("ERR", reason)
        self._csv("EMERGENCY_STOP", reason)
        self.link.send(CMD_STOP)
        self._add_log("TX", CMD_STOP)
        self._awaiting_command = None
        self._deliver_substep = ""
        self._return_substep = ""
        self._handover_substep = ""
        self._pick_retries = 0
        self._enter(State.IDLE, "비상 정지(STOP) — 현재 각도 유지. 확인 후 재개하세요.", tone="warn")
        self.target_part = None

    def mark_false_trigger(self) -> None:
        """현재 또는 직전 요청을 오인식으로 기록한다(성능 평가용)."""
        self._csv("FALSE_TRIGGER_MARKED", "manual_mark")
        self._add_log("SYS", "오인식 요청 기록됨 (지표 반영)")

    def _verify_part_before_pick(self, part: int) -> bool:
        """비전 대조기가 있으면 PICK 전에 트레이의 목표 부품을 확인한다. 실패 시 IDLE 복귀."""
        if self.part_verifier is None:
            return True
        try:
            ok, reason = self.part_verifier(part)
        except Exception as exc:  # 비전 오류가 로봇 제어를 막지 않도록
            ok, reason = None, f"비전 대조 오류: {exc}"
        if ok is False:
            self._add_log("ERR", reason)
            self._csv("DELIVERY_FAILED", f"vision_verify:part={part}")
            self._go_idle(f"{reason} — 트레이 확인 후 재시도하세요.", tone="warn", send_home=True)
            return False
        self._add_log("SYS", reason)
        if ok:
            self._csv("VISION_VERIFIED", f"part={part}")
        return True

    def request_part(self, part: int, source: str = "text", note: str = "") -> bool:
        """글(VLM) 등 명시적 요청: 홀드 없이 바로 CONFIRM(SHOW,N + 취소 창)으로 들어간다."""
        if self.state != State.IDLE or part not in (1, 2, 3):
            return False
        self.target_part = part
        self._pick_retries = 0
        detail = f"part={part};source={source}"
        if note:
            detail += f";text={note.replace(';', ',')[:80]}"
        self._csv("REQUEST_CONFIRMED", detail)
        self._add_log("SYS", f"[{source.upper()}] 요청 '{note}' → 부품 {part}" if note else f"[{source.upper()}] 부품 {part} 요청")
        self._send(format_command(CMD_SHOW, part), timeout=self.cfg.ack_timeout)
        self._enter(
            State.CONFIRM,
            f"글 요청 확정 ({part}번 부품) — {self.cfg.confirm_seconds:.0f}초 내 취소할 수 있어요",
            tone="accent",
            duration=self.cfg.confirm_seconds,
        )
        return True

    def trigger_proximity_pick(self, part: int) -> bool:
        """가상/비전 근접 감지 시 대기 상태에서 즉시 파지(PICK) 파이프라인을 시작한다."""
        if self.state != State.IDLE or part not in (1, 2, 3):
            return False
        self.target_part = part
        self._pick_retries = 0
        self._csv("REQUEST_CONFIRMED", f"part={part};source=proximity")
        if not self._verify_part_before_pick(part):
            return False
        self._add_log("SYS", f"비전 근접 감지(부품 {part}) — 자동 파지 시작")
        self._send(format_command(CMD_SHOW, part))
        self._send(format_command(CMD_PICK, part), timeout=self.cfg.pick_timeout)
        self._enter(
            State.PICK,
            f"근접 감지로 단일 AiHand가 부품 {part} 파지 중...",
            tone="pass",
            duration=self.cfg.pick_timeout,
        )
        return True

    def trigger_return(self) -> bool:
        """인계 대기(HANDOVER) 상태에서 즉시 반납(RETURN) 시나리오를 실행한다."""
        if self.state != State.HANDOVER or self._handover_substep != "WAIT" or self.target_part is None:
            return False
        self._add_log("SYS", "무응답 반납 테스트 트리거 — 부품 반납 시작")
        self._return_substep = "RETURN"
        self._send(format_command(CMD_RETURN, self.target_part), timeout=self.cfg.return_timeout)
        self._enter(
            State.RETURN,
            f"부품을 트레이 {self.target_part}에 되돌리는 중...",
            tone="accent",
            duration=self.cfg.return_timeout,
        )
        return True

    def reset_all(self) -> None:
        """진행 중인 시나리오, 오류 상태, 콘솔 로그 및 CSV 지표를 모두 초기 상태로 리셋한다."""
        reset_link = getattr(self.link, "reset", None)
        if callable(reset_link):
            reset_link()
        else:
            self.link.send(CMD_HOME)

        if self.logger is not None:
            self.logger.clear()

        self.state = State.IDLE
        self.target_part = None
        self.state_entered_at = self._clock()
        self.phase_duration = 0.0
        self.tone = "pass"
        self.message = "모든 시나리오 리셋 완료 — 대기(IDLE) 상태입니다."

        self._awaiting_command = None
        self._cmd_sent_at = 0.0
        self._cmd_timeout = 0.0
        self._pick_retries = 0
        self._deliver_substep = ""
        self._return_substep = ""
        self._handover_substep = ""
        self.last_gesture = GESTURE_NONE
        self.last_confidence = 0.0

        self.logs.clear()
        self._add_log("SYS", "모든 시나리오 · 로그 · KPI가 초기 상태로 리셋되었습니다.")

    def step(self, gesture: str = GESTURE_NONE, confidence: float = 0.0) -> None:
        """주기적 틱(인식 결과 + 시리얼 수신 + 타임아웃 검사)을 1회 처리한다."""
        now = self._clock()
        self.last_gesture = gesture
        self.last_confidence = confidence

        # 1. 로봇 응답/이벤트 수신 처리
        for raw in self.link.poll():
            self._handle_rx(raw)

        # 2. 명령 응답 타임아웃 검사
        if self._awaiting_command is not None and (now - self._cmd_sent_at) > self._cmd_timeout:
            timed_out = self._awaiting_command
            self._add_log("ERR", f"{timed_out} 응답 타임아웃 ({self._cmd_timeout:.1f}s)")
            self._csv("DELIVERY_FAILED", f"timeout:{timed_out}")
            self.emergency_stop(f"통신 타임아웃({timed_out}) — STOP 전송 후 수동 복구 필요")
            return

        # 3. 상태별 인식·타이머 전이
        if self.state == State.IDLE:
            if gesture in self.VALID_PARTS and confidence >= self.cfg.min_confidence:
                self.target_part = int(gesture)
                self._enter(
                    State.REQUEST,
                    f"손짓 {self.target_part} 감지 — {self.cfg.hold_seconds:.1f}초 홀드 유지 중...",
                    tone="accent",
                    duration=self.cfg.hold_seconds,
                )

        elif self.state == State.REQUEST:
            same_part = (
                gesture in self.VALID_PARTS
                and int(gesture) == self.target_part
                and confidence >= self.cfg.min_confidence
            )
            if not same_part:
                self._add_log("SYS", "홀드 중 손짓 변경/이탈 — 타이머 초기화")
                self._go_idle("손짓 이탈로 요청 초기화 — 대기 중")
                return

            if (now - self.state_entered_at) >= self.cfg.hold_seconds:
                self._csv("REQUEST_CONFIRMED", f"part={self.target_part}")
                self._send(format_command(CMD_SHOW, self.target_part), timeout=self.cfg.ack_timeout)
                self._enter(
                    State.CONFIRM,
                    f"요청 확정 ({self.target_part}번 부품) — {self.cfg.confirm_seconds:.0f}초 내 손바닥으로 취소 가능",
                    tone="accent",
                    duration=self.cfg.confirm_seconds,
                )

        elif self.state == State.CONFIRM:
            if gesture == GESTURE_PALM and confidence >= self.cfg.min_confidence:
                self._add_log("SYS", "손바닥 제스처 감지 — 요청 취소")
                self._csv("REQUEST_CANCELLED", "palm_cancel")
                self._go_idle("취소됨 — 대기(IDLE) 상태로 복귀", tone="pass", send_home=True)
                return

            if (now - self.state_entered_at) >= self.cfg.confirm_seconds:
                self._pick_retries = 0
                if not self._verify_part_before_pick(self.target_part):
                    return
                self._send(format_command(CMD_PICK, self.target_part), timeout=self.cfg.pick_timeout)
                self._enter(
                    State.PICK,
                    f"단일 AiHand가 트레이 {self.target_part}에서 부품 집는 중...",
                    tone="pass",
                    duration=self.cfg.pick_timeout,
                )

        elif self.state == State.HANDOVER and self._handover_substep == "WAIT":
            if (now - self.state_entered_at) >= self.cfg.handover_timeout:
                self._add_log("SYS", f"인계 무응답({self.cfg.handover_timeout:.0f}s) — 부품 반납 시작")
                self._return_substep = "RETURN"
                self._send(format_command(CMD_RETURN, self.target_part), timeout=self.cfg.return_timeout)
                self._enter(
                    State.RETURN,
                    f"무응답으로 부품을 트레이 {self.target_part}에 되돌리는 중...",
                    tone="accent",
                    duration=self.cfg.return_timeout,
                )

    def _handle_rx(self, raw: str) -> None:
        msg = parse_response(raw)
        direction = "ERR" if msg.kind == "ERR" else "RX"
        self._add_log(direction, msg.raw)
        self._csv("CMD_RX", msg.raw)

        if msg.kind == "ACK":
            if self._awaiting_command == CMD_SHOW:
                self._awaiting_command = None
            return

        if msg.kind == "ERR":
            self._handle_robot_error(msg.raw)
            return

        if msg.kind == "EVT" and msg.arg == "HAND":
            if self.state == State.HANDOVER and self._handover_substep == "WAIT":
                self._csv("HANDOVER_SUCCESS", f"part={self.target_part}")
                self._handover_substep = "HOME"
                self._send(CMD_HOME, timeout=self.cfg.ack_timeout + 2.0)
                self.message = "수령 확인(EVT,HAND) — 중립 자세(HOME)로 복귀 중..."
                self.tone = "pass"
            return

        if msg.kind == "DONE":
            self._handle_done(msg.arg)

    def _handle_robot_error(self, err_raw: str) -> None:
        if self.state == State.PICK and err_raw == ERR_GRIP and self._pick_retries < self.cfg.max_grip_retries:
            self._pick_retries += 1
            self._add_log("SYS", f"잡기 실패 — 1회 재시도 (PICK,{self.target_part})")
            self._send(format_command(CMD_PICK, self.target_part), timeout=self.cfg.pick_timeout)
            self.state_entered_at = self._clock()
            self.message = f"잡기 실패 — 트레이 {self.target_part} 재시도 중..."
            self.tone = "accent"
            return

        self._csv("DELIVERY_FAILED", err_raw)
        if err_raw == ERR_EMPTY:
            self._go_idle("오류(ERR,EMPTY): 트레이가 비어 있습니다. 보충 후 재시도하세요.", tone="warn", send_home=True)
        else:
            self._go_idle(f"오류({err_raw}) 알림 후 대기(IDLE) 복귀", tone="warn", send_home=True)

    def _handle_done(self, action: str) -> None:
        if self.state == State.PICK and action == "PICK":
            self._awaiting_command = None
            self._deliver_substep = "DELIVER"
            self._send(CMD_DELIVER, timeout=self.cfg.deliver_timeout)
            self._enter(
                State.DELIVER,
                "경사로(슈트) 상단으로 이동 중...",
                tone="pass",
                duration=self.cfg.deliver_timeout,
            )

        elif self.state == State.DELIVER and action == "DELIVER" and self._deliver_substep == "DELIVER":
            self._awaiting_command = None
            self._deliver_substep = "RELEASE"
            self._send(CMD_RELEASE, timeout=self.cfg.release_timeout)
            self.message = "손 펼침(RELEASE) — 경사로 슈트로 캡슐 투하 중..."

        elif self.state == State.DELIVER and action == "RELEASE" and self._deliver_substep == "RELEASE":
            self._awaiting_command = None
            self._deliver_substep = ""
            self._csv("DELIVERY_RELEASED", f"part={self.target_part}")
            self._handover_substep = "WAIT"
            self._enter(
                State.HANDOVER,
                "수령 트레이 도착 — 작업자 수령(EVT,HAND) 대기 중...",
                tone="accent",
                duration=self.cfg.handover_timeout,
            )

        elif self.state == State.HANDOVER and action == "HOME" and self._handover_substep == "HOME":
            self._go_idle("전달 완료 — 대기(IDLE) 상태로 복귀했습니다.", tone="pass")

        elif self.state == State.RETURN and action == "RETURN" and self._return_substep == "RETURN":
            self._awaiting_command = None
            self._return_substep = "HOME"
            self._send(CMD_HOME, timeout=self.cfg.ack_timeout + 2.0)
            self.message = "부품 반납 완료 — 중립 자세(HOME)로 복귀 중..."

        elif self.state == State.RETURN and action == "HOME" and self._return_substep == "HOME":
            self._csv("DELIVERY_FAILED", "handover_timeout_returned")
            self._go_idle("무응답 반납 완료 — 대기(IDLE) 복귀", tone="accent")

    def snapshot(self) -> dict:
        """대시보드 UI가 소비하는 현재 상태 스냅샷을 반환한다."""
        now = self._clock()
        elapsed = max(0.0, now - self.state_entered_at)
        progress = 0.0
        if self.phase_duration > 0:
            progress = min(100.0, (elapsed / self.phase_duration) * 100.0)

        return {
            "state": self.state.value,
            "target_part": self.target_part,
            "deliver_substep": self._deliver_substep,
            "handover_substep": self._handover_substep,
            "return_substep": self._return_substep,
            "pick_retries": self._pick_retries,
            "tone": self.tone,
            "message": self.message,
            "elapsed": round(elapsed, 2),
            "phase_duration": self.phase_duration,
            "progress": round(progress, 1),
            "last_gesture": self.last_gesture,
            "last_confidence": self.last_confidence,
            "is_mock": self.link.is_mock,
            "logs": [
                {"time": item.time, "dir": item.direction, "text": item.text}
                for item in self.logs
            ],
        }
