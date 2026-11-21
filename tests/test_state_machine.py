"""7개 상태 머신·타임아웃·예외 처리 단위 테스트."""
import unittest

from config import PipelineConfig
from controller.state_machine import DeliveryStateMachine, State
from link.mock_microbit import MockMicrobitLink
from link.protocol import ERR_EMPTY, ERR_GRIP


class FakeClock:
    def __init__(self, initial: float = 0.0):
        self.now = initial

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock(100.0)
        self.link = MockMicrobitLink(clock=self.clock)
        self.cfg = PipelineConfig()
        self.sm = DeliveryStateMachine(link=self.link, config=self.cfg, clock=self.clock)

    def _hold_to_confirm(self, digit: str = "2"):
        self.sm.step(digit, 0.95)
        self.assertEqual(self.sm.state, State.REQUEST)
        self.clock.advance(1.05)
        self.sm.step(digit, 0.95)
        self.assertEqual(self.sm.state, State.CONFIRM)
        self.assertEqual(self.link.sent_commands[-1], f"SHOW,{digit}")

    def test_happy_path_single_hand_chute_delivery(self):
        self._hold_to_confirm("2")
        # ACK 수신 + 2초 취소 창 경과 -> PICK,2
        self.clock.advance(0.1)
        self.sm.step("NONE", 0.0)
        self.clock.advance(2.0)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.PICK)
        self.assertEqual(self.link.sent_commands[-1], "PICK,2")

        # PICK 완료 -> DELIVER
        self.clock.advance(1.1)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.DELIVER)
        self.assertEqual(self.link.sent_commands[-1], "DELIVER")

        # DELIVER 완료 -> RELEASE (슈트 투하)
        self.clock.advance(1.0)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.DELIVER)
        self.assertEqual(self.link.sent_commands[-1], "RELEASE")

        # RELEASE 완료 -> HANDOVER
        self.clock.advance(0.6)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.HANDOVER)

        # 작업자 수령(EVT,HAND) -> HOME -> IDLE
        self.link.trigger_hand_event()
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.link.sent_commands[-1], "HOME")
        self.clock.advance(0.5)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.IDLE)

    def test_hold_resets_when_gesture_breaks(self):
        self.sm.step("1", 0.95)
        self.assertEqual(self.sm.state, State.REQUEST)
        self.clock.advance(0.4)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.IDLE)

    def test_palm_cancels_during_confirm(self):
        self._hold_to_confirm("1")
        self.clock.advance(0.5)
        self.sm.step("PALM", 0.95)
        self.assertEqual(self.sm.state, State.IDLE)
        self.assertEqual(self.link.sent_commands[-1], "HOME")

    def test_err_empty_returns_home_immediately(self):
        self.link.inject_error(ERR_EMPTY)
        self._hold_to_confirm("3")
        self.clock.advance(2.1)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.PICK)

        self.clock.advance(0.4)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.IDLE)
        self.assertEqual(self.sm.tone, "warn")
        self.assertEqual(self.link.sent_commands[-1], "HOME")

    def test_err_grip_retries_once_then_succeeds(self):
        self.link.inject_error(ERR_GRIP)
        self._hold_to_confirm("1")
        self.clock.advance(2.1)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.PICK)

        # 첫 실패 후 1회 재시도 (PICK,1 두 번 전송)
        self.clock.advance(0.4)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.PICK)
        self.assertEqual(self.link.sent_commands.count("PICK,1"), 2)

        # 재시도 성공 -> DELIVER
        self.clock.advance(1.1)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.DELIVER)

    def test_handover_timeout_triggers_return(self):
        self._hold_to_confirm("2")
        self.clock.advance(2.1); self.sm.step("NONE", 0.0)  # -> PICK
        self.clock.advance(1.1); self.sm.step("NONE", 0.0)  # -> DELIVER
        self.clock.advance(1.0); self.sm.step("NONE", 0.0)  # -> RELEASE
        self.clock.advance(0.6); self.sm.step("NONE", 0.0)  # -> HANDOVER
        self.assertEqual(self.sm.state, State.HANDOVER)

        # 10초 무응답 -> RETURN,2
        self.clock.advance(10.1)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.RETURN)
        self.assertEqual(self.link.sent_commands[-1], "RETURN,2")

        self.clock.advance(1.0); self.sm.step("NONE", 0.0)  # -> HOME
        self.clock.advance(0.5); self.sm.step("NONE", 0.0)  # -> IDLE
        self.assertEqual(self.sm.state, State.IDLE)

    def test_command_timeout_triggers_stop(self):
        self.link.drop_next_response = True
        self._hold_to_confirm("1")
        self.clock.advance(2.1)
        self.sm.step("NONE", 0.0)  # ACK 응답이 누락된 상태
        self.assertEqual(self.sm.state, State.IDLE)
        self.assertIn("STOP", self.link.sent_commands)

    def test_emergency_stop_from_active_state(self):
        self._hold_to_confirm("3")
        self.sm.emergency_stop()
        self.assertEqual(self.sm.state, State.IDLE)
        self.assertEqual(self.link.sent_commands[-1], "STOP")
        self.assertEqual(self.sm.tone, "warn")

    def test_proximity_pick_starts_delivery_pipeline_directly(self):
        self.assertTrue(self.sm.trigger_proximity_pick(2))
        self.assertEqual(self.sm.state, State.PICK)
        self.assertEqual(self.sm.target_part, 2)
        self.assertEqual(self.link.sent_commands[-2:], ["SHOW,2", "PICK,2"])

        self.clock.advance(1.1)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.DELIVER)
        self.assertEqual(self.link.sent_commands[-1], "DELIVER")

    def test_manual_trigger_return_from_handover(self):
        self.sm.trigger_proximity_pick(1)
        self.clock.advance(1.1); self.sm.step("NONE", 0.0)  # -> DELIVER
        self.clock.advance(1.0); self.sm.step("NONE", 0.0)  # -> RELEASE
        self.clock.advance(0.6); self.sm.step("NONE", 0.0)  # -> HANDOVER
        self.assertEqual(self.sm.state, State.HANDOVER)
        self.assertEqual(self.sm.snapshot()["handover_substep"], "WAIT")

        self.assertTrue(self.sm.trigger_return())
        self.assertEqual(self.sm.state, State.RETURN)
        self.assertEqual(self.link.sent_commands[-1], "RETURN,1")
        self.assertEqual(self.sm.snapshot()["return_substep"], "RETURN")

    def test_reset_all_clears_state_pending_replies_and_csv(self):
        from pathlib import Path
        import tempfile
        from logging_metrics.csv_logger import CsvEventLogger
        from logging_metrics.metrics import compute_metrics_from_csv

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "events.csv"
            logger = CsvEventLogger(log_path, clock=self.clock)
            sm = DeliveryStateMachine(link=self.link, config=self.cfg, logger=logger, clock=self.clock)

            sm.trigger_proximity_pick(3)
            sm.mark_false_trigger()
            self.assertEqual(sm.state, State.PICK)
            self.assertGreater(compute_metrics_from_csv(log_path).total_requests, 0)

            sm.reset_all()
            self.assertEqual(sm.state, State.IDLE)
            self.assertIsNone(sm.target_part)
            self.assertEqual(sm.tone, "pass")
            self.assertEqual(len(sm.logs), 1)

            # Scheduled DONE,PICK from before reset must not fire
            self.clock.advance(2.0)
            sm.step("NONE", 0.0)
            self.assertEqual(sm.state, State.IDLE)

            metrics = compute_metrics_from_csv(log_path)
            self.assertEqual(metrics.total_requests, 0)
            self.assertEqual(metrics.false_triggers, 0)


    def test_text_request_skips_hold_and_allows_cancel(self):
        self.assertTrue(self.sm.request_part(3, source="vlm", note="초록색 잠금 캡"))
        self.assertEqual(self.sm.state, State.CONFIRM)
        self.assertEqual(self.link.sent_commands[-1], "SHOW,3")
        # 취소 창 안에서 손바닥 → 취소
        self.clock.advance(0.5)
        self.sm.step("PALM", 0.95)
        self.assertEqual(self.sm.state, State.IDLE)

    def test_text_request_proceeds_to_pick(self):
        self.sm.request_part(1, source="vlm")
        self.clock.advance(2.1)
        self.sm.step("NONE", 0.0)
        self.assertEqual(self.sm.state, State.PICK)
        self.assertIn("PICK,1", self.link.sent_commands)

    def test_text_request_rejected_when_busy(self):
        self.sm.request_part(1)
        self.assertFalse(self.sm.request_part(2))
        self.assertEqual(self.sm.target_part, 1)


if __name__ == "__main__":
    unittest.main()
