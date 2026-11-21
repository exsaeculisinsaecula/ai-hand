"""CSV 로깅 및 성능 지표 자동 계산 단위 테스트."""
from pathlib import Path
import tempfile
import unittest

from config import PipelineConfig
from controller.state_machine import DeliveryStateMachine
from link.mock_microbit import MockMicrobitLink
from logging_metrics.csv_logger import CsvEventLogger
from logging_metrics.metrics import compute_metrics_from_csv
from tests.test_state_machine import FakeClock


class MetricsTests(unittest.TestCase):
    def test_csv_metrics_calculation(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "events.csv"
            clock = FakeClock(10.0)
            link = MockMicrobitLink(clock=clock)
            logger = CsvEventLogger(log_path, clock=clock)
            sm = DeliveryStateMachine(link=link, config=PipelineConfig(), logger=logger, clock=clock)

            # 1건 정상 전달 완료
            sm.step("1", 0.95)
            clock.advance(1.1); sm.step("1", 0.95)     # -> CONFIRM (REQUEST_CONFIRMED @ 11.1)
            clock.advance(2.1); sm.step("NONE", 0.0)  # -> PICK
            clock.advance(1.1); sm.step("NONE", 0.0)  # -> DELIVER
            clock.advance(1.0); sm.step("NONE", 0.0)  # -> RELEASE
            clock.advance(0.6); sm.step("NONE", 0.0)  # -> HANDOVER (DELIVERY_RELEASED @ 15.9)
            link.trigger_hand_event()
            sm.step("NONE", 0.0)                      # HANDOVER_SUCCESS
            clock.advance(0.5); sm.step("NONE", 0.0)  # -> IDLE

            # 1건 취소 + 1건 오인식 기록
            sm.step("2", 0.95)
            clock.advance(1.1); sm.step("2", 0.95)
            clock.advance(0.3); sm.step("PALM", 0.95)
            sm.mark_false_trigger()

            metrics = compute_metrics_from_csv(log_path)
            self.assertEqual(metrics.total_requests, 2)
            self.assertEqual(metrics.successful_deliveries, 1)
            self.assertEqual(metrics.cancelled_requests, 1)
            self.assertEqual(metrics.false_triggers, 1)
            self.assertEqual(metrics.delivery_success_rate, 50.0)
            self.assertAlmostEqual(metrics.avg_delivery_seconds, 4.8, places=2)


if __name__ == "__main__":
    unittest.main()
