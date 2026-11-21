"""가짜 VLM·AIHand 통합. 화면 없이 상태 기계만 호출한다."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dashboard.catalog import default_inventory
from dashboard.hand import MicrobitHand
from dashboard.logcsv import clear_log, summarize_log
from dashboard.pipeline import Pipeline
from dashboard.vlm import FakeVlm
from link.mock_microbit import MockMicrobitLink

ACCENT = (154, 52, 18)


class DashboardPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "events.csv"
        self.pipeline = Pipeline(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _csv(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def test_known_part_reaches_done(self) -> None:
        snaps = self.pipeline.run("2")
        last = snaps[-1]
        self.assertEqual(last.state, "DONE")
        self.assertEqual(last.match, "yes")
        self.assertEqual(last.pick, "yes")
        self.assertEqual(last.deliver, "yes")
        selected = next(s for s in snaps if s.state == "SELECT")
        self.assertEqual(selected.part, 2)
        text = self._csv()
        for name in ("LOOK", "SELECT", "VERIFY", "PICK", "DELIVER", "DONE"):
            self.assertIn(name, text)

    def test_unknown_text_does_not_pick(self) -> None:
        snaps = self.pipeline.run("볼트")
        self.assertEqual(snaps[-1].state, "FAIL")
        self.assertEqual(snaps[-1].detail, "인식 실패")
        self.assertEqual(self.pipeline.hand.pick_calls, 0)

    def test_blank_text_is_recognition_failure(self) -> None:
        snaps = self.pipeline.run("  ")
        self.assertEqual(snaps[-1].detail, "인식 실패")
        self.assertEqual(self.pipeline.hand.pick_calls, 0)

    def test_wrong_part_stops_before_deliver(self) -> None:
        snaps = self.pipeline.run("캡B", force_wrong=True)
        self.assertEqual(snaps[-1].state, "FAIL")
        self.assertEqual(snaps[-1].detail, "불일치")
        self.assertEqual(snaps[-1].match, "no")
        self.assertEqual(self.pipeline.hand.deliver_calls, 0)
        self.assertEqual(self.pipeline.hand.pick_calls, 0)

    def test_pick_failure_stops_before_deliver(self) -> None:
        snaps = self.pipeline.run("1", force_pick_fail=True)
        self.assertEqual(snaps[-1].state, "FAIL")
        self.assertEqual(snaps[-1].pick, "no")
        self.assertEqual(snaps[-1].detail, "집기 실패")
        self.assertEqual(self.pipeline.hand.deliver_calls, 0)

    def test_deliver_failure_keeps_pick(self) -> None:
        snaps = self.pipeline.run("캡C", force_deliver_fail=True)
        self.assertEqual(snaps[-1].state, "FAIL")
        self.assertEqual(snaps[-1].pick, "yes")
        self.assertEqual(snaps[-1].deliver, "no")
        self.assertEqual(snaps[-1].detail, "전달 실패")

    def test_select_frame_marks_only_requested_part(self) -> None:
        snaps = self.pipeline.run("2")
        frame = next(s.tray for s in snaps if s.state == "SELECT")
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(tuple(int(v) for v in frame[150, 260]), ACCENT)
        self.assertNotEqual(tuple(int(v) for v in frame[150, 80]), ACCENT)

    def test_idle_before_run_has_empty_verdicts(self) -> None:
        idle = self.pipeline.idle()
        self.assertEqual(idle.state, "IDLE")
        self.assertEqual(idle.match, "")
        self.assertEqual(idle.pick, "")
        self.assertEqual(idle.deliver, "")

    def test_vlm_supports_natural_phrases_and_rejects_ambiguous(self) -> None:
        vlm = FakeVlm()
        self.assertEqual(vlm.resolve("캡B 주세요"), 2)
        self.assertEqual(vlm.resolve("cap-c 가져와"), 3)
        self.assertEqual(vlm.resolve("첫번째 부품"), 1)
        self.assertIsNone(vlm.resolve("1번이랑 2번"))
        self.assertIsNone(vlm.resolve("11번 부품"))

    def test_inventory_decrements_and_blocks_out_of_stock(self) -> None:
        inv = default_inventory()
        inv[2] = 1
        snaps_ok = self.pipeline.run("2", inventory=inv)
        self.assertEqual(snaps_ok[-1].state, "DONE")
        self.assertEqual(snaps_ok[-1].stock_after[2], 0)

        snaps_empty = self.pipeline.run("2", inventory=snaps_ok[-1].stock_after)
        self.assertEqual(snaps_empty[-1].state, "FAIL")
        self.assertEqual(snaps_empty[-1].detail, "재고 없음")

    def test_microbit_hand_protocol_trace_and_errors(self) -> None:
        link = MockMicrobitLink()
        hand = MicrobitHand(link)
        snaps = self.pipeline.run("캡B", hand=hand)
        self.assertEqual(snaps[-1].state, "DONE")
        trace = "\n".join(snaps[-1].protocol_lines)
        for expected in ("TX SHOW,2", "TX PICK,2", "RX DONE,PICK", "TX DELIVER", "TX RELEASE", "TX HOME"):
            self.assertIn(expected, trace)

        err_hand = MicrobitHand(MockMicrobitLink(), force_pick_fail=True)
        err_snaps = self.pipeline.run("1", hand=err_hand)
        self.assertEqual(err_snaps[-1].state, "FAIL")
        self.assertEqual(err_snaps[-1].detail, "집기 실패")

    def test_log_summary_and_clear(self) -> None:
        self.pipeline.run("1")
        self.pipeline.run("볼트")
        stats = summarize_log(self.path)
        self.assertEqual(stats["total_runs"], 2)
        self.assertEqual(stats["done_runs"], 1)
        self.assertEqual(stats["fail_runs"], 1)
        self.assertEqual(stats["success_rate"], 50.0)
        clear_log(self.path)
        self.assertEqual(summarize_log(self.path)["total_runs"], 0)


if __name__ == "__main__":
    unittest.main()
