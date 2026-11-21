"""YOLO 클래스 매핑 · 검출 요약 · 상태 머신 부품 대조 테스트 (ultralytics 불필요)."""
import json
from pathlib import Path
import tempfile
import unittest

from config import PipelineConfig
from controller.state_machine import DeliveryStateMachine, State
from link.mock_microbit import MockMicrobitLink
from vision.classes import KIND_GESTURE, KIND_OTHER, KIND_PART, ClassMap
from vision.detector import Detection, describe_gesture, summarize


class FakeClock:
    def __init__(self, now: float = 100.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, s: float) -> None:
        self.now += s


class ClassMapTests(unittest.TestCase):
    def test_default_aliases(self):
        cm = ClassMap()
        self.assertEqual(cm.resolve("cap_a").value, 1)
        self.assertEqual(cm.resolve("Cap-B").value, 2)
        self.assertEqual(cm.resolve("part3").kind, KIND_PART)
        self.assertEqual(cm.resolve("one").value, "1")
        self.assertEqual(cm.resolve("palm").value, "PALM")
        self.assertEqual(cm.resolve("open_hand").value, "PALM")
        self.assertEqual(cm.resolve("person").kind, KIND_OTHER)

    def test_json_mapping_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps({
                "parts": {"red_capsule": 1, "blue_capsule": 2, "green_capsule": 3},
                "gestures": {"g_one": "1", "g_palm": "palm"},
                "ignore": ["hand"],
            }), encoding="utf-8")
            cm = ClassMap.load(path)
        self.assertEqual(cm.resolve("blue_capsule").value, 2)
        self.assertEqual(cm.resolve("g_palm").value, "PALM")
        # 명시 매핑이 있으면 기본 별칭은 쓰지 않는다
        self.assertEqual(cm.resolve("cap_a").kind, KIND_OTHER)
        self.assertEqual(cm.resolve("hand").kind, KIND_OTHER)

    def test_invalid_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps({"parts": {"x": 7}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                ClassMap.load(path)

    def test_example_file_parses(self):
        example = Path(__file__).resolve().parents[1] / "models" / "yolo_classes.example.json"
        cm = ClassMap.load(example)
        self.assertEqual(cm.resolve("cap_c").value, 3)


def _det(name, kind, value, conf):
    return Detection(name, kind, value, conf, (0, 0, 10, 10))


class SummaryTests(unittest.TestCase):
    def test_best_gesture_and_part_counts(self):
        dets = [
            _det("one", KIND_GESTURE, "1", 0.6),
            _det("two", KIND_GESTURE, "2", 0.9),
            _det("cap_a", KIND_PART, 1, 0.8),
            _det("cap_a", KIND_PART, 1, 0.7),
            _det("cap_c", KIND_PART, 3, 0.5),
        ]
        s = summarize(dets)
        self.assertEqual(s.gesture, "2")
        self.assertAlmostEqual(s.gesture_conf, 0.9)
        self.assertEqual(s.part_counts, {1: 2, 2: 0, 3: 1})
        ok, _ = describe_gesture(s, 0.7)
        self.assertTrue(ok)

    def test_low_conf_gesture_fails(self):
        s = summarize([_det("one", KIND_GESTURE, "1", 0.5)])
        ok, reason = describe_gesture(s, 0.7)
        self.assertFalse(ok)
        self.assertIn("신뢰도", reason)


class VerifierStateMachineTests(unittest.TestCase):
    def _run_to_pick_gate(self, verifier):
        clock = FakeClock()
        link = MockMicrobitLink(clock=clock)
        sm = DeliveryStateMachine(link=link, config=PipelineConfig(), clock=clock, part_verifier=verifier)
        sm.step("2", 0.95)
        clock.advance(1.05)
        sm.step("2", 0.95)
        self.assertEqual(sm.state, State.CONFIRM)
        clock.advance(2.1)
        sm.step("NONE", 0.0)
        return sm, link

    def test_verified_part_proceeds_to_pick(self):
        sm, link = self._run_to_pick_gate(lambda p: (True, f"ok {p}"))
        self.assertEqual(sm.state, State.PICK)
        self.assertIn("PICK,2", link.sent_commands)

    def test_unknown_vision_still_proceeds(self):
        sm, link = self._run_to_pick_gate(lambda p: (None, "no tray in view"))
        self.assertEqual(sm.state, State.PICK)

    def test_missing_part_aborts_before_pick(self):
        sm, link = self._run_to_pick_gate(lambda p: (False, "part missing"))
        self.assertEqual(sm.state, State.IDLE)
        self.assertNotIn("PICK,2", link.sent_commands)
        self.assertEqual(link.sent_commands[-1], "HOME")
        self.assertEqual(sm.tone, "warn")

    def test_verifier_exception_does_not_block(self):
        def boom(_):
            raise RuntimeError("camera gone")
        sm, _ = self._run_to_pick_gate(boom)
        self.assertEqual(sm.state, State.PICK)

    def test_proximity_pick_respects_verifier(self):
        clock = FakeClock()
        link = MockMicrobitLink(clock=clock)
        sm = DeliveryStateMachine(link=link, clock=clock, part_verifier=lambda p: (False, "missing"))
        self.assertFalse(sm.trigger_proximity_pick(3))
        self.assertEqual(sm.state, State.IDLE)
        self.assertNotIn("PICK,3", link.sent_commands)


if __name__ == "__main__":
    unittest.main()
