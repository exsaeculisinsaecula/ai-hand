"""VLM 연결 테스트 — Ollama 없이 가짜 응답으로 흐름만 확인."""
from concurrent.futures import Future
import unittest

import numpy as np

from controller.handover import HandoverTask
from vision.vlm_bridge import VlmBridge
from tests.test_handover import FakeClock, FakeLink


class FakeBridge(VlmBridge):
    def __init__(self, text_reply="", check_label=None, fail=False, **kw):
        super().__init__(**kw)
        self.text_reply, self.check_label, self.fail, self.calls = text_reply, check_label, fail, []

    def _chat(self, prompt, images=None, timeout=20.0, schema=None):
        self.calls.append("image" if images else "text")
        if self.fail:
            raise TimeoutError("timed out")
        if images:                                 # 손 위 확인: 사진 속 공(check_label)이 프롬프트 대상과 같으면 true
            name = {"red": "빨간 공", "blue": "파란 공", "green": "초록 공"}.get(self.check_label, "없음")
            if self.check_label == "garbage":
                return "잘 모르겠어요"
            return '{"found": %s}' % ("true" if f"'{name}'" in prompt else "false")
        return self.text_reply

    def submit_check(self, img):
        f = Future()
        f.set_result(self.check_part(img))
        return f

    def submit_found(self, img, target):
        f = Future()
        f.set_result(self.check_found(img, target))
        return f


class InterpretTest(unittest.TestCase):
    def test_vlm_interprets_every_request(self):
        b = FakeBridge('{"part": "빨간 공"}')
        t, reply, info = b.interpret("빨간 공 줘")
        self.assertEqual((t, info["method"]), ("red", "VLM"))
        self.assertEqual(b.calls, ["text"])

    def test_vlm_failure_uses_rule_for_clear_words(self):
        b = FakeBridge(fail=True)
        self.assertEqual(b.interpret("파란 공 줘")[0], "blue")

    def test_unknown_word_goes_to_vlm(self):
        b = FakeBridge('{"part": "빨간 공"}')
        self.assertEqual(b.interpret("사과 색깔 거 줘")[0], "red")

    def test_vlm_says_unknown(self):
        b = FakeBridge('{"part": "모름"}')
        self.assertIsNone(b.interpret("렌치 줘")[0])

    def test_vlm_failure_falls_back_to_rule(self):
        b = FakeBridge(fail=True)
        self.assertIsNone(b.interpret("동그란 거")[0])

    def test_vlm_unknown_asks_for_color(self):
        b = FakeBridge('{"part": "모름"}')
        t, reply, _ = b.interpret("노란 공 줘")
        self.assertIsNone(t)
        self.assertIn("빨간 공 / 파란 공 / 초록 공", reply)


class OfflineTest(unittest.TestCase):
    def test_offline_uses_word_rule_without_calling(self):
        b = FakeBridge('{"part": "초록 공"}')
        b.online = False
        t, reply, info = b.interpret("빨간 공 줘")
        self.assertEqual(t, "red")
        self.assertEqual(b.calls, [])                      # Ollama 를 기다리지 않음
        self.assertIn("VLM 꺼짐", info["method"])

    def test_offline_skips_palm_check(self):
        b = FakeBridge()
        b.online = False
        self.assertFalse(b.should_check(0.3))

    def test_parse_found(self):
        from vision.vlm_bridge import parse_found
        self.assertTrue(parse_found('{"found": true}'))
        self.assertFalse(parse_found('{"found": false}'))
        self.assertTrue(parse_found('answer: TRUE'))
        self.assertIsNone(parse_found('모르겠음'))


class GuardTest(unittest.TestCase):
    """실제 Gemma 3 4B 시험에서 틀렸던 두 경우 — VLM 에 묻기 전에 규칙으로 막음"""

    def test_two_colors_refused_without_vlm(self):
        b = FakeBridge('{"part": "빨간 공"}')
        t, reply, _ = b.interpret("빨간 공이랑 파란 공 둘 다")
        self.assertIsNone(t)
        self.assertEqual(b.calls, [])

    def test_no_color_clue_asks_again_without_vlm(self):
        b = FakeBridge('{"part": "빨간 공"}')
        for text in ("공 줘", "공 하나 주세요", "아무거나 줘", "give me a ball"):
            self.assertIsNone(b.interpret(text)[0], text)
        self.assertEqual(b.calls, [])

    def test_analogy_still_goes_to_vlm(self):
        b = FakeBridge('{"part": "빨간 공", "reason": "사과는 빨간색"}')
        self.assertEqual(b.interpret("사과 색깔 공 하나 줘")[0], "red")
        self.assertEqual(b.calls, ["text"])


class VerifyFlowTest(unittest.TestCase):
    """손 위 공 확인 — 팀 vlm-go 방식 (사진 + 대상 → found true/false)"""

    def make(self, on_palm, verify_below=1.0):
        clock, link = FakeClock(), FakeLink()
        task = HandoverTask(link, clock=clock, vlm=FakeBridge(check_label=on_palm, verify_below=verify_below))
        task.frame_source = lambda: np.full((480, 640, 3), 255, np.uint8)
        task.request("빨간 공 줘", pre=("red", "빨간 공", {"method": "VLM"}))
        link.rx.append("DONE,OPEN"); task.step(None)
        self.assertEqual(task.state, "RECEIVE")
        return task, link, clock

    def place(self, task, clock, label="red", conf=0.8):
        res = {"label": label, "confidence": conf, "box": (200, 150, 80, 80), "motion": 0.0}
        for _ in range(8):
            clock.t += 0.2
            task.step(dict(res))
            if task.state != "RECEIVE":
                break
        task.step(None)                            # VLM 결과 처리

    def test_vlm_found_true_sends_grip(self):
        task, link, clock = self.make("red")
        self.place(task, clock)
        self.assertEqual(task.state, "GRIP")
        self.assertEqual(task.vlm.calls, ["image"])
        self.assertEqual(link.sent[-1], "GRIP")
        self.assertTrue(task.snapshot()["vlm_last"]["found"])

    def test_vlm_found_false_sends_ng_even_if_color_rule_agrees(self):
        task, link, clock = self.make("blue")               # 손 위엔 파란 공, 색 규칙은 빨강이라고 착각
        self.place(task, clock, label="red")
        self.assertEqual(task.state, "VERIFY")
        self.assertEqual(link.sent[-1], "NG")
        self.assertNotIn("GRIP", link.sent)
        self.assertEqual((task.reject_count, task.vlm_blocked), (1, 1))

    def test_vlm_no_answer_falls_back_to_color_rule(self):
        task, link, clock = self.make("garbage")
        self.place(task, clock, label="red")
        self.assertEqual(task.state, "GRIP")
        self.assertEqual(task.vlm.stats["check_fail"], 1)

    def test_cancel_while_vlm_checking(self):
        task, link, clock = self.make("red")
        task.vlm.submit_found = lambda img, target: Future()     # 응답이 아직 안 옴
        res = {"label": "red", "confidence": 0.8, "box": (200, 150, 80, 80), "motion": 0.0}
        for _ in range(8):
            clock.t += 0.2
            task.step(dict(res))
        self.assertEqual(task.snapshot()["vlm_state"], "checking")
        task.cancel()
        self.assertEqual(task.state, "HOME")
        self.assertNotIn("GRIP", link.sent)

    def test_verify_off_uses_color_rule(self):
        task, link, clock = self.make("blue", verify_below=0)
        self.place(task, clock, label="red")
        self.assertEqual(task.state, "GRIP")
        self.assertEqual(task.vlm.calls, [])

    def test_request_reports_method_and_robot_command(self):
        clock, link = FakeClock(), FakeLink()
        task = HandoverTask(link, clock=clock, vlm=FakeBridge('{"part": "초록 공", "reason": "풀 색"}'))
        out = task.request("잔디 색 공 하나")
        self.assertEqual((out["target"], out["robot"], out["info"]["method"]), ("green", "OPEN", "VLM"))
        self.assertEqual(link.sent[-1], "OPEN")


class TrayTest(unittest.TestCase):
    def test_yolo_picks_requested_part_on_tray(self):
        clock, link = FakeClock(), FakeLink()
        task = HandoverTask(link, clock=clock)
        task.request("파란 공 줘")
        link.rx.append("DONE,OPEN"); task.step(None)
        items = [{"label": "blue", "conf": 0.8, "box": (900, 300, 100, 100)},
                 {"label": "red", "conf": 0.7, "box": (100, 300, 100, 100)},
                 {"label": "blue", "conf": 0.5, "box": (500, 300, 100, 100)}]
        task.update_tray(items)
        self.assertEqual(task.tray["pick"], 3)                 # 왼쪽부터 1 빨간 공, 2 파란 공(0.5), 3 파란 공(0.8)
        self.assertIn("쟁반 3번", task.message)
        self.assertTrue(any("[쟁반]" in l["text"] for l in task.logs))

    def test_not_on_tray(self):
        clock, link = FakeClock(), FakeLink()
        task = HandoverTask(link, clock=clock)
        task.request("빨간 공 줘")
        link.rx.append("DONE,OPEN"); task.step(None)
        task.update_tray([{"label": "blue", "conf": 0.8, "box": (100, 100, 50, 50)}])
        self.assertIsNone(task.tray["pick"])
        self.assertIn("안 보입니다", task.message)

    def test_idle_has_no_pick(self):
        task = HandoverTask(FakeLink(), clock=FakeClock())
        task.update_tray([{"label": "blue", "conf": 0.8, "box": (100, 100, 50, 50)}])
        self.assertIsNone(task.tray["pick"])


class PointTest(unittest.TestCase):
    def setUp(self):
        self.clock, self.link = FakeClock(), FakeLink()
        self.task = HandoverTask(self.link, clock=self.clock)
        self.task.set_pointing(135, 45, hold_s=1.5)
        self.task.request("빨간 공 줘")
        self.link.rx.append("DONE,OPEN"); self.task.step(None)

    def test_points_to_pick_then_receives(self):
        t, link, clock = self.task, self.link, self.clock
        self.assertEqual((t.state, t._point_phase), ("REQUEST", "wait_pick"))
        t.update_tray([{"label": "blue", "conf": 0.8, "box": (100, 0, 50, 50), "xn": 0.1},
                       {"label": "red", "conf": 0.7, "box": (900, 0, 50, 50), "xn": 1.0}])
        t.step(None)
        self.assertEqual(link.sent[-1], "PAN,45")                 # 오른쪽 끝 빨간 공 → 45°
        link.rx.append("DONE,PAN"); t.step(None)
        self.assertEqual(t._point_phase, "hold")
        clock.t += 1.6; t.step(None)
        self.assertEqual(link.sent[-1], "PAN,90")                 # 정면 복귀
        link.rx.append("DONE,PAN"); t.step(None)
        self.assertEqual(t.state, "RECEIVE")

    def test_middle_angle(self):
        t = self.task
        t.update_tray([{"label": "red", "conf": 0.7, "box": (0, 0, 10, 10), "xn": 0.5}])
        t.step(None)
        self.assertEqual(self.link.sent[-1], "PAN,90")

    def test_skip_when_not_found(self):
        t = self.task
        t.update_tray([{"label": "blue", "conf": 0.8, "box": (0, 0, 10, 10), "xn": 0.2}])
        self.clock.t += 2.5; t.step(None)
        self.assertEqual(t.state, "RECEIVE")
        self.assertNotIn("PAN", " ".join(self.link.sent))

    def test_cancel_during_point(self):
        t = self.task
        t.update_tray([{"label": "red", "conf": 0.7, "box": (0, 0, 10, 10), "xn": 0.0}])
        t.step(None)
        t.cancel()
        self.assertEqual(t.state, "HOME")
        self.assertIsNone(t._point_phase)


if __name__ == "__main__":
    unittest.main()
