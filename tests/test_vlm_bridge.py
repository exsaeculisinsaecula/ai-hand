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

    def _chat(self, prompt, images=None, timeout=20.0):
        self.calls.append("image" if images else "text")
        if self.fail:
            raise TimeoutError("timed out")
        if images:
            name = {"bolt": "볼트", "nut": "너트"}.get(self.check_label, "모름")
            return '{"seen": "가짜", "name": "%s"}' % name
        return self.text_reply

    def submit_check(self, img):
        f = Future()
        f.set_result(self.check_part(img))
        return f


class InterpretTest(unittest.TestCase):
    def test_vlm_interprets_every_request(self):
        b = FakeBridge('{"part": "볼트"}')
        t, reply, info = b.interpret("볼트 줘")
        self.assertEqual((t, info["method"]), ("bolt", "VLM"))
        self.assertEqual(b.calls, ["text"])

    def test_vlm_failure_uses_rule_for_clear_words(self):
        b = FakeBridge(fail=True)
        self.assertEqual(b.interpret("너트 줘")[0], "nut")

    def test_partner_phrase_goes_to_vlm(self):
        b = FakeBridge('{"part": "너트", "reason": "볼트의 짝"}')
        t, reply, info = b.interpret("볼트에 끼우는 거")
        self.assertEqual((t, info["method"]), ("nut", "VLM"))
        self.assertEqual(b.calls, ["text"])

    def test_unknown_word_goes_to_vlm(self):
        b = FakeBridge('{"part": "볼트"}')
        self.assertEqual(b.interpret("길쭉한 거 줘")[0], "bolt")

    def test_vlm_says_unknown(self):
        b = FakeBridge('{"part": "모름"}')
        self.assertIsNone(b.interpret("렌치 줘")[0])

    def test_vlm_failure_partner_asks_again(self):
        b = FakeBridge(fail=True)
        t, reply, info = b.interpret("볼트에 끼우는 거")
        self.assertIsNone(t)                       # 규칙만으로는 반대 부품이 될 수 있어 다시 묻기
        self.assertIn("실패", info["method"])

    def test_vlm_failure_falls_back_to_rule(self):
        b = FakeBridge(fail=True)
        self.assertIsNone(b.interpret("길쭉한 거")[0])

    def test_washer_not_supported(self):
        b = FakeBridge('{"part": "모름"}')
        t, reply, _ = b.interpret("와셔 줘")
        self.assertIsNone(t)
        self.assertIn("와셔", reply)


class VerifyFlowTest(unittest.TestCase):
    def make(self, check_label, verify_below=0.70):
        clock, link = FakeClock(), FakeLink()
        task = HandoverTask(link, clock=clock, vlm=FakeBridge(check_label=check_label, verify_below=verify_below))
        task.frame_source = lambda: np.full((480, 640, 3), 255, np.uint8)
        task.request("볼트 줘", pre=("bolt", "볼트", {"method": "VLM"}))
        link.rx.append("DONE,OPEN"); task.step(None)
        self.assertEqual(task.state, "RECEIVE")
        return task, link, clock

    def place(self, task, clock, conf):
        res = {"label": "bolt", "confidence": conf, "box": (200, 150, 80, 80), "motion": 0.0}
        for _ in range(8):
            clock.t += 0.2
            task.step(dict(res))
            if task.state != "RECEIVE":
                break
        task.step(None)                            # 교차 확인 결과 처리

    def test_high_conf_skips_vlm(self):
        task, link, clock = self.make("nut")
        self.place(task, clock, 0.9)
        self.assertEqual(task.state, "GRIP")
        self.assertEqual(task.vlm.calls, [])

    def test_low_conf_vlm_agrees_grips(self):
        task, link, clock = self.make("bolt")
        self.place(task, clock, 0.55)
        self.assertEqual(task.state, "GRIP")
        self.assertEqual(task.vlm.calls, ["image"])

    def test_low_conf_vlm_disagrees_waits_for_worker(self):
        task, link, clock = self.make("nut")
        self.place(task, clock, 0.55)
        self.assertEqual((task.state, task.snapshot()["vlm_state"]), ("VERIFY", "hold"))
        self.assertNotIn("GRIP", link.sent)
        task.manual_confirm()
        self.assertEqual(task.state, "GRIP")
        self.assertEqual(task.vlm_blocked, 1)

    def test_disagree_then_cancel(self):
        task, link, clock = self.make("nut")
        self.place(task, clock, 0.55)
        task.cancel()
        self.assertEqual(task.state, "HOME")
        self.assertNotIn("GRIP", link.sent)

    def test_verify_off(self):
        task, link, clock = self.make("nut", verify_below=0)
        self.place(task, clock, 0.5)
        self.assertEqual(task.state, "GRIP")


class TrayTest(unittest.TestCase):
    def test_yolo_picks_requested_part_on_tray(self):
        clock, link = FakeClock(), FakeLink()
        task = HandoverTask(link, clock=clock)
        task.request("너트 줘")
        link.rx.append("DONE,OPEN"); task.step(None)
        items = [{"label": "nut", "conf": 0.8, "box": (900, 300, 100, 100)},
                 {"label": "bolt", "conf": 0.7, "box": (100, 300, 100, 100)},
                 {"label": "nut", "conf": 0.5, "box": (500, 300, 100, 100)}]
        task.update_tray(items)
        self.assertEqual(task.tray["pick"], 3)                 # 왼쪽부터 1 볼트, 2 너트(0.5), 3 너트(0.8)
        self.assertIn("쟁반 3번", task.message)
        self.assertTrue(any("[쟁반 YOLO]" in l["text"] for l in task.logs))

    def test_not_on_tray(self):
        clock, link = FakeClock(), FakeLink()
        task = HandoverTask(link, clock=clock)
        task.request("볼트 줘")
        link.rx.append("DONE,OPEN"); task.step(None)
        task.update_tray([{"label": "nut", "conf": 0.8, "box": (100, 100, 50, 50)}])
        self.assertIsNone(task.tray["pick"])
        self.assertIn("안 보여요", task.message)

    def test_idle_has_no_pick(self):
        task = HandoverTask(FakeLink(), clock=FakeClock())
        task.update_tray([{"label": "nut", "conf": 0.8, "box": (100, 100, 50, 50)}])
        self.assertIsNone(task.tray["pick"])


class PointTest(unittest.TestCase):
    def setUp(self):
        self.clock, self.link = FakeClock(), FakeLink()
        self.task = HandoverTask(self.link, clock=self.clock)
        self.task.set_pointing(135, 45, hold_s=1.5)
        self.task.request("볼트 줘")
        self.link.rx.append("DONE,OPEN"); self.task.step(None)

    def test_points_to_pick_then_receives(self):
        t, link, clock = self.task, self.link, self.clock
        self.assertEqual((t.state, t._point_phase), ("REQUEST", "wait_pick"))
        t.update_tray([{"label": "nut", "conf": 0.8, "box": (100, 0, 50, 50), "xn": 0.1},
                       {"label": "bolt", "conf": 0.7, "box": (900, 0, 50, 50), "xn": 1.0}])
        t.step(None)
        self.assertEqual(link.sent[-1], "PAN,45")                 # 오른쪽 끝 볼트 → 45°
        link.rx.append("DONE,PAN"); t.step(None)
        self.assertEqual(t._point_phase, "hold")
        clock.t += 1.6; t.step(None)
        self.assertEqual(link.sent[-1], "PAN,90")                 # 정면 복귀
        link.rx.append("DONE,PAN"); t.step(None)
        self.assertEqual(t.state, "RECEIVE")

    def test_middle_angle(self):
        t = self.task
        t.update_tray([{"label": "bolt", "conf": 0.7, "box": (0, 0, 10, 10), "xn": 0.5}])
        t.step(None)
        self.assertEqual(self.link.sent[-1], "PAN,90")

    def test_skip_when_not_found(self):
        t = self.task
        t.update_tray([{"label": "nut", "conf": 0.8, "box": (0, 0, 10, 10), "xn": 0.2}])
        self.clock.t += 2.5; t.step(None)
        self.assertEqual(t.state, "RECEIVE")
        self.assertNotIn("PAN", " ".join(self.link.sent))

    def test_cancel_during_point(self):
        t = self.task
        t.update_tray([{"label": "bolt", "conf": 0.7, "box": (0, 0, 10, 10), "xn": 0.0}])
        t.step(None)
        t.cancel()
        self.assertEqual(t.state, "HOME")
        self.assertIsNone(t._point_phase)


if __name__ == "__main__":
    unittest.main()
