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
        if images:
            if self.check_label == "garbage":
                return "잘 모르겠어요"
            if schema and "ball" in schema.get("properties", {}):   # 손 위 공 인식: 사진 속 공(check_label)의 색
                return '{"ball": "%s"}' % {"red": "빨간색", "blue": "파란색", "green": "초록색"}.get(self.check_label, "없음")
            name = {"red": "빨간 공", "blue": "파란 공", "green": "초록 공"}.get(self.check_label, "없음")
            return '{"found": %s}' % ("true" if f"'{name}'" in prompt else "false")
        return self.text_reply

    def submit_check(self, img):
        f = Future()
        f.set_result(self.check_part(img))
        return f

    def ask_team(self, img, target):
        """팀원 vlm_go.ask_vlm 대역: 손 위 공(check_label)이 대상과 같으면 found=true"""
        self.calls.append("image")
        if self.fail or self.check_label == "garbage":
            return {"found": None, "raw": "응답 없음", "seconds": 1.0, "ok": False}
        return {"found": self.check_label == target, "raw": "{}", "seconds": 1.0, "ok": True}

    def submit_team(self, img, target):
        f = Future()
        f.set_result(self.ask_team(img, target))
        return f

    def submit_colors(self, img):
        f = Future()
        f.set_result(self.ask_team_color(img))
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


class CrossCheckTest(unittest.TestCase):
    """손 위 색은 팀원 ask_vlm 을 세 색에 한 번씩 물어 정한다. 색 규칙은 답이 없을 때만."""
    make = None

    def setUp(self):
        self.make = VerifyFlowTest.make.__get__(self)
        self.place = VerifyFlowTest.place.__get__(self)

    def test_matching_color_grips_after_three_asks(self):
        task, link, clock = self.make("red")
        self.place(task, clock, label="red")
        self.assertEqual((task.state, task.vlm.calls), ("GRIP", ["image", "image", "image"]))
        d = task.snapshot()["decision"]
        self.assertEqual((d["final"], d["color"]), ("쥐기", "빨간 공"))

    def test_other_color_is_ng_even_if_hue_says_requested(self):
        task, link, clock = self.make("blue")                  # VLM 은 파란 공. 화면 라벨이 빨강이어도 색은 VLM
        self.place(task, clock, label="red")
        self.assertEqual(task.vlm.calls, ["image", "image", "image"])
        self.assertEqual(link.sent[-1], "NG")
        d = task.snapshot()["decision"]
        self.assertEqual((d["vlm"], d["color"], d["final"]), ("없음", "파란 공", "NG"))
        self.assertEqual(task.snapshot()["vlm_last"]["saw"], "파란 공")

    def test_vlm_miss_is_not_overridden_by_hue(self):
        task, link, clock = self.make("red")
        real, n = task.vlm.ask_team, []

        def first_miss(img, target):                            # 빨간 공 질문만 놓침 → 세 색 모두 false → 색 규칙으로 쥐지 않음
            n.append(1)
            return {"found": False, "raw": "", "seconds": 1.0, "ok": True} if len(n) == 1 else real(img, target)

        task.vlm.ask_team = first_miss
        self.place(task, clock, label="red")
        self.assertEqual(task.state, "RECEIVE")
        self.assertNotIn("GRIP", link.sent)
        self.assertEqual(task.snapshot()["decision"]["final"], "대기")


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
        task.palm_crop = lambda: np.full((200, 200, 3), 255, np.uint8)
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
        task.step(None)                            # (규칙과 다르면) 두 번째 확인 결과 처리

    def test_vlm_found_true_sends_grip(self):
        task, link, clock = self.make("red")
        self.place(task, clock)
        self.assertEqual(task.state, "GRIP")
        self.assertEqual(task.vlm.calls, ["image", "image", "image"])
        self.assertEqual(link.sent[-1], "GRIP")
        self.assertEqual(task.snapshot()["vlm_last"]["saw"], "빨간 공")

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
        task.vlm.submit_colors = lambda img: Future()            # 응답이 아직 안 옴
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


class VlmPalmTest(unittest.TestCase):
    """VLM 이 손 위 공 색을 인식 (색 규칙은 VLM 이 답하지 못할 때만)"""
    make = VerifyFlowTest.make
    place = VerifyFlowTest.place

    def test_object_without_color_is_judged_by_vlm(self):
        task, link, clock = self.make("red")
        self.place(task, clock, label="object")                # 색 규칙 없이 '무언가 놓임'
        self.assertEqual(task.state, "GRIP")
        self.assertEqual(task.snapshot()["vlm_last"]["saw"], "빨간 공")

    def test_vlm_sees_no_ball_keeps_waiting(self):
        task, link, clock = self.make(None)                    # VLM: 없음
        self.place(task, clock, label="object")
        self.assertEqual(task.state, "RECEIVE")
        self.assertNotIn("NG", link.sent)
        self.assertNotIn("GRIP", link.sent)

    def test_empty_hand_check_by_vlm(self):
        task, link, clock = self.make("red")
        self.place(task, clock)
        link.rx.append("DONE,GRIP"); task.step(None)
        link.rx.append("DONE,PLACE"); task.step(None)
        self.assertEqual(task.state, "HOME")
        task.vlm.check_label = None                            # 내려놓은 뒤 손 위: 없음
        clock.t += 2.0
        task.step(None); task.step(None)
        self.assertEqual(task.state, "IDLE")
        self.assertEqual(task.done_count, 1)
        self.assertEqual(task.snapshot()["vlm_last"]["kind"], "empty")

    def test_ball_left_in_hand_by_vlm_is_failure(self):
        task, link, clock = self.make("red")
        self.place(task, clock)
        link.rx.append("DONE,GRIP"); task.step(None)
        link.rx.append("DONE,PLACE"); task.step(None)
        clock.t += 2.0
        task.step(None); task.step(None)                       # VLM 이 여전히 빨간 공을 봄
        self.assertEqual((task.state, task.fail_count), ("IDLE", 1))


class TrayTest(unittest.TestCase):
    def test_yolo_picks_requested_part_on_tray(self):
        clock, link = FakeClock(), FakeLink()
        task = HandoverTask(link, clock=clock)
        task.request("파란 공 줘")
        link.rx.append("DONE,OPEN"); task.step(None)
        items = [{"label": "blue", "conf": 0.8, "box": (900, 300, 100, 100)},
                 {"label": "red", "conf": 0.7, "box": (100, 300, 100, 100)},
                 {"label": "blue", "conf": 0.5, "box": (500, 300, 100, 100)}]
        for _ in range(4):                                    # 같은 판단이 이어질 때 기록
            task.update_tray([dict(d) for d in items])
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

    def test_pending_color_waits_without_logging_missing(self):
        clock, link = FakeClock(), FakeLink()
        task = HandoverTask(link, clock=clock)
        task.request("빨간 공 줘")
        link.rx.append("DONE,OPEN"); task.step(None)
        task.update_tray([{"label": "pending", "conf": 0, "box": (100, 100, 40, 40)}])
        self.assertIsNone(task.tray["pick"])
        self.assertIn("색을 확인하는 중", task.message)
        self.assertFalse(any("[쟁반]" in l["text"] for l in task.logs))


class ColorItemsTest(unittest.TestCase):
    """화면·쟁반에 그릴 색 이름은 덩어리의 HSV 가 아니라 팀원 ask_vlm 결과"""

    def test_pending_then_team_color(self):
        import threading
        from handover_app import Runtime
        rt = object.__new__(Runtime)
        rt._color_cache = {}
        rt._color_pending = set()
        rt._color_lock = threading.Lock()

        class Task:
            _vlm_job = None
            vlm = None

            def vlm_active(self):
                return True

        class Vlm:
            def crop(self, frame, box, margin=0.35):
                return frame

            def submit_colors_bg(self, img):
                f = Future()
                f.set_result({"ok": True, "label": "green", "ambiguous": False})
                return f

        rt.task = Task()
        rt.task.vlm = Vlm()
        items = [{"label": "red", "conf": 0.8, "box": (10, 10, 30, 30)}]
        frame = np.zeros((40, 40, 3), np.uint8)
        first = rt._vlm_color_items(frame, items)
        self.assertEqual(first[0]["label"], "pending")          # HSV 빨강을 그대로 쓰지 않음
        second = rt._vlm_color_items(frame, items)
        self.assertEqual((second[0]["label"], second[0]["by"]), ("green", "vlm"))


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


class TeamCodeImportTest(unittest.TestCase):
    """대시보드가 팀원 vlm_go/ 코드를 고치지 않고 그대로 불러 쓰는지"""

    def test_palm_judgement_calls_team_ask_vlm(self):
        from pathlib import Path
        from vision import vlm_bridge
        mod = vlm_bridge.team_vlm_go()
        self.assertEqual(Path(mod.__file__).resolve(), (Path(__file__).resolve().parents[1] / "vlm_go" / "vlm_go.py").resolve())
        seen = {}
        real = mod.ask_vlm
        def spy(frame, target, model, url, *a, **k):
            seen.update(target=target, model=model, prompt=mod.PROMPT.format(target=target))
            return True, '{"found": true}', 0.5
        mod.ask_vlm = spy
        try:
            out = VlmBridge().ask_team(np.zeros((10, 10, 3), np.uint8), "red")
        finally:
            mod.ask_vlm = real
        self.assertEqual((out["found"], seen["target"], seen["model"]), (True, "빨간 공", "gemma3:4b"))
        self.assertIn("'빨간 공'을(를) 찾고 있어", seen["prompt"])            # 팀원 프롬프트 그대로

    def test_color_calls_team_ask_vlm_for_each_color(self):
        from vision import vlm_bridge
        mod = vlm_bridge.team_vlm_go()
        seen = []
        real = mod.ask_vlm

        def spy(frame, target, model, url, *a, **k):
            seen.append((target, mod.PROMPT.format(target=target)))
            return target == "파란 공", '{"found": %s}' % str(target == "파란 공").lower(), 0.1

        mod.ask_vlm = spy
        try:
            out = VlmBridge().ask_team_color(np.zeros((10, 10, 3), np.uint8))
        finally:
            mod.ask_vlm = real
        self.assertEqual([t for t, _ in seen], ["빨간 공", "파란 공", "초록 공"])
        self.assertEqual(out["label"], "blue")
        self.assertFalse(out["ambiguous"])
        self.assertIn("'초록 공'을(를) 찾고 있어", seen[2][1])

    def test_two_true_colors_are_ambiguous(self):
        b = VlmBridge()
        b.ask_team = lambda img, target: {"found": target in ("red", "green"), "raw": "{}", "seconds": 0.1, "ok": True}
        out = b.ask_team_color(np.zeros((8, 8, 3), np.uint8))
        self.assertTrue(out["ambiguous"])
        self.assertIsNone(out["label"])
        self.assertEqual(out["founds"], {"red": True, "blue": False, "green": True})

    def test_dashboard_installs_team_firmware(self):
        import handover_app
        self.assertEqual(handover_app.FIRMWARE.name, "microbit_go.py")
        self.assertEqual(handover_app.FIRMWARE.parent.name, "vlm_go")
        self.assertEqual(handover_app.firmware_version(), "5-go")

    def test_team_files_match_delivery(self):
        """팀원 전달 폴더가 있으면 vlm_go/ 와 한 글자도 다르지 않은지"""
        from pathlib import Path
        src = Path.home() / "Documents" / "카카오톡 받은 파일" / "vlm_go_팀원전달" / "vlm_go"
        if not src.exists():
            self.skipTest("팀원 전달 폴더 없음")
        mine = Path(__file__).resolve().parents[1] / "vlm_go"
        for f in ("vlm_go.py", "microbit_go.py", "README.md", "requirements.txt", "requirements-voice.txt"):
            self.assertEqual((src / f).read_bytes(), (mine / f).read_bytes(), f)
