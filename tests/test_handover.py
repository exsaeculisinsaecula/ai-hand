"""받기 작업(입력 해석 · 상태 머신 · 자세 미러 · 판별기) 테스트. 로봇·카메라 불필요."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from controller.handover import HandoverTask
from vision.parts import PartClassifier, extract_features, find_balls, parse_request, rule_classify

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "neuralgrip_best_two.pt"
TEAM_TEST = Path("/Users/archailects/Downloads/NeuralGrip_Team/neralgrip_two/images/test")


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeLink:
    is_mock = True

    def __init__(self):
        self.sent, self.rx = [], []

    def send(self, cmd):
        self.sent.append(cmd)

    def poll(self):
        out, self.rx = self.rx, []
        return out

    def close(self):
        pass


def seen(label, x=100, y=100, conf=0.9, motion=0.5):
    return {"label": label, "label_ko": label, "confidence": conf, "reason": "test",
            "box": [x, y, 40, 40], "motion": motion}


EMPTY = {"label": "none", "confidence": 0.0, "box": None, "motion": 0.3}


class ParseTests(unittest.TestCase):
    def test_words(self):
        self.assertEqual(parse_request("빨간 공 줘")[0], "red")
        self.assertEqual(parse_request("파란 공 하나 받아")[0], "blue")
        self.assertEqual(parse_request("초록색")[0], "green")
        self.assertEqual(parse_request("give me a blue ball")[0], "blue")

    def test_ball_without_color_asks_again(self):
        target, reply = parse_request("공 주세요")
        self.assertIsNone(target)
        self.assertIn("빨간 공 / 파란 공 / 초록 공", reply)

    def test_negation_and_ambiguity(self):
        self.assertIsNone(parse_request("빨간 공이랑 파란 공")[0])
        self.assertIsNone(parse_request("빨간 공 말고")[0])               # 파란·초록 중 무엇인지 다시 물음
        self.assertEqual(parse_request("빨간 공 말고 파란 공")[0], "blue")
        self.assertEqual(parse_request("빨강 파랑 빼고")[0], "green")    # 남은 건 초록뿐
        self.assertIsNone(parse_request("안녕")[0])


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.link = FakeLink()
        self.task = HandoverTask(self.link, clock=self.clock, stable_frames=5)

    def _to_receive(self, text="빨간 공 줘"):
        self.assertTrue(self.task.request(text)["ok"])
        self.assertEqual(self.link.sent[-1], "OPEN")
        self.link.rx.append("DONE,OPEN")
        self.task.step(None)
        self.assertEqual(self.task.state, "RECEIVE")

    def _place_and_check(self, frames_after):
        self.link.rx.append("DONE,GRIP")
        self.task.step(None)
        self.assertEqual(self.task.state, "PLACE")
        self.assertEqual(self.link.sent[-1], f"PLACE,{self.task.place_angle}")
        self.link.rx.append("DONE,PLACE")
        self.task.step(None)
        self.assertEqual(self.task.state, "HOME")
        for f in frames_after:
            self.clock.t += 0.2
            self.task.step(f)

    def test_happy_path_grip_place_verify_empty(self):
        self._to_receive()
        for _ in range(5):
            self.task.step(seen("red"))
        self.assertEqual(self.task.state, "GRIP")
        self.assertEqual(self.link.sent[-1], "GRIP")
        self._place_and_check([EMPTY] * 9)
        self.assertEqual(self.task.state, "IDLE")
        st = self.task.stats()
        self.assertEqual((st["done"], st["failed"], st["success_rate"]), (1, 0, 100))
        self.assertIsNotNone(st["avg_cycle_s"])

    def test_part_left_in_hand_counts_as_failure(self):
        self._to_receive()
        for _ in range(5):
            self.task.step(seen("red"))
        self._place_and_check([seen("red")] * 9)
        self.assertEqual(self.task.state, "IDLE")
        self.assertEqual(self.task.tone, "warn")
        self.assertEqual(self.task.stats()["failed"], 1)

    def test_wrong_part_ng_then_wait_for_removal(self):
        self._to_receive("빨간 공 줘")
        for _ in range(5):
            self.task.step(seen("blue"))
        self.assertEqual(self.task.state, "VERIFY")
        self.assertEqual(self.link.sent[-1], "NG")
        self.assertEqual(self.task.reject_count, 1)
        self.link.rx.append("DONE,NG")
        self.task.step(None)
        self.assertEqual(self.task.state, "RECEIVE")
        for _ in range(10):                                  # 파란 공가 그대로면 다시 NG 하지 않음
            self.task.step(seen("blue"))
        self.assertEqual(self.link.sent.count("NG"), 1)
        for _ in range(5):                                   # 치우면 다시 판정
            self.task.step(EMPTY)
        for _ in range(5):
            self.task.step(seen("red"))
        self.assertEqual(self.task.state, "GRIP")

    def test_human_hand_motion_blocks_grip(self):
        self._to_receive()
        for _ in range(12):
            self.task.step(seen("red", motion=15.0))        # 사람 손이 아직 손바닥 위
        self.assertEqual(self.task.state, "RECEIVE")
        self.assertIn("손을 빼", self.task.message)

    def test_moving_object_does_not_trigger(self):
        self._to_receive()
        for i in range(12):
            self.task.step(seen("red", x=100 + i * 40))
        self.assertEqual(self.task.state, "RECEIVE")

    def test_flicker_resets_streak(self):
        self._to_receive()
        for lbl in ["red", "red", "none", "red", "red", "none", "red"]:
            self.task.step(seen(lbl) if lbl != "none" else EMPTY)
        self.assertEqual(self.task.state, "RECEIVE")

    def test_low_confidence_ignored(self):
        self._to_receive()
        for _ in range(12):
            self.task.step(seen("red", conf=0.3))
        self.assertEqual(self.task.state, "RECEIVE")

    def test_stop_and_aborted_reply(self):
        self._to_receive()
        for _ in range(5):
            self.task.step(seen("red"))
        self.task.stop()
        self.assertEqual(self.link.sent[-3:], ["STOP", "STOP", "STOP"])
        self.link.rx.append("ERR,ABORTED")                    # 펌웨어가 이동 중단을 알림
        self.task.step(None)
        self.assertIn("비상 정지", self.task.message)        # 정지 안내가 덮어써지지 않음
        self.assertFalse(self.task.request("파란 공")["ok"])
        self.task.resume()
        self.link.rx.append("DONE,HOME")
        self.task.step(None)
        self.assertEqual(self.task.state, "IDLE")
        self.assertTrue(self.task.request("파란 공")["ok"])

    def test_command_timeout_stops(self):
        self.task.request("빨간 공")
        self.clock.t += 11.0
        self.task.step(None)
        self.assertTrue(self.task.stopped)

    def test_button_a_confirms_quickly(self):
        self._to_receive()
        self.link.rx.append("EVT,BTN_A")
        self.task.step(seen("red", motion=20.0))
        self.assertEqual(self.task.state, "GRIP")

    def test_confirm_without_camera_grips_and_finishes(self):
        self._to_receive()
        self.task.confirm_without_camera()
        self.assertEqual(self.task.state, "GRIP")
        self._place_and_check([None] * 9)
        self.clock.t += 2
        self.task.step(None)
        self.assertEqual(self.task.state, "IDLE")
        self.assertEqual(self.task.stats()["done"], 1)

    def test_place_angle_is_configurable(self):
        self.task.place_angle = 130
        self._to_receive()
        for _ in range(5):
            self.task.step(seen("red"))
        self.link.rx.append("DONE,GRIP")
        self.task.step(None)
        self.assertEqual(self.link.sent[-1], "PLACE,130")


class PoseMirrorTests(unittest.TestCase):
    def test_place_timeline(self):
        clock = FakeClock()
        task = HandoverTask(FakeLink(), clock=clock)
        task._send("GRIP")
        clock.t += 0.7
        self.assertLess(task.pose()["grip"], 0.8)              # 1.5초에 걸쳐 쥐는 중
        clock.t += 1.0
        self.assertGreater(task.pose()["grip"], 0.9)
        task._send("PLACE,110")
        clock.t += 0.8
        self.assertTrue(92 < task.pose()["pan"] < 110)        # 손목 회전 중 (코사인 이징)
        clock.t += 1.0
        self.assertEqual(task.pose()["pan"], 110.0)            # 회전 끝 → 손 펴는 중
        clock.t += 3.0
        p = task.pose()
        self.assertEqual(p["pan"], 90.0)
        self.assertLess(p["grip"], 0.2)

    def test_easing_starts_slowly(self):
        clock = FakeClock()
        task = HandoverTask(FakeLink(), clock=clock)
        task._send("PAN,180")
        samples = []
        for k in range(0, 18):
            clock.t = 100.0 + k * 0.1
            samples.append(task.pose()["pan"])
        self.assertLess(samples[1] - samples[0], samples[8] - samples[7])
        self.assertEqual(samples[-1], 180.0)

    def test_ng_wiggle_returns_to_front(self):
        clock = FakeClock()
        task = HandoverTask(FakeLink(), clock=clock)
        task._send("NG")
        clock.t += 0.34
        self.assertAlmostEqual(task.pose()["pan"], 70.0, delta=1)
        clock.t += 0.34
        self.assertAlmostEqual(task.pose()["pan"], 110.0, delta=1)
        clock.t += 2.0
        self.assertEqual(task.pose()["pan"], 90.0)

    def test_stop_freezes_pose(self):
        clock = FakeClock()
        task = HandoverTask(FakeLink(), clock=clock)
        task._send("PAN,180")
        clock.t += 0.6
        mid = task.pose()["pan"]
        task.stop()
        clock.t += 2.0
        self.assertAlmostEqual(task.pose()["pan"], mid, places=1)


def canvas():
    return np.full((480, 640, 3), 235, np.uint8)


BALL_BGR = {"red": (70, 60, 200), "blue": (200, 110, 40), "green": (60, 170, 60)}


def draw_ball(color, center=(320, 240), r=80, img=None):
    img = canvas() if img is None else img
    cv2.circle(img, center, r, BALL_BGR[color], -1)
    return img


class ColorFallbackTests(unittest.TestCase):
    """VLM 연동 전 임시 판별기 (색 규칙)."""

    def test_rule_classifier_on_synthetic_balls(self):
        for want in ("red", "blue", "green"):
            f, _ = extract_features(draw_ball(want))
            self.assertEqual(rule_classify(f)[0], want, f"{want}: {f}")

    def test_gray_clutter_is_not_a_ball(self):
        img = canvas()
        cv2.rectangle(img, (100, 100), (400, 300), (60, 60, 60), -1)      # 회색 물체(채도 낮음)
        self.assertIsNone(extract_features(img)[0])

    def test_background_difference_ignores_static_clutter(self):
        bg = draw_ball("green", center=(80, 80), r=40)                     # 원래 있던 초록 물체
        frame = draw_ball("red", img=bg.copy())
        self.assertEqual(rule_classify(extract_features(frame, background=bg)[0])[0], "red")
        self.assertIsNone(extract_features(bg.copy(), background=bg)[0])

    def test_hand_zone_ignores_objects_outside(self):
        img = draw_ball("blue")
        self.assertIsNotNone(extract_features(img, roi=(0.3, 0.2, 0.7, 0.8))[0])
        self.assertIsNone(extract_features(img, roi=(0.0, 0.0, 0.2, 0.2))[0])

    def test_knn_after_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            clf = PartClassifier(data_dir=tmp)
            for _ in range(2):
                for c in ("red", "blue", "green"):
                    clf.add_sample(draw_ball(c), c)
            self.assertEqual(clf.classify(draw_ball("blue")).label, "blue")
            self.assertEqual(PartClassifier(data_dir=tmp).counts()["green"], 2)


class TrayColorTests(unittest.TestCase):
    """쟁반: 색 규칙으로 공을 모두 찾기 (find_balls)"""

    def tray(self):
        img = canvas()
        cv2.rectangle(img, (30, 170), (610, 400), (140, 130, 115), -1)
        for x, c in ((130, "red"), (320, "blue"), (510, "green")):
            draw_ball(c, center=(x, 285), r=50, img=img)
        return img

    def test_finds_every_ball_left_to_right(self):
        found = sorted(find_balls(self.tray()), key=lambda d: d["box"][0])
        self.assertEqual([d["label"] for d in found], ["red", "blue", "green"])

    def test_palm_zone_is_excluded(self):
        found = find_balls(self.tray(), exclude=(0.4, 0.0, 0.6, 1.0))     # 가운데(파랑)는 손 영역
        self.assertEqual(sorted(d["label"] for d in found), ["green", "red"])

    def test_strip_is_not_a_ball(self):
        img = canvas()
        cv2.rectangle(img, (0, 50), (640, 62), BALL_BGR["red"], -1)
        self.assertEqual(find_balls(img), [])


@unittest.skipUnless(MODEL.exists(), "YOLO 가중치 없음")
class YoloTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from vision.parts import YoloParts
            cls.yolo = YoloParts(MODEL)
        except ImportError as exc:
            raise unittest.SkipTest(f"ultralytics 없음: {exc}")

    def test_classes(self):
        self.assertEqual(sorted(self.yolo.names.values()), ["bolt", "nut"])

    def test_empty_frame_has_no_part(self):
        self.assertEqual(self.yolo.classify(canvas()).label, "none")

    @unittest.skipUnless(TEAM_TEST.exists(), "팀 test 이미지 없음")
    def test_team_test_images_detect_nut_without_false_positive(self):
        imgs = sorted(TEAM_TEST.glob("*.jpg"))
        nut_hits = 0
        for p in imgs:
            labels = {d["label"] for d in self.yolo.detect(cv2.imread(str(p))) if d["conf"] >= 0.45}
            self.assertTrue(labels <= {"bolt", "nut"})
            nut_hits += "nut" in labels
        self.assertGreaterEqual(nut_hits, len(imgs) - 2)       # 빈 트레이 1장 포함

    @unittest.skipUnless(TEAM_TEST.exists(), "팀 test 이미지 없음")
    def test_roi_filters_by_box_center(self):
        img = cv2.imread(str(sorted(TEAM_TEST.glob("*.jpg"))[0]))
        dets = self.yolo.detect(img)
        self.assertTrue(dets)
        self.assertEqual(self.yolo.classify(img, roi=(0.0, 0.0, 0.01, 0.01), dets=dets).label, "none")


if __name__ == "__main__":
    unittest.main()


class ManualStepTests(unittest.TestCase):
    """인식이 실패해도 버튼만으로 끝까지 진행할 수 있는지"""

    def setUp(self):
        self.clock = FakeClock()
        self.link = FakeLink()
        self.task = HandoverTask(self.link, clock=self.clock)

    def keys(self):
        return [o["key"] for o in self.task.manual_options()]

    def test_full_cycle_with_buttons_only_and_no_robot_replies(self):
        self.assertEqual(self.keys(), ["req_red", "req_blue", "req_green"])
        self.assertTrue(self.task.manual("req_red")["ok"])
        self.assertEqual(self.task.state, "REQUEST")
        self.task.manual("skip_open")                 # OPEN 응답이 안 와도
        self.assertEqual(self.task.state, "RECEIVE")
        self.task.manual("grip")                      # 카메라가 못 알아봐도
        self.assertEqual((self.task.state, self.link.sent[-1]), ("GRIP", "GRIP"))
        self.task.manual("place")
        self.assertEqual((self.task.state, self.link.sent[-1]), ("PLACE", "PLACE,180"))
        self.task.manual("placed")
        self.assertEqual(self.task.state, "HOME")
        self.task.manual("done")
        self.assertEqual(self.task.state, "IDLE")
        self.assertEqual(self.task.stats()["done"], 1)

    def test_ng_then_receive_again_and_fail(self):
        self.task.manual("req_blue")
        self.task.manual("skip_open")
        self.task.manual("ng")
        self.assertEqual((self.task.state, self.link.sent[-1]), ("VERIFY", "NG"))
        self.task.manual("receive")
        self.assertEqual(self.task.state, "RECEIVE")
        self.task.manual("grip"); self.task.manual("place"); self.task.manual("placed")
        self.task.manual("fail")
        self.assertEqual(self.task.stats()["failed"], 1)

    def test_wrong_step_rejected_and_stop_hides_buttons(self):
        self.assertFalse(self.task.manual("grip")["ok"])
        self.task.stop()
        self.assertEqual(self.task.manual_options(), [])

    def test_late_robot_reply_is_ignored_after_manual_skip(self):
        self.task.manual("req_red"); self.task.manual("skip_open"); self.task.manual("grip"); self.task.manual("place")
        self.link.rx.append("DONE,GRIP")              # 늦게 온 쥐기 완료 응답
        self.task.step(None)
        self.assertEqual(self.task.state, "PLACE")


class ManualBusyTests(unittest.TestCase):
    def test_manual_request_refused_while_robot_moving(self):
        task = HandoverTask(FakeLink(), clock=FakeClock())
        task._send("PLACE,180")                       # 직접 조작으로 움직이는 중
        out = task.manual("req_red")
        self.assertFalse(out["ok"])
        self.assertEqual(task.state, "IDLE")


class HandZoneFromMotionTests(unittest.TestCase):
    """[손 자동 찾기]: 폈을 때와 쥐었을 때 사진 차이로 손 자리 찾기"""

    def frames(self, person_walks=False):
        from vision.parts import hand_zone_from_motion  # noqa: F401
        bg = canvas()
        cv2.rectangle(bg, (0, 380), (640, 480), (40, 40, 40), -1)              # 검은 책상 (움직이지 않음)
        opened = bg.copy()
        for i in range(4):                                                     # 펴진 손가락 4개
            cv2.rectangle(opened, (300 + i * 22, 120), (314 + i * 22, 230), (30, 30, 30), -1)
        cv2.rectangle(opened, (290, 230), (390, 300), (30, 30, 30), -1)          # 손바닥
        grip = bg.copy()
        cv2.rectangle(grip, (290, 200), (390, 300), (30, 30, 30), -1)            # 접힌 손가락이 손바닥 쪽으로
        opened2 = opened.copy()
        if person_walks:
            cv2.rectangle(grip, (20, 20), (120, 200), (90, 60, 40), -1)          # 쥘 때만 사람이 지나감
        return opened, grip, opened2

    def test_finds_finger_area(self):
        from vision.parts import hand_zone_from_motion
        zone, box = hand_zone_from_motion(*self.frames())
        x0, y0, x1, y1 = zone
        self.assertTrue(0.40 < (x0 + x1) / 2 < 0.60, zone)                       # 손 가운데
        self.assertTrue((y0 + y1) / 2 < 380 / 480, zone)                          # 책상이 아님

    def test_no_motion_returns_none(self):
        from vision.parts import hand_zone_from_motion
        f = canvas()
        self.assertIsNone(hand_zone_from_motion(f, f.copy(), f.copy()))


class CancelDuringMotionTests(unittest.TestCase):
    """움직이는 중에 취소하면 펌웨어가 HOME 을 버리므로, 동작이 끝난 뒤 HOME 을 보내야 함 (멈춤 방지)"""

    def test_cancel_while_opening_sends_home_after_done(self):
        clock, link = FakeClock(), FakeLink()
        task = HandoverTask(link, clock=clock)
        task.request("빨간 공 줘")
        self.assertEqual(link.sent[-1], "OPEN")
        task.cancel()                                   # 펴는 중에 취소
        self.assertEqual(link.sent[-1], "OPEN")         # 아직 HOME 을 보내지 않음
        link.rx.append("DONE,OPEN"); task.step(None)
        self.assertEqual(link.sent[-1], "HOME")
        link.rx.append("DONE,HOME"); task.step(None)
        self.assertEqual(task.state, "IDLE")
