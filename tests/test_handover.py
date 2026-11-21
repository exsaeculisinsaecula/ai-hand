"""받기 작업(입력 해석 · 상태 머신 · 자세 미러 · 판별기) 테스트. 로봇·카메라 불필요."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from controller.handover import HandoverTask
from vision.parts import PartClassifier, extract_features, parse_request, rule_classify

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
        self.assertEqual(parse_request("볼트 줘")[0], "bolt")
        self.assertEqual(parse_request("너트 하나 받아")[0], "nut")
        self.assertEqual(parse_request("give me a nut")[0], "nut")

    def test_washer_is_not_supported_by_model(self):
        target, reply = parse_request("와셔 주세요")
        self.assertIsNone(target)
        self.assertIn("볼트·너트", reply)

    def test_negation_and_ambiguity(self):
        self.assertIsNone(parse_request("볼트랑 너트")[0])
        self.assertEqual(parse_request("볼트 말고")[0], "nut")      # 남은 건 너트뿐
        self.assertEqual(parse_request("볼트 말고 너트")[0], "nut")
        self.assertIsNone(parse_request("안녕")[0])


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.link = FakeLink()
        self.task = HandoverTask(self.link, clock=self.clock, stable_frames=5)

    def _to_receive(self, text="볼트 줘"):
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
            self.task.step(seen("bolt"))
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
            self.task.step(seen("bolt"))
        self._place_and_check([seen("bolt")] * 9)
        self.assertEqual(self.task.state, "IDLE")
        self.assertEqual(self.task.tone, "warn")
        self.assertEqual(self.task.stats()["failed"], 1)

    def test_wrong_part_ng_then_wait_for_removal(self):
        self._to_receive("볼트 줘")
        for _ in range(5):
            self.task.step(seen("nut"))
        self.assertEqual(self.task.state, "VERIFY")
        self.assertEqual(self.link.sent[-1], "NG")
        self.assertEqual(self.task.reject_count, 1)
        self.link.rx.append("DONE,NG")
        self.task.step(None)
        self.assertEqual(self.task.state, "RECEIVE")
        for _ in range(10):                                  # 너트가 그대로면 다시 NG 하지 않음
            self.task.step(seen("nut"))
        self.assertEqual(self.link.sent.count("NG"), 1)
        for _ in range(5):                                   # 치우면 다시 판정
            self.task.step(EMPTY)
        for _ in range(5):
            self.task.step(seen("bolt"))
        self.assertEqual(self.task.state, "GRIP")

    def test_human_hand_motion_blocks_grip(self):
        self._to_receive()
        for _ in range(12):
            self.task.step(seen("bolt", motion=15.0))        # 사람 손이 아직 손바닥 위
        self.assertEqual(self.task.state, "RECEIVE")
        self.assertIn("손을 빼", self.task.message)

    def test_moving_object_does_not_trigger(self):
        self._to_receive()
        for i in range(12):
            self.task.step(seen("bolt", x=100 + i * 40))
        self.assertEqual(self.task.state, "RECEIVE")

    def test_flicker_resets_streak(self):
        self._to_receive()
        for lbl in ["bolt", "bolt", "none", "bolt", "bolt", "none", "bolt"]:
            self.task.step(seen(lbl) if lbl != "none" else EMPTY)
        self.assertEqual(self.task.state, "RECEIVE")

    def test_low_confidence_ignored(self):
        self._to_receive()
        for _ in range(12):
            self.task.step(seen("bolt", conf=0.3))
        self.assertEqual(self.task.state, "RECEIVE")

    def test_stop_and_aborted_reply(self):
        self._to_receive()
        for _ in range(5):
            self.task.step(seen("bolt"))
        self.task.stop()
        self.assertEqual(self.link.sent[-3:], ["STOP", "STOP", "STOP"])
        self.link.rx.append("ERR,ABORTED")                    # 펌웨어가 이동 중단을 알림
        self.task.step(None)
        self.assertIn("비상 정지", self.task.message)        # 정지 안내가 덮어써지지 않음
        self.assertFalse(self.task.request("너트")["ok"])
        self.task.resume()
        self.link.rx.append("DONE,HOME")
        self.task.step(None)
        self.assertEqual(self.task.state, "IDLE")
        self.assertTrue(self.task.request("너트")["ok"])

    def test_command_timeout_stops(self):
        self.task.request("볼트")
        self.clock.t += 11.0
        self.task.step(None)
        self.assertTrue(self.task.stopped)

    def test_button_a_confirms_quickly(self):
        self._to_receive()
        self.link.rx.append("EVT,BTN_A")
        self.task.step(seen("bolt", motion=20.0))
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
            self.task.step(seen("bolt"))
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


def draw_bolt():
    img = canvas()
    cv2.rectangle(img, (200, 225), (460, 255), (40, 40, 40), -1)
    cv2.rectangle(img, (170, 205), (205, 275), (40, 40, 40), -1)
    return img


def draw_nut():
    img = canvas()
    c, r = (320, 240), 90
    pts = np.array([[c[0] + r * np.cos(np.pi / 3 * k), c[1] + r * np.sin(np.pi / 3 * k)] for k in range(6)], np.int32)
    cv2.fillPoly(img, [pts], (40, 40, 40))
    cv2.circle(img, c, 40, (235, 235, 235), -1)
    return img


def draw_washer():
    img = canvas()
    cv2.circle(img, (320, 240), 95, (40, 40, 40), -1)
    cv2.circle(img, (320, 240), 42, (235, 235, 235), -1)
    return img


class ShapeFallbackTests(unittest.TestCase):
    """YOLO 를 못 쓸 때의 대체 판별기."""

    def test_rule_classifier_on_synthetic_shapes(self):
        for img, want in ((draw_bolt(), "bolt"), (draw_nut(), "nut"), (draw_washer(), "washer")):
            f, _ = extract_features(img)
            self.assertEqual(rule_classify(f)[0], want, f"{want}: {f}")

    def test_background_difference_ignores_static_clutter(self):
        bg = canvas()
        cv2.line(bg, (0, 50), (640, 80), (120, 120, 120), 3)
        frame = bg.copy()
        washer = draw_washer()
        mask = washer[:, :, 0] < 100
        frame[mask] = washer[mask]
        self.assertEqual(rule_classify(extract_features(frame, background=bg)[0])[0], "washer")
        self.assertIsNone(extract_features(bg.copy(), background=bg)[0])

    def test_hand_zone_ignores_objects_outside(self):
        img = draw_washer()
        self.assertIsNotNone(extract_features(img, roi=(0.3, 0.2, 0.7, 0.8))[0])
        self.assertIsNone(extract_features(img, roi=(0.0, 0.0, 0.2, 0.2))[0])

    def test_knn_after_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            clf = PartClassifier(data_dir=tmp)
            for _ in range(2):
                clf.add_sample(draw_bolt(), "bolt")
                clf.add_sample(draw_nut(), "nut")
                clf.add_sample(draw_washer(), "washer")
            self.assertEqual(clf.classify(draw_nut()).label, "nut")
            self.assertEqual(PartClassifier(data_dir=tmp).counts()["washer"], 2)


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
