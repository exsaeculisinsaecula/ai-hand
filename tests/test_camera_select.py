"""대시보드 [카메라 선택]: 카메라 찾기 · 바꾸기 · 저장 (실제 카메라 없이 가짜 VideoCapture 로)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import handover_app as app


class FakeCap:
    """0번=720p, 2번=1080p 만 있는 것처럼 동작"""
    SIZES = {0: (720, 1280), 2: (1080, 1920)}

    def __init__(self, idx):
        self.idx = idx

    def isOpened(self):
        return self.idx in self.SIZES

    def read(self):
        if self.idx not in self.SIZES:
            return False, None
        h, w = self.SIZES[self.idx]
        return True, np.full((h, w, 3), 120, np.uint8)

    def release(self):
        pass


class FakeTask:
    def __init__(self):
        self.logs = []
        self.place_angle = 110

    def _log(self, d, t):
        self.logs.append(t)


def make_runtime(tmp):
    rt = app.Runtime.__new__(app.Runtime)          # 스레드·로봇 없이 필요한 것만
    rt.task = FakeTask()
    rt.camera_index = 1
    rt._cam_switch = False
    rt.frame = np.zeros((480, 640, 3), np.uint8)    # 1번(지금 쓰는 카메라)의 최근 프레임
    return rt


class CameraSelectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch_file = mock.patch.object(app.Runtime, "SETTINGS_FILE", Path(self.tmp.name) / "s.json")
        self.patch_file.start()
        self.patch_cap = mock.patch.object(app.cv2, "VideoCapture", FakeCap)
        self.patch_cap.start()
        self.patch_sleep = mock.patch.object(app.time, "sleep", lambda s: None)
        self.patch_sleep.start()
        self.rt = make_runtime(self.tmp.name)

    def tearDown(self):
        self.patch_sleep.stop()
        self.patch_cap.stop()
        self.patch_file.stop()
        self.tmp.cleanup()

    def test_scan_lists_cameras_with_thumbnails(self):
        cams = self.rt.scan_cameras()
        self.assertEqual([c["index"] for c in cams], [0, 1, 2])
        in_use = next(c for c in cams if c["index"] == 1)
        self.assertEqual(in_use["role"], "hand")                    # 쓰는 중인 카메라는 다시 열지 않음
        self.assertEqual((in_use["width"], in_use["height"]), (640, 480))
        self.assertEqual((cams[2]["width"], cams[2]["height"]), (1920, 1080))
        self.assertTrue(all(c["thumb"].startswith("data:image/jpeg;base64,") for c in cams))

    def test_select_switches_and_remembers(self):
        out = self.rt.select_camera("hand", 2)
        self.assertTrue(out["ok"])
        self.assertEqual(self.rt.camera_index, 2)
        self.assertTrue(self.rt._cam_switch)                       # 카메라 루프가 다시 엶
        self.assertEqual(json.loads(app.Runtime.SETTINGS_FILE.read_text())["camera_index"], 2)
        self.assertEqual(app.Runtime.read_settings()["place_angle"], 110)   # 다른 설정 유지

    def test_select_same_camera_is_noop(self):
        out = self.rt.select_camera("hand", 1)
        self.assertTrue(out["ok"])
        self.assertFalse(self.rt._cam_switch)

    def test_single_camera_mode_has_no_tray_role(self):
        self.assertFalse(self.rt.select_camera("tray", 2)["ok"])


if __name__ == "__main__":
    unittest.main()


class TwoCameraSelectTests(unittest.TestCase):
    def setUp(self):
        import handover_app_2cam as two
        self.tmp = tempfile.TemporaryDirectory()
        self.patch_file = mock.patch.object(app.Runtime, "SETTINGS_FILE", Path(self.tmp.name) / "s.json")
        self.patch_file.start()
        rt = two.Runtime2Cam.__new__(two.Runtime2Cam)
        rt.task = FakeTask()
        rt.camera_index, rt._cam_switch = 1, False
        rt.tray_source, rt._tray_switch = 2, False
        rt.frame = np.zeros((480, 640, 3), np.uint8)
        rt.tray_frame = np.zeros((480, 640, 3), np.uint8)
        self.rt = rt

    def tearDown(self):
        self.patch_file.stop()
        self.tmp.cleanup()

    def test_roles_in_use(self):
        self.assertEqual(self.rt.cameras_in_use(), {1: "hand", 2: "tray"})

    def test_tray_switch_and_conflicts(self):
        self.assertFalse(self.rt.select_camera("tray", 1)["ok"])      # 손 카메라와 같은 번호 거부
        self.assertFalse(self.rt.select_camera("hand", 2)["ok"])      # 쟁반 카메라와 같은 번호 거부
        self.assertTrue(self.rt.select_camera("tray", 3)["ok"])
        self.assertEqual(self.rt.tray_source, 3)
        self.assertTrue(self.rt._tray_switch)
        self.assertEqual(app.Runtime.read_settings()["tray_camera"], 3)
