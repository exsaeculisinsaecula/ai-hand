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
    rt.cam_disabled = set()
    rt.frame = np.zeros((480, 640, 3), np.uint8)    # 1번(지금 쓰는 카메라)의 최근 프레임
    rt.views, rt.tray_slot = {}, None               # 분할 화면 칸 (없음)
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
        rt.cam_disabled = set()
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


class CameraPowerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch_file = mock.patch.object(app.Runtime, "SETTINGS_FILE", Path(self.tmp.name) / "s.json")
        self.patch_file.start()
        self.opened = []

        class Spy(FakeCap):
            def __init__(cap, idx):
                self.opened.append(idx)
                super().__init__(idx)
        self.patch_cap = mock.patch.object(app.cv2, "VideoCapture", Spy)
        self.patch_cap.start()
        self.patch_sleep = mock.patch.object(app.time, "sleep", lambda s: None)
        self.patch_sleep.start()
        self.rt = make_runtime(self.tmp.name)
        self.rt.cam_disabled = set()

    def tearDown(self):
        self.patch_sleep.stop()
        self.patch_cap.stop()
        self.patch_file.stop()
        self.tmp.cleanup()

    def test_off_camera_is_never_opened_by_scan(self):
        self.rt.set_camera_power(0, False)
        cams = self.rt.scan_cameras()
        self.assertNotIn(0, self.opened)                          # 내장 카메라 불이 켜지지 않음
        self.assertTrue(next(c for c in cams if c["index"] == 0)["off"])
        self.assertEqual(app.Runtime.read_settings()["camera_disabled"], [0])

    def test_turning_off_current_camera_and_back_on(self):
        self.rt.set_camera_power(1, False)
        self.assertTrue(self.rt.camera_off())
        self.rt.set_camera_power(1, True)
        self.assertFalse(self.rt.camera_off())
        self.assertTrue(self.rt._cam_switch)                      # 카메라 루프가 다시 엶

    def test_cannot_select_off_camera(self):
        self.rt.set_camera_power(2, False)
        self.assertFalse(self.rt.select_camera("hand", 2)["ok"])


class CameraWatchdogTests(unittest.TestCase):
    def test_stalled_camera_is_reopened_in_new_thread(self):
        rt = make_runtime(None)
        rt.running = True
        rt.jpeg, rt.fps, rt.cam_error = b"x", 25.0, ""
        rt._cam_gen, rt._last_frame_t = 0, 0.0
        rt._cam_opened_t = app.time.monotonic() - 10       # 10초 전에 열렸는데 영상이 한 장도 없음
        started = []
        with mock.patch.object(app.threading, "Thread", lambda target, args, daemon: mock.Mock(start=lambda: started.append(args))):
            rt._camera_watchdog()
        self.assertEqual(rt._cam_gen, 1)
        self.assertEqual(started, [(1,)])
        self.assertIsNone(rt.jpeg)
        self.assertIn("영상을 보내지 않아", rt.cam_error)

    def test_healthy_or_off_camera_is_left_alone(self):
        rt = make_runtime(None)
        rt._cam_gen = 0
        rt._cam_opened_t = app.time.monotonic() - 10
        rt._last_frame_t = app.time.monotonic()            # 방금 프레임이 옴
        rt._camera_watchdog()
        self.assertEqual(rt._cam_gen, 0)
        rt._last_frame_t = 0.0
        rt.cam_disabled = {1}                               # 꺼 둔 카메라는 감시하지 않음
        rt._camera_watchdog()
        self.assertEqual(rt._cam_gen, 0)


class PortDiscoveryTests(unittest.TestCase):
    """micro:bit 포트 자동 찾기 — Windows COM / macOS / Linux"""

    def _ports(self, *items):
        return [mock.Mock(device=d, vid=v, description=desc, manufacturer=None, product=None) for d, v, desc in items]

    def test_windows_com_port_by_vid(self):
        ports = self._ports(("COM3", None, "Bluetooth 링크를 통한 표준 직렬"), ("COM5", 0x0D28, "USB Serial Device (COM5)"))
        with mock.patch("serial.tools.list_ports.comports", return_value=ports):
            self.assertEqual(app.RobotLink.list_microbit_ports(), ["COM5"])
            link = app.RobotLink.__new__(app.RobotLink)
            link.port_hint = ""
            self.assertEqual(link._find_port(), "COM5")

    def test_windows_manual_port_hint(self):
        ports = self._ports(("COM7", None, "USB-SERIAL CH340"))
        with mock.patch("serial.tools.list_ports.comports", return_value=ports):
            link = app.RobotLink.__new__(app.RobotLink)
            link.port_hint = "COM7"                  # --port COM7
            self.assertEqual(link._find_port(), "COM7")

    def test_macos_and_linux_names(self):
        ports = self._ports(("/dev/cu.Bluetooth-Incoming-Port", None, "n/a"), ("/dev/cu.usbmodem145102", None, "n/a"))
        with mock.patch("serial.tools.list_ports.comports", return_value=ports):
            self.assertEqual(app.RobotLink.list_microbit_ports(), ["/dev/cu.usbmodem145102"])
        ports = self._ports(("/dev/ttyACM0", None, "n/a"))
        with mock.patch("serial.tools.list_ports.comports", return_value=ports):
            self.assertEqual(app.RobotLink.list_microbit_ports(), ["/dev/ttyACM0"])


class FakeView:
    """ViewCam 대역 (스레드·카메라 없이)"""

    def __init__(self, rt, slot, src):
        self.slot, self.src, self.stopped = slot, src, False

    def stop(self):
        self.stopped = True


class MultiCamTests(unittest.TestCase):
    """CAM 1~4 분할 화면: 칸에 카메라 넣기 · 자리 바꾸기 · 분할 줄이면 닫기 · 쟁반 칸"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [mock.patch.object(app.Runtime, "SETTINGS_FILE", Path(self.tmp.name) / "s.json"),
                        mock.patch.object(app, "ViewCam", FakeView)]
        for p in self.patches:
            p.start()
        self.rt = make_runtime(self.tmp.name)
        self.rt.split, self.rt.view_order = 4, ["hand", "2", "3", "4"]

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_set_view_and_saved(self):
        self.assertTrue(self.rt.set_view(3, "2")["ok"])
        self.assertEqual(self.rt.views[3].src, 2)                       # 숫자는 웹캠 번호로
        saved = json.loads(app.Runtime.SETTINGS_FILE.read_text())
        self.assertEqual(saved["view_sources"], {"3": 2})

    def test_reorder_then_shrink_closes_hidden_cameras(self):
        self.rt.set_view(2, "0")
        self.rt.set_view(4, "3")
        self.assertTrue(self.rt.set_view_order(["4", "hand", "2", "3"])["ok"])   # CAM 1 자리에 4번 칸
        cam_slot2 = self.rt.views[2]
        self.rt.set_split(2)                                            # 보이는 건 "4", "hand" 뿐
        self.assertEqual(sorted(self.rt.views), [4])
        self.assertTrue(cam_slot2.stopped)

    def test_bad_order_rejected(self):
        self.assertFalse(self.rt.set_view_order(["hand", "2", "2", "4"])["ok"])

    def test_tray_slot_toggles_and_needs_camera(self):
        self.assertFalse(self.rt.set_tray_slot(3)["ok"])                # 빈 칸은 쟁반 불가
        self.rt.set_view(3, "2")
        self.rt.set_tray_slot(3)
        self.assertEqual(self.rt.tray_slot, 3)
        self.rt.set_tray_slot(3)                                        # 다시 누르면 해제
        self.assertIsNone(self.rt.tray_slot)

    def test_picker_can_fill_a_tile(self):
        out = self.rt.select_camera("view2", 4)
        self.assertTrue(out["ok"])
        self.assertEqual(self.rt.views[2].src, 4)


class AutoCamTests(MultiCamTests):
    """카메라가 새로 잡히면 빈 CAM 칸에 자동 배치"""

    def setUp(self):
        super().setUp()
        self.rt.split = 1

    def test_missing_hand_camera_takes_first_found_and_rest_fill_tiles(self):
        notes = self.rt.auto_place([0, 2, 3])                   # 손 카메라(1번)가 없음
        self.assertEqual(self.rt.camera_index, 0)
        self.assertEqual(self.rt.split, 3)
        self.assertEqual(sorted(v.src for v in self.rt.views.values()), [2, 3])
        self.assertTrue(any("손 카메라" in n for n in notes))

    def test_working_hand_camera_is_kept(self):
        self.rt.auto_place([1, 4])
        self.assertEqual(self.rt.camera_index, 1)
        self.assertEqual([v.src for v in self.rt.views.values()], [4])
        self.assertEqual(self.rt.split, 2)

    def test_already_shown_cameras_are_not_duplicated(self):
        self.rt.auto_place([1, 4])
        self.rt.auto_place([1, 4])
        self.assertEqual(len(self.rt.views), 1)

    def test_more_than_four_cameras(self):
        notes = self.rt.auto_place([1, 2, 3, 4, 5])
        self.assertEqual(self.rt.split, 4)
        self.assertEqual(len(self.rt.views), 3)
        self.assertTrue(any("모자라" in n for n in notes))

    def test_switched_off_hand_camera_is_not_replaced(self):
        self.rt.cam_disabled = {1}
        self.rt.auto_place([0])
        self.assertEqual(self.rt.camera_index, 1)
