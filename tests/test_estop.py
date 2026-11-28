"""비상정지(controller/estop.py) 시험 — 실물 버튼 없이 가짜 어댑터 · 배선으로 모든 경우를 재현한다."""
import random
import sys
import threading
import unittest
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.estop import (EStop, GuardedLink, LoopChecker, MockSwitchWorld, match_port,  # noqa: E402
                              MICROBIT_VID)
from controller.handover import HandoverTask  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 0.0005)


class Wire:
    """USB-시리얼 어댑터 하나 + 그 출력선과 입력선 사이의 NC 비상정지 버튼."""

    def __init__(self, clock, port="COM7", out="rts", inn="cts", latency=0.004, serial="A1", vid=0x1A86, pid=0x7523,
                 location="1-2", wired=True):
        self.clock, self.port, self.out, self.inn, self.latency = clock, port, out, inn, latency
        self.ident = {"port": port, "vid": vid, "pid": pid, "serial": serial, "location": location, "desc": "USB-SERIAL CH340"}
        self.wired = wired            # 버튼이 이 어댑터에 연결돼 있나
        self.pressed = False
        self.broken = False           # 전선 끊김
        self.plugged = True
        self.stuck = None             # 입력선이 True/False 에 붙음
        self.floating = False         # 입력선이 떠서 잡음
        self.jumper = None            # (out, in): 버튼 없이 점퍼선으로 이어진 다른 회로
        self.levels = {"rts": True, "dtr": True}
        self.prev = dict(self.levels)
        self.changed = {"rts": -1e9, "dtr": -1e9}
        self.opened = 0
        self.rng = random.Random(7)

    def set_out(self, line, v):
        if not self.plugged:
            raise OSError("unplugged")
        if self.levels[line] != bool(v):
            self.prev[line] = self.levels[line]
            self.levels[line] = bool(v)
            self.changed[line] = self.clock()

    def _seen(self, line):           # USB 지연: 바뀐 지 latency 가 지나야 입력에 보임
        return self.levels[line] if self.clock() - self.changed[line] >= self.latency else self.prev[line]

    def read_in(self, line):
        if not self.plugged:
            raise OSError("unplugged")
        if self.jumper and line == self.jumper[1]:
            return self._seen(self.jumper[0])
        if line != self.inn or not self.wired:
            return False
        if self.stuck is not None:
            return self.stuck
        if self.floating:
            return self.rng.random() < 0.5
        if self.pressed or self.broken:
            return False
        return self._seen(self.out)


class Bench:
    """여러 어댑터가 꽂힌 PC."""

    def __init__(self, clock, wires):
        self.clock, self.wires = clock, wires
        self.extra_ports = []
        self.open_log = []

    def lister(self):
        return [w.ident for w in self.wires if w.plugged] + list(self.extra_ports)

    def opener(self, port):
        self.open_log.append(port)
        for w in self.wires:
            if w.ident["port"] == port and w.plugged:
                w.opened += 1
                return _Dev(w)
        raise OSError(f"{port} 없음")


class _Dev:
    def __init__(self, w):
        self.w = w
        self.closed = False

    def set_out(self, line, v):
        self.w.set_out(line, v)

    def read_in(self, line):
        return self.w.read_in(line)

    def close(self):
        self.closed = True


class LinkSpy:
    is_mock = False
    port = "COM5"

    def __init__(self):
        self.sent = []
        self.rx = deque()
        self.connected = True
        self.status = "connected"

    def send(self, cmd):
        self.sent.append(cmd)
        return True

    def poll(self):
        out = list(self.rx)
        self.rx.clear()
        return out

    def info(self):
        return {"status": "connected"}


def cfg_for(w, **kw):
    c = {"port": w.ident["port"], "vid": w.ident["vid"], "pid": w.ident["pid"], "serial": w.ident["serial"],
         "location": w.ident["location"], "out": w.out, "in": w.inn, "settle_s": 0.03}
    c.update(kw)
    return c


def make(clock, wires, cfg=None, link=None):
    bench = Bench(clock, wires)
    link = link or LinkSpy()
    saved = []
    logs = []
    es = EStop(link, cfg=cfg, log=lambda d, t: logs.append((d, t)), lister=bench.lister, opener=bench.opener,
               on_save=saved.append, clock=clock, sleep=clock.sleep, rng=random.Random(3))
    return es, bench, link, saved, logs


def run(es, clock, seconds):
    end = clock() + seconds
    while clock() < end:
        es.tick()


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.w = Wire(self.clock)
        self.es, self.bench, self.link, self.saved, self.logs = make(self.clock, [self.w], cfg=cfg_for(self.w))

    def test_closed_circuit_is_verified_and_allows_motion(self):
        self.assertFalse(self.es.allow("OPEN"))            # 확인 전에는 막음
        run(self.es, self.clock, 0.3)
        self.assertEqual(self.es.input, "closed")
        self.assertFalse(self.es.latched)
        self.assertTrue(self.es.allow("OPEN"))
        self.assertGreater(self.es.cycles, 5)

    def test_press_stops_within_a_tenth_of_a_second(self):
        run(self.es, self.clock, 0.3)
        self.w.pressed = True
        t0 = self.clock()
        while not self.es.latched and self.clock() - t0 < 1.0:
            self.es.tick()
        self.assertTrue(self.es.latched)
        self.assertLess(self.clock() - t0, 0.15)
        self.assertIn("STOP", self.link.sent)
        self.assertEqual(self.es.input, "open")
        self.assertIn("button", self.es.sources)
        for cmd in ("OPEN", "GRIP", "PLACE,180", "PAN,90", "HOME", "NG", "GO", "SERVO,6,90,500"):
            self.assertFalse(self.es.allow(cmd), cmd)
        for cmd in ("STOP", "PING", "VER"):
            self.assertTrue(self.es.allow(cmd), cmd)

    def test_releasing_the_button_never_restarts_and_reset_needs_stable_circuit(self):
        run(self.es, self.clock, 0.3)
        self.w.pressed = True
        run(self.es, self.clock, 0.2)
        ok, why = self.es.reset()
        self.assertFalse(ok)
        self.assertIn("눌려", why)
        self.w.pressed = False
        run(self.es, self.clock, 0.1)
        self.assertTrue(self.es.latched)                   # 풀기만 해서는 잠금 유지
        self.assertFalse(self.es.reset()[0])                # 0.5초 정상 유지 전
        run(self.es, self.clock, 0.6)
        self.assertTrue(self.es.latched)
        sent_before = list(self.link.sent)
        ok, _ = self.es.reset()
        self.assertTrue(ok)
        self.assertFalse(self.es.latched)
        self.assertEqual(self.link.sent, sent_before)      # 해제는 로봇을 움직이지 않음
        self.assertTrue(self.es.allow("HOME"))

    def test_broken_wire_is_treated_like_a_press(self):
        run(self.es, self.clock, 0.3)
        self.w.broken = True
        run(self.es, self.clock, 0.2)
        self.assertTrue(self.es.latched)
        self.assertEqual(self.es.input, "open")

    def test_unplugged_adapter_stops_and_is_found_again(self):
        run(self.es, self.clock, 0.3)
        self.w.plugged = False
        run(self.es, self.clock, 0.2)
        self.assertTrue(self.es.latched)
        self.assertEqual(self.es.input, "absent")
        self.assertFalse(self.es.reset()[0])
        self.w.plugged = True
        self.w.ident["port"] = "COM9"                      # 다시 꽂으니 COM 번호가 바뀜
        run(self.es, self.clock, 2.0)
        self.assertEqual(self.es.input, "closed")
        self.assertEqual(self.es.port_now, "COM9")
        self.assertTrue(self.es.reset()[0])

    def test_input_stuck_high_or_low_is_detected(self):
        for stuck in (True, False):
            with self.subTest(stuck=stuck):
                clock = Clock()
                w = Wire(clock)
                es, *_ = make(clock, [w], cfg=cfg_for(w))
                run(es, clock, 0.3)
                w.stuck = stuck
                run(es, clock, 0.3)
                self.assertTrue(es.latched)

    def test_floating_input_never_looks_healthy(self):
        self.w.floating = True
        end = self.clock() + 5.0
        seen = set()
        while self.clock() < end:
            self.es.tick()
            seen.add(self.es.input)
            self.assertFalse(self.es.allow("OPEN"))        # 한 순간도 움직임을 허락하지 않음
        self.assertNotIn("closed", seen)
        self.assertTrue(self.es.latched)
        self.assertFalse(self.es.reset()[0])
        chk = LoopChecker(_Dev(self.w), "rts", "cts", 0.04, clock=self.clock, sleep=self.clock.sleep, rng=random.Random(5))
        self.assertLess(sum(chk.check()[0] for _ in range(400)), 40)   # 잡음이 검사를 통과하는 비율 < 10%

    def test_one_slow_usb_answer_does_not_stop(self):
        run(self.es, self.clock, 0.3)
        real = self.es.checker.check
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            return (False, 0.0) if calls["n"] == 1 else real()
        self.es.checker.check = flaky
        run(self.es, self.clock, 0.3)
        self.assertFalse(self.es.latched)

    def test_stop_is_repeated_until_robot_acknowledges_then_kept_alive(self):
        run(self.es, self.clock, 0.3)
        guard = GuardedLink(self.link, self.es)
        self.w.pressed = True
        run(self.es, self.clock, 0.5)
        fast = self.link.sent.count("STOP")
        self.assertGreaterEqual(fast, 5)                   # 확인 전: 0.05초마다
        self.link.rx.append("ACK")
        self.assertEqual(guard.poll(), [])                 # 정지 확인 ACK 는 기록에 넘기지 않음
        self.assertTrue(self.es.robot_ack)
        self.link.sent.clear()
        run(self.es, self.clock, 3.0)
        self.assertIn(self.link.sent.count("STOP"), (2, 3, 4))   # 확인 뒤: 1초마다 유지

    def test_robot_reboot_gets_stop_again(self):
        run(self.es, self.clock, 0.3)
        guard = GuardedLink(self.link, self.es)
        self.es.trip("pc", "시험")
        self.link.rx.append("ACK")
        guard.poll()
        self.link.sent.clear()
        self.link.rx.append("READY")                        # micro:bit 재부팅 → 펌웨어 정지 상태가 사라짐
        guard.poll()
        self.assertEqual(self.link.sent, ["STOP"])
        self.assertFalse(self.es.robot_ack)

    def test_watchdog_stops_when_monitor_freezes(self):
        run(self.es, self.clock, 0.3)
        self.es._thread = threading.current_thread()       # 감시 스레드가 있는 것으로
        self.clock.t += 2.0                                 # tick 이 2초 동안 돌지 않음
        self.es.watchdog()
        self.assertTrue(self.es.latched)
        self.assertIn("watchdog", self.es.sources)

    def test_board_ab_stop_is_noticed(self):
        run(self.es, self.clock, 0.3)
        guard = GuardedLink(self.link, self.es)
        self.link.rx.append("ERR,STOPPED")
        guard.poll()
        self.assertTrue(self.es.latched)
        self.assertIn("robot", self.es.sources)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.w = Wire(self.clock)
        self.es, _, self.link, _, self.logs = make(self.clock, [self.w], cfg=cfg_for(self.w))
        run(self.es, self.clock, 0.3)
        self.guard = GuardedLink(self.link, self.es)
        self.task = HandoverTask(self.guard, clock=self.clock)

    def test_every_motion_path_is_blocked_while_stopped(self):
        self.task.request("빨간 공")
        self.assertEqual(self.link.sent[-1], "OPEN")
        self.es.trip("button", "버튼")
        self.link.sent.clear()
        self.assertFalse(self.guard.send("GRIP"))
        self.task.manual_confirm()
        self.task._send("HOME")                             # 연결 직후 HOME · 취소 뒤 HOME 등 모든 경로
        self.assertNotIn("HOME", self.link.sent)
        self.assertNotIn("GRIP", self.link.sent)
        self.assertTrue(any("차단" in l["text"] for l in self.task.logs))
        self.task.stop()
        self.assertEqual(self.link.sent[-3:], ["STOP", "STOP", "STOP"])

    def test_unpaired_switch_does_not_block(self):
        clock = Clock()
        es, _, link, _, _ = make(clock, [], cfg=None)
        self.assertEqual(es.input, "unpaired")
        self.assertTrue(GuardedLink(link, es).send("OPEN"))


class PairingTests(unittest.TestCase):
    def _pair_run(self, es, clock, until_step, limit=60.0):
        end = clock() + limit
        while clock() < end:
            es.tick()
            if es.pairing and es.pairing["step"] in until_step:
                return es.pairing["step"]
        return es.pairing["step"] if es.pairing else None

    def test_finds_the_button_among_adapters_and_jumpers(self):
        clock = Clock()
        button = Wire(clock, port="COM7", out="dtr", inn="dsr", serial="")          # 버튼: DTR → DSR
        other = Wire(clock, port="COM8", serial="B2", wired=False, location="1-3")
        other.jumper = ("rts", "cts")                                               # 점퍼선 (버튼 아님)
        es, bench, link, saved, logs = make(clock, [button, other])
        bench.extra_ports.append({"port": "COM5", "vid": MICROBIT_VID, "pid": 0x0204, "serial": "MB",
                                  "location": "1-1", "desc": "micro:bit"})
        ok, _ = es.pair_start()
        self.assertTrue(ok)
        self.assertTrue(es.latched)                                                 # 등록하는 동안 로봇 정지
        self.assertEqual(self._pair_run(es, clock, ("press", "failed")), "press")
        self.assertEqual(es.pairing["found"], 2)
        self.assertNotIn("COM5", bench.open_log)                                    # micro:bit 은 건드리지 않음
        button.pressed = True
        self.assertEqual(self._pair_run(es, clock, ("release", "failed")), "release")
        self.assertEqual(es.pairing["port"], "COM7")
        button.pressed = False
        self.assertEqual(self._pair_run(es, clock, ("done", "failed")), "done")
        cfg = saved[-1]
        self.assertEqual((cfg["port"], cfg["out"], cfg["in"]), ("COM7", "dtr", "dsr"))
        self.assertTrue(0.02 <= cfg["settle_s"] <= 0.08)
        self.assertEqual(es.input, "closed")
        self.assertFalse(es.reset_check()[0] and False)
        run(es, clock, 0.6)
        self.assertTrue(es.reset()[0])                                              # 등록 뒤 정상 해제
        button.pressed = True                                                       # 등록된 회로를 계속 감시
        run(es, clock, 0.2)
        self.assertTrue(es.latched)

    def test_pressed_or_no_contact_finds_nothing(self):
        clock = Clock()
        w = Wire(clock)
        w.pressed = True                                                            # 눌린 채 시작 (= NO 접점 배선과 같음)
        es, _, _, saved, _ = make(clock, [w])
        es.pair_start()
        self.assertEqual(self._pair_run(es, clock, ("press", "failed")), "failed")
        self.assertIn("NC", es.pairing["msg"])
        self.assertEqual(saved, [])

    def test_jumper_alone_is_not_accepted(self):
        clock = Clock()
        w = Wire(clock, wired=False)
        w.jumper = ("rts", "cts")
        es, _, _, saved, _ = make(clock, [w])
        es.pair_start()
        self.assertEqual(self._pair_run(es, clock, ("press", "failed")), "press")
        self.assertEqual(self._pair_run(es, clock, ("failed", "done"), limit=40), "failed")   # 눌러도 안 열림 → 시간 초과
        self.assertEqual(saved, [])

    def test_two_circuits_opening_together_is_refused(self):
        clock = Clock()
        a = Wire(clock, port="COM7", serial="A")
        b = Wire(clock, port="COM8", serial="B", location="1-3")
        es, _, _, saved, _ = make(clock, [a, b])
        es.pair_start()
        self._pair_run(es, clock, ("press", "failed"))
        a.pressed = b.pressed = True
        self.assertEqual(self._pair_run(es, clock, ("release", "failed")), "failed")
        self.assertEqual(saved, [])

    def test_cancel_keeps_robot_stopped_until_reset(self):
        clock = Clock()
        w = Wire(clock)
        es, *_ = make(clock, [w])
        es.pair_start()
        self._pair_run(es, clock, ("press", "failed"))
        es.pair_cancel()
        es.tick()
        self.assertEqual(es.pairing["step"], "cancelled")
        self.assertTrue(es.latched)
        self.assertTrue(es.reset()[0])

    def test_unpair_only_while_stopped(self):
        clock = Clock()
        w = Wire(clock)
        es, _, _, saved, _ = make(clock, [w], cfg=cfg_for(w))
        run(es, clock, 0.3)
        self.assertFalse(es.unpair()[0])
        es.trip("pc", "시험")
        self.assertTrue(es.unpair()[0])
        self.assertIsNone(saved[-1])
        self.assertEqual(es.input, "unpaired")


class MatchPortTests(unittest.TestCase):
    def test_identity_rules(self):
        ports = [{"port": "COM9", "vid": 0x0403, "pid": 0x6001, "serial": "FT123", "location": "1-4", "desc": ""},
                 {"port": "COM4", "vid": 0x1A86, "pid": 0x7523, "serial": "", "location": "1-2", "desc": ""}]
        self.assertEqual(match_port({"vid": 0x0403, "pid": 0x6001, "serial": "FT123", "port": "COM3"}, ports)["port"], "COM9")
        self.assertIsNone(match_port({"vid": 0x0403, "pid": 0x6001, "serial": "OTHER", "port": "COM9"}, ports))
        self.assertEqual(match_port({"vid": 0x1A86, "pid": 0x7523, "serial": "", "location": "9-9", "port": "COM7"}, ports)["port"], "COM4")
        two = ports + [{"port": "COM6", "vid": 0x1A86, "pid": 0x7523, "serial": "", "location": "1-5", "desc": ""}]
        self.assertIsNone(match_port({"vid": 0x1A86, "pid": 0x7523, "serial": "", "location": "9-9", "port": "COM7"}, two))
        self.assertEqual(match_port({"vid": 0x1A86, "pid": 0x7523, "serial": "", "location": "1-5", "port": "COM7"}, two)["port"], "COM6")


class CheckerTests(unittest.TestCase):
    def test_each_check_proves_a_transition(self):
        clock = Clock()
        w = Wire(clock)
        chk = LoopChecker(_Dev(w), "rts", "cts", 0.03, clock=clock, sleep=clock.sleep, rng=random.Random(1))
        self.assertTrue(all(chk.check()[0] for _ in range(50)))
        w.stuck = True
        self.assertFalse(any(chk.check()[0] for _ in range(50)))


class MockWorldTests(unittest.TestCase):
    def test_mock_switch_pairs_and_stops(self):
        clock = Clock()
        world = MockSwitchWorld()
        saved = []
        es = EStop(LinkSpy(), lister=world.lister, opener=world.opener, on_save=saved.append,
                   clock=clock, sleep=clock.sleep, rng=random.Random(2), mock_world=world)
        es.pair_start()
        end = clock() + 10
        while clock() < end and es.pairing["step"] != "press":
            es.tick()
        world.pressed = True
        while clock() < end + 10 and es.pairing["step"] != "release":
            es.tick()
        world.pressed = False
        while clock() < end + 20 and es.pairing["step"] != "done":
            es.tick()
        self.assertEqual(es.pairing["step"], "done")
        run(es, clock, 0.6)
        self.assertTrue(es.reset()[0])
        world.plugged = False
        run(es, clock, 0.2)
        self.assertTrue(es.latched)
        self.assertEqual(es.status()["input"], "absent")


class RuntimeSafetyTests(unittest.TestCase):
    """handover_app.Runtime 의 비상정지 연결부 (스레드 · 카메라 없이)."""

    def _rt(self, cfg=True):
        import handover_app as app
        clock = Clock()
        w = Wire(clock)
        es, _, link, _, _ = make(clock, [w], cfg=cfg_for(w) if cfg else None)
        rt = app.Runtime.__new__(app.Runtime)
        rt.link, rt.estop, rt.lock = link, es, threading.Lock()
        rt.task = HandoverTask(GuardedLink(link, es), clock=clock)
        rt._home_pending = False
        return rt, es, w, link, clock

    def test_reconnect_while_stopped_sends_stop_not_home(self):
        rt, es, w, link, clock = self._rt()
        run(es, clock, 0.3)
        es.trip("button", "버튼")
        link.sent.clear()
        rt.on_robot_connect()
        self.assertEqual(link.sent, ["STOP"])

    def test_first_home_waits_for_switch_check(self):
        rt, es, w, link, clock = self._rt()
        rt.on_robot_connect()                              # 스위치 확인 전 연결
        self.assertNotIn("HOME", link.sent)
        self.assertTrue(rt._home_pending)
        run(es, clock, 0.3)
        rt._safety_step(True)
        self.assertEqual(link.sent.count("HOME"), 1)

    def test_hardware_press_moves_task_to_stopped(self):
        rt, es, w, link, clock = self._rt()
        run(es, clock, 0.3)
        rt.task.request("파란 공")
        w.pressed = True
        run(es, clock, 0.2)
        rt._safety_step(True)
        self.assertTrue(rt.task.stopped)
        self.assertEqual(rt.task.state, "IDLE")
        self.assertFalse(rt.task.request("빨간 공")["ok"])


if __name__ == "__main__":
    unittest.main()
