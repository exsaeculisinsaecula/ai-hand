"""받기 작업 상태 머신 (서 있는 AiHand, 팔 없음 — 사람이 손바닥에 부품을 쥐어 준다).

  IDLE ─요청(볼트/너트)→ REQUEST(손 펴는 중) → RECEIVE(YOLO 가 쟁반에서 고른 부품을 사람이 집어 손 영역에 올릴 때까지)
      → 맞는 부품: GRIP(쥐기) → PLACE(손목 회전 · 펴서 내려놓기 · 복귀) → HOME(손이 비었는지 확인) → IDLE
      → 틀린 부품: VERIFY(NG — 손목 흔들기) → RECEIVE(그 부품을 치울 때까지 다시 판정하지 않음)
      → (VLM 켜짐) YOLO 점수가 낮으면: VERIFY(VLM 교차 확인) → 일치: GRIP / 불일치: 작업자 확인 버튼 → GRIP
  (가리키기 켜짐) REQUEST 안에서: 손 펴기 → 쟁반에서 YOLO 가 고른 부품 쪽으로 손목 회전 → 잠깐 멈춤 → 정면 복귀 → RECEIVE

link        : send(cmd) -> bool|None / poll() -> list[str]
frame_result: {"label", "confidence", "box", "motion"}  (손 영역 기준, 카메라가 없으면 None)
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
import math
import time
from typing import Callable

from vision.parts import LABEL_KO, parse_request

# 펌웨어(microbit/handover.py v5, 팀 보정값)와 같은 동작 시간 (초, +80ms 여유)
PAN_S = 1.58     # 손목 회전 1500ms
OPEN_S = 0.78    # 펴기 700ms
GRIP_S = 1.58    # 쥐기 1500ms
NG_SEG_S = 0.34  # NG 흔들기 한 구간 260ms
OPEN_GRIP = 0.1
CLOSED_GRIP = 0.92
PLACE_ANGLE = 110


def _ease(a: float, b: float, t: float) -> float:
    """펌웨어와 같은 코사인 이징 (천천히 출발 → 가속 → 천천히 정지)."""
    t = max(0.0, min(1.0, t))
    return a + (b - a) * (1 - math.cos(math.pi * t)) / 2


STEPS = ("IDLE", "REQUEST", "RECEIVE", "VERIFY", "GRIP", "PLACE", "HOME")
STEP_KO = {"IDLE": "대기", "REQUEST": "요청", "RECEIVE": "받기", "VERIFY": "확인",
           "GRIP": "잡기", "PLACE": "내려놓기", "HOME": "복귀"}
IDLE_MSG = "대기 중 — 받을 부품(볼트 / 너트)을 입력하세요."


class HandoverTask:
    def __init__(self, link, clock: Callable[[], float] = time.monotonic,
                 stable_frames: int = 6, min_conf: float = 0.45, cmd_timeout: float = 10.0,
                 max_move_px: int = 24, motion_still: float = 6.0, place_angle: int = PLACE_ANGLE,
                 empty_check_s: float = 1.5, rearm_frames: int = 5, vlm=None):
        self.link = link
        self.clock = clock
        self.stable_frames = stable_frames
        self.min_conf = min_conf
        self.cmd_timeout = cmd_timeout
        self.max_move_px = max_move_px
        self.motion_still = motion_still        # 손 영역 움직임이 이보다 작아야 "사람 손이 빠짐"
        self.place_angle = place_angle
        self.empty_check_s = empty_check_s
        self.rearm_frames = rearm_frames
        self.vlm = vlm                          # vision.vlm_bridge.VlmBridge (없으면 기존과 똑같이 동작)
        self.frame_source: Callable[[], object] | None = None   # 교차 확인용 현재 카메라 프레임

        self.state = "IDLE"
        self.target: str | None = None
        self.message = IDLE_MSG
        self.tone = "pass"
        self.stopped = False
        self.entered = clock()
        self.last_seen: dict | None = None
        self.parse_reply = ""

        self._awaiting: str | None = None
        self._sent_at = 0.0
        self._prev: dict | None = None
        self._streak = 0
        self._force_verify = False
        self._rearm_wait = False                # NG 뒤: 손 영역이 빌 때까지 판정 보류
        self._empty_frames = 0
        self._check_seen = 0                    # 내려놓기 후 확인: 부품이 보인 프레임 수
        self._check_total = 0
        self._cycle_t0: float | None = None
        self._vlm_job = None                    # 교차 확인 중인 작업 (Future, 확인할 YOLO 결과)
        self._vlm_hold = False                  # VLM 과 YOLO 가 달라 작업자 확인을 기다리는 중
        self.vlm_blocked = 0                    # VLM 이 막은 횟수 (불일치 → 작업자 확인)
        self.tray: dict = {"items": [], "pick": None}   # 쟁반 위 YOLO 탐지 (번호는 왼쪽부터)
        self._tray_logged = None
        # 가리키기: 쟁반 왼쪽 끝/오른쪽 끝을 향하는 손목 각도 (None 이면 끔)
        self.point_cfg: dict | None = None
        self._point_phase: str | None = None    # wait_pick → turning → hold → back
        self._point_t = 0.0
        self._point_angle: int | None = None

        # 실적
        self.requests = 0
        self.done_count = 0
        self.fail_count = 0
        self.reject_count = 0
        self.cycle_times: list[float] = []

        # 실물 자세 (3D 미러링용)
        self._pose_from = {"pan": 90.0, "grip": OPEN_GRIP}
        self._motion: tuple[str, float, float] | None = None

        self.logs: deque[dict] = deque(maxlen=200)
        self._log("SYS", "받기 작업 준비 완료")

    # ---- helpers ----
    def _log(self, d: str, text: str) -> None:
        self.logs.append({"time": datetime.now().strftime("%H:%M:%S"), "dir": d, "text": text})

    def _send(self, cmd: str, wait: bool = True) -> bool:
        if self.link.send(cmd) is False:
            self._log("ERR", f"{cmd} 전송 실패 — 로봇 미연결")
            return False
        self._log("TX", cmd)
        self._start_motion(cmd)
        if wait:
            self._awaiting = cmd.split(",")[0]
            self._sent_at = self.clock()
        return True

    def _enter(self, state: str, msg: str, tone: str = "accent") -> None:
        self.state, self.message, self.tone = state, msg, tone
        self.entered = self.clock()

    def _reset_detection(self) -> None:
        self._prev = None
        self._streak = 0
        self._force_verify = False

    def _finish(self, ok: bool, note: str) -> None:
        name = LABEL_KO.get(self.target or "", "부품")
        if ok:
            self.done_count += 1
            if self._cycle_t0 is not None:
                self.cycle_times.append(self.clock() - self._cycle_t0)
            self._log("SYS", f"{name} 전달 완료 {note}".strip())
            self._enter("IDLE", f"{name} 전달 완료 — 다음 부품을 입력하세요", "pass")
        else:
            self.fail_count += 1
            self._log("ERR", f"{name} 전달 실패 — {note}")
            self._enter("IDLE", f"{name}이(가) 손에 남아 있어요 — 꺼낸 뒤 다시 요청하세요", "warn")
        self.target = None
        self._cycle_t0 = None

    # ---- pose mirror ----
    def _start_motion(self, cmd: str) -> None:
        parts = cmd.split(",")
        head = parts[0]
        if head not in ("OPEN", "GRIP", "PAN", "PLACE", "HOME", "NG", "SERVO"):
            return
        arg = 90.0
        if head == "PAN" and len(parts) > 1:
            arg = float(parts[1])
        elif head == "PLACE":
            arg = float(parts[1]) if len(parts) > 1 else float(PLACE_ANGLE)
        elif head == "SERVO":
            if len(parts) > 2 and parts[1] == "6":
                head, arg = "PAN", float(parts[2])
            else:
                return
        self._pose_from = self.pose()
        self._motion = (head, self.clock(), arg)

    def pose(self) -> dict:
        p0 = self._pose_from
        if not self._motion:
            return dict(p0)
        head, t0, arg = self._motion
        t = self.clock() - t0
        pan, grip = p0["pan"], p0["grip"]
        if head == "OPEN":
            grip = _ease(p0["grip"], OPEN_GRIP, t / OPEN_S)
        elif head == "GRIP":
            grip = _ease(p0["grip"], CLOSED_GRIP, t / GRIP_S)
        elif head == "PAN":
            pan = _ease(p0["pan"], arg, t / PAN_S)
        elif head == "HOME":
            pan = _ease(p0["pan"], 90.0, t / PAN_S)
            grip = _ease(p0["grip"], OPEN_GRIP, t / OPEN_S)
        elif head == "NG":
            seq = [70.0, 110.0, 70.0, 110.0, 90.0]
            k = int(t // NG_SEG_S)
            if k >= len(seq):
                pan = 90.0
            else:
                frm = p0["pan"] if k == 0 else seq[k - 1]
                pan = _ease(frm, seq[k], (t - k * NG_SEG_S) / NG_SEG_S)
        elif head == "PLACE":
            # 손목 회전 → 펴서 내려놓기 → 0.3초 대기 → 정면 복귀
            t1 = PAN_S
            t3 = PAN_S + OPEN_S + 0.3
            if t < t1:
                pan = _ease(p0["pan"], arg, t / PAN_S)
            elif t < t3:
                pan, grip = arg, _ease(p0["grip"], OPEN_GRIP, (t - t1) / OPEN_S)
            else:
                pan, grip = _ease(arg, 90.0, (t - t3) / PAN_S), OPEN_GRIP
        return {"pan": round(pan, 1), "grip": round(grip, 3)}

    # ---- operator actions ----
    def request(self, text: str, pre: tuple | None = None) -> dict:
        """pre: 미리 해석한 결과 (target, reply, info). VLM 해석은 느릴 수 있어 화면 잠금 밖에서 먼저 함."""
        if pre is not None:
            target, reply, info = pre
        elif self.vlm is not None:
            target, reply, info = self.vlm.interpret(text)
        else:
            (target, reply), info = parse_request(text), {"method": "규칙"}
        self.parse_reply = reply
        how = info.get("method", "규칙") + (f" {info['seconds']}s" if info.get("seconds") else "")
        self._log("SYS", f"[해석] '{text}' → {LABEL_KO[target] if target else '판단 불가'} ({how})"
                  + (f" — {info['reason']}" if info.get("method", "").startswith("VLM") and info.get("reason") else ""))
        if target is None:
            return {"ok": False, "reply": reply}
        if self.stopped:
            return {"ok": False, "reply": "비상 정지 중이에요. 대기 복귀 후 다시 요청하세요."}
        if self.state != "IDLE" or self._awaiting:
            return {"ok": False, "reply": "지금 다른 작업 중이에요."}
        self.target = target
        self.requests += 1
        self._cycle_t0 = self.clock()
        self._reset_detection()
        self._rearm_wait = False
        self._enter("REQUEST", f"{LABEL_KO[target]} 요청 — 손을 펴는 중")
        self._send("OPEN")
        return {"ok": True, "reply": reply, "target": target}

    def set_pointing(self, left_angle: float, right_angle: float, hold_s: float = 1.5, wait_s: float = 2.0) -> None:
        """쟁반에서 고른 부품 쪽으로 손목을 돌려 '알아봤다'를 보여 주기. 쟁반 왼쪽 끝·오른쪽 끝 각도로 보정."""
        self.point_cfg = {"left": float(left_angle), "right": float(right_angle), "hold": hold_s, "wait": wait_s}

    def _point_reset(self) -> None:
        self._point_phase, self._point_angle = None, None

    def _point_step(self, now: float) -> None:
        """REQUEST 상태에서 가리키기 진행"""
        name = LABEL_KO[self.target]
        if self._point_phase == "wait_pick":
            pick = next((d for d in self.tray["items"] if d["n"] == self.tray["pick"]), None)
            if pick is not None and pick.get("xn") is not None:
                c = self.point_cfg
                ang = int(round(max(0.0, min(180.0, c["left"] + (c["right"] - c["left"]) * pick["xn"]))))
                self._point_phase, self._point_angle = "turning", ang
                self._log("SYS", f"[가리키기] 쟁반 {pick['n']}번 {name} → 손목 {ang}°")
                self.message = f"쟁반 {pick['n']}번({name}) 쪽을 가리키는 중"
                self._send(f"PAN,{ang}")
            elif now - self._point_t > self.point_cfg["wait"]:
                self._log("SYS", f"[가리키기] 쟁반에서 {name}을(를) 못 찾아 건너뜀")
                self._point_reset()
                self._enter_receive()
        elif self._point_phase == "hold" and now >= self._point_t:
            self._point_phase = "back"
            self._send("PAN,90")

    def _enter_receive(self) -> None:
        self._enter("RECEIVE", f"손을 폈어요 — {LABEL_KO[self.target]}을(를) 손 영역에 올려 주세요. 닿으면 쥐어요")

    def manual_confirm(self) -> None:
        """'쥐어줬어요' / micro:bit 버튼 A: 정지·움직임 조건을 건너뛰고 다음 탐지로 바로 판정."""
        if self.state == "VERIFY" and self._vlm_hold:
            self._vlm_hold = False
            self._log("SYS", f"작업자 확인 — YOLO 판단대로 {LABEL_KO[self.target]} 쥐기")
            self._enter("GRIP", f"작업자 확인 — {LABEL_KO[self.target]} 쥐는 중", "pass")
            self._send("GRIP")
            return
        if self.state == "RECEIVE":
            self._force_verify = True
            self._rearm_wait = False
            self._log("SYS", "작업자 확인 — 다음 탐지로 바로 판정합니다")
            self.message = "작업자 확인 — 손 영역에서 부품을 찾는 중이에요 (안 보이면 쥐지 않아요)"

    def confirm_without_camera(self) -> None:
        """카메라가 없을 때: 작업자 확인만으로 쥐기."""
        if self.state == "RECEIVE" and self.target:
            self._log("SYS", f"카메라 없음 — 작업자 확인으로 {LABEL_KO[self.target]} 쥐기 진행")
            self._enter("GRIP", f"작업자 확인 — {LABEL_KO[self.target]} 쥐는 중", "pass")
            self._send("GRIP")

    def cancel(self) -> None:
        self._vlm_job, self._vlm_hold = None, False
        self._point_reset()
        if self.state in ("REQUEST", "RECEIVE", "VERIFY"):
            self._log("SYS", "요청 취소")
            self.target = None
            self._cycle_t0 = None
            self._send("HOME")
            self._enter("HOME", "취소 — 손을 펴고 정면으로 복귀 중", "pass")

    def stop(self) -> None:
        self.stopped = True
        self._vlm_job, self._vlm_hold = None, False
        self._point_reset()
        self._pose_from = self.pose()
        self._motion = None
        for _ in range(3):                       # 이동 중 펌웨어가 USB 를 잠깐씩만 보므로 여러 번
            self.link.send("STOP")
        self._log("ERR", "비상 정지(STOP)")
        self._awaiting = None
        if self.target and self.state not in ("IDLE",):
            self.fail_count += 1
        self._enter("IDLE", "비상 정지 — 현재 자세 유지. 확인 후 대기 복귀하세요.", "warn")
        self.target = None
        self._cycle_t0 = None

    def link_lost(self) -> None:
        self._awaiting = None
        self._pose_from = self.pose()
        self._motion = None
        if self.state != "IDLE":
            self._log("ERR", f"{STEP_KO[self.state]} 단계에서 로봇 연결이 끊겨 작업을 중단했어요")
            if self.target:
                self.fail_count += 1
        self._enter("IDLE", "로봇 연결이 끊겼어요 — USB · 전원을 확인하세요", "warn")
        self.target = None
        self._cycle_t0 = None

    def resume(self) -> None:
        self.stopped = False
        self._send("HOME")
        self._enter("HOME", "대기 복귀 중", "pass")

    # ---- periodic ----
    def step(self, frame_result: dict | None) -> None:
        now = self.clock()
        for raw in self.link.poll():
            self._on_rx(raw.strip())
        if self._awaiting and now - self._sent_at > self.cmd_timeout:
            self._log("ERR", f"{self._awaiting} 응답 없음 — 정지")
            self.stop()
            return
        if frame_result is not None:
            self.last_seen = frame_result
        if self.state == "REQUEST" and self._point_phase in ("wait_pick", "hold") and not self._awaiting:
            self._point_step(now)
        elif self.state == "VERIFY" and self._vlm_job is not None:
            self._vlm_result()
        elif self.state == "RECEIVE" and frame_result is not None:
            self._receive(frame_result)
        elif self.state == "HOME" and self._check_total >= 0 and self.target:
            self._empty_check(frame_result, now)

    def _receive(self, res: dict) -> None:
        label = res.get("label")
        conf = res.get("confidence", 0.0)
        seen = label not in (None, "none") and conf >= self.min_conf
        if self._rearm_wait:
            self._empty_frames = self._empty_frames + 1 if not seen else 0
            if self._empty_frames >= self.rearm_frames:
                self._rearm_wait = False
                self._reset_detection()
                self.message = f"{LABEL_KO[self.target]}을(를) 손 영역에 올려 주세요. 닿으면 쥐어요"
                self.tone = "accent"
            return
        if self._force_verify and seen:
            self._reset_detection()
            self._verify(res)
            return
        prev = self._prev
        still_box = bool(prev and prev.get("label") == label and prev.get("box") and res.get("box")
                         and abs(prev["box"][0] - res["box"][0]) <= self.max_move_px
                         and abs(prev["box"][1] - res["box"][1]) <= self.max_move_px)
        hand_clear = res.get("motion", 0.0) <= self.motion_still
        if seen and hand_clear and (still_box or self._streak == 0):
            self._streak += 1
        else:
            self._streak = 1 if seen and hand_clear else 0
        self._prev = res
        if seen and not hand_clear:
            self.message = "손 영역에 움직임이 있어요 — 부품을 올린 뒤 손을 빼 주세요"
        if self._streak >= self.stable_frames:
            self._reset_detection()
            self._verify(res)

    def _verify(self, res: dict) -> None:
        label = res["label"]
        self._log("SYS", f"[닿음] 손 영역에 {LABEL_KO[label]} ({res.get('confidence', 0):.2f}) — {res.get('reason', '')}")
        if label == self.target:
            conf = float(res.get("confidence", 0.0))
            if self.vlm is not None and self.vlm.should_check(conf) and self._start_vlm_check(res):
                return
            self._enter("GRIP", f"{LABEL_KO[label]} 확인 — 쥐는 중", "pass")
            self._send("GRIP")
        else:
            self.reject_count += 1
            self._enter("VERIFY", f"손에 놓인 건 {LABEL_KO[label]}이에요 — {LABEL_KO[self.target]}을(를) 주세요", "warn")
            self._send("NG")

    # ---- 쟁반 안내: YOLO 가 쟁반 위 부품 중 요청한 것을 골라 사람에게 알려 줌 ----
    def update_tray(self, items: list[dict]) -> None:
        """items: [{"label", "conf", "box"}] (손 영역 밖 탐지). 번호를 붙이고 요청 부품을 고른다."""
        items = sorted(items, key=lambda d: d["box"][0] + d["box"][2] / 2)
        for i, d in enumerate(items, 1):
            d["n"] = i
        pick = None
        if self.target and self.state in ("REQUEST", "RECEIVE"):
            same = [d for d in items if d["label"] == self.target]
            pick = max(same, key=lambda d: d["conf"])["n"] if same else None
        self.tray = {"items": items, "pick": pick}
        if not self.target or self.state not in ("REQUEST", "RECEIVE"):
            self._tray_logged = None
            return
        key = (self.target, pick, len(items))
        if key != self._tray_logged:                    # 판단이 바뀔 때만 기록 (대시보드 로그)
            self._tray_logged = key
            seen = ", ".join(f"{d['n']}번 {LABEL_KO.get(d['label'], d['label'])} {d['conf']:.2f}" for d in items) or "없음"
            res = f"{LABEL_KO[self.target]}은(는) {pick}번" if pick else f"{LABEL_KO[self.target]} 없음"
            self._log("SYS", f"[쟁반 YOLO] {seen} → {res}")
        if self.state == "RECEIVE" and not self._rearm_wait and self._streak == 0:
            name = LABEL_KO[self.target]
            self.message = (f"쟁반 {pick}번({name})을 집어 손 영역에 올려 주세요" if pick else
                            f"쟁반에서 {name}이(가) 안 보여요 — {name}을(를) 손 영역에 올려 주세요")

    # ---- VLM 교차 확인 ----
    def _start_vlm_check(self, res: dict) -> bool:
        frame = self.frame_source() if self.frame_source else None
        img = self.vlm.crop(frame, res.get("box"))
        if img is None:
            return False                                 # 사진이 없으면 YOLO 판단대로
        self._vlm_job = (self.vlm.submit_check(img), res)
        self._enter("VERIFY", f"YOLO {LABEL_KO[res['label']]} {res.get('confidence', 0):.2f} — "
                              f"점수가 낮아 VLM이 한 번 더 확인하는 중", "accent")
        return True

    def _vlm_result(self) -> None:
        fut, res = self._vlm_job
        if not fut.done():
            return
        self._vlm_job = None
        try:
            v = fut.result()
        except Exception as exc:
            v = {"label": None, "seen": str(exc), "seconds": 0, "ok": False}
        yolo = res["label"]
        st = self.vlm.stats
        if v["label"] == yolo:
            st["check_agree"] += 1
            self._log("SYS", f"[VLM 확인] {LABEL_KO[yolo]} 일치 ({v['seconds']}s) — {v['seen']}")
            self._enter("GRIP", f"{LABEL_KO[yolo]} 확인 (YOLO·VLM 일치) — 쥐는 중", "pass")
            self._send("GRIP")
            return
        if v["ok"]:
            st["check_disagree"] += 1
        else:
            st["check_fail"] += 1
        self.vlm_blocked += 1
        vl = LABEL_KO.get(v["label"], "모름") if v["label"] else "모름"
        self._log("SYS", f"[VLM 확인] YOLO {LABEL_KO[yolo]} / VLM {vl} ({v['seconds']}s) — 작업자 확인 대기")
        self._vlm_hold = True
        self._enter("VERIFY", f"YOLO는 {LABEL_KO[yolo]}, VLM은 {vl} — 맞으면 [쥐어줬어요] 버튼, 아니면 취소", "warn")

    def _empty_check(self, res: dict | None, now: float) -> None:
        if res is not None:
            self._check_total += 1
            if res.get("label") not in (None, "none") and res.get("confidence", 0) >= self.min_conf:
                self._check_seen += 1
        if now - self.entered < self.empty_check_s:
            return
        if self._check_total == 0:
            self._finish(True, "(카메라 확인 없음)")
        elif self._check_seen >= max(2, self._check_total // 2):
            self._finish(False, "내려놓은 뒤에도 손 영역에 부품이 보여요")
        else:
            self._finish(True, "· 손 비었음 확인")

    def _on_rx(self, raw: str) -> None:
        if not raw:
            return
        self._log("ERR" if raw.startswith("ERR") else "RX", raw)
        if raw == "EVT,BTN_A":
            self.manual_confirm()
            return
        if raw.startswith("ERR"):
            self._awaiting = None
            if raw == "ERR,ABORTED" and self.stopped:
                return                                  # 정지로 인한 중단 — 이미 처리됨
            if raw == "ERR,STOPPED" and self.stopped:
                return
            if self.target:
                self.fail_count += 1
            self._enter("IDLE", f"로봇 오류({raw}) — 대기 복귀하세요", "warn")
            self.target = None
            self._cycle_t0 = None
            return
        if not raw.startswith("DONE,"):
            return
        what = raw.split(",", 1)[1]
        self._awaiting = None
        if what == "OPEN" and self.state == "REQUEST":
            if self.point_cfg and self.target:
                self._point_phase, self._point_t = "wait_pick", self.clock()
                self.message = f"쟁반에서 {LABEL_KO[self.target]}을(를) 찾는 중"
            else:
                self._enter_receive()
        elif what == "PAN" and self.state == "REQUEST" and self._point_phase == "turning":
            self._point_phase, self._point_t = "hold", self.clock() + self.point_cfg["hold"]
        elif what == "PAN" and self.state == "REQUEST" and self._point_phase == "back":
            self._point_reset()
            self._enter_receive()
        elif what == "NG" and self.state == "VERIFY":
            self._rearm_wait = True
            self._empty_frames = 0
            self._enter("RECEIVE", f"그 부품을 치우고 {LABEL_KO[self.target]}을(를) 올려 주세요", "warn")
        elif what == "GRIP" and self.state == "GRIP":
            self._enter("PLACE", f"손목을 {self.place_angle}°로 돌려 내려놓는 중", "pass")
            self._send(f"PLACE,{self.place_angle}")
        elif what == "PLACE" and self.state == "PLACE":
            self._check_seen = 0
            self._check_total = 0
            self._enter("HOME", "내려놓기 완료 — 손이 비었는지 확인 중", "pass")
        elif what == "HOME":
            if self.state == "HOME" and self.target:
                return                                  # (확인 중) 무시
            self._enter("IDLE", IDLE_MSG, "pass")
            self.target = None

    def stats(self) -> dict:
        finished = self.done_count + self.fail_count
        return {
            "requests": self.requests,
            "done": self.done_count,
            "failed": self.fail_count,
            "rejected": self.reject_count,
            "success_rate": round(100 * self.done_count / finished) if finished else None,
            "vlm_blocked": self.vlm_blocked,
            "vlm": dict(self.vlm.stats) if self.vlm is not None else None,
            "avg_cycle_s": round(sum(self.cycle_times) / len(self.cycle_times), 1) if self.cycle_times else None,
        }

    def snapshot(self) -> dict:
        return {
            "mode": "handover",
            "state": self.state,
            "state_ko": STEP_KO[self.state],
            "steps": [{"key": s, "ko": STEP_KO[s]} for s in STEPS],
            "target": self.target,
            "target_ko": LABEL_KO.get(self.target, "") if self.target else "",
            "message": self.message,
            "tone": "warn" if self.stopped else self.tone,
            "stopped": self.stopped,
            "elapsed": round(self.clock() - self.entered, 1),
            "parse_reply": self.parse_reply,
            "seen": self.last_seen,
            "pose": self.pose(),
            "moving": bool(self._motion and self._awaiting),
            "place_angle": self.place_angle,
            "rearm_wait": self._rearm_wait,
            "vlm_state": "checking" if self._vlm_job is not None else ("hold" if self._vlm_hold else ""),
            "point": {"phase": self._point_phase, "angle": self._point_angle} if self.point_cfg else None,
            "tray": {"items": [{k: v for k, v in d.items()} for d in self.tray["items"]], "pick": self.tray["pick"]},
            "stats": self.stats(),
            "done": self.done_count,
            "rejected": self.reject_count,
            "logs": list(self.logs),
        }
