"""AiHand 받기 작업 — 웹캠 2대 버전 (쟁반 카메라 + 손 카메라).

handover_app.py (웹캠 1대) 는 그대로 두고, 이 파일만 따로 실행합니다. 기존 파일은 고치지 않습니다.

  웹캠 1 (쟁반)  쟁반 위 볼트·너트 → YOLO 가 요청한 부품을 골라 "PICK THIS" 표시
  웹캠 2 (손)    로봇 손바닥 → YOLO 판정 (점수 낮으면 Gemma 교차 확인) → 쥐기 / NG

  python3 handover_app_2cam.py --hand-camera 1 --tray-camera 2 --vlm          # 실물
  python3 handover_app_2cam.py --mock --hand-camera 0 --tray-camera 사진.jpg   # 웹캠 1대 + 쟁반은 사진으로 시험
  ... --point --point-left 135 --point-right 45     # 고른 부품 쪽으로 손목 돌려 알려 주기 (각도는 보정값)

대시보드 영상: 왼쪽 = 손 카메라, 오른쪽 = 쟁반 카메라 (나란히 한 화면)
맥 내장캠이 0번이라 USB 웹캠은 보통 1, 2번입니다.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time
import webbrowser
import http.server

import cv2
import numpy as np

import handover_app as one
from vision.parts import YoloParts, YOLO_LABELS

ROOT = Path(__file__).resolve().parent
TRAY_MIN_CONF = one.TRAY_MIN_CONF
TRAY_FPS = 3.0                     # 쟁반 YOLO 는 요청 중에만, 초당 3번 (노트북 부담 줄이기)


class _EmptyTray:
    """손 카메라 그림에는 쟁반 표시를 빼기 위한 대리 객체"""

    def __init__(self, task):
        self._task = task
        self.tray = {"items": [], "pick": None}

    def __getattr__(self, name):
        return getattr(self._task, name)


class _HandView:
    def __init__(self, rt):
        self._rt = rt
        self.task = _EmptyTray(rt.task)
        self.tray_roi = None

    def __getattr__(self, name):
        return getattr(self._rt, name)


class Runtime2Cam(one.Runtime):
    def __init__(self, link, hand_camera, tray_camera, model_path=None, conf=0.35, vlm=None):
        self.tray_source = tray_camera            # 웹캠 번호(int) 또는 사진/영상 경로(str)
        self._tray_switch = False
        self.tray_frame = None
        self.tray_error = ""
        self._tray_latest: list[dict] = []
        self._hand_frac = 0.5                     # 합친 화면에서 손 화면이 차지하는 가로 비율 (영역 드래그 보정)
        self.tray_yolo = None
        super().__init__(link, hand_camera, model_path=model_path, conf=conf, vlm=vlm)
        if self.yolo is not None:
            try:
                self.tray_yolo = YoloParts(self.yolo.weights, conf=conf)   # 쟁반 전용 (스레드끼리 같은 모델 공유 안 함)
            except Exception as exc:
                self.tray_error = f"쟁반 YOLO 불러오기 실패: {exc.__class__.__name__}"
        self.task._log("SYS", f"웹캠 2대 모드 — 손 {hand_camera}번 · 쟁반 {tray_camera}")
        threading.Thread(target=self._tray_loop, daemon=True).start()

    # ---- 쟁반 카메라 ----
    def _open_tray(self):
        self._tray_switch = False
        src = self.tray_source
        if isinstance(src, str) and Path(src).suffix.lower() in (".jpg", ".jpeg", ".png"):
            img = cv2.imread(src)
            if img is None:
                self.tray_error = f"쟁반 사진을 읽을 수 없어요: {src}"
            return ("image", img)
        while self.running:
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                self.tray_error = ""
                self.task._log("SYS", f"쟁반 웹캠({src}) 연결됨")
                return ("cap", cap)
            cap.release()
            self.tray_error = f"쟁반 웹캠({src})이 연결되지 않았어요"
            time.sleep(2.0)
        return ("none", None)

    def _tray_loop(self) -> None:
        kind, src = self._open_tray()
        last_infer, fails = 0.0, 0
        while self.running:
            if self._tray_switch:                 # 대시보드에서 쟁반 카메라를 바꿈
                if kind == "cap":
                    src.release()
                self.tray_frame = None
                kind, src = self._open_tray()
                fails = 0
                continue
            if kind == "image":
                frame = src
                time.sleep(0.1)
            elif kind == "cap":
                ok, frame = src.read()
                if not ok:
                    fails += 1
                    if fails > 40:
                        src.release()
                        self.task._log("SYS", "쟁반 웹캠 연결이 끊겼어요 — 다시 연결을 기다려요")
                        kind, src = self._open_tray()
                        fails = 0
                    time.sleep(0.05)
                    continue
                fails = 0
            else:
                return
            if frame is None:
                time.sleep(0.5)
                continue
            self.tray_frame = frame
            active = self.task.target is not None and self.task.state in ("REQUEST", "RECEIVE")
            now = time.monotonic()
            if not active:
                self._tray_latest = []
            elif self.tray_yolo is not None and now - last_infer >= 1.0 / TRAY_FPS:
                last_infer = now
                try:
                    dets = self.tray_yolo.detect(frame)
                except Exception as exc:
                    self.task._log("ERR", f"쟁반 YOLO 오류: {exc}")
                    dets = []
                fw = frame.shape[1]
                self._tray_latest = [{"label": d["label"], "conf": round(float(d["conf"]), 2), "box": tuple(d["box"]),
                                      "xn": round((d["box"][0] + d["box"][2] / 2) / fw, 3)}   # 쟁반 화면 안 가로 위치
                                     for d in dets if d["label"] in YOLO_LABELS and d["conf"] >= TRAY_MIN_CONF]
            with self.lock:                       # 손 카메라가 멈춰 있어도 쟁반 안내는 갱신
                self.task.update_tray(list(self._tray_latest))

    def _tray_items(self, frame, dets) -> list[dict]:
        """(부모의 추론 루프가 호출) 손 카메라가 아니라 쟁반 카메라 결과를 돌려줌"""
        return list(self._tray_latest)

    # ---- 화면: 왼쪽 손 · 오른쪽 쟁반 ----
    def _draw_tray(self, h_target: int):
        tf = self.tray_frame
        if tf is None:
            img = np.full((h_target, int(h_target * 4 / 3), 3), 40, np.uint8)
            cv2.putText(img, "TRAY CAMERA", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            cv2.putText(img, "no signal", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 200), 2)
            return img
        vis = tf.copy()
        h, w = vis.shape[:2]
        k = max(1.0, w / 640)
        th = max(2, int(2 * k))
        tray = self.task.tray
        for d in tray["items"]:
            bx, by, bw, bh = d["box"]
            picked = tray["pick"] == d["n"]
            c = (40, 200, 40) if picked else (150, 150, 150)
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), c, th + (3 if picked else 0))
            cv2.putText(vis, f"#{d['n']} {d['label'].upper()} {d['conf']:.2f}", (bx, by + bh + int(28 * k)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, c, th, cv2.LINE_AA)
            if picked:
                ty = by - int(14 * k) if by > int(50 * k) else by + bh + int(64 * k)   # 위가 좁으면 아래에
                cv2.putText(vis, "PICK THIS", (bx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * k, c, th + 1, cv2.LINE_AA)
        cv2.putText(vis, "TRAY", (int(12 * k), int(36 * k)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * k, (230, 230, 230), th + 1)
        return cv2.resize(vis, (int(w * h_target / h), h_target), interpolation=cv2.INTER_AREA)

    def _draw(self, frame):
        hand = one.Runtime._draw(_HandView(self), frame)            # 손 화면 (원래 그림, 쟁반 표시 제외)
        hh = hand.shape[0]
        k = max(1.0, hand.shape[1] / 640)
        cv2.putText(hand, "HAND", (int(12 * k), int(36 * k)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * k, (230, 230, 230),
                    max(2, int(2 * k)) + 1)
        tray = self._draw_tray(hh)
        self._hand_frac = hand.shape[1] / (hand.shape[1] + tray.shape[1])
        gap = np.full((hh, 8, 3), 255, np.uint8)
        return np.hstack([hand, gap, tray])

    def set_roi(self, x0, y0, x1, y1) -> bool:
        """대시보드에서 드래그한 좌표는 합친 화면 기준 → 손 화면 기준으로 바꿈"""
        f = self._hand_frac
        return super().set_roi(min(1.0, float(x0) / f), y0, min(1.0, float(x1) / f), y1)

    # ---- 카메라 선택 (손 · 쟁반) ----
    def cameras_in_use(self) -> dict:
        used = {self.camera_index: "hand"}
        if isinstance(self.tray_source, int):
            used[self.tray_source] = "tray"
        return used

    def _frame_for_role(self, role: str):
        return self.tray_frame if role == "tray" else self.frame

    def select_camera(self, role: str, index: int) -> dict:
        if role == "hand":
            if index == self.tray_source:
                return {"ok": False, "reply": f"{index}번은 쟁반 카메라로 쓰고 있어요"}
            return super().select_camera(role, index)
        if role == "tray":
            if index == self.camera_index:
                return {"ok": False, "reply": f"{index}번은 손 카메라로 쓰고 있어요"}
            if index == self.tray_source:
                return {"ok": True, "reply": f"이미 {index}번을 쟁반 카메라로 쓰고 있어요"}
            self.tray_source = index
            self._tray_switch = True
            self._save_settings(tray_camera=index)
            self.task._log("SYS", f"쟁반 카메라를 {index}번으로 바꿨어요")
            return {"ok": True, "reply": f"쟁반 카메라를 {index}번으로 바꾸는 중"}
        return {"ok": False, "reply": "알 수 없는 카메라 역할이에요"}

    def state(self) -> dict:
        snap = super().state()
        snap["camera_roles"] = ["hand", "tray"]
        snap["tray_camera"] = {"source": str(self.tray_source), "error": self.tray_error,
                               "running": self.tray_frame is not None}
        return snap


def main() -> None:
    ap = argparse.ArgumentParser(description="AiHand 받기 작업 — 웹캠 2대 (쟁반 + 손)")
    ap.add_argument("--port", default="", help="micro:bit 시리얼 포트 (생략하면 자동 탐색)")
    ap.add_argument("--mock", action="store_true", help="로봇 없이 모의 micro:bit 로 실행")
    ap.add_argument("--hand-camera", type=int, default=None, help="손 웹캠 번호 (생략하면 대시보드에서 마지막으로 고른 번호, 처음엔 1)")
    ap.add_argument("--tray-camera", default=None, help="쟁반 웹캠 번호 또는 시험용 사진 경로 (생략하면 마지막 선택, 처음엔 2)")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "neuralgrip_best_two.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--vlm", action="store_true", help="Gemma: 요청 해석 + 손 위 교차 확인")
    ap.add_argument("--vlm-model", default="gemma3:4b")
    ap.add_argument("--vlm-url", default="http://localhost:11434")
    ap.add_argument("--vlm-verify-below", type=float, default=0.70)
    ap.add_argument("--point", action="store_true", help="쟁반에서 고른 부품 쪽으로 손목을 돌려 알려 주기")
    ap.add_argument("--point-left", type=float, default=135, help="쟁반 화면 왼쪽 끝을 향하는 손목 각도 (보정)")
    ap.add_argument("--point-right", type=float, default=45, help="쟁반 화면 오른쪽 끝을 향하는 손목 각도 (보정)")
    ap.add_argument("--point-hold", type=float, default=1.5, help="가리킨 채 멈추는 시간 (초)")
    ap.add_argument("--http-port", type=int, default=8090)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    saved = one.Runtime.read_settings()
    if a.hand_camera is None:
        a.hand_camera = int(saved.get("camera_index", 1))
    if a.tray_camera is None:
        a.tray_camera = str(saved.get("tray_camera", 2))
    tray_src = int(a.tray_camera) if a.tray_camera.isdigit() else a.tray_camera
    if tray_src == a.hand_camera:
        raise SystemExit("손 카메라와 쟁반 카메라 번호가 같아요. 서로 다른 번호를 주세요.")
    vlm = None
    if a.vlm:
        from vision.vlm_bridge import VlmBridge
        vlm = VlmBridge(a.vlm_model, a.vlm_url, verify_below=a.vlm_verify_below)
    link = one.MockLink() if a.mock else one.RobotLink(port_hint=a.port)
    rt = Runtime2Cam(link, a.hand_camera, tray_src, model_path=a.model, conf=a.conf, vlm=vlm)
    if a.point:
        rt.task.set_pointing(a.point_left, a.point_right, a.point_hold)
        rt.task._log("SYS", f"가리키기 켜짐 — 쟁반 왼쪽 끝 {a.point_left:.0f}° · 오른쪽 끝 {a.point_right:.0f}°")
    if not link.is_mock:
        link.log = rt.task._log
        link.on_connect = rt.on_robot_connect
        link.start()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.http_port), one.make_handler(rt))
    url = f"http://127.0.0.1:{a.http_port}/index.html"
    print(f"[AI-HAND 2CAM] 로봇: {'모의' if link.is_mock else '자동 연결'} · 손 {a.hand_camera}번 · 쟁반 {a.tray_camera} · {url}")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rt.running = False
        link.close()


if __name__ == "__main__":
    main()
