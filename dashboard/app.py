"""AI-HAND 부품 전달 제어 대시보드 (Streamlit)."""
from __future__ import annotations

import base64
import html
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components

from dashboard.catalog import PARTS, by_id, default_inventory
from config import YOLO_CLASSES_PATH, YOLO_MODEL_PATH
from dashboard.hand import FakeHand, MicrobitHand, VisionHand
from dashboard.logcsv import DEFAULT_LOG, HEADER, clear_log, read_all, summarize_log
from dashboard.pipeline import Pipeline, idle_snapshot
from gesture.webcam import analyze_image_bytes
from link.mock_microbit import MockMicrobitLink
from link.serial_link import SerialMicrobitLink, find_microbit_port
from vision.detector import YoloDetector, analyze_image_bytes_yolo

STATE_KO = {
    "IDLE": "대기",
    "LOOK": "인식",
    "SELECT": "선택",
    "VERIFY": "대조",
    "PICK": "집기",
    "DELIVER": "전달",
    "DONE": "완료",
    "FAIL": "실패",
}

YOLO_FRAME_SOURCES = ("카메라 1프레임 캡처", "업로드한 트레이 이미지")

MODE_OPTIONS = (
    "가상 파이프라인 (FakeHand)",
    "모의 micro:bit V2 (프로토콜 시뮬레이션)",
    "실물 USB 시리얼 (micro:bit V2)",
)

CSS = """
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
#MainMenu,
footer,
div[data-testid="stStatusWidget"] {
  display: none;
}
.stApp, [data-testid="stAppViewContainer"] {
  background: #f4f1ea;
  color: #1c1c1c;
}
.block-container {
  max-width: 1420px;
  padding-top: 16px;
  padding-bottom: 44px;
}

/* Header Bar */
.wb-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 16px;
  background: #faf8f4;
  border: 1px solid #1c1c1c;
  margin-bottom: 14px;
}
.wb-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}
.wb-tag {
  background: #1c1c1c;
  color: #f4f1ea;
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 12px;
  font-weight: 600;
  padding: 4px 10px;
  letter-spacing: 0.06em;
}
.wb-title {
  font-size: 18px;
  font-weight: 600;
  color: #1c1c1c;
  margin: 0;
}
.wb-sub {
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 11px;
  color: #6b665e;
}
.wb-badges {
  display: flex;
  align-items: center;
  gap: 8px;
}
.pill {
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 11px;
  font-weight: 600;
  padding: 4px 10px;
  border: 1px solid #1c1c1c;
  background: #eae5dc;
  color: #1c1c1c;
}
.pill-idle { background: #eae5dc; color: #1c1c1c; }
.pill-active { background: #ffedd5; color: #9a3412; border-color: #9a3412; }
.pill-done { background: #dcfce7; color: #15803d; border-color: #15803d; }
.pill-fail { background: #fee2e2; color: #b91c1c; border-color: #b91c1c; }

/* KPI Strip */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
@media (max-width: 780px) {
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.kpi-card {
  background: #faf8f4;
  border: 1px solid #d5cfc2;
  padding: 10px 14px;
}
.kpi-label {
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 11px;
  color: #6b665e;
  margin-bottom: 4px;
}
.kpi-value {
  font-size: 20px;
  font-weight: 600;
  color: #1c1c1c;
  line-height: 1.2;
}
.kpi-sub {
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 11px;
  color: #78716c;
  margin-top: 3px;
}

/* Section Cards */
.sec-label {
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: #57534e;
  margin: 4px 0 6px;
}
.cam-cap {
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 11px;
  color: #57534e;
  margin: 0 0 4px;
}

/* Stepper */
.stepper {
  display: flex;
  align-items: stretch;
  gap: 4px;
  margin: 6px 0 10px;
}
.step-node {
  flex: 1;
  padding: 6px 6px;
  border: 1px solid #cfc8ba;
  background: #eae5dc;
  text-align: center;
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 11px;
  color: #6b665e;
}
.step-node.done {
  background: #faf8f4;
  border-color: #1c1c1c;
  color: #1c1c1c;
}
.step-node.active {
  background: #9a3412;
  border-color: #9a3412;
  color: #faf8f4;
  font-weight: 600;
}
.step-node.success {
  background: #15803d;
  border-color: #15803d;
  color: #faf8f4;
  font-weight: 600;
}
.step-node.fail {
  background: #b91c1c;
  border-color: #b91c1c;
  color: #faf8f4;
  font-weight: 600;
}

/* Verdict Panel */
.verdict-box {
  background: #faf8f4;
  border: 1px solid #1c1c1c;
  padding: 10px 14px;
  margin-top: 10px;
}
.verdict-row {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  align-items: center;
}
.verdict-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 12px;
}
.verdict-note {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #e2ddd2;
  font-size: 13px;
  color: #292524;
}

/* Part Card */
.part-card {
  background: #faf8f4;
  border: 1px solid #d5cfc2;
  padding: 8px 10px;
  margin-bottom: 6px;
}
.part-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  font-weight: 600;
}
.part-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin-right: 6px;
  border: 1px solid #1c1c1c;
}
.part-spec {
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 11px;
  color: #6b665e;
  margin-top: 2px;
}

/* Protocol Trace */
pre.proto {
  background: #1c1c1c;
  color: #e7e5e4;
  padding: 10px 12px;
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 11px;
  line-height: 1.5;
  margin: 0;
  overflow-x: auto;
}

/* Streamlit Buttons & Inputs */
div.stButton > button, div.stFormSubmitButton > button, div.stDownloadButton > button {
  height: 36px;
  min-height: 36px;
  border-radius: 0 !important;
  border: 1px solid #1c1c1c !important;
  background: #faf8f4 !important;
  color: #1c1c1c !important;
  box-shadow: none !important;
  font-family: "Segoe UI", "Malgun Gothic", sans-serif;
  font-size: 13px;
  font-weight: 500;
  padding: 0 12px;
}
div.stButton > button:hover, div.stFormSubmitButton > button:hover, div.stDownloadButton > button:hover {
  background: #1c1c1c !important;
  color: #f4f1ea !important;
}
div.stButton > button:focus,
div.stButton > button:focus-visible,
div.stButton > button:active {
  box-shadow: none !important;
  outline: none;
}
div[data-testid="stTextInput"] input {
  border-radius: 0;
  border: 1px solid #1c1c1c;
  background: #faf8f4;
  color: #1c1c1c;
  height: 36px;
}
div[data-testid="stTextInput"] input:focus {
  border-color: #9a3412;
  box-shadow: none;
}
div[data-testid="stImage"] img {
  border: 1px solid #1c1c1c;
  border-radius: 0;
}
div[data-testid="stTabs"] button[role="tab"] {
  border-radius: 0 !important;
  font-size: 13px;
  font-weight: 500;
}
"""


def _word_match(value: str) -> tuple[str, str]:
    if value == "yes":
        return "일치", "pill-done"
    if value == "no":
        return "불일치", "pill-fail"
    return "—", "pill-idle"


def _word_ok(value: str) -> tuple[str, str]:
    if value == "yes":
        return "성공", "pill-done"
    if value == "no":
        return "실패", "pill-fail"
    return "—", "pill-idle"


def _state_pill_class(state: str) -> str:
    if state == "DONE":
        return "pill-done"
    if state == "FAIL":
        return "pill-fail"
    if state == "IDLE":
        return "pill-idle"
    return "pill-active"


def _init_state() -> None:
    st.session_state.setdefault("snaps", None)
    st.session_state.setdefault("idx", 0)
    st.session_state.setdefault("inventory", default_inventory())
    st.session_state.setdefault("mode", MODE_OPTIONS[0])
    st.session_state.setdefault("serial_port", find_microbit_port() or "COM3")
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("gesture_result", None)
    st.session_state.setdefault("yolo_model_path", str(YOLO_MODEL_PATH))
    st.session_state.setdefault("yolo_conf", 0.5)
    st.session_state.setdefault("yolo_verify", False)
    st.session_state.setdefault("yolo_frame_src", YOLO_FRAME_SOURCES[0])
    st.session_state.setdefault("yolo_camera", 0)
    st.session_state.setdefault("yolo_tray_bytes", None)
    st.session_state.setdefault("yolo_test_result", None)


@st.cache_resource(show_spinner="YOLO 모델 로드 중...")
def _load_yolo(model_path: str, conf: float) -> YoloDetector:
    det = YoloDetector(model_path=Path(model_path), classes_path=YOLO_CLASSES_PATH, conf=conf)
    det.load()
    return det


def _yolo() -> YoloDetector:
    return _load_yolo(st.session_state.yolo_model_path.strip(), float(st.session_state.yolo_conf))


def _grab_yolo_frame():
    """VERIFY 대조용 BGR 프레임: 업로드한 트레이 이미지 또는 카메라 1프레임."""
    import cv2
    import numpy as np

    if st.session_state.yolo_frame_src == YOLO_FRAME_SOURCES[1]:
        raw = st.session_state.get("yolo_tray_bytes")
        if not raw:
            return None
        return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    cap = cv2.VideoCapture(int(st.session_state.yolo_camera))
    try:
        frame = None
        for _ in range(5):  # 자동 노출 안정화
            ok, img = cap.read()
            if ok:
                frame = img
        return frame
    finally:
        cap.release()


def _build_hand() -> tuple[object, object | None]:
    """선택된 연동 모드에 맞는 Hand 인스턴스와 닫아야 할 링크를 반환한다."""
    mode = st.session_state.get("mode", MODE_OPTIONS[0])
    wrong = bool(st.session_state.get("force_wrong", False))
    pick_fail = bool(st.session_state.get("force_pick_fail", False))
    deliver_fail = bool(st.session_state.get("force_deliver_fail", False))

    link = None
    if mode == MODE_OPTIONS[1]:
        link = MockMicrobitLink()
        hand = MicrobitHand(link, force_wrong=wrong, force_pick_fail=pick_fail, force_deliver_fail=deliver_fail)
    elif mode == MODE_OPTIONS[2]:
        port = st.session_state.get("serial_port", "").strip() or None
        link = SerialMicrobitLink(port=port)
        hand = MicrobitHand(link, force_wrong=wrong, force_pick_fail=pick_fail, force_deliver_fail=deliver_fail)
    else:
        hand = FakeHand(wrong, pick_fail, deliver_fail)

    if st.session_state.get("yolo_verify"):
        det = _yolo()
        if det.available and det.has_part_classes:
            hand = VisionHand(hand, det, _grab_yolo_frame)
    return hand, link


def _execute_pipeline(cmd_text: str, *, source: str = "text") -> None:
    st.session_state.last_error = ""
    link_to_close = None
    try:
        hand, link_to_close = _build_hand()
        snaps = Pipeline(DEFAULT_LOG).run(
            cmd_text,
            inventory=st.session_state.inventory,
            hand=hand,
            source=source,
        )
        st.session_state.snaps = snaps
        st.session_state.idx = len(snaps) - 1
        if snaps and snaps[-1].stock_after is not None:
            st.session_state.inventory = dict(snaps[-1].stock_after)
    except Exception as exc:
        st.session_state.last_error = f"실행 오류: {exc}"
    finally:
        if link_to_close is not None:
            link_to_close.close()


def _reset_to_idle() -> None:
    st.session_state.snaps = None
    st.session_state.idx = 0
    st.session_state.last_error = ""
    st.session_state.gesture_result = None


def _render_header(view) -> None:
    state_ko = STATE_KO.get(view.state, view.state)
    pill_cls = _state_pill_class(view.state)
    mode_short = {
        MODE_OPTIONS[0]: "SIM · FAKE",
        MODE_OPTIONS[1]: "MOCK · 115200bps",
        MODE_OPTIONS[2]: f"USB · {st.session_state.get('serial_port', 'AUTO')}",
    }.get(st.session_state.get("mode", MODE_OPTIONS[0]), "SIM")

    c_head, c_reset = st.columns([9, 1.4])
    with c_head:
        st.markdown(
            f"""
            <div class="wb-header">
              <div class="wb-brand">
                <span class="wb-tag">AI-HAND</span>
                <div>
                  <div class="wb-title">부품 전달 제어 워크벤치</div>
                  <div class="wb-sub">단일 AiHand × 경사 슈트 전달 파이프라인</div>
                </div>
              </div>
              <div class="wb-badges">
                <span class="pill">{html.escape(mode_short)}</span>
                <span class="pill {pill_cls}">상태: {html.escape(state_ko)} ({html.escape(view.state)})</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_reset:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("대기 복귀", key="reset_idle", width="stretch"):
            _reset_to_idle()
            st.rerun()


def _render_kpis(view, stats: dict) -> None:
    inv = st.session_state.inventory
    total_stock = sum(inv.values())
    max_stock = sum(p.default_stock for p in PARTS)
    if view.part is not None:
        p = by_id(view.part)
        target_text = f"#{p.id} {p.name}"
        target_sub = f"{p.code} · 재고 {inv.get(p.id, 0)}개"
    else:
        target_text = "—"
        target_sub = "선택된 부품 없음"

    last_res = STATE_KO.get(view.state, view.state)
    last_sub = view.detail if view.detail else ("정상 처리" if view.state == "DONE" else "대기 중")

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-label">총 실행 횟수</div>
            <div class="kpi-value">{stats['total_runs']}건</div>
            <div class="kpi-sub">완료 {stats['done_runs']} · 실패 {stats['fail_runs']}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">전달 성공률</div>
            <div class="kpi-value">{stats['success_rate']:.1f}%</div>
            <div class="kpi-sub">CSV 누적 로그 기준</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">대상 부품 / 결과</div>
            <div class="kpi-value">{html.escape(target_text)}</div>
            <div class="kpi-sub">{html.escape(last_res)} · {html.escape(last_sub)}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">쟁반 잔여 재고</div>
            <div class="kpi-value">{total_stock} / {max_stock}</div>
            <div class="kpi-sub">A:{inv[1]}  B:{inv[2]}  C:{inv[3]} ({html.escape(target_sub)})</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_stepper(snaps, idx: int) -> None:
    if not snaps:
        nodes = [
            f"<div class='step-node'>{STATE_KO[s]}<br><span style='font-size:10px'>{s}</span></div>"
            for s in ("LOOK", "SELECT", "VERIFY", "PICK", "DELIVER", "DONE")
        ]
        st.markdown(f"<div class='stepper'>{''.join(nodes)}</div>", unsafe_allow_html=True)
        return

    nodes = []
    for i, snap in enumerate(snaps):
        if i == idx:
            if snap.state == "FAIL":
                cls = "fail"
            elif snap.state == "DONE":
                cls = "success"
            else:
                cls = "active"
        elif i < idx:
            cls = "done"
        else:
            cls = ""
        nodes.append(
            f"<div class='step-node {cls}'>{STATE_KO.get(snap.state, snap.state)}"
            f"<br><span style='font-size:10px'>{snap.state}</span></div>"
        )
    st.markdown(f"<div class='stepper'>{''.join(nodes)}</div>", unsafe_allow_html=True)

    nav_cols = st.columns([1, 1, 1, 3])
    with nav_cols[0]:
        if st.button("◀ 이전", key="prev_step", disabled=(idx <= 0), width="stretch"):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()
    with nav_cols[1]:
        if st.button("다음 ▶", key="next_step", disabled=(idx >= len(snaps) - 1), width="stretch"):
            st.session_state.idx = min(len(snaps) - 1, idx + 1)
            st.rerun()
    with nav_cols[2]:
        if st.button("최종 단계", key="last_step", disabled=(idx == len(snaps) - 1), width="stretch"):
            st.session_state.idx = len(snaps) - 1
            st.rerun()
    with nav_cols[3]:
        step_cols = st.columns(len(snaps))
        for i, snap in enumerate(snaps):
            label = f"{i+1}.{STATE_KO.get(snap.state, snap.state)}"
            if step_cols[i].button(label, key=f"ph_{i}", width="stretch"):
                st.session_state.idx = i
                st.rerun()


def _render_verdict_card(view) -> None:
    m_txt, m_cls = _word_match(view.match)
    p_txt, p_cls = _word_ok(view.pick)
    d_txt, d_cls = _word_ok(view.deliver)
    part_label = f"#{view.part} {by_id(view.part).name}" if view.part else "미지정"
    note_txt = view.note or view.detail or "실행 대기 중"

    detail_badge = (
        f"<span class='pill pill-fail'>{html.escape(view.detail)}</span>"
        if view.state == "FAIL" and view.detail
        else ""
    )

    st.markdown(
        f"""
        <div class="verdict-box">
          <div class="verdict-row">
            <div class="verdict-item">대상 <span class="pill">{html.escape(part_label)}</span></div>
            <div class="verdict-item">대조 <span class="pill {m_cls}">{m_txt}</span></div>
            <div class="verdict-item">집기 <span class="pill {p_cls}">{p_txt}</span></div>
            <div class="verdict-item">전달 <span class="pill {d_cls}">{d_txt}</span></div>
            {detail_badge}
          </div>
          <div class="verdict-note">{html.escape(note_txt)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_gesture_tab() -> None:
    src_mode = st.radio(
        "입력 소스",
        ("이미지 파일 업로드", "웹캠 스냅샷 촬영"),
        horizontal=True,
        key="gesture_src_mode",
    )
    det = _yolo()
    yolo_gesture_ok = det.available and det.has_gesture_classes
    engine = st.radio(
        "인식 엔진",
        ("MediaPipe 21점 규칙", "YOLO 학습 모델"),
        horizontal=True,
        key="gesture_engine",
        index=1 if yolo_gesture_ok else 0,
    )
    if engine == "YOLO 학습 모델" and not yolo_gesture_ok:
        st.warning(det.error or "YOLO 모델에 손짓 클래스가 없습니다. '하드웨어 · 시험' 탭의 YOLO 클래스 매핑을 확인하세요.")
    auto_run = st.checkbox("1·2·3 손짓 인식 성공 시 즉시 전달 실행", value=True, key="gesture_auto_run")

    raw_bytes: bytes | None = None
    if src_mode == "웹캠 스냅샷 촬영":
        cam_file = st.camera_input("손짓을 카메라에 보여주고 촬영하세요 (1 / 2 / 3 / 손바닥)", key="cam_snap")
        if cam_file is not None:
            raw_bytes = cam_file.getvalue()
    else:
        up_file = st.file_uploader("손 사진 업로드 (PNG / JPG)", type=["png", "jpg", "jpeg"], key="gesture_upload")
        if up_file is not None:
            raw_bytes = up_file.getvalue()

    if raw_bytes is not None and st.button("손짓 분석 실행", key="analyze_gesture", width="stretch"):
        if engine == "YOLO 학습 모델" and yolo_gesture_ok:
            res = analyze_image_bytes_yolo(raw_bytes, det)
        else:
            res = analyze_image_bytes(raw_bytes)
        st.session_state.gesture_result = res
        if res.get("ok") and res.get("passed"):
            label = res.get("label")
            if label == "PALM":
                _reset_to_idle()
                st.session_state.gesture_result = res
            elif auto_run and label in {"1", "2", "3"}:
                _execute_pipeline(label, source="gesture")
        st.rerun()

    res = st.session_state.get("gesture_result")
    if res:
        if not res.get("ok"):
            st.error(res.get("error", "인식 실패"))
        else:
            data_url = res.get("image_data_url", "")
            if data_url.startswith("data:image/jpeg;base64,"):
                b64 = data_url.split(",", 1)[1]
                st.image(base64.b64decode(b64), caption=res.get("reason", ""), width="stretch")
            label = res.get("label", "NONE")
            conf = res.get("confidence", 0.0)
            if res.get("passed"):
                st.success(f"{res.get('reason')} · 신뢰도 {conf:.2f}")
                if label in {"1", "2", "3"} and st.button(f"인식된 #{label} 부품 다시 실행", key="rerun_gesture_part", width="stretch"):
                    _execute_pipeline(label, source="gesture")
                    st.rerun()
            else:
                st.warning(res.get("reason", "인식 조건 미충족"))


def _render_yolo_section() -> None:
    st.markdown("<div class='sec-label' style='margin-top:10px'>YOLO 비전 (학습 모델)</div>", unsafe_allow_html=True)
    c_path, c_conf = st.columns([3, 1.3])
    with c_path:
        st.text_input("가중치 경로 (.pt / .onnx)", key="yolo_model_path")
    with c_conf:
        st.number_input("신뢰도", min_value=0.05, max_value=0.95, step=0.05, key="yolo_conf")

    det = _yolo()
    if not det.available:
        st.info(f"{det.error}\n\nYOLO 없이도 기존 가짜 대조로 계속 동작합니다.")
        return

    st.success(f"모델 로드됨: {det.model_path.name} · 매핑 {'기본 별칭' if det.class_map.source == 'default' else Path(det.class_map.source).name}")
    kind_ko = {"part": "부품", "gesture": "손짓", "other": "무시"}
    st.dataframe(
        [{"id": c["id"], "클래스": c["name"], "용도": kind_ko[c["kind"]], "값": "" if c["value"] is None else str(c["value"])}
         for c in det.classes],
        hide_index=True,
        width="stretch",
    )
    if not det.has_part_classes:
        st.warning("부품으로 해석된 클래스가 없어 VERIFY 대조에 쓸 수 없습니다. models/yolo_classes.json 으로 매핑하세요.")
        return

    st.checkbox("VERIFY 단계에서 YOLO로 트레이 부품 대조", key="yolo_verify")
    st.radio("대조 프레임", YOLO_FRAME_SOURCES, horizontal=True, key="yolo_frame_src")
    if st.session_state.yolo_frame_src == YOLO_FRAME_SOURCES[0]:
        st.number_input("카메라 인덱스", min_value=0, max_value=9, step=1, key="yolo_camera")
    tray_file = st.file_uploader("트레이 사진 (대조/검출 테스트용)", type=["png", "jpg", "jpeg"], key="yolo_tray_upload")
    if tray_file is not None:
        st.session_state.yolo_tray_bytes = tray_file.getvalue()

    if st.button("YOLO 검출 테스트", key="yolo_test", width="stretch"):
        import cv2

        if st.session_state.yolo_frame_src == YOLO_FRAME_SOURCES[1] and st.session_state.yolo_tray_bytes:
            raw = st.session_state.yolo_tray_bytes
        else:
            frame = _grab_yolo_frame()
            ok, buf = cv2.imencode(".jpg", frame) if frame is not None else (False, None)
            raw = buf.tobytes() if ok else None
        st.session_state.yolo_test_result = (
            analyze_image_bytes_yolo(raw, det) if raw else {"ok": False, "error": "프레임을 얻지 못했습니다."}
        )

    res = st.session_state.get("yolo_test_result")
    if res:
        if not res.get("ok"):
            st.error(res.get("error", "검출 실패"))
        else:
            data_url = res.get("image_data_url", "")
            if data_url:
                st.image(base64.b64decode(data_url.split(",", 1)[1]), width="stretch")
            counts = res.get("part_counts", {})
            st.caption(
                f"부품 1: {counts.get(1, 0)}개 · 2: {counts.get(2, 0)}개 · 3: {counts.get(3, 0)}개"
                f" · 손짓: {res.get('label')} ({res.get('confidence', 0):.2f})"
            )


def _render_control_workbench() -> None:
    st.markdown("<div class='sec-label'>CONTROL WORKBENCH // 입력 · 하드웨어 · 재고</div>", unsafe_allow_html=True)
    tab_cmd, tab_gesture, tab_hw = st.tabs(["명령 · 빠른 선택", "손짓 인식", "하드웨어 · 시험"])

    with tab_cmd:
        with st.form("cmd_form", clear_on_submit=False):
            c_in, c_btn = st.columns([3.4, 1.2])
            with c_in:
                cmd_val = st.text_input(
                    "명령어",
                    placeholder="예: 캡B, 2번 부품 줘, cap c, 1",
                    label_visibility="collapsed",
                    key="cmd",
                )
            with c_btn:
                submitted = st.form_submit_button("실행", width="stretch")
        if submitted:
            _execute_pipeline(cmd_val, source="text")
            st.rerun()

        st.markdown("<div class='sec-label' style='margin-top:10px'>원클릭 부품 선택 & 재고</div>", unsafe_allow_html=True)
        inv = st.session_state.inventory
        for part in PARTS:
            qty = inv.get(part.id, 0)
            r, g, b = part.color_rgb
            stock_badge = f"재고 {qty}개" if qty > 0 else "품절"
            c_info, c_go = st.columns([2.7, 1.3])
            with c_info:
                st.markdown(
                    f"""
                    <div class="part-card">
                      <div class="part-head">
                        <span><span class="part-dot" style="background:rgb({r},{g},{b})"></span>#{part.id} {part.name}</span>
                        <span class="pill {'pill-fail' if qty == 0 else ''}">{stock_badge}</span>
                      </div>
                      <div class="part-spec">{part.code} · {part.spec}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_go:
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                if st.button(f"#{part.id} 전달", key=f"quick_{part.id}", width="stretch"):
                    _execute_pipeline(str(part.id), source="quick")
                    st.rerun()

    with tab_gesture:
        _render_gesture_tab()

    with tab_hw:
        st.radio("로봇 연결 모드", MODE_OPTIONS, key="mode")
        if st.session_state.mode == MODE_OPTIONS[2]:
            st.text_input("시리얼 포트 (예: COM3, /dev/ttyACM0)", key="serial_port")

        _render_yolo_section()

        st.markdown("<div class='sec-label' style='margin-top:10px'>고장 주입 시험 (Fault Injection)</div>", unsafe_allow_html=True)
        st.checkbox("다른 부품을 집게 한다 (VERIFY 불일치)", key="force_wrong")
        st.checkbox("집기를 실패시킨다 (ERR,GRIP)", key="force_pick_fail")
        st.checkbox("전달을 실패시킨다 (DELIVER 타임아웃)", key="force_deliver_fail")

        st.markdown("<div class='sec-label' style='margin-top:10px'>쟁반 재고 관리</div>", unsafe_allow_html=True)
        inv = st.session_state.inventory
        inv_cols = st.columns(4)
        for i, part in enumerate(PARTS):
            with inv_cols[i]:
                if st.button(f"{part.name} +1", key=f"add_stock_{part.id}", width="stretch"):
                    inv[part.id] = inv.get(part.id, 0) + 1
                    st.rerun()
        with inv_cols[3]:
            if st.button("전체 보충", key="refill_all", width="stretch"):
                st.session_state.inventory = default_inventory()
                st.rerun()

    if st.session_state.get("last_error"):
        st.error(st.session_state.last_error)


def _render_history_and_analytics(view, stats: dict) -> None:
    st.markdown("<div class='sec-label' style='margin-top:14px'>OPERATIONS // 이력 · 통계 · 프로토콜</div>", unsafe_allow_html=True)
    tab_log, tab_stat, tab_proto = st.tabs(["작업 이력 (CSV)", "통계 분석", "시리얼 프로토콜"])

    rows = read_all(DEFAULT_LOG)
    with tab_log:
        f_col, d_col, c_col = st.columns([2, 1.2, 1])
        with f_col:
            flt = st.selectbox("상태 필터", ["전체", "완료 (DONE)", "실패 (FAIL)", "진행 단계 포함"], key="log_filter", label_visibility="collapsed")
        with d_col:
            csv_text = "\n".join([",".join(HEADER)] + [",".join(r) for r in rows])
            st.download_button(
                "CSV 내보내기",
                data=csv_text.encode("utf-8"),
                file_name="dashboard_events.csv",
                mime="text/csv",
                width="stretch",
            )
        with c_col:
            if st.button("로그 비우기", key="clear_csv", width="stretch"):
                clear_log(DEFAULT_LOG)
                st.rerun()

        filtered = rows
        if flt == "완료 (DONE)":
            filtered = [r for r in rows if len(r) > 1 and r[1] == "DONE"]
        elif flt == "실패 (FAIL)":
            filtered = [r for r in rows if len(r) > 1 and r[1] == "FAIL"]
        elif flt == "전체":
            filtered = [r for r in rows if len(r) > 1 and r[1] in {"DONE", "FAIL"}]

        if filtered:
            table_rows = [
                {
                    "시각": r[0].replace("T", " "),
                    "상태": STATE_KO.get(r[1], r[1]),
                    "부품": f"#{r[2]}" if r[2] else "—",
                    "대조": r[4] or "—",
                    "집기": r[5] or "—",
                    "전달": r[6] or "—",
                    "상세": r[7] or "—",
                }
                for r in reversed(filtered[-50:])
            ]
            st.dataframe(table_rows, width="stretch", hide_index=True)
        else:
            st.info("표시할 작업 이력이 없습니다. 위에서 부품을 선택해 실행해 보세요.")

    with tab_stat:
        c_part, c_fail = st.columns(2)
        with c_part:
            st.markdown("<div class='cam-cap'>부품별 요청 분포</div>", unsafe_allow_html=True)
            total_parts = max(1, sum(stats["part_counts"].values()))
            for part in PARTS:
                cnt = stats["part_counts"].get(str(part.id), 0)
                st.write(f"#{part.id} {part.name} ({part.code}) — {cnt}회")
                st.progress(cnt / total_parts)
        with c_fail:
            st.markdown("<div class='cam-cap'>실패 사유 분석</div>", unsafe_allow_html=True)
            if stats["fail_reasons"]:
                for reason, cnt in sorted(stats["fail_reasons"].items(), key=lambda x: -x[1]):
                    st.write(f"• {reason}: {cnt}건")
            else:
                st.write("기록된 실패 이력이 없습니다.")

    with tab_proto:
        lines = view.protocol_lines
        if lines:
            st.markdown(f"<pre class='proto'>{html.escape(chr(10).join(lines))}</pre>", unsafe_allow_html=True)
        else:
            st.info("현재 실행은 가상 파이프라인(FakeHand)으로 동작했습니다. '하드웨어 · 시험' 탭에서 '모의 micro:bit V2' 또는 '실물 USB 시리얼'을 선택하면 실제 TX/RX 프로토콜 추적을 확인할 수 있습니다.")


def _load_bundled_simulator_html() -> str:
    """index.html과 sim/*.js를 단일 HTML로 번들링해 Streamlit iframe에서도 60fps 구동한다."""
    index_path = ROOT / "index.html"
    raw_html = index_path.read_text(encoding="utf-8")

    def _inline_script(match: re.Match[str]) -> str:
        rel_src = match.group(1)
        script_path = ROOT / rel_src
        if script_path.exists():
            js_code = script_path.read_text(encoding="utf-8")
            return f"<script>\n{js_code}\n</script>"
        return match.group(0)

    return re.sub(r'<script\s+src="(sim/[^"]+)"\s*></script>', _inline_script, raw_html)


def main() -> None:
    st.set_page_config(page_title="AI-HAND // 실시간 3D 시뮬레이션 & 제어 콘솔", layout="wide")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    _init_state()

    tab_sim, tab_workbench = st.tabs([
        "실시간 3D 와이어프레임 & 카메라 시뮬레이터 (60FPS)",
        "부품 전달 제어 워크벤치 (VLM · 스냅샷 · CSV)",
    ])

    with tab_sim:
        st.caption(
            "선 기반 3D 와이어프레임 5지 로봇손 · 비전 카메라 객체 네모칸 근접 자동 파지 · 경사로 슈트 실시간 공 궤적 시뮬레이션"
            "  (실물 웹캠/USB 시리얼 실시간 연동은 터미널에서 `python app.py` 실행)"
        )
        components.html(_load_bundled_simulator_html(), height=1180, scrolling=True)

    with tab_workbench:
        snaps = st.session_state.snaps
        if snaps:
            idx = min(st.session_state.idx, len(snaps) - 1)
            view = snaps[idx]
        else:
            idx = 0
            view = idle_snapshot(st.session_state.inventory)

        stats = summarize_log(DEFAULT_LOG)
        _render_header(view)
        _render_kpis(view, stats)

        left_col, right_col = st.columns([1.45, 1.0], gap="medium")
        with left_col:
            st.markdown("<div class='sec-label'>PIPELINE STAGE // 단계 타임라인 & 카메라 뷰</div>", unsafe_allow_html=True)
            _render_stepper(snaps, idx)

            tray_col, hand_col = st.columns(2)
            with tray_col:
                st.markdown("<p class='cam-cap'>CAM-01 · 쟁반 검사</p>", unsafe_allow_html=True)
                st.image(view.tray, width="stretch")
            with hand_col:
                st.markdown("<p class='cam-cap'>CAM-02 · 로봇 핸드 & 슈트</p>", unsafe_allow_html=True)
                st.image(view.hand, width="stretch")

            _render_verdict_card(view)

        with right_col:
            _render_control_workbench()

        _render_history_and_analytics(view, stats)


if __name__ == "__main__":
    main()
