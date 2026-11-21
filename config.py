"""프로젝트 전역 설정 — 타이밍, 시리얼 통신, 인식 임계값."""
from dataclasses import dataclass
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "hand_landmarker.task"
LOG_DIR = ROOT_DIR / "logs"

# YOLO 학습 모델 (ultralytics .pt / .onnx). 환경변수로 덮어쓸 수 있다.
YOLO_MODEL_PATH = Path(os.environ.get("AIHAND_YOLO_MODEL", ROOT_DIR / "models" / "best.pt"))
# 클래스 이름 → 부품 번호 / 손짓 매핑 (없으면 기본 별칭 규칙 사용)
YOLO_CLASSES_PATH = Path(os.environ.get("AIHAND_YOLO_CLASSES", ROOT_DIR / "models" / "yolo_classes.json"))


@dataclass(frozen=True)
class PipelineConfig:
    # 상태 머신 타이밍 (초)
    hold_seconds: float = 1.0
    confirm_seconds: float = 2.0
    pick_timeout: float = 5.0
    deliver_timeout: float = 5.0
    release_timeout: float = 3.0
    handover_timeout: float = 10.0
    return_timeout: float = 5.0
    ack_timeout: float = 2.0

    # 인식 설정
    min_confidence: float = 0.70
    camera_index: int = 0

    # YOLO 비전 설정
    yolo_conf: float = 0.50
    yolo_imgsz: int = 640
    verify_parts: bool = True   # PICK 전 트레이에 목표 부품이 보이는지 YOLO로 대조

    # 시리얼 통신 설정
    serial_port: str = ""
    baudrate: int = 115200

    # 예외 처리 설정
    max_grip_retries: int = 1
