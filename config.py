"""프로젝트 전역 설정 — 타이밍, 시리얼 통신, 인식 임계값."""
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "hand_landmarker.task"
LOG_DIR = ROOT_DIR / "logs"


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

    # 시리얼 통신 설정
    serial_port: str = ""
    baudrate: int = 115200

    # 예외 처리 설정
    max_grip_retries: int = 1
