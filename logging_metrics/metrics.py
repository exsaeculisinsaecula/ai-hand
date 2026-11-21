"""CSV 로그 기반 성능 지표 자동 계산기 (PDF 5페이지 지표 기준)."""
import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PipelineMetrics:
    total_requests: int = 0
    successful_deliveries: int = 0
    cancelled_requests: int = 0
    failed_deliveries: int = 0
    false_triggers: int = 0
    delivery_success_rate: float | None = None
    avg_delivery_seconds: float | None = None
    last_delivery_seconds: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics_from_csv(log_path: Path) -> PipelineMetrics:
    """CSV 로그의 타임스탬프를 분석해 전달 소요 시간, 성공률, 오인식/취소 건수를 계산한다."""
    metrics = PipelineMetrics()
    path = Path(log_path)
    if not path.exists():
        return metrics

    confirm_start_sec: float | None = None
    durations: list[float] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ev = row.get("event_type", "")
            detail = row.get("detail", "")
            try:
                t = float(row.get("mono_sec", "0"))
            except ValueError:
                continue

            if ev == "REQUEST_CONFIRMED":
                metrics.total_requests += 1
                confirm_start_sec = t
            elif ev == "DELIVERY_RELEASED":
                if confirm_start_sec is not None:
                    durations.append(max(0.0, t - confirm_start_sec))
                    confirm_start_sec = None
            elif ev == "HANDOVER_SUCCESS":
                metrics.successful_deliveries += 1
            elif ev == "REQUEST_CANCELLED":
                metrics.cancelled_requests += 1
            elif ev == "DELIVERY_FAILED":
                metrics.failed_deliveries += 1
                confirm_start_sec = None
            elif ev == "FALSE_TRIGGER_MARKED":
                metrics.false_triggers += 1

    if metrics.total_requests > 0:
        metrics.delivery_success_rate = round(
            (metrics.successful_deliveries / metrics.total_requests) * 100.0, 1
        )
    if durations:
        metrics.avg_delivery_seconds = round(sum(durations) / len(durations), 2)
        metrics.last_delivery_seconds = round(durations[-1], 2)

    return metrics
