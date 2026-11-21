"""YOLO 클래스 이름 → 부품 번호(1~3) / 손짓(1·2·3·PALM) 매핑.

학습 데이터의 클래스 이름은 프로젝트마다 다르므로 두 단계로 해석한다.
1. models/yolo_classes.json 이 있으면 그 매핑을 그대로 사용한다.
2. 없으면 아래 기본 별칭 규칙으로 클래스 이름을 추정한다.

yolo_classes.json 형식 (모든 키는 선택):
{
  "parts":    {"cap_a": 1, "cap_b": 2, "cap_c": 3},
  "gestures": {"one": "1", "two": "2", "three": "3", "palm": "PALM"},
  "ignore":   ["hand", "tray"]
}
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import re

from gesture.classifier import GESTURE_1, GESTURE_2, GESTURE_3, GESTURE_PALM

KIND_PART = "part"
KIND_GESTURE = "gesture"
KIND_OTHER = "other"

# 정규화된 이름(소문자, 영숫자/한글만) 기준 기본 별칭
_DEFAULT_PARTS: dict[str, int] = {}
for _pid, _letter in ((1, "a"), (2, "b"), (3, "c")):
    for _alias in (
        f"cap{_letter}", f"capsule{_letter}", f"part{_letter}", f"캡{_letter}",
        f"part{_pid}", f"cap{_pid}", f"capsule{_pid}", f"tray{_pid}", f"item{_pid}", f"부품{_pid}",
    ):
        _DEFAULT_PARTS[_alias] = _pid

_DEFAULT_GESTURES: dict[str, str] = {}
for _label, _aliases in (
    (GESTURE_1, ("1", "one", "gesture1", "finger1", "fingers1", "hand1", "g1", "손짓1", "하나")),
    (GESTURE_2, ("2", "two", "gesture2", "finger2", "fingers2", "hand2", "g2", "v", "peace", "손짓2", "둘")),
    (GESTURE_3, ("3", "three", "gesture3", "finger3", "fingers3", "hand3", "g3", "손짓3", "셋")),
    (GESTURE_PALM, ("palm", "open", "openpalm", "openhand", "five", "5", "stop", "cancel", "손바닥", "취소")),
):
    for _alias in _aliases:
        _DEFAULT_GESTURES[_alias] = _label


def normalize(name: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", str(name).lower())


@dataclass(frozen=True)
class ClassInfo:
    kind: str           # "part" | "gesture" | "other"
    value: int | str | None = None


@dataclass
class ClassMap:
    parts: dict[str, int] = field(default_factory=dict)
    gestures: dict[str, str] = field(default_factory=dict)
    ignore: set[str] = field(default_factory=set)
    source: str = "default"

    @classmethod
    def load(cls, path: Path | None) -> "ClassMap":
        if path is None or not Path(path).exists():
            return cls(source="default")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        parts = {normalize(k): int(v) for k, v in (data.get("parts") or {}).items()}
        gestures = {normalize(k): str(v).upper() for k, v in (data.get("gestures") or {}).items()}
        ignore = {normalize(k) for k in (data.get("ignore") or [])}
        for name, pid in parts.items():
            if pid not in (1, 2, 3):
                raise ValueError(f"{path}: 부품 번호는 1~3이어야 합니다 ({name}={pid})")
        for name, label in gestures.items():
            if label not in (GESTURE_1, GESTURE_2, GESTURE_3, GESTURE_PALM):
                raise ValueError(f"{path}: 손짓은 1/2/3/PALM 중 하나여야 합니다 ({name}={label})")
        return cls(parts=parts, gestures=gestures, ignore=ignore, source=str(path))

    def resolve(self, class_name: str) -> ClassInfo:
        key = normalize(class_name)
        if key in self.ignore:
            return ClassInfo(KIND_OTHER)
        if self.source != "default":
            # 명시 매핑이 있으면 그것만 신뢰한다 (추정으로 인한 오작동 방지)
            if key in self.parts:
                return ClassInfo(KIND_PART, self.parts[key])
            if key in self.gestures:
                return ClassInfo(KIND_GESTURE, self.gestures[key])
            return ClassInfo(KIND_OTHER)
        if key in _DEFAULT_PARTS:
            return ClassInfo(KIND_PART, _DEFAULT_PARTS[key])
        if key in _DEFAULT_GESTURES:
            return ClassInfo(KIND_GESTURE, _DEFAULT_GESTURES[key])
        return ClassInfo(KIND_OTHER)

    def describe(self, model_names: dict[int, str] | None) -> list[dict]:
        """모델 클래스 목록과 해석 결과 (대시보드 표시용)."""
        rows = []
        for idx, name in sorted((model_names or {}).items()):
            info = self.resolve(name)
            rows.append({"id": int(idx), "name": name, "kind": info.kind, "value": info.value})
        return rows
