"""글 → 부품 번호. 모르는 글은 None."""
import re


class Vlm:
    def resolve(self, text: str) -> int | None:
        raise NotImplementedError


class FakeVlm(Vlm):
    _WORDS = {
        "1": 1,
        "2": 2,
        "3": 3,
        "캡A": 1,
        "캡B": 2,
        "캡C": 3,
    }

    _PATTERNS: tuple[tuple[int, tuple[str, ...]], ...] = (
        (1, (r"캡\s*a", r"cap[\s\-_]*a", r"(?<!\d)1\s*번", r"첫\s*번\s*째", r"하나", r"일번")),
        (2, (r"캡\s*b", r"cap[\s\-_]*b", r"(?<!\d)2\s*번", r"두\s*번\s*째", r"둘", r"이번")),
        (3, (r"캡\s*c", r"cap[\s\-_]*c", r"(?<!\d)3\s*번", r"세\s*번\s*째", r"셋", r"삼번")),
    )

    _TOKEN_MAP = {
        "1": 1,
        "a": 1,
        "2": 2,
        "b": 2,
        "3": 3,
        "c": 3,
    }

    def explain(self, text: str) -> tuple[int | None, str]:
        cleaned = text.strip()
        if not cleaned:
            return None, "입력이 비어 있습니다."

        if cleaned in self._WORDS:
            pid = self._WORDS[cleaned]
            return pid, f"직접 지정 매칭 → #{pid}"

        lowered = cleaned.lower()
        if lowered in {"a", "b", "c"}:
            pid = self._TOKEN_MAP[lowered]
            return pid, f"코드 매칭 → #{pid}"

        if re.search(r"(?<!\d)[04-9]\s*번", lowered) or re.search(r"\d{2,}\s*번", lowered) or re.search(r"캡\s*[d-z]", lowered):
            return None, "등록되지 않은 부품 번호/코드입니다."

        matched: set[int] = set()
        for pid, patterns in self._PATTERNS:
            for pat in patterns:
                if re.search(pat, lowered):
                    matched.add(pid)

        for tok in re.findall(r"[a-z0-9]+", lowered):
            if tok in self._TOKEN_MAP:
                matched.add(self._TOKEN_MAP[tok])

        if len(matched) == 1:
            pid = next(iter(matched))
            return pid, f"자연어 인식 매칭 → #{pid}"
        if len(matched) > 1:
            return None, "여러 부품이 동시에 언급되어 모호합니다."
        return None, "일치하는 부품을 찾지 못했습니다."

    def resolve(self, text: str) -> int | None:
        part_id, _ = self.explain(text)
        return part_id
