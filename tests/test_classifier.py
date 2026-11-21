"""21점 손 랜드마크 규칙 분류기 단위 테스트."""
import unittest

from gesture.classifier import (
    GESTURE_1,
    GESTURE_2,
    GESTURE_3,
    GESTURE_NONE,
    GESTURE_PALM,
    classify_landmarks,
)


def make_hand(thumb=False, index=False, middle=False, ring=False, pinky=False):
    """손목(0)을 (0.5, 0.9)로 두고 5개 손가락 펼침 여부에 따라 21개 좌표를 생성한다."""
    pts = [(0.5, 0.9)] * 21

    # 엄지 (1..4)
    pts[1] = (0.44, 0.84)
    pts[2] = (0.40, 0.78)
    pts[3] = (0.38, 0.72) if thumb else (0.44, 0.74)
    pts[4] = (0.31, 0.65) if thumb else (0.47, 0.73)

    # 검지(5..8), 중지(9..12), 약지(13..16), 소지(17..20)
    xs = [0.44, 0.50, 0.56, 0.62]
    open_flags = [index, middle, ring, pinky]
    bases = [5, 9, 13, 17]
    for base, x, is_open in zip(bases, xs, open_flags):
        pts[base] = (x, 0.68)       # MCP
        pts[base + 1] = (x, 0.58)   # PIP
        pts[base + 2] = (x, 0.50 if is_open else 0.65)  # DIP
        pts[base + 3] = (x, 0.36 if is_open else 0.72)  # TIP
    return pts


class ClassifierTests(unittest.TestCase):
    def test_digits_and_palm_and_none(self):
        self.assertEqual(classify_landmarks(make_hand(index=True)).label, GESTURE_1)
        self.assertEqual(classify_landmarks(make_hand(index=True, middle=True)).label, GESTURE_2)
        self.assertEqual(classify_landmarks(make_hand(index=True, middle=True, ring=True)).label, GESTURE_3)
        self.assertEqual(
            classify_landmarks(make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True)).label,
            GESTURE_PALM,
        )
        self.assertEqual(classify_landmarks(make_hand()).label, GESTURE_NONE)

    def test_confidence_bounds_and_invalid_length(self):
        pred = classify_landmarks(make_hand(index=True), hand_score=0.95)
        self.assertGreaterEqual(pred.confidence, 0.70)
        self.assertLessEqual(pred.confidence, 1.0)
        self.assertEqual(classify_landmarks([(0.0, 0.0)] * 5).label, GESTURE_NONE)


if __name__ == "__main__":
    unittest.main()
