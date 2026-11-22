"""VLM 효과 측정 (발표용)

1) 요청 해석: 단어 규칙만 vs 규칙 + VLM   (tests/request_cases.csv: request,answer)
   python3 vlm_eval.py request
2) 교차 확인: 손 위 사진 폴더에서 YOLO 결과와 VLM 판단 비교   (사진 이름에 bolt / nut 포함)
   python3 vlm_eval.py check --images data/captures
옵션: --model gemma3:4b
"""
import argparse
import csv
from pathlib import Path
import statistics

import cv2

from vision.parts import LABEL_KO, parse_request
from vision.vlm_bridge import VlmBridge

ROOT = Path(__file__).resolve().parent


def eval_request(b: VlmBridge, path: Path) -> None:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    ok_rule = ok_vlm = wrong_rule = wrong_vlm = 0
    times = []
    for r in rows:
        ans = r["answer"].strip()
        rt, _ = parse_request(r["request"])
        vt, _, info = b.interpret(r["request"])
        rp, vp = (LABEL_KO[rt] if rt else "모름"), (LABEL_KO[vt] if vt else "모름")
        ok_rule += rp == ans; ok_vlm += vp == ans
        wrong_rule += rp not in (ans, "모름"); wrong_vlm += vp not in (ans, "모름")
        if info.get("seconds"):
            times.append(info["seconds"])
        mark = "O" if vp == ans else "X"
        print(f"{mark} {r['request']:<18} 정답 {ans:<3} | 규칙 {rp:<3} | +VLM {vp:<3} ({info['method']} {info.get('seconds', 0)}s)")
    n = len(rows)
    print(f"\n===== 요청 해석 ({b.model}, {n}문장)")
    print(f"단어 규칙만     : {ok_rule / n:.0%} ({ok_rule}/{n}) · 반대 부품으로 오해 {wrong_rule}건")
    print(f"규칙 + VLM      : {ok_vlm / n:.0%} ({ok_vlm}/{n}) · 반대 부품으로 오해 {wrong_vlm}건")
    if times:
        print(f"VLM 사용 {len(times)}건 · 평균 {statistics.mean(times):.1f}s · 최대 {max(times):.1f}s")


def eval_check(b: VlmBridge, folder: Path, model_path: Path) -> None:
    from vision.parts import YoloParts
    yolo = YoloParts(model_path)
    rows = []
    for p in sorted(folder.glob("*.jpg")):
        truth = "bolt" if "bolt" in p.name.lower() else "nut" if "nut" in p.name.lower() else None
        img = cv2.imread(str(p))
        dets = yolo.detect(img)
        if not dets:
            print(f"- {p.name}: YOLO 탐지 없음"); continue
        d = max(dets, key=lambda x: x["conf"])
        v = b.check_part(b.crop(img, d["box"]))
        rows.append((truth, d["label"], d["conf"], v["label"]))
        print(f"{p.name:<28} 정답 {truth} | YOLO {d['label']} {d['conf']:.2f} | VLM {v['label']} ({v['seconds']}s) {v['seen']}")
    if rows:
        n = len(rows)
        print(f"\nYOLO 정확도 {sum(t == y for t, y, _, _ in rows)}/{n} · VLM 정확도 {sum(t == v for t, _, _, v in rows)}/{n}"
              f" · 둘이 일치 {sum(y == v for _, y, _, v in rows)}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["request", "check"])
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--cases", type=Path, default=ROOT / "tests" / "request_cases.csv")
    ap.add_argument("--images", type=Path, default=ROOT / "data" / "captures")
    ap.add_argument("--yolo", type=Path, default=ROOT / "models" / "neuralgrip_best_two.pt")
    a = ap.parse_args()
    b = VlmBridge(a.model)
    if a.what == "request":
        eval_request(b, a.cases)
    else:
        eval_check(b, a.images, a.yolo)


if __name__ == "__main__":
    main()
