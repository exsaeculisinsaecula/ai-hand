# 손짓 요청 부품 전달 로봇 (단일 AiHand × 1)

작업자가 웹캠 앞에서 손가락으로 **1 · 2 · 3**을 표시하면 이를 인식하고, 단일 **Hiwonder AiHand**가 해당 트레이의 부품(캡슐)을 집어 **경사로(슈트)** 를 통해 작업자에게 전달하는 파이프라인입니다.

## 구성 요약
- **PC / 라즈베리 파이 (3모듈)**
  - `gesture/` : MediaPipe Hands 21점 랜드마크 기반 `1 / 2 / 3 / PALM / NONE` 규칙 분류기 + 웹캠 인식기
  - `controller/` : 7개 상태 머신(`IDLE → REQUEST → CONFIRM → PICK → DELIVER → HANDOVER → RETURN`), 1초 홀드, 2초 취소 창, 타임아웃 및 예외 처리
  - `logging_metrics/` : CSV 타임스탬프 로깅(`logs/pipeline_events.csv`) 및 전달 성공률·소요 시간 자동 계산
  - `vision/` : YOLO(ultralytics) 학습 모델로 부품(트레이 캡슐)·손짓 검출, 클래스 매핑, PICK 전 부품 대조
- **통신 (`link/`)**
  - USB 시리얼 `115200bps` 줄 단위 프로토콜 (`SHOW,N`, `PICK,N`, `DELIVER`, `RELEASE`, `RETURN,N`, `HOME`, `STOP` ↔ `ACK`, `DONE,*`, `ERR,*`, `EVT,HAND`)
  - 로봇 없이도 테스트 가능한 가짜 micro:bit 모의 객체 (`MockMicrobitLink`) 포함
- **micro:bit V2 (`microbit/main.py`)**
  - 단일 AiHand 서보 각도 클램프, 자세 프리셋, 초음파 연속 N회 필터, 버튼 A+B 비상 정지
- **실시간 3D 와이어프레임 & 비전 시뮬레이션 대시보드 (`index.html`, `sim/`, `server.py`, `dashboard/app.py`)**
  - 선 기반 3D 와이어프레임 5지 로봇손(실시간 손가락 접힘·팬틸트 회전) + 경사로 슈트 공 물리·3D 궤적 시각화
  - 가상 비전 카메라(객체 네모칸 추적 · 근접 거리 임계값 자동 파지 · 21점 손짓 PiP) 및 실물 웹캠 연동
  - 계획서 7-상태 머신 유기적 시뮬레이션, 자동 시나리오 데모, 수동 5지 관절 테스트, micro:bit 5×5 LED 미러링

## 빠른 실행
```powershell
# 1. 실시간 3D 와이어프레임 & 비전 시뮬레이션 대시보드 실행 (모의 micro:bit 포함)
python app.py

# 2. 웹캠 실시간 손짓 인식과 함께 실행
python app.py --webcam

# 3. 실물 micro:bit V2 USB 시리얼 연결 실행
python app.py --serial
# 또는 포트 직접 지정: python app.py --port COM3

# 4. Streamlit 통합 콘솔 실행 (3D 시뮬레이터 + VLM/재고 워크벤치)
streamlit run dashboard/app.py
```

## YOLO 학습 모델 연동
학습한 YOLO 가중치가 있으면 대시보드 두 곳에 자동으로 붙습니다. 모델이 없으면 기존 MediaPipe 손짓 인식으로 그대로 동작합니다.

1. `pip install ultralytics`
2. 가중치를 `models/best.pt` 에 두거나 `--yolo-model 경로` 로 지정합니다 (`.onnx` 도 가능).
3. 클래스 이름이 `cap_a/cap_b/cap_c`, `one/two/three/palm` 같은 흔한 이름이면 자동 인식됩니다.
   다르면 `models/yolo_classes.example.json` 을 `models/yolo_classes.json` 으로 복사해 매핑하세요.

```powershell
# 실물 로봇(micro:bit 시리얼) + YOLO 실시간 비전
python app.py --serial --webcam --yolo-model models/best.pt
# 옵션: --vision yolo|mediapipe|auto  --yolo-conf 0.5  --no-verify-parts
```

| 위치 | YOLO 쓰임 |
|---|---|
| `python app.py` 대시보드 (`index.html`) | 실물 웹캠 영상에 검출 박스 표시, 손짓 클래스 → 상태 머신 입력, 트레이별 부품 개수 패널 |
| 상태 머신 PICK 직전 | 트레이가 보이는데 목표 부품이 없으면 PICK 대신 HOME 후 IDLE (경고). 트레이가 안 보이면 대조를 건너뜀 |
| Streamlit 콘솔 → 하드웨어 · 시험 | 모델 상태·클래스 표, 검출 테스트, VERIFY 단계를 YOLO 대조로 교체 (카메라 1프레임 또는 업로드한 트레이 사진) |
| Streamlit 콘솔 → 손짓 인식 | 인식 엔진을 MediaPipe / YOLO 중에서 선택 |

모델에 손짓 클래스가 없으면 손짓은 같은 카메라 프레임에서 MediaPipe로 인식하고, YOLO는 부품 검출만 맡습니다.

## 실물 받기 작업 (서 있는 AiHand + micro:bit + USB 웹캠)
> **받을 물건: 빨간 공 · 파란 공 · 초록 공.** 검색창 문장 이해와 손 위 공 확인은 VLM(로컬 Ollama · Gemma 3 4B, 팀 vlm-go 방식)이 맡고,
> Ollama 가 꺼져 있으면 단어 규칙·색 규칙으로 동작합니다.
> 요청은 글자 또는 **음성**(입력창 [말하기], 팀원 `vlm_go/` 의 Whisper)으로 할 수 있습니다. 팀원의 단독 시험 도구는 `vlm_go/README.md`.
> 이어서 작업하는 분은 먼저 [`docs/HANDOFF.md`](docs/HANDOFF.md) 를 읽어 주십시오.

로봇손은 팔 없이 서 있고, 사람이 손바닥에 공을 쥐어 줍니다. 작업자가 대시보드에 "빨간 공 줘"처럼 입력하면
손을 펴고 기다리다가, 손 영역에 놓인 공의 색을 확인해 맞으면 쥐고 → 손목을 돌려 내려놓고 → 정면으로 돌아온 뒤 손이 비었는지 카메라로 확인합니다.
다른 색 공이면 손목을 좌우로 흔들어(NG) 알려 주고, 그 공을 치울 때까지 다시 판정하지 않습니다.

```bash
python3 -m venv .venv
.venv/bin/pip install pyserial "numpy<2" "opencv-python-headless==4.10.0.84" microfs
.venv/bin/pip install "torch==2.2.2" "torchvision==0.17.2" ultralytics     # Intel Mac 은 torch 2.2.2 가 마지막
.venv/bin/python handover_app.py            # micro:bit 자동 탐색 · 펌웨어 자동 설치 · USB 웹캠 1번
.venv/bin/python handover_app.py --mock     # 로봇 없이 화면 흐름 시험
# 옵션: --camera 1  --vlm  --port /dev/cu.usbmodemXXXX   (--model 은 red/blue/green 클래스 YOLO 가 생겼을 때만)
```

**Windows**: `setup_windows.bat`(처음 한 번) → `run_windows.bat`. micro:bit 는 COM 포트를 자동으로 찾고(직접 지정: `run_windows.bat --port COM5`),
카메라는 대시보드 [카메라 선택]에서 고르고, [분할 1~4]로 CAM 1~4 에 여러 대를 동시에 띄웁니다 (이름표를 끌어 자리 바꾸기, [쟁반]으로 쟁반 카메라 지정).
대시보드는 휴대폰·태블릿·노트북 화면 크기에 맞춰 배치가 바뀝니다.

| 항목 | 값 |
|---|---|
| 인식 | VLM (Gemma 3 4B): 손 위 사진 + 대상 → found true/false. Ollama 가 꺼져 있으면 색 규칙 (`docs/HANDOFF.md`) |
| 쥐는 조건 | 손 영역 안 탐지 신뢰도 ≥ 0.45, 같은 자리에서 연속 6회(약 1초), 손 영역 움직임 없음(사람 손이 빠짐) |
| 동작 | 쥐기 엄지 130°·손가락 50° 1.5초 / 펴기 0.7초 (팀 보정값) / 손목 내려놓기 180°(최대) → 정면 90°, 회전 1.5초 (코사인 이징) |
| 전달 확인 | 내려놓은 뒤 1.5초 동안 손 영역이 비어야 "완료", 공이 남아 있으면 "실패"로 기록 |
| 정지 | 대시보드 버튼 / Space / micro:bit A+B. 펌웨어가 이동 중 단계마다 STOP 을 확인해 중단 |
| 연결 | micro:bit 자동 탐색·재연결, 2초 PING, 펌웨어 버전(`VER`) 확인 후 자동 설치 |

- 서보 전원 어댑터를 켜야 움직입니다. micro:bit 방향(앞뒤)을 꼭 확인하십시오.
- 카메라 칸 [손 영역]으로 손바닥 자리를 드래그해 지정합니다 (`data/palm_roi.json`, 기본값은 팀 zones.json 의 palm).
- 로봇 창 아래 [조작] 줄: 펴기·쥐기·내려놓기·NG·손목 슬라이더·내려놓기 각도(`data/handover_settings.json`).
- 카메라 칸 [빨간 공][파란 공][초록 공] 버튼은 지금 손 영역의 공을 그 색 샘플로 저장합니다 (`data/parts/<색>/`, 색마다 2장 이상이면 kNN 사용).

## 테스트 실행
```powershell
python -m unittest discover -s tests -v
```
