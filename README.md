# 손짓 요청 부품 전달 로봇 (단일 AiHand × 1)

작업자가 웹캠 앞에서 손가락으로 **1 · 2 · 3**을 표시하면 이를 인식하고, 단일 **Hiwonder AiHand**가 해당 트레이의 부품(캡슐)을 집어 **경사로(슈트)** 를 통해 작업자에게 전달하는 파이프라인입니다.

## 구성 요약
- **PC / 라즈베리 파이 (3모듈)**
  - `gesture/` : MediaPipe Hands 21점 랜드마크 기반 `1 / 2 / 3 / PALM / NONE` 규칙 분류기 + 웹캠 인식기
  - `controller/` : 7개 상태 머신(`IDLE → REQUEST → CONFIRM → PICK → DELIVER → HANDOVER → RETURN`), 1초 홀드, 2초 취소 창, 타임아웃 및 예외 처리
  - `logging_metrics/` : CSV 타임스탬프 로깅(`logs/pipeline_events.csv`) 및 전달 성공률·소요 시간 자동 계산
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

## 테스트 실행
```powershell
python -m unittest discover -s tests -v
```
