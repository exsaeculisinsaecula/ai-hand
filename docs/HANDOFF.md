# 인수인계 메모 — 받을 물건: 볼트·너트 → 빨간 공 · 파란 공 · 초록 공

이 프로젝트를 이어받는 분이 먼저 읽을 문서입니다. 마지막 갱신: 2026-11-24

## 1. 바뀐 점 (2026-11-24)
- 받을 물건이 **볼트·너트(·와셔)** 에서 **빨간 공 / 파란 공 / 초록 공** 으로 바뀌었습니다.
- 내부 라벨: `red` / `blue` / `green` (`vision/parts.py` 의 `LABELS`, `LABEL_KO`)
  - 대시보드 번호(색 칠하기용): 빨강=1, 파랑=2, 초록=3 (`index.html` 의 `HO_PART`)
- 요청 해석(`parse_request`)은 색 단어로 판단합니다: 빨간/빨강/붉은/레드/red, 파란/파랑/푸른/블루/blue, 초록/녹색/그린/green.
  "A 말고 B", "A 빼고" 같은 부정 표현도 처리합니다. 색이 없는 "공 줘"는 다시 묻습니다.
- 수동 버튼 키: `req_red`, `req_blue`, `req_green` (`controller/handover.py`).
- 대시보드는 요청한 공 색(물빠진 톤)으로 진행 막대·현재 단계·활성 버튼이 물듭니다.
  - 색 값: 빨강 `#C0716A`, 파랑 `#6F8FB5`, 초록 `#79A07A` (`index.html` 맨 아래 CSS `/* ===== 공 색 */`, `sim/wireframe_3d.js` 의 `HELD_BALL`)
- 화면·로그 문구는 모두 하십시오체(~습니다 / ~십시오 / ~니까)로 통일했습니다. 새 문구도 같은 말투로 써 주십시오.

## 2. VLM 연동 (2026-11-24 적용 — 팀 vlm-go 반영)
출처: https://github.com/jihustar12-design/vlm-go (`vlm_go.py`, `microbit_go.py`, 2026-11-22)

### 팀원 코드를 그대로 쓰는 곳 (2026-11-24 확인)
| 무엇 | 팀원 코드 | 대시보드에서 |
|---|---|---|
| 손 위 공 판정 VLM | `vlm_go/vlm_go.py` `ask_vlm` · `PROMPT` · `SCHEMA` · `parse_found` | 그대로 불러 씀 (`vision/vlm_bridge.py` `team_vlm_go()`) |
| 음성 인식 | `vlm_go/vlm_go.py` `Voice` (Whisper small) | 그대로 불러 씀 (`handover_app.py` `_team_vlm_go()`) |
| micro:bit 펌웨어 | `vlm_go/microbit_go.py` (5-go) | 그대로 설치 (`handover_app.py` `FIRMWARE`) |
| 문장 이해 · VLM 관찰 | (팀원 코드에 없음) | 대시보드 기능 — 같은 Gemma 사용 |
- `vlm_go/` 는 팀원 전달본과 한 글자도 다르지 않음 (`tests/test_vlm_bridge.py` `TeamCodeImportTest` 가 확인).
- GitHub `jihustar12-design/vlm-go` 와 카카오톡 전달본: VLM 부분(프롬프트·스키마·parse_found·대상 추출)과 펌웨어는 같고,
  전달본은 기다리는 한도(120→300초) · 음성 입력 · 카메라 고르기가 추가된 새 버전 → 전달본을 씀.

### 손 위 공 인식 = VLM (2026-11-24 변경)
| 단계 | 누가 판단 |
|---|---|
| 공이 놓였는지 | **화면 변화** (`detect_change`: 손을 편 직후의 빈 손 사진과 달라진 덩어리 + 1초 정지) — 색과 무관 |
| 요청한 공인지 → 쥐기/NG | **팀원 `vlm_go/vlm_go.py` 의 `ask_vlm()` 을 고치지 않고 그대로 불러 씀** (`vision/vlm_bridge.py` `ask_team`: importlib 로 파일을 불러 같은 프롬프트·스키마·`parse_found`·gemma3:4b). found=true → GRIP, false → 색 있는 공이면 NG / 공이 아니면 기다림. 이 PC 실제 Gemma: 12/12 |
| VLM 이 '없음' | 쥐지도 NG 하지도 않고 계속 기다림 (2.5초 뒤 다시 확인) |
| 내려놓은 뒤 손 빔 확인 | **VLM** (`_empty_check_vlm`) |
| 규칙 교차 확인 | `ask_vlm` 판단과 색 규칙(점수 0.6 이상)이 다르면, 색 규칙이 찾은 공만 가까이 잘라 `ask_vlm` 을 한 번 더 → 그 답으로 최종 결정. 최종 판단은 언제나 VLM |
| 색 규칙 단독 | Ollama 가 꺼졌거나 VLM 이 답하지 못한 그 한 번만 (화면·로그에 표시) |
| 손 자동 찾기 (숨김) | 실물에서 쥐는 자리를 정확히 못 잡아 버튼을 숨기고, 쥘 자리는 화면 가운데 고정 사각형 (`DEFAULT_ROI = (0.30, 0.22, 0.70, 0.78)`, [손 영역]으로 직접 바꿀 수 있음). 코드는 남겨 둠 — [손 자동 찾기]: 손을 폈다·쥐었다·폈다 하며 찍은 사진 차이 = 손가락이 움직인 자리 (`hand_zone_from_motion`) → VLM 이 "로봇 손" 인지 예/아니오 → 맞으면 손 영역으로. Gemma 에게 좌표를 직접 물으면 틀려서(격자 0칸, 상자 어긋남) 이 방식을 씀 |
| VLM 관찰 | 쥐기 승인 직후 자동 + 칸의 [관찰] 버튼: 물체·색·개수·위치·상태·주변·메모 (약 2~3초, **판정에는 안 씀** — 넓은 화면의 주변 설명은 틀릴 수 있음) |
- 실물 기록 (팀원 테스트, 이전 found 방식): 13번 중 12번 맞음. 바뀐 '어떤 공' 방식은 실물 공으로 다시 재야 함.

### 대시보드 검색창 → 로봇까지 실제 흐름
```
[검색창] "사과 색깔 공 하나 줘"
   → POST /api/action {action: request}            (index.html submitAsk)
   → VlmBridge.interpret: Gemma 3 4B 가 문장 이해 → "빨간 공" (JSON 스키마로 답 강제)
   → HandoverTask.request → 로봇에 OPEN (손 펴기)     ← 화면에 "이해: 빨간 공 · VLM 0.8s — 근거 → 로봇에 OPEN 전송"
[작업자가 공을 손바닥에 올림 → 손 영역에서 1초 정지]
   → 색 규칙은 "무언가 놓였다" 감지만
   → VlmBridge.check_found: 손 위 사진 + "빨간 공" → {"found": true/false}   (vlm-go 프롬프트 그대로)
   → true: 로봇에 GRIP → PLACE → 손 빔 확인 / false: 로봇에 NG (손목 흔들기)
```
- VLM 은 **기본으로 켜져** 있습니다 (`--no-vlm` 으로 끔). Ollama 연결 상태를 10초마다 확인하고,
  대시보드 검색창 옆에 `VLM gemma3:4b 연결됨` / `VLM 꺼짐` 으로 표시합니다.
- **Ollama 가 꺼져 있으면** 기다리지 않고 바로 단어 규칙(요청)·색 규칙(손 위)으로 동작합니다.
  VLM 이 손 위 확인에 응답하지 않으면 그 한 번만 색 규칙으로 판단합니다.
- 주소는 `http://127.0.0.1:11434` 입니다. Windows 에서 `localhost` 로 쓰면 IPv6 를 먼저 시도해 매 요청 2초가 늦어집니다.

### 처음 설치 (2026-11-24 이 PC 에 Ollama 0.40.16 + gemma3:4b 설치 완료)
```powershell
winget install Ollama.Ollama        # 또는 https://ollama.com/download
ollama pull gemma3:4b               # 약 3.3GB
python handover_app.py              # 대시보드가 Ollama 를 자동으로 찾아 붙습니다
```

### 카메라 여러 대 — CAM 1~4
- 카메라 창 제목줄 **[분할 1 2 3 4]**. 칸 이름은 자리 번호 **CAM 1~4** (3분할은 CAM 1 이 크게, 4분할은 2×2).
- **자동 배치 [자동]** (기본 켬): 운영체제가 아는 카메라 수를 4초마다 셈 (카메라를 열지 않아 불이 안 켜짐).
  수가 바뀌면 한 번만 열어 보고, 손 판정 카메라(손 영역 인식·VLM 손 위 확인에 쓰는 카메라)가 없으면 첫 카메라를,
  나머지는 빈 CAM 칸에 넣고 분할을 늘림 (최대 4대, 넘으면 로그로 알림). 쓰는 중인 카메라는 다시 열지 않음.
  `Runtime.os_camera_count` · `probe_cameras` · `auto_place` · `_discovery_loop`.
- **아무 카메라나 아무 자리에:** 칸 아래 목록 또는 [미리보기](카메라 선택 창이 그 칸 탭으로 열림)에서 고름.
  손 판정 카메라도 칸 하나일 뿐이라 원하는 자리에 둘 수 있음 (화면에는 '손' 표시 없음, 코드에서는 `view_order` 의 `"hand"`).
- **자리 바꾸기:** 칸 왼쪽 위 이름표(⠿ CAM n)를 끌어 다른 칸에 놓으면 서로 바뀜 (마우스·터치). 배치는 저장됨.
- **[쟁반]:** 그 칸을 쟁반 카메라로 → 색 규칙(`find_balls`)으로 공을 모두 찾아 공 색 상자·번호·PICK THIS, 요청 중에만 초당 5번,
  손 카메라와 따로 돎 (`_tray_view_loop`). 쟁반 칸이 없으면 손 카메라 화면의 손 영역 밖에서 찾음.
- 칸마다 [끄기/켜기] (저장), 꺼진·끊긴·빈 칸은 손 카메라와 같은 모양의 상태 표시.
- **웹캠 여러 대:** 추가 칸은 MJPG 640×480 으로 열어 한 USB 허브에 3~4대를 꽂아도 대역폭이 남게 함 (`VIEW_SIZE`).
  그래도 끊기면 웹캠을 서로 다른 USB 포트(컨트롤러)에 나눠 꽂을 것.
- 칸마다 자기 스레드 · 자기 영상 (`/api/video?view=N`). 손 카메라와 같은 번호를 고르면 다시 열지 않고 같은 영상을 씀.
  분할을 줄여 화면에서 빠진 칸의 카메라는 닫음.
- 서버 설정: `data/handover_settings.json` 의 `split`, `view_order`, `view_sources`, `tray_slot` (git 에 안 올라감).
- 시험용으로 사진·영상 파일도 칸에 넣을 수 있음: `python handover_app.py --mock --view 쟁반사진.jpg --view 2 --tray-view 2 --point`
- `handover_app_2cam.py` (PR #1 의 웹캠 2대 버전)는 이 기능으로 대체됐지만 남겨 둠.

### 영상 위 표시 (대시보드가 직접 그림)
- 서버는 영상에 아무것도 그리지 않음 (`Runtime.DRAW_IN_VIDEO = False`) → 대시보드가 공·쥘 자리 위치(`hand_overlay`)를 받아 HUD 로 그림.
- 공: 그 공 색의 모서리 눈금 상자 + 선으로 이어진 카드 ("초록색 · 공", VLM 확인/색 추정 · 상태), 최근 1.2초 궤적(얇은 선).
- 쥘 자리: 같은 모양의 점선 상자 (요청한 공 색), 단계별 안내 문구. 손 근처가 아닌 곳의 공은 점수 0.72 이상만.
- 판정 방식 패널(알람 바로 위): 1 놓임 감지(규칙) → 2 VLM 판정 → 3 규칙 교차 확인 → 4 로봇 신호, 지금 단계 강조, 최근 판정 한 줄. [자세히] 로 설명 펼침.

### 대시보드 동작 (2026-11-24)
- 상태 칸은 실물 받기 작업에서 `renderHandover` 만 씀. 예전에는 시뮬레이터 HUD(초당 15번)와 번갈아 써서 글자·막대가 깜빡였음.
- 진행 막대는 단계 안에서도 예상 시간(`HO_TAU`)에 따라 계속 차오르고, 끝나면 잠깐 꽉 찼다가 비움.
- 로그는 새 줄만 덧붙임(전체 다시 그리기 없음), 글자 바뀜·버튼 등장은 짧은 페이드, 현재 단계 표시는 은은하게 맥동.
  시스템 '동작 줄이기' 설정이면 애니메이션 끔. 알람 줄은 화면 맨 아래.
- 상태 확인은 0.12초마다, 이전 요청이 끝나기 전에는 다시 묻지 않고, 탭이 가려지면 멈춤.
- `--mock` 으로 켜면 헤더에 "모의 로봇" 이 뜸. 실물 상태를 보려면 `--mock` 없이 실행 (`run_windows.bat`).

### 펌웨어 · vlm_go.py
- **펌웨어 = 팀원 `vlm_go/microbit_go.py` 그대로 (5-go, 팀 펌웨어 + GO 명령)**. 대시보드도 이 파일을 설치하므로
  대시보드와 vlm_go.py 가 같은 펌웨어를 씀 (서로 다시 설치하지 않음). 예전에 버전만 바꿔 만든 `microbit/handover.py`(v6)는 지웠습니다.
- **`vlm_go/` (팀원 전달본, 2026-11-24 — 고치지 않고 그대로 넣음)**: 대시보드 없이 "보이면 GO" 를 시험하는 팀 도구.
  글자 입력 + **음성 입력**(Enter → 4초 녹음 → faster-whisper `small`, 한국어, 노트북 안에서). 자세한 사용법은 `vlm_go/README.md`.
  ```powershell
  pip install -r vlm_go/requirements.txt -r vlm_go/requirements-voice.txt
  cd vlm_go
  python vlm_go.py --dry --camera 0                 # 판단만 (Enter = 음성, 글자 입력도 됨)
  python vlm_go.py --install                        # 팀원 펌웨어(5-go) 설치 (대시보드도 같은 펌웨어)
  python vlm_go.py --camera 0                       # 보이면 GO → 손 쥐었다 폄
  ```
  - 2026-11-24 이 PC 시험: 글자 입력 · 음성(Whisper 가 문장을 그대로 받아씀, 1.4초) · micro:bit 자동 탐색(COM4) · `--install` · 실제 로봇 GO(`DONE,GO`) 모두 통과.
  - 대시보드와 동시에 쓰지 말 것 (micro:bit·카메라를 한 프로그램만 열 수 있음).
- **대시보드 음성 입력**: 입력창 [말하기] → 팀원 `vlm_go/vlm_go.py` 의 `Voice` 클래스를 고치지 않고 그대로 불러 씀
  (Whisper small · 한국어 · 4초, 마이크는 이 PC 기본 입력 = Osmo 카메라 마이크) → 알아들은 문장을 글자 입력과 같은 VLM 해석으로.
  음성 패키지가 없으면 버튼이 꺼지고 글자 입력은 그대로.

### 남은 일
| 항목 | 상태 |
|---|---|
| 실제 Gemma 정확도 (2026-11-24, RTX 4060) | 문장 이해 **20/20** (평균 0.5초 — 비유 "사과 색깔"·"바다 같은 색", 오타, 영어, 부정 포함) · 손 위 found **12/12** (평균 0.9초). 단, 손 위 사진은 **합성 그림**이었음 → **실물 공·웹캠 사진으로 다시 재야 함** |
| 규칙으로 막아 둔 Gemma 약점 | ① 색 없는 "공 줘" 를 빨강으로 짐작 → `has_clue` 로 VLM 전에 다시 물음 ② "빨간 공이랑 파란 공" 에서 하나를 고름 → `mentioned_colors` 로 거절 |
| 공 학습 데이터 | **없음** (data/ 는 git 에 안 올라감, PC 에도 없음). 색 규칙 kNN 을 쓰려면 대시보드 [빨간 공][파란 공][초록 공] 버튼으로 색마다 2장 이상 찍기 |
| 응답 시간 | Gemma 3 4B 사진 판단은 GPU 에 따라 1~5초. 그동안 화면에 "VLM이 손 위 공을 확인하는 중" 표시 |
| 쟁반 안내 (`--point`, `handover_app_2cam.py`) | **색 규칙으로 동작** (`vision/parts.py` `find_balls`: 채도 높은 둥근 덩어리를 모두 찾아 색 판정, 손 영역·길쭉한 띠 제외). 사진 시험: 빨·파·초 3개 모두 찾고 요청한 공 번호 → 손목 가리키기까지 확인. **실물 쟁반·조명은 미확인** |
| 웹캠 2대 화면 | 손+쟁반 영상을 손 카메라 루프에서 한 장으로 합쳐 보냄 → **손 웹캠이 끊기면 쟁반 화면도 안 보임** (인식·안내는 계속 동작). 카메라별 영상 분리가 다음 개선 거리 |

### 색 규칙 (VLM 이 꺼졌을 때 판정, 켜져 있을 때는 '놓임' 감지만)
- `vision/parts.py` `extract_features` + `rule_classify` (HSV 채도 ≥ 110 덩어리의 평균 색상).
  [빨간 공][파란 공][초록 공] 버튼으로 찍은 샘플이 색마다 2장 이상이면 kNN.
- 피부색·주황 계열이 빨강으로 보일 수 있습니다 → 채도 기준(`SAT_MIN = 110`)과 "손 영역 움직임 있음" 판정으로 막고 있습니다.

## 3. 쓰지 않게 된 것
- `models/neuralgrip_best_two.pt` (팀 YOLO, 볼트·너트 2클래스): 공에는 쓸 수 없어 `handover_app.py --model` 기본값에서 뺐습니다.
  파일과 `tests/test_handover.py` 의 `YoloTests`(모델 자체 검사)는 남겨 두었습니다.
  공 YOLO 를 새로 학습한다면 클래스 이름을 `red` / `blue` / `green` 으로 하면 그대로 붙습니다 (`--model 경로`).
- 시뮬레이션 모드(`python app.py`, 캡 A·B·C 트레이 데모)는 이번 변경 대상이 아닙니다.

## 4. 확인 방법
```powershell
python -m unittest discover -s tests            # 143개 (팀 test 이미지가 없으면 2개 건너뜀)
python handover_app.py --mock                   # 로봇 없이 대시보드 흐름 확인 → http://127.0.0.1:8090/index.html
```
