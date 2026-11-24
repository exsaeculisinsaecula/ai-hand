# VLM이 보면 GO — 초간단 파이프라인

```
[1] 입력 "영상에 빨간 공이 있으면 인식해줘" + 웹캠 사진 한 장
[2] Gemma 3 4B (노트북 Ollama) → {"found": true} 또는 {"found": false} 만 답함
[3] true → micro:bit 로 GO 전송 / false → 대기
[4] micro:bit: 손가락 접기 → 3초 → 펴기 (1회)
```

| 파일 | 어디서 | 하는 일 |
|---|---|---|
| `vlm_go.py` | 노트북 | 사진 → VLM 판단 → GO 전송, 결과 사진·기록 저장 |
| `microbit_go.py` | micro:bit | 팀 펌웨어 + GO 매크로 |

## 입력 방법 (글자 / 음성 둘 다)
```
찾을 것을 말해 주세요 (글자 입력 / 그냥 Enter = 음성 / q = 종료):
```
- **글자**: `영상에 빨간 공이 있으면 인식해줘` 입력 후 Enter
- **음성**: 아무것도 안 치고 Enter → `🎤 듣는 중...` 이 나오면 4초 안에 말하기 → `[음성] ...` 으로 알아들은 문장 표시 → 같은 흐름

## 처음 한 번
```bash
pip3 install pyserial opencv-python
ollama pull gemma3:4b
python3 vlm_go.py --install          # micro:bit 에 GO 펌웨어 설치 (USB 연결, 서보 전원 켜기)
```
설치가 안 되면 python.microbit.org → `microbit_go.py` 내용 붙여 넣기 → Send to micro:bit.

## 음성으로 입력하기 (선택)
```bash
pip3 install faster-whisper sounddevice
```
- 시스템 설정 → 개인정보 보호 및 보안 → **마이크** → **터미널** 켜기 (켠 뒤 터미널을 다시 열기)
- 음성 → 글자 변환은 **Whisper**(노트북 안에서 동작). 처음 쓸 때 모델(small, 약 500MB)을 한 번 내려받아요
- 옵션: `--record-seconds 5` (녹음 시간), `--stt-model medium` (더 정확·느림, tiny/base/small/medium)
- 마이크 없이 시험: `--voice-file 녹음.wav`
- 설치 안 해도 글자 입력은 그대로 돼요

## 카메라 고르기
- 기본은 `--camera 0` (이 맥에서 USB 웹캠 번호). 번호가 바뀌면 `--camera auto`로 이름으로 자동 선택 (맥 내장·아이폰 제외)
- 이름을 알려면 한 번: `pip3 install pyobjc-framework-AVFoundation` (없으면 1번을 씀)
- 번호를 눈으로 확인: `python3 vlm_go.py --camera-check` → 번호마다 찍은 사진이 열려요 → `--camera 번호`로 직접 지정

## 실행
```bash
python3 vlm_go.py --dry --show                 # micro:bit 없이 판단만 시험
python3 vlm_go.py --show                       # 입력할 때마다 한 번 확인 → true 면 AiHand 쥐기
python3 vlm_go.py --watch 3                    # 3초마다 자동 확인 (보이면 GO, 사라졌다 다시 보이면 또 GO)
```
- 결과: `runs/vlm_go/` 에 판단한 사진(위에 found=true/false)과 음성(.wav), `log_v2.csv` 에 시간·입력 방식(text/voice)·알아들은 문장·대상·답·걸린 초·GO 여부
- 팀 대시보드(`handover_app.py`)를 다시 실행하면 micro:bit 펌웨어가 팀 것으로 자동 재설치돼요

## 다른 컴퓨터에서 처음 실행할 때 (팀원용 체크리스트)
1. 설치: `pip3 install -r requirements.txt` (음성까지: `pip3 install -r requirements-voice.txt`)
   - 가상환경(.venv)을 쓰면 그 안에서 설치·실행
2. Ollama 설치 → `ollama pull gemma3:4b` → Ollama 앱 켜 두기
   - 인텔 맥·GPU 없는 PC는 판단 한 번에 1~3분 걸릴 수 있어요 (기다리는 한도 5분)
3. 카메라 번호 확인: `python3 vlm_go.py --camera-check` → 웹캠이 찍힌 번호로 `--camera 번호`
4. micro:bit: `python3 vlm_go.py --install` → LED 웃는 얼굴 확인
   - 팀 대시보드(`handover_app.py`)를 실행한 뒤에는 펌웨어가 바뀌므로 다시 `--install`
   - 포트는 자동으로 찾아요 (맥·윈도우·리눅스). 안 되면 `--port /dev/cu.usbmodem… ` 또는 `--port COM3`
5. 손만 먼저 시험: 서보 전원 켜고 아래 실행 → 손이 쥐었다 펴면 OK
   ```bash
   python3 -c "import vlm_go as v; L=v.MicrobitLink(); print(L.version(), L.send('GO','DONE,GO',15))"
   ```
6. 전체: `python3 vlm_go.py --dry --show` (판단만) → `python3 vlm_go.py --show` (로봇까지)

