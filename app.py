"""손짓 요청 부품 전달 로봇 — 통합 실행 진입점.

실행 예:
  python app.py                 # 가짜 micro:bit 모의 객체 + 웹 대시보드
  python app.py --webcam        # 웹캠 실시간 손짓 인식 켜고 시작
  python app.py --serial        # 실물 micro:bit V2 USB 시리얼 자동 연결
  python app.py --port COM4     # 지정 시리얼 포트로 연결
"""
import argparse
import webbrowser

from config import ROOT_DIR, PipelineConfig
from link.mock_microbit import MockMicrobitLink
from link.serial_link import SerialMicrobitLink
from server import PipelineRuntime, create_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="손짓 요청 부품 전달 로봇 파이프라인")
    parser.add_argument("--serial", action="store_true", help="실물 micro:bit USB 시리얼 연결 사용")
    parser.add_argument("--port", default="", help="시리얼 포트 지정 (예: COM3, /dev/ttyACM0)")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 인덱스 (기본 0)")
    parser.add_argument("--webcam", action="store_true", help="시작 시 웹캠 손짓 인식 자동 활성화")
    parser.add_argument("--http-port", type=int, default=0, help="웹 대시보드 포트 (기본 자동 탐색)")
    parser.add_argument("--no-browser", action="store_true", help="브라우저 자동 열기 비활성화")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(camera_index=args.camera, serial_port=args.port)

    if args.serial or args.port:
        link = SerialMicrobitLink(port=args.port or None, baudrate=cfg.baudrate)
        mode_desc = f"실물 USB 시리얼 ({link.port}, {cfg.baudrate}bps)"
    else:
        link = MockMicrobitLink()
        mode_desc = "가짜 micro:bit 모의 객체 (115200bps 시뮬레이션)"

    runtime = PipelineRuntime(link=link, config=cfg)
    runtime.start(start_webcam=args.webcam)

    server, http_port = create_server(runtime=runtime, root_dir=ROOT_DIR, port=args.http_port)
    url = f"http://127.0.0.1:{http_port}/index.html"
    print(f"[AI-HAND] 실행 모드: {mode_desc}")
    print(f"[AI-HAND] 대시보드 주소: {url}  (종료: Ctrl+C)")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[AI-HAND] 종료 중...")
    finally:
        server.shutdown()
        runtime.stop()


if __name__ == "__main__":
    main()
