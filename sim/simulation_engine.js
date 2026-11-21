/**
 * sim/simulation_engine.js
 * Organic 60fps Simulation Controller:
 *  - Smooth 5-finger & Pan-Tilt kinematic animation
 *  - Ball gravity, chute rolling spin, receiving tray bounce physics
 *  - Real-time Proximity Auto-Grasp ("네모칸 쳐서 가까이 오면 잡고")
 *  - Standalone 7-State Machine + Live Backend (/api/state) seamless sync
 *  - Auto-Loop Demo, Fault Injection, Manual Servo Control & KPI tracking
 */
(function (global) {
  'use strict';

  const LED_PATTERNS = {
    IDLE: [
      '00000',
      '00000',
      '00100',
      '00000',
      '00000',
    ],
    '1': [
      '00100',
      '01100',
      '00100',
      '00100',
      '01110',
    ],
    '2': [
      '01110',
      '00010',
      '01110',
      '01000',
      '01110',
    ],
    '3': [
      '01110',
      '00010',
      '01110',
      '00010',
      '01110',
    ],
    OK: [
      '00000',
      '00001',
      '00010',
      '10100',
      '01000',
    ],
    X: [
      '10001',
      '01010',
      '00100',
      '01010',
      '10001',
    ],
    PROX: [
      '00100',
      '01010',
      '10101',
      '01010',
      '00100',
    ],
    ARROW: [
      '00100',
      '01000',
      '11111',
      '01000',
      '00100',
    ],
  };

  const GESTURE_FINGERS = {
    NONE: [0.18, 0.16, 0.15, 0.14, 0.14],
    '1': [0.15, 1.0, 0.08, 0.08, 0.08],
    '2': [0.15, 1.0, 1.0, 0.08, 0.08],
    '3': [0.15, 1.0, 1.0, 1.0, 0.08],
    PALM: [1.0, 1.0, 1.0, 1.0, 1.0],
  };

  const ZONE_KO = {
    TRAY: '부품 트레이 대기',
    APPROACH: '근접 접근 중',
    GRIP: '로봇손 파지 중',
    AIR: '슈트 상단 투하',
    CHUTE: '경사로 슈트 이동 중',
    BAY: '수령 트레이 도착',
    HANDED: '작업자 수령 완료',
  };

  class AiHandSimulator {
    constructor(callbacks) {
      this.H3D = global.AiHand3D;
      this.cb = callbacks || {};

      // World state
      this.time = 0;
      this.speed = 1.0;
      this.state = 'IDLE';
      this.tone = 'pass';
      this.message = '대기 중 — 손짓(1·2·3), 공 네모칸 클릭, 또는 근접 자동 잡기를 시작하세요.';
      this.targetPart = null;
      this.deliverSubstep = '';
      this.handoverSubstep = '';
      this.returnSubstep = '';
      this.progress = 0;
      this.elapsed = 0;
      this.phaseDuration = 0;
      this.emergencyStopped = false;

      // Modes & toggles
      this.liveMode = false;
      this.proximityMode = false;
      this.autoDemo = false;
      this.autoHandover = true;
      this.manualServo = false;
      this.showTrail = true;
      this.graspThresholdCm = 8.5;

      // Internal phase clock (standalone)
      this._phaseElapsed = 0;
      this._pickSub = ''; // 'DESCEND' | 'CLOSE' | 'RETRY_SLIP' | 'LIFT'
      this._pickTargetPos = null;
      this._pickRetryPending = false;
      this._autoDemoNextPart = 1;
      this._autoDemoCooldown = 1.5;
      this._autoDemoStepCount = 0;
      this._handoverAutoTimer = 0;
      this._runStartTime = 0;
      this._lastLiveState = 'IDLE';
      this._lastLiveDeliverSub = '';

      // Stock & local KPI metrics (used in standalone; overlaid by server metrics in live mode)
      this.stock = { 1: 3, 2: 3, 3: 3 };
      this.emptyAlertPart = null;
      this._emptyAlertTimer = 0;
      this.localMetrics = {
        totalRequests: 0,
        successfulDeliveries: 0,
        cancelledRequests: 0,
        falseTriggers: 0,
        deliveryTimes: [],
        lastConfidence: 0.96,
      };

      // Robot pose (live interpolated + target)
      this.pose = {
        pan: 90,
        tilt: 90,
        wristPitch: 0,
        fingers: [0.12, 0.12, 0.12, 0.12, 0.12],
        splay: 0.25,
      };
      this.targetPose = {
        pan: 90,
        tilt: 90,
        wristPitch: 0,
        fingers: [0.12, 0.12, 0.12, 0.12, 0.12],
        splay: 0.25,
      };
      this.kinematics = this.H3D.computeHandKinematics(this.pose);

      // Balls
      this.balls = this.H3D.PARTS.map((p) => ({
        id: p.id,
        name: p.name,
        code: p.code,
        color: p.color,
        dimColor: p.dimColor,
        radius: this.H3D.BALL_RADIUS,
        x: p.slot.x,
        y: p.slot.y,
        z: p.slot.z,
        vx: 0,
        vy: 0,
        vz: 0,
        spinX: 0,
        spinZ: 0,
        zone: 'TRAY',
        visible: true,
        distToHand: 20,
      }));
      this.heldBallId = null;
      this.activeBallId = null;
      this.focusBallId = null;
      this.draggingBallId = null;
      this.ballTrail = [];
      this._trailSampleAccum = 0;

      // Worker hand & Ultrasonic sensor
      this.workerHand = {
        visible: false,
        pos: { x: -45, y: 5.5, z: 0 },
        open: 0.85,
        carryingBallId: null,
        phase: 'IDLE', // 'APPROACH' | 'GRAB' | 'EXIT'
      };
      this.ultrasonicCm = 42.0;
      this.ultrasonicHits = 0;

      // Virtual gesture camera state (21-pt hand PiP)
      this.virtualGesture = {
        label: 'NONE',
        confidence: 0.0,
        fingersTarget: GESTURE_FINGERS.NONE.slice(),
        fingersSmooth: GESTURE_FINGERS.NONE.slice(),
        reason: '가상 카메라 감시 중 — 손짓(1·2·3) 또는 공 근접을 감지합니다.',
        passed: false,
      };

      // Active architecture module highlight
      this.activeModules = new Set(['ui']);
      this.ledKey = 'IDLE';

      // Initial proximity check
      this._updateDistances();
    }

    _log(dir, text) {
      if (this.cb.onLog) this.cb.onLog(dir, text);
    }

    getActiveBall() {
      if (this.activeBallId == null) return null;
      return this.balls.find((b) => b.id === this.activeBallId) || null;
    }

    getFocusBall() {
      if (this.focusBallId != null) {
        const fb = this.balls.find((b) => b.id === this.focusBallId && b.visible);
        if (fb) return fb;
      }
      if (this.targetPart != null) {
        const tb = this.balls.find((b) => b.id === this.targetPart && b.visible);
        if (tb) return tb;
      }
      // Closest visible ball to hand
      let best = null;
      for (const b of this.balls) {
        if (!b.visible) continue;
        if (!best || b.distToHand < best.distToHand) best = b;
      }
      return best;
    }

    get ballZone() {
      const b = this.getActiveBall();
      return b ? b.zone : 'TRAY';
    }

    ballZoneLabel() {
      const b = this.getActiveBall() || this.getFocusBall();
      return b ? (ZONE_KO[b.zone] || b.zone) : '대기';
    }

    setVirtualGesture(label, confidence, reason, passed) {
      this.virtualGesture.label = label;
      this.virtualGesture.confidence = confidence;
      this.virtualGesture.fingersTarget = (GESTURE_FINGERS[label] || GESTURE_FINGERS.NONE).slice();
      this.virtualGesture.reason = reason;
      this.virtualGesture.passed = Boolean(passed);
    }

    resetBallToTray(partId) {
      const part = this.H3D.PARTS.find((p) => p.id === partId);
      const ball = this.balls.find((b) => b.id === partId);
      if (!part || !ball) return;
      ball.x = part.slot.x;
      ball.y = part.slot.y;
      ball.z = part.slot.z;
      ball.vx = 0;
      ball.vy = 0;
      ball.vz = 0;
      ball.spinX = 0;
      ball.spinZ = 0;
      ball.zone = 'TRAY';
      ball.visible = (this.stock[partId] || 0) > 0;
    }

    refillStock() {
      this.stock = { 1: 3, 2: 3, 3: 3 };
      this.emptyAlertPart = null;
      for (const p of this.H3D.PARTS) {
        if (this.heldBallId !== p.id && this.activeBallId !== p.id) {
          this.resetBallToTray(p.id);
        }
      }
      this._log('SYS', '트레이 재고 전체 보충 완료 (캡A:3 · 캡B:3 · 캡C:3)');
    }

    setManualPose(partial) {
      this.manualServo = true;
      if (partial.pan != null) this.targetPose.pan = this.H3D.clamp(partial.pan, 0, 180);
      if (partial.tilt != null) this.targetPose.tilt = this.H3D.clamp(partial.tilt, 20, 160);
      if (partial.wristPitch != null) this.targetPose.wristPitch = this.H3D.clamp(partial.wristPitch, -20, 80);
      if (partial.gripAll != null) {
        const c = this.H3D.clamp(partial.gripAll, 0, 1);
        this.targetPose.fingers = [c, c, c, c, c];
      }
      if (partial.fingerIndex != null && partial.value != null) {
        this.targetPose.fingers[partial.fingerIndex] = this.H3D.clamp(partial.value, 0, 1);
      }
    }

    toggleProximityMode(forceVal) {
      this.proximityMode = forceVal != null ? Boolean(forceVal) : !this.proximityMode;
      this.manualServo = false;
      if (this.proximityMode) {
        this.ledKey = 'PROX';
        this._log('SYS', `근접 자동 잡기 모드 활성 (임계값 ≤ ${this.graspThresholdCm.toFixed(1)} cm)`);
        if (this.state === 'IDLE') {
          this.message = '근접 감지 대기 — 공을 로봇손 가까이 드래그하거나 [공 가까이 보내기]를 누르세요.';
        }
      } else {
        this.ledKey = 'IDLE';
        this.focusBallId = null;
        if (this.state === 'IDLE') {
          this.message = '대기 중 — 손짓(1·2·3)을 입력하거나 자동 시나리오를 실행하세요.';
        }
      }
    }

    /**
     * Spawns a ball rolling steadily toward the robotic hand's grasp zone to demonstrate
     * bounding box tracking -> distance countdown -> proximity auto-grasp -> chute delivery!
     */
    launchProximityApproach(partId) {
      if (this.state !== 'IDLE' && !this.emergencyStopped) {
        this._log('SYS', `현재 ${this.state} 진행 중입니다. 대기(IDLE) 상태에서 근접 테스트를 시작하세요.`);
        return;
      }
      this.emergencyStopped = false;
      this.manualServo = false;
      this.proximityMode = true;
      const pid = partId || ((this._autoDemoStepCount % 3) + 1);
      const ball = this.balls.find((b) => b.id === pid);
      if (!ball) return;

      // Ensure stock & clean reset
      if ((this.stock[pid] || 0) <= 0) this.stock[pid] = 1;
      for (const b of this.balls) {
        if (b.id !== pid && b.zone !== 'TRAY') this.resetBallToTray(b.id);
      }

      ball.visible = true;
      ball.x = (pid - 2) * 3.8; // slight angle offset so hand tracks pan smoothly!
      ball.y = this.H3D.BALL_RADIUS + 0.2;
      ball.z = 24.5;
      ball.vx = 0;
      ball.vy = 0;
      ball.vz = -7.8; // steady approach toward palm at z ~ 11.5
      ball.zone = 'APPROACH';

      this.activeBallId = pid;
      this.focusBallId = pid;
      this.ballTrail = [{ x: ball.x, y: ball.y, z: ball.z }];
      this.ledKey = 'PROX';
      this.tone = 'accent';
      this.message = `비전 카메라 객체 #${pid} (${ball.name}) 네모칸 추적 중 — 근접 임계값(${this.graspThresholdCm.toFixed(1)}cm) 진입 시 자동 파지!`;
      this.setVirtualGesture('NONE', 0.96, `비전 추적 활성: #${pid} ${ball.name} 접근 감지 중`, true);
      this._log('SYS', `[VIS] 객체 #${pid} (${ball.code}) 네모칸 추적 시작 — 접근 대기 중`);
    }

    /**
     * Triggers an immediate proximity grasp when an object enters graspThresholdCm.
     */
    _triggerProximityGrasp(ball) {
      if (this.state !== 'IDLE') return;
      const dist = ball.distToHand;
      ball.vx = 0;
      ball.vy = 0;
      ball.vz = 0;
      this.draggingBallId = null;
      this._log('SYS', `[VIS] 근접 임계값 진입! #${ball.id} ${ball.code} (거리 ${dist.toFixed(1)}cm ≤ ${this.graspThresholdCm.toFixed(1)}cm) → 자동 파지 실행`);
      this.setVirtualGesture(String(ball.id), 0.99, `근접 감지 성공 (#${ball.id} ${ball.name} · ${dist.toFixed(1)}cm) → 파지`, true);

      if (this.liveMode && this.cb.onBackendAction) {
        this.targetPart = ball.id;
        this.activeBallId = ball.id;
        this._pickTargetPos = { x: ball.x, y: ball.y, z: ball.z };
        this._pickSub = 'DESCEND';
        this._phaseElapsed = 0;
        this.cb.onBackendAction('proximity_pick', ball.id);
        return;
      }

      // Standalone organic proximity pick -> deliver -> chute -> handover!
      this.localMetrics.totalRequests += 1;
      this._runStartTime = this.time;
      this.targetPart = ball.id;
      this.activeBallId = ball.id;
      this.focusBallId = ball.id;
      this._pickTargetPos = { x: ball.x, y: ball.y, z: ball.z };
      this._enterState('PICK', `근접 감지 (${dist.toFixed(1)}cm) — 5지 와이어프레임 손가락 파지 실행!`, 'pass', 1.6);
      this._pickSub = 'DESCEND';
      this.ledKey = String(ball.id);
      this._log('TX', `SHOW,${ball.id}`);
      this._log('RX', 'ACK');
      this._log('TX', `PICK,${ball.id}`);
    }

    /**
     * Starts a standard gesture request (1, 2, or 3) from the project plan.
     */
    triggerGesture(part) {
      if (this.emergencyStopped) {
        this.emergencyStopped = false;
      }
      this.manualServo = false;
      if (this.liveMode && this.cb.onBackendAction) {
        this.cb.onBackendAction('gesture', part);
        return;
      }
      if (this.state !== 'IDLE') {
        this._log('SYS', `현재 ${this.state} 상태입니다. IDLE에서만 새 요청을 시작할 수 있습니다.`);
        return;
      }

      const b = this.balls.find((item) => item.id === part);
      if (b && b.zone !== 'TRAY') this.resetBallToTray(part);

      this.targetPart = part;
      this.activeBallId = part;
      this.focusBallId = part;
      this.ballTrail = b ? [{ x: b.x, y: b.y, z: b.z }] : [];
      this.emptyAlertPart = null;
      this.setVirtualGesture(String(part), 0.97, `인식 성공: 손짓 ${part} (1.0초 홀드 중)`, true);
      this._enterState('REQUEST', `손짓 ${part} 감지 — 1.0초 홀드 유지 중...`, 'accent', 1.0);
      this._log('SYS', `[CAM] 손짓 ${part} 21점 랜드마크 인식 (신뢰도 0.97) — 1초 홀드 시작`);
    }

    triggerCancel() {
      if (this.liveMode && this.cb.onBackendAction) {
        this.cb.onBackendAction('cancel');
        return;
      }
      if (this.state === 'REQUEST' || this.state === 'CONFIRM') {
        this.localMetrics.cancelledRequests += 1;
        this.setVirtualGesture('PALM', 0.96, '손바닥(PALM) 감지 — 요청 취소됨', true);
        this._log('SYS', '손바닥 제스처 감지 — 요청 취소');
        if (this.state === 'CONFIRM') {
          this._log('TX', 'HOME');
          this._log('RX', 'DONE,HOME');
        }
        this._goIdle('취소됨 — 중립 자세(HOME)로 복귀했습니다.', 'pass');
      } else {
        this._log('SYS', '취소는 REQUEST 또는 CONFIRM 상태에서만 동작합니다.');
      }
    }

    triggerHandover() {
      if (this.liveMode && this.cb.onBackendAction) {
        this._startWorkerHandApproach();
        this.cb.onBackendAction('hand');
        return;
      }
      if (this.state !== 'HANDOVER') {
        this._log('SYS', '인계 대기(HANDOVER) 상태가 아닙니다.');
        return;
      }
      if (this.handoverSubstep === 'WAIT') {
        this._startWorkerHandApproach();
      }
    }

    triggerReturn() {
      if (this.liveMode && this.cb.onBackendAction) {
        this.cb.onBackendAction('return');
        return;
      }
      if (this.state !== 'HANDOVER' || !this.targetPart) {
        this._log('SYS', '반납(RETURN)은 HANDOVER 대기 상태에서 실행됩니다.');
        return;
      }
      this._startReturnStandalone();
    }

    triggerError(errCode) {
      if (this.liveMode && this.cb.onBackendAction) {
        this.cb.onBackendAction(errCode === 'ERR,EMPTY' ? 'err_empty' : 'err_grip');
        return;
      }
      if (errCode === 'ERR,EMPTY') {
        const pid = this.targetPart || 1;
        this.emptyAlertPart = pid;
        this._emptyAlertTimer = 4.0;
        this.stock[pid] = 0;
        this.resetBallToTray(pid);
        this._log('ERR', `ERR,EMPTY — 트레이 ${pid} 비어 있음 (WonderCam 감지)`);
        this._log('TX', 'HOME');
        this._log('RX', 'DONE,HOME');
        this.ledKey = 'X';
        this._goIdle(`오류(ERR,EMPTY): 트레이 ${pid}이 비어 있습니다. 보충 후 재시도하세요.`, 'warn');
        return;
      }
      if (errCode === 'ERR,GRIP') {
        if (this.state === 'PICK') {
          this._pickRetryPending = true;
          this._pickSub = 'RETRY_SLIP';
          this._phaseElapsed = 0;
          this.heldBallId = null;
          this._log('ERR', 'ERR,GRIP — 잡기 미끄러짐 감지! 계획서 규칙에 따라 1회 자동 재시도합니다.');
          this._log('TX', `PICK,${this.targetPart}`);
          this.message = `잡기 실패(ERR,GRIP) — 트레이 ${this.targetPart} 1회 재시도 중...`;
          this.tone = 'accent';
        } else {
          this._log('ERR', 'ERR,GRIP 시뮬레이션 — 안전한 자세(HOME) 복귀');
          this._log('TX', 'HOME');
          this._log('RX', 'DONE,HOME');
          this.ledKey = 'X';
          this._goIdle('오류(ERR,GRIP) 알림 후 대기(IDLE) 복귀', 'warn');
        }
      }
    }

    triggerEmergencyStop() {
      this.autoDemo = false;
      this.emergencyStopped = true;
      this.draggingBallId = null;
      this.ledKey = 'X';
      if (this.liveMode && this.cb.onBackendAction) {
        this.cb.onBackendAction('stop');
        return;
      }
      this._log('ERR', '비상 정지(STOP) 트리거됨 — 모든 관절 현재 각도 고정!');
      this._log('TX', 'STOP');
      this._log('RX', 'ACK');
      this.state = 'IDLE';
      this.tone = 'warn';
      this.progress = 0;
      this.message = '비상 정지(STOP) — 현재 각도 유지. 새 요청 또는 대기 복귀로 재개하세요.';
    }

    resumeFromStop() {
      this.emergencyStopped = false;
      this.manualServo = false;
      this.emptyAlertPart = null;
      for (const p of this.H3D.PARTS) {
        this.resetBallToTray(p.id);
      }
      this.heldBallId = null;
      this.activeBallId = null;
      this.ballTrail = [];
      this.workerHand.visible = false;
      this.ledKey = 'IDLE';
      this._goIdle('중립 자세(HOME) 복귀 완료 — 대기 중', 'pass');
    }

    /**
     * Resets EVERY scenario, mode, ball, servo pose, stock, trail, ultrasonic state, and KPI counter.
     */
    resetAll() {
      this.autoDemo = false;
      this.proximityMode = false;
      this.manualServo = false;
      this.emergencyStopped = false;
      this.autoHandover = true;
      this.speed = 1.0;
      this.showTrail = true;
      this.graspThresholdCm = 8.5;

      this.state = 'IDLE';
      this.tone = 'pass';
      this.message = '모든 시나리오 리셋 완료 — 손짓(1·2·3), 공 네모칸 클릭, 또는 근접 자동 잡기를 시작하세요.';
      this.targetPart = null;
      this.deliverSubstep = '';
      this.handoverSubstep = '';
      this.returnSubstep = '';
      this.progress = 0;
      this.elapsed = 0;
      this.phaseDuration = 0;

      this._phaseElapsed = 0;
      this._pickSub = '';
      this._pickTargetPos = null;
      this._pickRetryPending = false;
      this._autoDemoNextPart = 1;
      this._autoDemoCooldown = 1.5;
      this._autoDemoStepCount = 0;
      this._handoverAutoTimer = 0;
      this._runStartTime = 0;
      this._lastLiveState = 'IDLE';
      this._lastLiveDeliverSub = '';

      // Stock & KPIs
      this.stock = { 1: 3, 2: 3, 3: 3 };
      this.emptyAlertPart = null;
      this._emptyAlertTimer = 0;
      this.localMetrics = {
        totalRequests: 0,
        successfulDeliveries: 0,
        cancelledRequests: 0,
        falseTriggers: 0,
        deliveryTimes: [],
        lastConfidence: 0.96,
      };

      // Reset robot target pose to HOME
      this.targetPose = {
        pan: 90,
        tilt: 90,
        wristPitch: 0,
        fingers: [0.12, 0.12, 0.12, 0.12, 0.12],
        splay: 0.25,
      };
      this.pose = {
        pan: 90,
        tilt: 90,
        wristPitch: 0,
        fingers: [0.12, 0.12, 0.12, 0.12, 0.12],
        splay: 0.25,
      };
      this.kinematics = this.H3D.computeHandKinematics(this.pose);

      // Reset all balls & trajectory
      this.heldBallId = null;
      this.activeBallId = null;
      this.focusBallId = null;
      this.draggingBallId = null;
      this.ballTrail = [];
      for (const p of this.H3D.PARTS) {
        this.resetBallToTray(p.id);
      }

      // Reset worker hand, ultrasonic, virtual gesture, LED
      this.workerHand.visible = false;
      this.workerHand.pos = { x: -45, y: 5.5, z: 0 };
      this.workerHand.open = 0.85;
      this.workerHand.carryingBallId = null;
      this.workerHand.phase = 'IDLE';
      this.ultrasonicCm = 42.0;
      this.ultrasonicHits = 0;
      this.ledKey = 'IDLE';
      this.setVirtualGesture('NONE', 0.0, '가상 카메라 감시 중 — 손짓(1·2·3) 또는 공 근접을 감지합니다.', false);
      this.activeModules = new Set(['ui']);
      this._updateDistances();

      if (this.liveMode && this.cb.onBackendAction) {
        this.cb.onBackendAction('reset_all', null);
      }
    }

    markFalseTrigger() {
      this.localMetrics.falseTriggers += 1;
      if (this.liveMode && this.cb.onBackendAction) {
        this.cb.onBackendAction('false_trigger');
      } else {
        this._log('SYS', '오인식 요청 기록됨 (KPI-02 반영)');
      }
    }

    _enterState(newState, msg, tone, duration) {
      this.state = newState;
      this.message = msg;
      this.tone = tone || 'pass';
      this.phaseDuration = duration || 0;
      this._phaseElapsed = 0;
      this.elapsed = 0;
      this.progress = 0;
    }

    _goIdle(msg, tone) {
      this.state = 'IDLE';
      this.tone = tone || 'pass';
      this.message = msg;
      this.phaseDuration = 0;
      this._phaseElapsed = 0;
      this.elapsed = 0;
      this.progress = 0;
      this.deliverSubstep = '';
      this.handoverSubstep = '';
      this.returnSubstep = '';
      this.heldBallId = null;
      this._pickSub = '';
      this._pickTargetPos = null;
      this.targetPart = null;
      this.workerHand.visible = false;
      this.ultrasonicHits = 0;
      this.ultrasonicCm = 42.0;
      if (this.ledKey !== 'X') this.ledKey = this.proximityMode ? 'PROX' : 'IDLE';
      // Replenish any handed ball back into tray if stock remains
      for (const p of this.H3D.PARTS) {
        const b = this.balls.find((item) => item.id === p.id);
        if (b && (b.zone === 'HANDED' || b.zone === 'BAY')) {
          this.resetBallToTray(p.id);
        }
      }
    }

    _startWorkerHandApproach() {
      this.workerHand.visible = true;
      this.workerHand.pos = { x: -45.5, y: 5.5, z: 0 };
      this.workerHand.open = 0.85;
      this.workerHand.carryingBallId = null;
      this.workerHand.phase = 'APPROACH';
      this.ultrasonicHits = 0;
    }

    _startReturnStandalone() {
      const pid = this.targetPart || 1;
      this.returnSubstep = 'RETURN';
      this._log('SYS', `인계 무응답 — 부품 ${pid} 트레이 반납 시작`);
      this._log('TX', `RETURN,${pid}`);
      this._enterState('RETURN', `무응답으로 부품 #${pid}을 트레이 ${pid}에 되돌리는 중...`, 'accent', 2.2);
    }

    /**
     * Syncs simulation visuals when running connected to the real Python backend (`python app.py`).
     */
    applyBackendSnapshot(snap) {
      this.liveMode = true;
      const prevState = this._lastLiveState;
      const prevDeliverSub = this._lastLiveDeliverSub;
      this._lastLiveState = snap.state;
      this._lastLiveDeliverSub = snap.deliver_substep || '';

      this.state = snap.state;
      this.tone = snap.tone || 'pass';
      this.message = snap.message || '';
      this.targetPart = snap.target_part || null;
      this.deliverSubstep = snap.deliver_substep || '';
      this.handoverSubstep = snap.handover_substep || '';
      this.returnSubstep = snap.return_substep || '';
      this.elapsed = Number(snap.elapsed || 0);
      this.phaseDuration = Number(snap.phase_duration || 0);
      this.progress = Number(snap.progress || 0);

      if (snap.tone === 'warn' && snap.message && snap.message.includes('비상 정지')) {
        this.emergencyStopped = true;
        this.ledKey = 'X';
      } else if (snap.state !== 'IDLE') {
        this.emergencyStopped = false;
      }

      // React to backend state transitions to drive exact 3D kinematics & ball physics!
      if (snap.state !== prevState) {
        this._phaseElapsed = 0;
        if (snap.state === 'REQUEST' && this.targetPart) {
          this.activeBallId = this.targetPart;
          this.focusBallId = this.targetPart;
          const b = this.balls.find((item) => item.id === this.targetPart);
          if (b && b.zone !== 'TRAY') this.resetBallToTray(this.targetPart);
          this.ballTrail = b ? [{ x: b.x, y: b.y, z: b.z }] : [];
          this.setVirtualGesture(String(this.targetPart), snap.last_confidence || 0.96, `손짓 ${this.targetPart} 홀드 감지 중`, true);
        } else if (snap.state === 'CONFIRM' && this.targetPart) {
          this.ledKey = String(this.targetPart);
        } else if (snap.state === 'PICK' && this.targetPart) {
          this.activeBallId = this.targetPart;
          const b = this.balls.find((item) => item.id === this.targetPart);
          if (b) {
            if (!this._pickTargetPos) this._pickTargetPos = { x: b.x, y: b.y, z: b.z };
            this.ballTrail = [{ x: b.x, y: b.y, z: b.z }];
          }
          this._pickSub = 'DESCEND';
          this.ledKey = String(this.targetPart);
        } else if (snap.state === 'DELIVER' && this.targetPart) {
          const b = this.balls.find((item) => item.id === this.targetPart);
          if (b) {
            this.heldBallId = b.id;
            b.zone = 'GRIP';
          }
          this.ledKey = 'ARROW';
        } else if (snap.state === 'HANDOVER') {
          this.heldBallId = null;
          this.ledKey = 'OK';
          // Ensure ball was released if backend jumped quickly
          const b = this.getActiveBall();
          if (b && b.zone === 'GRIP') {
            this._releaseBallOntoChute(b);
          }
        } else if (snap.state === 'RETURN' && this.targetPart) {
          this.ledKey = 'ARROW';
        } else if (snap.state === 'IDLE') {
          this._pickTargetPos = null;
          this.heldBallId = null;
          this.setVirtualGesture('NONE', 0.0, '대기 중 — 손짓 또는 근접 객체를 감시합니다.', false);
          if (!this.emergencyStopped) {
            this.ledKey = this.proximityMode ? 'PROX' : 'IDLE';
            for (const p of this.H3D.PARTS) {
              const b = this.balls.find((item) => item.id === p.id);
              if (b && (b.zone === 'HANDED' || b.zone === 'BAY' || b.zone === 'GRIP')) {
                this.resetBallToTray(p.id);
              }
            }
          }
        }
      }

      // Detect DELIVER -> RELEASE substep in live mode!
      if (snap.state === 'DELIVER' && this.deliverSubstep === 'RELEASE' && prevDeliverSub !== 'RELEASE') {
        const b = this.getActiveBall();
        if (b && this.heldBallId === b.id) {
          this._releaseBallOntoChute(b);
        }
      }
    }

    _releaseBallOntoChute(ball) {
      this.heldBallId = null;
      ball.zone = 'AIR';
      ball.vx = -6.5;
      ball.vy = -2.0;
      ball.vz = 0;
      this.ledKey = 'ARROW';
    }

    /**
     * Main 60fps simulation tick.
     */
    update(rawDt) {
      const dt = Math.min(rawDt, 0.05) * (this.emergencyStopped ? 0 : this.speed);
      this.time += rawDt;

      if (this._emptyAlertTimer > 0) {
        this._emptyAlertTimer -= rawDt;
        if (this._emptyAlertTimer <= 0) this.emptyAlertPart = null;
      }

      // Smooth virtual 21-point hand skeleton
      for (let i = 0; i < 5; i++) {
        this.virtualGesture.fingersSmooth[i] = this.H3D.lerp(
          this.virtualGesture.fingersSmooth[i],
          this.virtualGesture.fingersTarget[i],
          Math.min(1, rawDt * 10)
        );
      }

      if (this.emergencyStopped) {
        this.activeModules = new Set(['robot', 'ui']);
        this._updateDistances();
        return;
      }

      // 1. Standalone state progression (when not driven by Python server)
      if (!this.liveMode) {
        this._stepStandaloneStateMachine(dt);
      } else {
        this._phaseElapsed += dt;
      }

      // 2. Auto-Demo loop (organic continuous showcase across parts + proximity grasp!)
      if (this.autoDemo && this.state === 'IDLE' && !this.draggingBallId) {
        const approachingBall = this.balls.find((b) => b.zone === 'APPROACH');
        if (!approachingBall) {
          this._autoDemoCooldown -= dt;
          if (this._autoDemoCooldown <= 0) {
            this._autoDemoCooldown = 1.6;
            this._autoDemoStepCount += 1;
            // Every 4th auto-demo cycle showcases Proximity Auto-Grasp!
            if (this._autoDemoStepCount % 4 === 0) {
              this.launchProximityApproach(2);
            } else {
              const p = this._autoDemoNextPart;
              this._autoDemoNextPart = (this._autoDemoNextPart % 3) + 1;
              if ((this.stock[p] || 0) <= 0) this.stock[p] = 3;
              this.triggerGesture(p);
            }
          }
        }
      }

      // 3. Determine target robotic hand pose for current state & substep
      if (!this.manualServo) {
        this._updateTargetPose(dt);
      }

      // 4. Smoothly interpolate robot joints & compute exact 3D forward kinematics
      const kJoint = Math.min(1, dt * 8.2);
      const kFinger = Math.min(1, dt * 10.5);
      this.pose.pan = this.H3D.lerp(this.pose.pan, this.targetPose.pan, kJoint);
      this.pose.tilt = this.H3D.lerp(this.pose.tilt, this.targetPose.tilt, kJoint);
      this.pose.wristPitch = this.H3D.lerp(this.pose.wristPitch, this.targetPose.wristPitch, kJoint);
      this.pose.splay = this.H3D.lerp(this.pose.splay, this.targetPose.splay, kJoint);
      for (let i = 0; i < 5; i++) {
        this.pose.fingers[i] = this.H3D.lerp(this.pose.fingers[i], this.targetPose.fingers[i], kFinger);
      }
      this.kinematics = this.H3D.computeHandKinematics(this.pose);

      // 5. Update ball physics, chute roll, worker handover & proximity detection
      this._updateBallPhysics(dt);
      this._updateWorkerHand(dt);
      this._updateDistances();
      this._checkProximityAutoGrasp();
      this._updateActiveModules();
    }

    _stepStandaloneStateMachine(dt) {
      if (this.state === 'IDLE') return;

      this._phaseElapsed += dt;
      this.elapsed = this._phaseElapsed;
      if (this.phaseDuration > 0) {
        this.progress = Math.min(100, (this._phaseElapsed / this.phaseDuration) * 100);
      }

      if (this.state === 'REQUEST') {
        if (this._phaseElapsed >= 1.0) {
          const p = this.targetPart;
          this.localMetrics.totalRequests += 1;
          this._runStartTime = this.time;
          this.ledKey = String(p);
          this._log('TX', `SHOW,${p}`);
          this._log('RX', 'ACK');
          this.setVirtualGesture(String(p), 0.98, `요청 확정 (#${p}) — 2초 내 손바닥으로 취소 가능`, true);
          this._enterState('CONFIRM', `요청 확정 (${p}번 부품) — 2.0초 내 손바닥(PALM)으로 취소 가능`, 'accent', 2.0);
        }
      } else if (this.state === 'CONFIRM') {
        if (this._phaseElapsed >= 2.0) {
          const p = this.targetPart;
          if ((this.stock[p] || 0) <= 0) {
            this.triggerError('ERR,EMPTY');
            return;
          }
          const ball = this.balls.find((item) => item.id === p);
          this._pickTargetPos = ball ? { x: ball.x, y: ball.y, z: ball.z } : null;
          this._pickSub = 'DESCEND';
          this._log('TX', `PICK,${p}`);
          this.setVirtualGesture('NONE', 0.0, `트레이 ${p} 객체 네모칸 추적 · 파지 진행 중`, true);
          this._enterState('PICK', `단일 AiHand가 트레이 ${p}의 공에 접근해 5지를 접는 중...`, 'pass', 1.85);
        }
      } else if (this.state === 'PICK') {
        const ball = this.balls.find((item) => item.id === this.targetPart);
        if (this._pickSub === 'RETRY_SLIP') {
          if (this._phaseElapsed >= 0.75) {
            this._pickSub = 'DESCEND';
            this._phaseElapsed = 0;
            this.message = `재시도 접근 — 트레이 ${this.targetPart} 공 다시 파지 중...`;
            this.tone = 'pass';
          }
          return;
        }
        if (this._pickSub === 'DESCEND' && this._phaseElapsed >= 0.58) {
          this._pickSub = 'CLOSE';
          this.message = `근접 범위 진입 — 5개 와이어프레임 손가락 실시간 접힘(GRASP)!`;
        } else if (this._pickSub === 'CLOSE') {
          const avg = this.pose.fingers.reduce((a, b) => a + b, 0) / 5;
          if (avg >= 0.68 && ball && this.heldBallId !== ball.id) {
            this.heldBallId = ball.id;
            ball.zone = 'GRIP';
            this._log('SYS', `[GRIP] 5지 파지 완료 — 공 #${ball.id} (${ball.name}) 들어올리는 중`);
          }
          if (this._phaseElapsed >= 1.25) {
            this._pickSub = 'LIFT';
          }
        } else if (this._pickSub === 'LIFT' && this._phaseElapsed >= 1.85) {
          if (ball && this.heldBallId !== ball.id) {
            this.heldBallId = ball.id;
            ball.zone = 'GRIP';
          }
          this._log('RX', 'DONE,PICK');
          this._log('TX', 'DELIVER');
          this.deliverSubstep = 'DELIVER';
          this.ledKey = 'ARROW';
          this._enterState('DELIVER', '공을 쥔 채 경사로(슈트) 상단으로 180° 선회 중...', 'pass', 2.0);
        }
      } else if (this.state === 'DELIVER') {
        const ball = this.balls.find((item) => item.id === this.targetPart);
        if (this.deliverSubstep === 'DELIVER' && this._phaseElapsed >= 1.15) {
          this.deliverSubstep = 'RELEASE';
          this._log('RX', 'DONE,DELIVER');
          this._log('TX', 'RELEASE');
          this.message = '손 펼침(RELEASE) — 5지 펴지며 공을 경사로 슈트로 투하!';
        } else if (this.deliverSubstep === 'RELEASE') {
          const avg = this.pose.fingers.reduce((a, b) => a + b, 0) / 5;
          if (ball && this.heldBallId === ball.id && avg <= 0.42) {
            this._releaseBallOntoChute(ball);
          }
          if (this._phaseElapsed >= 2.0) {
            if (ball && this.heldBallId === ball.id) {
              this._releaseBallOntoChute(ball);
            }
            this._log('RX', 'DONE,RELEASE');
            this.deliverSubstep = '';
            this.handoverSubstep = 'WAIT';
            this._handoverAutoTimer = 0;
            this.ledKey = 'OK';
            this._enterState(
              'HANDOVER',
              '공이 경사로를 굴러 수령 트레이에 도착 — 초음파 근접 수령(EVT,HAND) 대기 중...',
              'accent',
              10.0
            );
          }
        }
      } else if (this.state === 'HANDOVER') {
        if (this.handoverSubstep === 'WAIT') {
          const ball = this.balls.find((item) => item.id === this.targetPart);
          if (this.autoHandover && ball && ball.zone === 'BAY' && !this.workerHand.visible) {
            this._handoverAutoTimer += dt;
            if (this._handoverAutoTimer >= 1.35) {
              this._startWorkerHandApproach();
            }
          }
          if (this._phaseElapsed >= 10.0 && !this.workerHand.visible) {
            this._startReturnStandalone();
          }
        } else if (this.handoverSubstep === 'HOME') {
          if (this._phaseElapsed >= 0.7) {
            this._log('RX', 'DONE,HOME');
            this._goIdle('전달 완료 — 중립 자세(HOME)로 복귀했습니다.', 'pass');
          }
        }
      } else if (this.state === 'RETURN') {
        const ball = this.balls.find((item) => item.id === this.targetPart);
        const part = this.H3D.PARTS.find((p) => p.id === this.targetPart);
        if (this.returnSubstep === 'RETURN') {
          if (ball && part) {
            // Smooth visual return arc back to tray slot
            const t = this.H3D.clamp(this._phaseElapsed / 1.5, 0, 1);
            ball.x = this.H3D.lerp(this.H3D.LAYOUT.bay.center.x, part.slot.x, t);
            ball.z = this.H3D.lerp(0, part.slot.z, t);
            ball.y = this.H3D.BALL_RADIUS + 0.2 + Math.sin(t * Math.PI) * 11.0;
            ball.zone = 'GRIP';
            this._sampleTrail(ball);
          }
          if (this._phaseElapsed >= 1.55) {
            if (ball) this.resetBallToTray(ball.id);
            this._log('RX', 'DONE,RETURN');
            this._log('TX', 'HOME');
            this.returnSubstep = 'HOME';
            this.message = '부품 반납 완료 — 중립 자세(HOME)로 복귀 중...';
          }
        } else if (this.returnSubstep === 'HOME' && this._phaseElapsed >= 2.2) {
          this._log('RX', 'DONE,HOME');
          this._goIdle('무응답 반납 완료 — 대기(IDLE) 상태로 복귀했습니다.', 'accent');
        }
      }
    }

    _updateTargetPose() {
      const H3D = this.H3D;
      const breathe = 0.03 * Math.sin(this.time * 2.2);

      if (this.state === 'IDLE') {
        const focus = this.getFocusBall();
        const trackingBall =
          (this.proximityMode || this.draggingBallId != null) &&
          focus &&
          (focus.zone === 'APPROACH' || this.draggingBallId === focus.id || this.proximityMode);

        if (trackingBall && focus) {
          // Smoothly face & track the approaching / dragged ball!
          const approachPose = H3D.solvePoseForPoint(focus.x, Math.max(4.5, focus.y + 3.0), focus.z, 'pregrasp');
          // Keep hand facing forward neatly while awaiting approach
          const radial = Math.hypot(focus.x, focus.z);
          if (radial > 18) {
            this.targetPose.pan = approachPose.pan;
            this.targetPose.tilt = 114;
            this.targetPose.wristPitch = 30;
          } else {
            this.targetPose.pan = approachPose.pan;
            this.targetPose.tilt = approachPose.tilt;
            this.targetPose.wristPitch = approachPose.wristPitch;
          }
          // Open fingers wider in anticipation as ball comes closer
          const anticipation = H3D.clamp((16 - focus.distToHand) / 10, 0, 1);
          const openCurl = 0.14 - anticipation * 0.09;
          this.targetPose.fingers = [openCurl, openCurl, openCurl, openCurl, openCurl];
          this.targetPose.splay = 0.28 + anticipation * 0.45;
          return;
        }

        // Standard IDLE (HOME)
        this.targetPose.pan = 90;
        this.targetPose.tilt = 90;
        this.targetPose.wristPitch = 0;
        this.targetPose.fingers = [
          0.12 + breathe,
          0.10 + breathe,
          0.11 + breathe,
          0.13 + breathe,
          0.15 + breathe,
        ];
        this.targetPose.splay = 0.25;
        return;
      }

      if (this.state === 'REQUEST') {
        const part = H3D.PARTS.find((p) => p.id === this.targetPart);
        this.targetPose.pan = part ? H3D.lerp(90, part.pan, 0.28) : 90;
        this.targetPose.tilt = 96;
        this.targetPose.wristPitch = 8;
        this.targetPose.fingers = [0.08, 0.06, 0.06, 0.08, 0.10];
        this.targetPose.splay = 0.45;
        return;
      }

      if (this.state === 'CONFIRM') {
        const part = H3D.PARTS.find((p) => p.id === this.targetPart);
        if (part) {
          const pre = H3D.solvePoseForPoint(part.slot.x, part.slot.y, part.slot.z, 'pregrasp');
          this.targetPose.pan = pre.pan;
          this.targetPose.tilt = pre.tilt;
          this.targetPose.wristPitch = pre.wristPitch;
        }
        this.targetPose.fingers = [0.06, 0.05, 0.05, 0.06, 0.08];
        this.targetPose.splay = 0.55;
        return;
      }

      if (this.state === 'PICK') {
        const part = H3D.PARTS.find((p) => p.id === this.targetPart);
        const targetPos = this._pickTargetPos || (part ? part.slot : { x: 14.2, y: 2.2, z: 0 });

        // Live mode subphase estimation based on elapsed time
        let sub = this._pickSub;
        if (this.liveMode) {
          if (this._phaseElapsed < 0.36) sub = 'DESCEND';
          else if (this._phaseElapsed < 0.78) sub = 'CLOSE';
          else sub = 'LIFT';
        }

        if (sub === 'RETRY_SLIP') {
          const pre = H3D.solvePoseForPoint(targetPos.x, targetPos.y, targetPos.z, 'pregrasp');
          this.targetPose.pan = pre.pan;
          this.targetPose.tilt = pre.tilt;
          this.targetPose.wristPitch = pre.wristPitch;
          this.targetPose.fingers = [0.12, 0.10, 0.10, 0.12, 0.14];
          this.targetPose.splay = 0.5;
        } else if (sub === 'DESCEND') {
          const grasp = H3D.solvePoseForPoint(targetPos.x, targetPos.y, targetPos.z, 'grasp');
          this.targetPose.pan = grasp.pan;
          this.targetPose.tilt = grasp.tilt;
          this.targetPose.wristPitch = grasp.wristPitch;
          this.targetPose.fingers = [0.10, 0.08, 0.08, 0.10, 0.12];
          this.targetPose.splay = 0.5;
        } else if (sub === 'CLOSE') {
          const grasp = H3D.solvePoseForPoint(targetPos.x, targetPos.y, targetPos.z, 'grasp');
          this.targetPose.pan = grasp.pan;
          this.targetPose.tilt = grasp.tilt;
          this.targetPose.wristPitch = grasp.wristPitch;
          // Staggered organic 5-finger closing around the ball!
          this.targetPose.fingers = [0.88, 0.90, 0.92, 0.88, 0.84];
          this.targetPose.splay = 0.05;
          if (this.liveMode) {
            const b = this.balls.find((item) => item.id === this.targetPart);
            const avg = this.pose.fingers.reduce((a, c) => a + c, 0) / 5;
            if (b && avg > 0.62) {
              this.heldBallId = b.id;
              b.zone = 'GRIP';
            }
          }
        } else {
          // LIFT
          const lift = H3D.solvePoseForPoint(targetPos.x, targetPos.y, targetPos.z, 'lift');
          this.targetPose.pan = lift.pan;
          this.targetPose.tilt = lift.tilt;
          this.targetPose.wristPitch = lift.wristPitch;
          this.targetPose.fingers = [0.90, 0.92, 0.94, 0.90, 0.86];
          this.targetPose.splay = 0.05;
        }
        return;
      }

      if (this.state === 'DELIVER') {
        const dropPose = H3D.solvePoseForPoint(-11.8, 12.6, 0, 'grasp');
        this.targetPose.pan = dropPose.pan;
        this.targetPose.tilt = dropPose.tilt;
        this.targetPose.wristPitch = dropPose.wristPitch;

        if (this.deliverSubstep === 'RELEASE') {
          // Smooth real-time 5-finger unfolding to drop ball onto chute!
          this.targetPose.fingers = [0.06, 0.05, 0.05, 0.06, 0.08];
          this.targetPose.splay = 0.65;
        } else {
          // Tightly gripping ball while rotating over to the chute
          this.targetPose.fingers = [0.90, 0.92, 0.94, 0.90, 0.86];
          this.targetPose.splay = 0.05;
        }
        return;
      }

      if (this.state === 'HANDOVER') {
        this.targetPose.pan = this.handoverSubstep === 'HOME' ? 90 : 72;
        this.targetPose.tilt = 92;
        this.targetPose.wristPitch = 4;
        this.targetPose.fingers = [0.12, 0.10, 0.10, 0.12, 0.14];
        this.targetPose.splay = 0.30;
        return;
      }

      if (this.state === 'RETURN') {
        const part = H3D.PARTS.find((p) => p.id === this.targetPart);
        if (this.returnSubstep === 'HOME' || !part) {
          this.targetPose.pan = 90;
          this.targetPose.tilt = 90;
          this.targetPose.wristPitch = 0;
          this.targetPose.fingers = [0.12, 0.12, 0.12, 0.12, 0.12];
        } else {
          const ret = H3D.solvePoseForPoint(part.slot.x, part.slot.y, part.slot.z, 'pregrasp');
          this.targetPose.pan = ret.pan;
          this.targetPose.tilt = ret.tilt;
          this.targetPose.wristPitch = ret.wristPitch;
          this.targetPose.fingers = [0.15, 0.12, 0.12, 0.14, 0.16];
        }
      }
    }

    _sampleTrail(ball) {
      const last = this.ballTrail[this.ballTrail.length - 1];
      if (!last || this.H3D.vDist(last, ball) > 0.65) {
        this.ballTrail.push({ x: ball.x, y: ball.y, z: ball.z });
        if (this.ballTrail.length > 80) this.ballTrail.shift();
      }
    }

    _updateBallPhysics(dt) {
      const H3D = this.H3D;
      const ch = H3D.LAYOUT.chute;
      const bay = H3D.LAYOUT.bay;
      const R = H3D.BALL_RADIUS;

      for (const b of this.balls) {
        if (!b.visible) continue;

        if (this.heldBallId === b.id) {
          b.zone = 'GRIP';
          // Smoothly nest ball into exact wireframe gripCenter
          const blend = Math.min(1, dt * 18);
          b.x = H3D.lerp(b.x, this.kinematics.gripCenter.x, blend);
          b.y = H3D.lerp(b.y, this.kinematics.gripCenter.y, blend);
          b.z = H3D.lerp(b.z, this.kinematics.gripCenter.z, blend);
          b.vx = 0;
          b.vy = 0;
          b.vz = 0;
          this._sampleTrail(b);
          continue;
        }

        if (b.zone === 'APPROACH' && this.draggingBallId !== b.id) {
          // Roll smoothly toward the open robotic hand
          b.x += b.vx * dt;
          b.z += b.vz * dt;
          b.y = R + 0.2;
          b.spinX -= (b.vz * dt) / R;
          this._sampleTrail(b);
          // Stop gently if it passes under without triggering
          if (b.z < 5.5) {
            b.vz = 0;
          }
          continue;
        }

        if (b.zone === 'AIR') {
          b.vy -= 68 * dt;
          b.x += b.vx * dt;
          b.y += b.vy * dt;
          b.z += b.vz * dt;
          b.spinZ += (b.vx * dt) / R;

          // Contact with inclined chute surface
          if (b.x <= ch.topX + 1.2 && b.x >= ch.bottomX - 0.5 && Math.abs(b.z) <= ch.halfZ) {
            const surfY = H3D.chuteSurfaceY(b.x) + R;
            if (b.y <= surfY) {
              b.y = surfY;
              b.vy = 0;
              b.vx = Math.min(b.vx, -9.0);
              b.zone = 'CHUTE';
            }
          } else if (b.y <= R + 0.2) {
            b.y = R + 0.2;
            b.vy = 0;
            b.vx *= 0.6;
            b.zone = b.x <= bay.maxX ? 'BAY' : 'TRAY';
          }
          this._sampleTrail(b);
          continue;
        }

        if (b.zone === 'CHUTE') {
          // Gravity acceleration down the 20° inclined chute
          b.vx -= 28.0 * dt;
          b.x += b.vx * dt;
          b.z *= Math.max(0, 1 - 5 * dt);
          b.y = H3D.chuteSurfaceY(b.x) + R;
          b.spinZ += (b.vx * dt) / R;
          this._sampleTrail(b);

          if (b.x <= ch.bottomX) {
            b.zone = 'BAY';
            b.vy = -5.0;
          }
          continue;
        }

        if (b.zone === 'BAY') {
          const floorY = bay.floorY + R;
          if (b.y > floorY || b.vy !== 0) {
            b.vy -= 72 * dt;
            b.y += b.vy * dt;
            if (b.y <= floorY) {
              b.y = floorY;
              if (Math.abs(b.vy) > 4.5) {
                b.vy = -b.vy * 0.34;
              } else {
                b.vy = 0;
              }
            }
          }
          b.x += b.vx * dt;
          b.vx *= Math.max(0, 1 - 3.6 * dt);
          if (Math.abs(b.vx) < 0.25) b.vx = 0;
          b.spinZ += (b.vx * dt) / R;

          // Keep inside receiving bay walls
          if (b.x < bay.minX + R + 0.5) {
            b.x = bay.minX + R + 0.5;
            b.vx = Math.abs(b.vx) * 0.38;
          }
          if (b.x > bay.maxX - R) {
            b.x = bay.maxX - R;
          }
          if (Math.abs(b.vx) > 0.4 || Math.abs(b.vy) > 0.4) {
            this._sampleTrail(b);
          }
        }
      }
    }

    _updateWorkerHand(dt) {
      const wh = this.workerHand;
      if (!wh.visible) {
        this.ultrasonicCm = 42.0;
        return;
      }

      const bayCenter = this.H3D.LAYOUT.bay.center;
      const usPos = this.H3D.LAYOUT.bay.ultrasonic;

      if (wh.phase === 'APPROACH') {
        wh.pos.x += 13.5 * dt;
        wh.pos.y = this.H3D.lerp(wh.pos.y, 3.6, Math.min(1, dt * 4));
        this.ultrasonicCm = Math.max(6.2, this.H3D.vDist(wh.pos, usPos));

        // Consecutive 3-hit ultrasonic proximity filter (<= 15cm), matching microbit/main.py!
        if (this.ultrasonicCm <= 15.0) {
          this.ultrasonicHits = Math.min(3, this.ultrasonicHits + dt * 5.0);
        }

        if (wh.pos.x >= bayCenter.x - 1.2) {
          wh.pos.x = bayCenter.x - 1.2;
          wh.phase = 'GRAB';
          wh.open = 0.22;
          const ball = this.balls.find((item) => item.id === this.targetPart);
          if (ball) {
            wh.carryingBallId = ball.id;
            ball.zone = 'HANDED';
            if (this.stock[ball.id] != null && this.stock[ball.id] > 0) {
              this.stock[ball.id] -= 1;
            }
          }

          if (!this.liveMode && this.state === 'HANDOVER' && this.handoverSubstep === 'WAIT') {
            this.localMetrics.successfulDeliveries += 1;
            if (this._runStartTime > 0) {
              this.localMetrics.deliveryTimes.push(Math.max(1.0, this.time - this._runStartTime));
            }
            this._log('RX', 'EVT,HAND');
            this._log('SYS', '초음파 연속 3회 감지 (≤ 15cm) — 작업자 수령 확인!');
            this._log('TX', 'HOME');
            this.handoverSubstep = 'HOME';
            this._phaseElapsed = 0;
            this.message = '수령 확인(EVT,HAND) — 중립 자세(HOME)로 복귀 중...';
            this.tone = 'pass';
          }
        }
      } else if (wh.phase === 'GRAB') {
        wh.open = this.H3D.lerp(wh.open, 0.15, Math.min(1, dt * 8));
        wh.pos.y += 5.0 * dt;
        if (wh.pos.y >= 6.5) wh.phase = 'EXIT';
      } else if (wh.phase === 'EXIT') {
        wh.pos.x -= 16.0 * dt;
        if (wh.pos.x <= -47.0) {
          wh.visible = false;
          if (wh.carryingBallId != null) {
            const b = this.balls.find((item) => item.id === wh.carryingBallId);
            if (b) b.visible = false;
            wh.carryingBallId = null;
          }
        }
      }

      // Move carried ball with worker hand
      if (wh.carryingBallId != null) {
        const b = this.balls.find((item) => item.id === wh.carryingBallId);
        if (b) {
          b.x = wh.pos.x + 1.4;
          b.y = wh.pos.y - 1.2;
          b.z = wh.pos.z;
        }
      }
    }

    _updateDistances() {
      const g = this.kinematics.gripCenter;
      for (const b of this.balls) {
        b.distToHand = this.H3D.vDist(g, b);
      }
    }

    _checkProximityAutoGrasp() {
      if (this.state !== 'IDLE' || this.manualServo) return;
      for (const b of this.balls) {
        if (!b.visible) continue;
        const eligible = b.zone === 'APPROACH' || this.draggingBallId === b.id;
        if (eligible && b.distToHand <= this.graspThresholdCm) {
          this._triggerProximityGrasp(b);
          return;
        }
      }
    }

    _updateActiveModules() {
      const m = new Set(['ui']);
      if (this.state === 'REQUEST' || this.virtualGesture.label !== 'NONE') m.add('gesture');
      if (this.state !== 'IDLE') {
        m.add('controller');
        m.add('link');
      }
      if (this.state === 'PICK' || this.state === 'DELIVER' || this.state === 'RETURN') m.add('robot');
      if (this.state === 'HANDOVER' || this.proximityMode || this.emptyAlertPart) m.add('handover');
      this.activeModules = m;
    }

    // Drag helpers for direct ball dragging on either canvas
    startDragBall(ballId) {
      if (this.state !== 'IDLE') return false;
      const b = this.balls.find((item) => item.id === ballId && item.visible);
      if (!b) return false;
      this.draggingBallId = ballId;
      this.focusBallId = ballId;
      this.activeBallId = ballId;
      this.proximityMode = true;
      this.manualServo = false;
      b.zone = 'APPROACH';
      b.vx = 0;
      b.vy = 0;
      b.vz = 0;
      this.ballTrail = [{ x: b.x, y: b.y, z: b.z }];
      return true;
    }

    moveDraggedBall(wx, wz) {
      if (this.draggingBallId == null) return;
      const b = this.balls.find((item) => item.id === this.draggingBallId);
      if (!b) return;
      const prevX = b.x;
      const prevZ = b.z;
      b.x = this.H3D.clamp(wx, -16, 22);
      b.z = this.H3D.clamp(wz, -13, 25);
      b.y = this.H3D.BALL_RADIUS + 0.2;
      b.spinZ += (b.x - prevX) / b.radius;
      b.spinX -= (b.z - prevZ) / b.radius;
      this._sampleTrail(b);
    }

    endDragBall() {
      this.draggingBallId = null;
    }

    getLedMatrix() {
      return LED_PATTERNS[this.ledKey] || LED_PATTERNS.IDLE;
    }

    getLocalKpis() {
      const m = this.localMetrics;
      const rate = m.totalRequests > 0 ? Math.round((m.successfulDeliveries / m.totalRequests) * 1000) / 10 : null;
      const avgTime = m.deliveryTimes.length > 0
        ? Math.round((m.deliveryTimes.reduce((a, b) => a + b, 0) / m.deliveryTimes.length) * 100) / 100
        : null;
      return {
        total_requests: m.totalRequests,
        successful_deliveries: m.successfulDeliveries,
        cancelled_requests: m.cancelledRequests,
        false_triggers: m.falseTriggers,
        delivery_success_rate: rate,
        avg_delivery_seconds: avgTime,
      };
    }
  }

  const api = {
    AiHandSimulator,
    LED_PATTERNS,
    ZONE_KO,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    global.AiHandSim = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
