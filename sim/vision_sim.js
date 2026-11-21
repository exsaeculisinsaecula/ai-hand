/**
 * sim/vision_sim.js
 * Real-Time Simulated AI Vision Camera Feed:
 *  - Overhead/Front USB Webcam + WonderCam Fusion view of the live 3D scene
 *  - Real-time Object Detection Bounding Boxes (네모칸) with distance & proximity lock
 *  - 21-Point MediaPipe-style Worker Hand Gesture Skeleton PiP
 */
(function (global) {
  'use strict';

  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17],
  ];

  /**
   * Computes 21-point normalized 2D hand landmarks for a virtual gesture
   * fingersExt: [thumb, index, middle, ring, pinky] each in [0..1] (1=extended, 0=folded)
   */
  function buildVirtualHand21(fingersExt, sway) {
    const ext = fingersExt || [0.2, 0.2, 0.2, 0.2, 0.2];
    const s = sway || 0;
    const pts = new Array(21);

    // 0: Wrist
    pts[0] = { x: 0.5 + s * 0.015, y: 0.87 };

    // Thumb (1..4)
    const t = ext[0];
    pts[1] = { x: 0.39, y: 0.78 };
    pts[2] = { x: 0.31 - t * 0.04, y: 0.67 - t * 0.03 };
    pts[3] = { x: 0.24 - t * 0.07, y: 0.56 - t * 0.05 };
    pts[4] = { x: 0.22 + (1 - t) * 0.16 - t * 0.06, y: 0.45 + (1 - t) * 0.14 };

    // 4 main fingers: MCP -> PIP -> DIP -> TIP
    const bases = [
      { idx: 5, x: 0.38, y: 0.56, len: 0.34, dx: -0.04, e: ext[1] }, // Index
      { idx: 9, x: 0.48, y: 0.53, len: 0.38, dx: -0.005, e: ext[2] }, // Middle
      { idx: 13, x: 0.58, y: 0.55, len: 0.35, dx: 0.025, e: ext[3] }, // Ring
      { idx: 17, x: 0.67, y: 0.60, len: 0.28, dx: 0.055, e: ext[4] }, // Pinky
    ];

    for (const b of bases) {
      const e = b.e;
      // When folded (e -> 0), PIP rises slightly then DIP & TIP curl down into palm!
      const mcp = { x: b.x + s * 0.01, y: b.y };
      const pip = {
        x: mcp.x + b.dx * (0.38 + 0.2 * e),
        y: mcp.y - b.len * (0.20 + 0.22 * e),
      };
      const dip = {
        x: pip.x + b.dx * (0.35 * e),
        y: e > 0.45 ? pip.y - b.len * 0.28 * e : mcp.y - b.len * (0.06 * e) + 0.04 * (1 - e),
      };
      const tip = {
        x: dip.x + b.dx * (0.30 * e),
        y: e > 0.45 ? dip.y - b.len * 0.28 * e : mcp.y + 0.09 * (1 - e),
      };
      pts[b.idx] = mcp;
      pts[b.idx + 1] = pip;
      pts[b.idx + 2] = dip;
      pts[b.idx + 3] = tip;
    }
    return pts;
  }

  /**
   * Draws corner-bracketed CV detection bounding box (네모칸).
   */
  function drawDetectionBox(ctx, x1, y1, x2, y2, color, opts) {
    const bw = Math.max(12, x2 - x1);
    const bh = Math.max(12, y2 - y1);
    const corner = Math.min(11, bw * 0.32, bh * 0.32);
    const active = opts && opts.active;
    const pulse = opts && opts.pulse ? opts.pulse : 0;
    const dashed = opts && opts.dashed;

    ctx.save();
    // Subtle tinted box interior fill
    if (opts && opts.fill) {
      ctx.fillStyle = opts.fill;
      ctx.fillRect(x1, y1, bw, bh);
    }

    // Thin outer bounding frame
    ctx.strokeStyle = color;
    ctx.globalAlpha = active ? 0.65 + pulse * 0.35 : 0.48;
    ctx.lineWidth = active ? 1.5 : 1.0;
    if (dashed) ctx.setLineDash([4, 3]);
    ctx.strokeRect(x1, y1, bw, bh);
    ctx.setLineDash([]);

    // Bold corner brackets (네모칸 강조)
    ctx.globalAlpha = 1.0;
    ctx.lineWidth = active ? 2.5 : 1.8;
    ctx.beginPath();
    // Top-left
    ctx.moveTo(x1, y1 + corner);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1 + corner, y1);
    // Top-right
    ctx.moveTo(x2 - corner, y1);
    ctx.lineTo(x2, y1);
    ctx.lineTo(x2, y1 + corner);
    // Bottom-right
    ctx.moveTo(x2, y2 - corner);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2 - corner, y2);
    // Bottom-left
    ctx.moveTo(x1 + corner, y2);
    ctx.lineTo(x1, y2);
    ctx.lineTo(x1, y2 - corner);
    ctx.stroke();

    // Optional center lock reticle
    if (active) {
      const cx = (x1 + x2) * 0.5;
      const cy = (y1 + y2) * 0.5;
      ctx.lineWidth = 1.1;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.moveTo(cx - 7, cy);
      ctx.lineTo(cx - 2, cy);
      ctx.moveTo(cx + 2, cy);
      ctx.lineTo(cx + 7, cy);
      ctx.moveTo(cx, cy - 7);
      ctx.lineTo(cx, cy - 2);
      ctx.moveTo(cx, cy + 2);
      ctx.lineTo(cx, cy + 7);
      ctx.stroke();
    }

    // Top label tag
    if (opts && opts.topLabel) {
      ctx.globalAlpha = 1.0;
      ctx.font = '600 9.5px "IBM Plex Mono", Consolas, monospace';
      const tw = ctx.measureText(opts.topLabel).width + 8;
      const ty = Math.max(24, y1 - 16);
      ctx.fillStyle = active ? color : 'rgba(18, 22, 27, 0.88)';
      ctx.fillRect(x1, ty, tw, 15);
      if (!active) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(x1, ty, tw, 15);
      }
      ctx.fillStyle = active ? '#121519' : color;
      ctx.fillText(opts.topLabel, x1 + 4, ty + 11);
    }

    // Bottom status / distance tag
    if (opts && opts.bottomLabel) {
      ctx.globalAlpha = 1.0;
      ctx.font = '600 9px "IBM Plex Mono", Consolas, monospace';
      const tw = ctx.measureText(opts.bottomLabel).width + 8;
      const by = y2 + 2;
      ctx.fillStyle = 'rgba(15, 19, 24, 0.90)';
      ctx.fillRect(x1, by, tw, 14);
      ctx.strokeStyle = color;
      ctx.lineWidth = 0.9;
      ctx.strokeRect(x1, by, tw, 14);
      ctx.fillStyle = color;
      ctx.fillText(opts.bottomLabel, x1 + 4, by + 10);
    }

    ctx.restore();
  }

  /**
   * Draws the 21-point Worker Gesture Camera PiP in the top-right corner of the Vision feed.
   */
  function drawGesturePiP(ctx, w, h, world) {
    const pw = Math.min(154, Math.round(w * 0.30));
    const ph = Math.min(126, Math.round(h * 0.37));
    const px = w - pw - 10;
    const py = 32;

    ctx.save();
    ctx.fillStyle = 'rgba(14, 18, 23, 0.90)';
    ctx.fillRect(px, py, pw, ph);

    const gLabel = world.virtualGesture.label || 'NONE';
    const isActive = gLabel !== 'NONE';
    const borderCol = gLabel === 'PALM'
      ? '#D9534F'
      : isActive
        ? '#6FBF73'
        : '#3C424B';

    ctx.strokeStyle = borderCol;
    ctx.lineWidth = isActive ? 1.4 : 1;
    ctx.strokeRect(px, py, pw, ph);

    // Header bar of PiP
    ctx.fillStyle = 'rgba(24, 27, 32, 0.95)';
    ctx.fillRect(px, py, pw, 17);
    ctx.font = '600 9px "IBM Plex Mono", Consolas, monospace';
    ctx.fillStyle = isActive ? borderCol : '#8D959E';
    ctx.fillText(`GESTURE · ${gLabel}`, px + 6, py + 12);
    ctx.fillStyle = '#6E7680';
    ctx.textAlign = 'right';
    ctx.fillText('21-PT', px + pw - 5, py + 12);
    ctx.textAlign = 'left';

    // Build 21-point virtual hand skeleton
    const sway = Math.sin(world.time * 2.4) * 0.4;
    const lms = buildVirtualHand21(world.virtualGesture.fingersSmooth, sway);

    const ox = px + 8;
    const oy = py + 19;
    const iw = pw - 16;
    const ih = ph - 24;

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    const sPts = lms.map((p) => {
      const sx = ox + p.x * iw;
      const sy = oy + p.y * ih;
      if (sx < minX) minX = sx;
      if (sy < minY) minY = sy;
      if (sx > maxX) maxX = sx;
      if (sy > maxY) maxY = sy;
      return { x: sx, y: sy };
    });

    // Hand detection bounding box inside PiP
    drawDetectionBox(
      ctx,
      minX - 6,
      minY - 6,
      maxX + 6,
      maxY + 6,
      isActive ? borderCol : 'rgba(130, 138, 148, 0.45)',
      { active: isActive }
    );

    // Skeleton connections
    ctx.strokeStyle = isActive ? '#E4E7EB' : 'rgba(142, 150, 158, 0.5)';
    ctx.lineWidth = 1.4;
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(sPts[a].x, sPts[a].y);
      ctx.lineTo(sPts[b].x, sPts[b].y);
      ctx.stroke();
    }

    // 21 landmark dots
    for (let i = 0; i < sPts.length; i++) {
      const isTip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
      ctx.fillStyle = isActive ? (isTip ? '#ECEEEB' : borderCol) : '#69717B';
      ctx.beginPath();
      ctx.arc(sPts[i].x, sPts[i].y, isTip ? 2.5 : 1.8, 0, Math.PI * 2);
      ctx.fill();
    }

    // Hold progress ring during REQUEST / CONFIRM
    if (world.state === 'REQUEST' || world.state === 'CONFIRM') {
      const pct = Math.min(1, (world.progress || 0) / 100);
      const r = 11;
      const cx = px + pw - 16;
      const cy = py + ph - 16;
      ctx.strokeStyle = 'rgba(255,255,255,0.14)';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();

      ctx.strokeStyle = world.state === 'REQUEST' ? '#D69E72' : '#6FBF73';
      ctx.beginPath();
      ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + pct * Math.PI * 2);
      ctx.stroke();

      ctx.font = '600 8px "IBM Plex Mono", Consolas, monospace';
      ctx.fillStyle = '#E4E7EB';
      ctx.textAlign = 'center';
      ctx.fillText(world.state === 'REQUEST' ? 'H' : 'C', cx, cy + 3);
      ctx.textAlign = 'left';
    }

    ctx.restore();
  }

  /**
   * Renders the full Simulated AI Vision Camera feed with live bounding boxes & proximity lock.
   * Returns { detections: [{ ball, x1, y1, x2, y2, cx, cy }] } for direct mouse click/drag interaction.
   */
  function renderVisionCamera(ctx, w, h, world) {
    const H3D = global.AiHand3D;
    if (!H3D) return { detections: [] };

    ctx.clearRect(0, 0, w, h);

    // Camera monochrome optical backdrop
    const bg = ctx.createRadialGradient(w * 0.5, h * 0.5, 20, w * 0.5, h * 0.5, Math.max(w, h) * 0.72);
    bg.addColorStop(0, '#161A20');
    bg.addColorStop(1, '#0E1115');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    const cam = H3D.CAM_PRESETS.vision;
    const basis = H3D.getCameraBasis(cam);
    const kin = world.kinematics || H3D.computeHandKinematics(world.pose);

    // 1. Render the live 3D workspace as captured by the overhead USB/WonderCam
    H3D.drawWorkstationEnvironment(ctx, world, basis, cam, w, h, true);

    // 2. Render trajectory inside camera feed
    if (world.ballTrail && world.ballTrail.length > 1) {
      H3D.drawBallTrajectory3D(ctx, world.ballTrail, '#D69E72', basis, cam, w, h);
    }

    // 3. Draw Proximity Grasp Zone ring projected into camera space
    const threshold = world.graspThresholdCm || 8.5;
    const focusBall = world.getFocusBall();
    const minDist = focusBall ? focusBall.distToHand : 99;
    const anyInRange = minDist <= threshold;

    H3D.drawCircle3D(
      ctx,
      kin.gripCenter,
      threshold,
      { x: 1, y: 0, z: 0 },
      { x: 0, y: 0, z: 1 },
      24,
      basis,
      cam,
      w,
      h,
      anyInRange ? 'rgba(111, 191, 115, 0.80)' : 'rgba(166, 174, 184, 0.30)',
      anyInRange ? 1.6 : 1.0,
      anyInRange ? 'rgba(111, 191, 115, 0.08)' : null,
      [5, 4]
    );

    // 4. Render 3D balls and collect exact screen bounding boxes
    const detections = [];
    for (const ball of world.balls) {
      if (!ball.visible) continue;
      const isGrasped = world.heldBallId === ball.id;
      const isTarget = world.targetPart === ball.id || ball.id === world.focusBallId;
      const sb = H3D.drawWireBall(ctx, ball, basis, cam, w, h, { isTarget, isGrasped });
      if (sb) {
        const pad = Math.max(8, sb.r * 0.45);
        detections.push({
          ball,
          cx: sb.cx,
          cy: sb.cy,
          r: sb.r,
          x1: sb.cx - sb.r - pad,
          y1: sb.cy - sb.r - pad,
          x2: sb.cx + sb.r + pad,
          y2: sb.cy + sb.r + pad,
        });
      }
    }

    // 5. Render the wireframe robotic hand in camera space
    H3D.drawWireframeHand(ctx, kin, world.tone, world.emergencyStopped, basis, cam, w, h, true);

    // 6. Subtle monotone robot hand tracker box
    const pHand = H3D.projectPoint(kin.palmCenter, basis, cam, w, h);
    if (pHand) {
      const hr = Math.max(24, 5.2 * pHand.scale);
      drawDetectionBox(
        ctx,
        pHand.x - hr,
        pHand.y - hr,
        pHand.x + hr,
        pHand.y + hr,
        world.emergencyStopped ? '#D9534F' : 'rgba(160, 168, 178, 0.48)',
        {
          topLabel: `AI-HAND · ${world.emergencyStopped ? 'STOP' : world.state}`,
          dashed: true,
        }
      );
    }

    // 7. Object Detection Bounding Boxes (네모칸):
    //    Standby = crisp monotone silver; Target/Approach/Chute = muted amber; Proximity Lock/Grasped/Bay = sage-green
    const pulse = 0.5 + 0.5 * Math.sin(world.time * 9.0);
    const pGrip = H3D.projectPoint(kin.gripCenter, basis, cam, w, h);

    for (const det of detections) {
      const b = det.ball;
      const isGrasped = world.heldBallId === b.id;
      const isTarget = world.targetPart === b.id;
      const inRange = !isGrasped && b.distToHand <= threshold;
      const approaching = !isGrasped && b.distToHand <= threshold + 6.5;
      const onChute = b.zone === 'CHUTE';
      const inBay = b.zone === 'BAY';

      let boxColor = '#939BA6';
      let fill = 'rgba(180, 188, 198, 0.04)';
      let active = false;
      let statusTag = `D:${b.distToHand.toFixed(1)}cm`;

      if (isGrasped) {
        boxColor = '#6FBF73';
        fill = 'rgba(111, 191, 115, 0.12)';
        active = true;
        statusTag = 'GRASPED · 파지 중';
      } else if (onChute) {
        boxColor = '#D69E72';
        fill = 'rgba(214, 158, 114, 0.12)';
        active = true;
        const spd = Math.hypot(b.vx || 0, b.vy || 0, b.vz || 0);
        statusTag = `CHUTE · ${spd.toFixed(0)}cm/s`;
      } else if (inBay) {
        boxColor = '#6FBF73';
        fill = 'rgba(111, 191, 115, 0.12)';
        active = true;
        statusTag = 'ARRIVED · 수령 대기';
      } else if (inRange) {
        boxColor = '#6FBF73';
        fill = 'rgba(111, 191, 115, 0.16)';
        active = true;
        statusTag = `근접 ${b.distToHand.toFixed(1)}cm → GRASP`;
      } else if (isTarget || approaching) {
        boxColor = '#D69E72';
        fill = 'rgba(214, 158, 114, 0.10)';
        active = true;
        statusTag = `${isTarget ? 'LOCK' : 'APPROACH'} · ${b.distToHand.toFixed(1)}cm`;
      }

      if (pGrip && (active || b.id === world.focusBallId) && !isGrasped && !onChute && !inBay) {
        ctx.save();
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = inRange ? 1.6 : 1.0;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(pGrip.x, pGrip.y);
        ctx.lineTo(det.cx, det.cy);
        ctx.stroke();
        ctx.restore();
      }

      const spd = Math.hypot(b.vx || 0, b.vy || 0, b.vz || 0);
      if (spd > 2.5 && !isGrasped) {
        const nextPt = H3D.projectPoint(
          { x: b.x + b.vx * 0.22, y: b.y + b.vy * 0.22, z: b.z + b.vz * 0.22 },
          basis,
          cam,
          w,
          h
        );
        if (nextPt) {
          ctx.save();
          ctx.strokeStyle = '#ECEEEB';
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(det.cx, det.cy);
          ctx.lineTo(nextPt.x, nextPt.y);
          ctx.stroke();
          ctx.restore();
        }
      }

      const conf = (0.95 + (b.id * 0.011) + Math.sin(world.time * 2 + b.id) * 0.008).toFixed(2);
      drawDetectionBox(ctx, det.x1, det.y1, det.x2, det.y2, boxColor, {
        active,
        pulse,
        fill,
        topLabel: `#${b.id} ${b.code} ${Math.round(Number(conf) * 100)}%`,
        bottomLabel: statusTag,
      });
    }

    // Highlight empty slot alert box if ERR,EMPTY recently occurred
    if (world.emptyAlertPart) {
      const part = H3D.PARTS.find((p) => p.id === world.emptyAlertPart);
      if (part) {
        const pe = H3D.projectPoint(part.slot, basis, cam, w, h);
        if (pe) {
          drawDetectionBox(ctx, pe.x - 24, pe.y - 24, pe.x + 24, pe.y + 24, '#D9534F', {
            active: true,
            pulse,
            dashed: true,
            fill: 'rgba(217, 83, 79, 0.14)',
            topLabel: `SLOT #${part.id} EMPTY`,
            bottomLabel: 'ERR,EMPTY · 보충 필요',
          });
        }
      }
    }

    // 8. Subtle camera scanline & monochrome optical HUD overlay
    ctx.save();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.016)';
    for (let y = 0; y < h; y += 4) {
      ctx.fillRect(0, y, w, 1);
    }

    // Top camera telemetry bar
    ctx.fillStyle = 'rgba(14, 17, 21, 0.88)';
    ctx.fillRect(0, 0, w, 24);
    ctx.fillStyle = '#D9534F';
    ctx.beginPath();
    ctx.arc(12, 12, 3.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.font = '600 10px "IBM Plex Mono", Consolas, monospace';
    ctx.fillStyle = '#DCE0E5';
    ctx.fillText('REC  CAM-01 // VISION BBOX + PROXIMITY', 22, 15.5);

    const now = new Date();
    const ts = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
    ctx.textAlign = 'right';
    ctx.fillStyle = '#8E97A2';
    ctx.fillText(`OBJ:${detections.length}  THR:${threshold.toFixed(1)}cm  ${ts}`, w - 10, 15.5);
    ctx.textAlign = 'left';

    // Bottom interaction hint strip inside camera
    ctx.fillStyle = 'rgba(14, 17, 21, 0.80)';
    ctx.fillRect(0, h - 20, w, 20);
    ctx.font = '500 9.5px "IBM Plex Mono", Consolas, monospace';
    ctx.fillStyle = anyInRange ? '#6FBF73' : '#858E98';
    const hint = world.proximityMode
      ? 'PROXIMITY MODE: 공 네모칸을 드래그해 로봇손 가까이 가져가면 자동 파지'
      : 'CLICK BBOX: 부품 즉시 요청  ·  DRAG BALL: 근접 거리 테스트';
    ctx.fillText(hint, 10, h - 6.5);
    ctx.restore();

    // 9. Virtual 21-Point Gesture PiP (top-right)
    drawGesturePiP(ctx, w, h, world);

    return { detections, cam, basis };
  }

  const api = {
    HAND_CONNECTIONS,
    buildVirtualHand21,
    drawDetectionBox,
    renderVisionCamera,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    global.AiHandVision = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
