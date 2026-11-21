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

  let visLabels = [];

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

    // Labels are laid out later (no overlaps)
    if (opts && opts.topLabel) {
      visLabels.push({ x: x1 + bw / 2, y: y1 - 2, text: opts.topLabel, priority: active ? 1 : 4, color, active, prefer: 'up' });
    }
    if (opts && opts.bottomLabel) {
      visLabels.push({ x: x1 + bw / 2, y: y2 + 2, text: opts.bottomLabel, priority: active ? 2 : 5, color, active: false, prefer: 'below' });
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
  /**
   * Tray inspection camera: only the parts tray and what the camera recognizes on it.
   * Returns { detections, cam, basis } (detections are clickable to request a part).
   */
  function renderVisionCamera(ctx, w, h, world) {
    const H3D = global.AiHand3D;
    if (!H3D) return { detections: [] };

    ctx.clearRect(0, 0, w, h);
    visLabels = [];

    const bg = ctx.createRadialGradient(w * 0.5, h * 0.5, 20, w * 0.5, h * 0.5, Math.max(w, h) * 0.72);
    bg.addColorStop(0, '#161A20');
    bg.addColorStop(1, '#0E1115');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    const cam = H3D.CAM_PRESETS.tray;
    const basis = H3D.getCameraBasis(cam);
    const t = H3D.LAYOUT.tray;
    const active = world.state !== 'IDLE' && !world.emergencyStopped;

    // 1. Tray body + slot dividers
    H3D.drawWireBox(ctx, t.minX, t.maxX, 0, 1.1, t.minZ, t.maxZ, basis, cam, w, h, 'rgba(160, 168, 178, 0.75)', 1.3, 'rgba(32, 37, 44, 0.45)');
    H3D.drawLine3D(ctx, { x: t.minX, y: 1.1, z: -3.2 }, { x: t.maxX, y: 1.1, z: -3.2 }, basis, cam, w, h, 'rgba(130, 138, 148, 0.6)', 1);
    H3D.drawLine3D(ctx, { x: t.minX, y: 1.1, z: 3.2 }, { x: t.maxX, y: 1.1, z: 3.2 }, basis, cam, w, h, 'rgba(130, 138, 148, 0.6)', 1);

    const pulse = 0.5 + 0.5 * Math.sin(world.time * 9.0);
    const detections = [];
    const inTrayView = (b) => b.x >= t.minX - 1 && b.x <= t.maxX + 1 && b.z >= t.minZ - 1 && b.z <= t.maxZ + 1 && b.y < 14;

    for (const part of H3D.PARTS) {
      const stock = world.stock[part.id] != null ? world.stock[part.id] : 3;
      const isTarget = active && world.targetPart === part.id;
      const ball = world.balls.find((b) => b.id === part.id);
      const visible = ball && ball.visible && inTrayView(ball);

      // slot ring
      H3D.drawCircle3D(ctx, { x: part.slot.x, y: 0.25, z: part.slot.z }, 2.55, { x: 1, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }, 20, basis, cam, w, h,
        stock <= 0 ? '#D9534F' : isTarget ? '#D69E72' : 'rgba(130, 138, 148, 0.45)', isTarget ? 1.8 : 1.0, null);

      if (visible) {
        const lifted = world.heldBallId === part.id;
        const sb = H3D.drawWireBall(ctx, ball, basis, cam, w, h, { isTarget, isGrasped: lifted });
        if (!sb) continue;
        const pad = Math.max(8, sb.r * 0.45);
        const det = { ball, cx: sb.cx, cy: sb.cy, r: sb.r, x1: sb.cx - sb.r - pad, y1: sb.cy - sb.r - pad, x2: sb.cx + sb.r + pad, y2: sb.cy + sb.r + pad };
        detections.push(det);

        let color = '#939BA6';
        let fill = 'rgba(180, 188, 198, 0.04)';
        let status = null;
        let on = false;
        if (lifted) { color = '#6FBF73'; fill = 'rgba(111, 191, 115, 0.12)'; status = '집는 중'; on = true; }
        else if (isTarget) { color = '#D69E72'; fill = 'rgba(214, 158, 114, 0.12)'; status = '요청됨'; on = true; }
        const conf = Math.round((0.95 + part.id * 0.011 + Math.sin(world.time * 2 + part.id) * 0.008) * 100);
        drawDetectionBox(ctx, det.x1, det.y1, det.x2, det.y2, color, {
          active: on, pulse, fill,
          topLabel: `${part.name} ${conf}%`,
          bottomLabel: status,
        });
      } else {
        // nothing in the slot
        const pe = H3D.projectPoint({ x: part.slot.x, y: 0.6, z: part.slot.z }, basis, cam, w, h);
        if (!pe) continue;
        const r = Math.max(16, 2.6 * pe.scale);
        const empty = stock <= 0;
        drawDetectionBox(ctx, pe.x - r, pe.y - r, pe.x + r, pe.y + r, empty ? '#D9534F' : 'rgba(150, 158, 168, 0.7)', {
          active: empty, pulse, dashed: true,
          fill: empty ? 'rgba(217, 83, 79, 0.10)' : null,
          topLabel: part.name,
          bottomLabel: empty ? '비어 있음 · 보충 필요' : '로봇이 집어 감',
        });
      }
    }

    // 2. Camera HUD (top bar + bottom hint)
    ctx.save();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.016)';
    for (let y = 0; y < h; y += 4) ctx.fillRect(0, y, w, 1);

    ctx.fillStyle = 'rgba(14, 17, 21, 0.88)';
    ctx.fillRect(0, 0, w, 24);
    ctx.fillStyle = '#D9534F';
    ctx.beginPath();
    ctx.arc(12, 12, 3.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = '600 10px "IBM Plex Mono", Consolas, monospace';
    ctx.fillStyle = '#DCE0E5';
    const leftTxt = 'CAM-01 · 부품 트레이';
    ctx.fillText(leftTxt, 22, 15.5);
    const leftEnd = 22 + ctx.measureText(leftTxt).width + 16;
    const now = new Date();
    const ts = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
    const rightOptions = [`인식 ${detections.length}/3  ${ts}`, `인식 ${detections.length}/3`, ''];
    const rightTxt = rightOptions.find((x) => ctx.measureText(x).width <= w - 10 - leftEnd) || '';
    ctx.textAlign = 'right';
    ctx.fillStyle = '#8E97A2';
    if (rightTxt) ctx.fillText(rightTxt, w - 10, 15.5);
    ctx.textAlign = 'left';

    ctx.fillStyle = 'rgba(14, 17, 21, 0.80)';
    ctx.fillRect(0, h - 20, w, 20);
    ctx.font = '500 9.5px "IBM Plex Mono", Consolas, monospace';
    ctx.fillStyle = '#858E98';
    const hint = ['박스를 누르면 그 부품을 바로 요청해요', '박스 클릭: 부품 요청', ''].find((x) => ctx.measureText(x).width <= w - 20) || '';
    if (hint) ctx.fillText(hint, 10, h - 6.5);
    ctx.restore();

    // 3. Labels without overlap
    const layout = H3D.createLabelLayout(w, h, { top: 26, bottom: 22 });
    for (const d of detections) layout.reserve(d.cx - d.r, d.cy - d.r, d.r * 2, d.r * 2);
    visLabels.sort((a, b) => a.priority - b.priority);
    for (const L of visLabels) {
      H3D.drawLayoutLabel(ctx, layout, L.x, L.y, L.text, {
        prefer: L.prefer === 'below' ? 'below' : undefined,
        font: L.active ? '700 11px "IBM Plex Mono", Consolas, monospace' : '600 10.5px "IBM Plex Mono", Consolas, monospace',
        color: L.active ? '#121519' : L.color,
        plate: L.active ? L.color : 'rgba(18, 22, 27, 0.9)',
        border: L.active ? null : L.color,
        height: 17,
        padX: 5,
      });
    }
    visLabels = [];

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
