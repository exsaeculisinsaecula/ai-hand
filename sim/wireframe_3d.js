/**
 * sim/wireframe_3d.js
 * 3D Wireframe Kinematic Engine & Line-Based Hiwonder AiHand Renderer.
 * Zero external dependencies — works over http:// and file://.
 */
(function (global) {
  'use strict';

  const DEG = Math.PI / 180;
  const RAD = 180 / Math.PI;

  const BALL_RADIUS = 2.0;

  // 3D workstation layout (in cm, Y=0 table surface, +X right/tray, -X left/chute, +Z front)
  // Restrained monotone base tones for standby balls; semantic accent highlights only the active ball.
  const PALETTE = {
    wirePrimary: '#DDE1E6',
    wireSecondary: '#8B939E',
    wireDim: 'rgba(130, 138, 148, 0.38)',
    accent: '#D69E72',       // Active target / motion emphasis
    accentDim: 'rgba(214, 158, 114, 0.18)',
    pass: '#6FBF73',         // Grasp lock / handover success
    passDim: 'rgba(111, 191, 115, 0.18)',
    warn: '#D9534F',         // Fault / Emergency STOP only
  };

  // 실물 받기 작업에서 손에 쥔 공 (물빠진 색)
  const HELD_BALL = {
    red: { name: '빨간 공', color: '#C0716A', dim: 'rgba(192, 113, 106, 0.18)' },
    blue: { name: '파란 공', color: '#6F8FB5', dim: 'rgba(111, 143, 181, 0.18)' },
    green: { name: '초록 공', color: '#79A07A', dim: 'rgba(121, 160, 122, 0.18)' },
  };

  const PARTS = [
    {
      id: 1,
      name: '캡A',
      code: 'CAP-A',
      color: '#C8CED6',
      dimColor: 'rgba(200, 206, 214, 0.14)',
      pan: 135,
      slot: { x: 13.1, y: BALL_RADIUS + 0.2, z: 5.43 },
    },
    {
      id: 2,
      name: '캡B',
      code: 'CAP-B',
      color: '#AEB6C0',
      dimColor: 'rgba(174, 182, 192, 0.14)',
      pan: 150,
      slot: { x: 14.2, y: BALL_RADIUS + 0.2, z: 0.0 },
    },
    {
      id: 3,
      name: '캡C',
      code: 'CAP-C',
      color: '#919AA6',
      dimColor: 'rgba(145, 154, 166, 0.14)',
      pan: 165,
      slot: { x: 13.1, y: BALL_RADIUS + 0.2, z: -5.43 },
    },
  ];

  const LAYOUT = {
    ballRadius: BALL_RADIUS,
    turretHeight: 6.2,
    forearmLen: 10.8,
    palmLen: 5.6,
    tray: { minX: 9.6, maxX: 18.8, minZ: -9.4, maxZ: 9.4, y: 0.2 },
    chute: {
      topX: -10.4,
      topY: 9.6,
      bottomX: -26.0,
      bottomY: 2.8,
      halfZ: 3.4,
      wallH: 1.8,
    },
    bay: {
      minX: -36.8,
      maxX: -26.0,
      halfZ: 5.2,
      wallH: 3.0,
      floorY: 0.2,
      center: { x: -31.5, y: BALL_RADIUS + 0.2, z: 0.0 },
      ultrasonic: { x: -26.8, y: 8.8, z: 0.0 },
    },
    cameraStand: {
      x: 19.5,
      z: -13.0,
      topY: 24.0,
      lens: { x: 6.0, y: 23.0, z: -11.0 },
    },
  };

  function chuteSurfaceY(x) {
    const c = LAYOUT.chute;
    const t = clamp((c.topX - x) / (c.topX - c.bottomX), 0, 1);
    return c.topY + (c.bottomY - c.topY) * t;
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  // Vector helpers
  function vAdd(a, b) {
    return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
  }
  function vSub(a, b) {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  }
  function vScale(a, s) {
    return { x: a.x * s, y: a.y * s, z: a.z * s };
  }
  function vDot(a, b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }
  function vCross(a, b) {
    return {
      x: a.y * b.z - a.z * b.y,
      y: a.z * b.x - a.x * b.z,
      z: a.x * b.y - a.y * b.x,
    };
  }
  function vLen(a) {
    return Math.hypot(a.x, a.y, a.z);
  }
  function vNorm(a) {
    const l = vLen(a) || 1e-6;
    return { x: a.x / l, y: a.y / l, z: a.z / l };
  }
  function vDist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  /**
   * Maps micro:bit pan servo angle [0..180] to 3D azimuth yawDeg:
   *  40° -> -90° (facing Chute on -X)
   *  90° ->   0° (facing Front +Z)
   * 135° -> +67.5° (Tray 1)
   * 150° -> +90°   (Tray 2)
   * 165° -> +112.5°(Tray 3)
   */
  function panToYawDeg(pan) {
    const p = clamp(pan, 0, 180);
    if (p <= 90) {
      return ((p - 90) / 50) * 90;
    }
    return ((p - 90) / 60) * 90;
  }

  function yawDegToPan(yawDeg) {
    if (yawDeg <= 0) {
      return clamp(90 + (yawDeg / 90) * 50, 0, 180);
    }
    return clamp(90 + (yawDeg / 90) * 60, 0, 180);
  }

  /**
   * Computes full 3D forward kinematics for the 5-finger wireframe Hiwonder AiHand.
   * pose: { pan (0..180), tilt (20..160), wristPitch (deg), fingers: [5 curls 0..1], splay (0..1) }
   */
  function computeHandKinematics(pose) {
    const pan = pose.pan != null ? pose.pan : 90;
    const tilt = pose.tilt != null ? pose.tilt : 90;
    const wristPitch = pose.wristPitch != null ? pose.wristPitch : 0;
    const curls = pose.fingers || [0.1, 0.1, 0.1, 0.1, 0.1];
    const splay = pose.splay != null ? pose.splay : 0.25;

    const yaw = panToYawDeg(pan) * DEG;
    const sinY = Math.sin(yaw);
    const cosY = Math.cos(yaw);

    // Local turret orthonormal basis
    const fwd = { x: sinY, y: 0, z: cosY }; // direction turret points
    const right = { x: cosY, y: 0, z: -sinY }; // right side when looking along fwd
    const upWorld = { x: 0, y: 1, z: 0 };

    const turretOrigin = { x: 0, y: LAYOUT.turretHeight, z: 0 };

    // Tilt servo: 90° -> near vertical (10° lean), 130° -> reaches forward (65° lean), 155° -> deep reach (98° lean)
    const armAngle = clamp((tilt - 82) * 1.38, -25, 102) * DEG;
    const armDir = vAdd(vScale(fwd, Math.sin(armAngle)), vScale(upWorld, Math.cos(armAngle)));
    const armPerp = vAdd(vScale(fwd, Math.cos(armAngle)), vScale(upWorld, -Math.sin(armAngle)));

    const wrist = vAdd(turretOrigin, vScale(armDir, LAYOUT.forearmLen));

    // Palm orientation: starts aligned with armDir, pitched further forward by wristPitch
    const palmAngle = armAngle + wristPitch * DEG;
    const palmUp = vAdd(vScale(fwd, Math.sin(palmAngle)), vScale(upWorld, Math.cos(palmAngle)));
    const palmFront = vAdd(vScale(fwd, Math.cos(palmAngle)), vScale(upWorld, -Math.sin(palmAngle)));
    const palmRight = right;

    function palmToWorld(u, v, w) {
      return {
        x: wrist.x + palmRight.x * u + palmUp.x * v + palmFront.x * w,
        y: wrist.y + palmRight.y * u + palmUp.y * v + palmFront.y * w,
        z: wrist.z + palmRight.z * u + palmUp.z * v + palmFront.z * w,
      };
    }

    // Palm wireframe cage (front and back faces)
    const palmOutline = [
      [-2.1, 0.4],
      [-2.8, 2.2],
      [-2.6, 5.2],
      [-1.0, 5.8],
      [1.0, 5.8],
      [2.6, 5.3],
      [2.8, 2.0],
      [2.1, 0.4],
    ];
    const palmFrontPts = palmOutline.map(([u, v]) => palmToWorld(u, v, 0.65));
    const palmBackPts = palmOutline.map(([u, v]) => palmToWorld(u, v, -0.65));
    const palmCenter = palmToWorld(0, 3.1, 0.0);
    const ledPos = palmToWorld(0, 1.5, 0.72);

    // Exact grip pocket where a ball of radius 2.0cm sits comfortably inside curled 5 fingers
    const gripCenter = palmToWorld(0.0, 5.6, 2.55);

    // 4 main fingers (Index, Middle, Ring, Pinky) + Opposing Thumb
    // Order: [0: Thumb, 1: Index, 2: Middle, 3: Ring, 4: Pinky]
    const mainSpecs = [
      { idx: 1, name: '검지', u: 2.0, v: 5.5, baseSplay: 11, lens: [2.5, 1.85, 1.35], widths: [0.58, 0.50, 0.42, 0.34] },
      { idx: 2, name: '중지', u: 0.7, v: 5.85, baseSplay: 2, lens: [2.75, 2.05, 1.45], widths: [0.60, 0.52, 0.44, 0.35] },
      { idx: 3, name: '약지', u: -0.7, v: 5.65, baseSplay: -5, lens: [2.55, 1.9, 1.35], widths: [0.56, 0.48, 0.40, 0.33] },
      { idx: 4, name: '소지', u: -2.05, v: 5.15, baseSplay: -13, lens: [2.05, 1.55, 1.15], widths: [0.50, 0.42, 0.36, 0.29] },
    ];

    const fingerChains = new Array(5);

    // Build 4 main fingers
    for (const sp of mainSpecs) {
      const c = clamp(curls[sp.idx] != null ? curls[sp.idx] : 0.1, 0, 1);
      const splayRad = sp.baseSplay * (0.35 + 0.9 * splay) * (1 - 0.65 * c) * DEG;

      // Local finger plane basis (finger bends toward +palmFront)
      const fUp0 = vNorm(vAdd(vScale(palmUp, Math.cos(splayRad)), vScale(palmRight, Math.sin(splayRad))));
      const fSide = vNorm(vCross(palmFront, fUp0));
      const fIn = palmFront;

      const mcpDeg = lerp(5, 48, c);
      const pipDeg = lerp(8, 66, c);
      const dipDeg = lerp(6, 54, c);
      const segAngles = [mcpDeg, pipDeg, dipDeg];

      const p0 = palmToWorld(sp.u, sp.v, 0.15);
      const joints = [p0];
      const dirs = [];
      let accum = 0;

      for (let i = 0; i < 3; i++) {
        accum += segAngles[i] * DEG;
        const dir = vAdd(vScale(fUp0, Math.cos(accum)), vScale(fIn, Math.sin(accum)));
        dirs.push(dir);
        joints.push(vAdd(joints[joints.length - 1], vScale(dir, sp.lens[i])));
      }

      fingerChains[sp.idx] = {
        id: sp.idx,
        name: sp.name,
        curl: c,
        joints,
        dirs,
        side: fSide,
        widths: sp.widths,
        anglesDeg: { mcp: Math.round(mcpDeg), pip: Math.round(pipDeg), dip: Math.round(dipDeg) },
        tendonAnchor: palmToWorld(sp.u * 0.72, 1.3, 0.25),
      };
    }

    // Build opposing thumb (idx 0)
    {
      const c = clamp(curls[0] != null ? curls[0] : 0.1, 0, 1);
      const thumbBase = palmToWorld(2.65, 2.2, 0.55);

      // Thumb rotates from lateral abduction toward opposition across the front of the palm
      const oppAngle = lerp(22, 68, c) * DEG;
      const tOut0 = vNorm(
        vAdd(
          vAdd(vScale(palmRight, Math.cos(oppAngle) * 0.85), vScale(palmUp, 0.52)),
          vScale(palmFront, Math.sin(oppAngle) * 0.95 + 0.3)
        )
      );
      const tIn = vNorm(vAdd(vScale(palmRight, -0.78), vScale(palmFront, 0.45)));
      // Re-orthogonalize tIn against tOut0
      const tInOrtho = vNorm(vSub(tIn, vScale(tOut0, vDot(tIn, tOut0))));
      const tSide = vNorm(vCross(tInOrtho, tOut0));

      const lens = [2.1, 1.75, 1.35];
      const widths = [0.68, 0.58, 0.48, 0.38];
      const a0 = lerp(6, 26, c);
      const a1 = lerp(8, 44, c);
      const a2 = lerp(6, 40, c);
      const segAngles = [a0, a1, a2];

      const joints = [thumbBase];
      const dirs = [];
      let accum = 0;
      for (let i = 0; i < 3; i++) {
        accum += segAngles[i] * DEG;
        const dir = vAdd(vScale(tOut0, Math.cos(accum)), vScale(tInOrtho, Math.sin(accum)));
        dirs.push(dir);
        joints.push(vAdd(joints[joints.length - 1], vScale(dir, lens[i])));
      }

      fingerChains[0] = {
        id: 0,
        name: '엄지',
        curl: c,
        joints,
        dirs,
        side: tSide,
        widths,
        anglesDeg: { mcp: Math.round(a0), pip: Math.round(a1), dip: Math.round(a2) },
        tendonAnchor: palmToWorld(1.6, 1.1, 0.3),
      };
    }

    return {
      pan,
      tilt,
      wristPitch,
      yawDeg: panToYawDeg(pan),
      turretOrigin,
      wrist,
      palmCenter,
      gripCenter,
      ledPos,
      fwd,
      right,
      armDir,
      armPerp,
      palmUp,
      palmFront,
      palmRight,
      palmFrontPts,
      palmBackPts,
      fingerChains,
    };
  }

  /**
   * Exact 2-link analytical IK in the vertical radial plane (r, y) so gripCenter aligns with (tx, ty, tz).
   */
  function solvePoseForPoint(tx, ty, tz, mode) {
    const yawDeg = Math.atan2(tx, tz) * RAD;
    const pan = yawDegToPan(yawDeg);
    let r = Math.hypot(tx, tz);
    let y = ty;

    if (mode === 'lift') {
      r = Math.min(r, 12.0);
      y = Math.max(ty + 9.5, 12.5);
    } else if (mode === 'pregrasp') {
      y = ty + 3.6;
    }

    const L1 = LAYOUT.forearmLen; // 10.8
    const gv = 5.6;
    const gw = 2.55;
    const L2 = Math.hypot(gv, gw); // ~6.15
    const gripOffsetRad = Math.atan2(gw, gv); // ~0.427 rad

    const dy = y - LAYOUT.turretHeight;
    let dist = Math.hypot(r, dy);
    const maxR = L1 + L2 - 0.15;
    const minR = Math.abs(L1 - L2) + 0.4;
    if (dist > maxR) {
      const s = maxR / dist;
      r *= s;
      dist = maxR;
    } else if (dist < minR) {
      dist = minR;
    }

    // Angle from +Y vertical toward +r radial
    const targetAng = Math.atan2(r, y - LAYOUT.turretHeight);
    // Law of cosines for elbow-up solution (arm stays high, palm pitches down to grasp from above!)
    const cosA = clamp((L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist), -1, 1);
    const cosB = clamp((L1 * L1 + L2 * L2 - dist * dist) / (2 * L1 * L2), -1, 1);
    const alpha = Math.acos(cosA);
    const beta = Math.acos(cosB);

    const armAngleRad = targetAng - alpha;
    const effPalmAngleRad = armAngleRad + (Math.PI - beta);
    const palmAngleRad = effPalmAngleRad - gripOffsetRad;

    const tilt = clamp(82 + (armAngleRad * RAD) / 1.38, 25, 155);
    const wristPitch = clamp((palmAngleRad - armAngleRad) * RAD, -20, 88);
    return { pan, tilt, wristPitch };
  }

  // Camera presets
  const CAM_PRESETS = {
    iso: { yaw: -0.26, pitch: 0.42, dist: 58, target: { x: -5.0, y: 6.5, z: 0.5 }, fov: 44 },
    hand: { yaw: -0.22, pitch: 0.28, dist: 30, target: { x: 2.5, y: 10.5, z: 2.0 }, fov: 42 },
    side: { yaw: 0.0, pitch: 0.14, dist: 62, target: { x: -6.5, y: 7.0, z: 0.0 }, fov: 42 },
    top: { yaw: 0.0, pitch: 1.30, dist: 66, target: { x: -6.0, y: 2.0, z: 0.0 }, fov: 44 },
    // Vision camera mounted on overhead-front USB webcam stand
    vision: { yaw: 0.18, pitch: 0.68, dist: 50, target: { x: -2.0, y: 4.0, z: 1.5 }, fov: 50 },
    // Tray inspection camera: looking down at the parts tray, slots 1-2-3 left to right
    tray: { yaw: Math.PI / 2, pitch: 1.47, dist: 27, target: { x: 14.2, y: 0.4, z: 0.0 }, fov: 40 },
  };

  function getCameraBasis(cam) {
    const cy = Math.cos(cam.yaw);
    const sy = Math.sin(cam.yaw);
    const cp = Math.cos(cam.pitch);
    const sp = Math.sin(cam.pitch);

    // Camera position orbiting around cam.target
    const pos = {
      x: cam.target.x + cam.dist * cp * sy,
      y: cam.target.y + cam.dist * sp,
      z: cam.target.z + cam.dist * cp * cy,
    };

    const forward = vNorm(vSub(cam.target, pos));
    const worldUp = { x: 0, y: 1, z: 0 };
    const right = vNorm(vCross(forward, worldUp));
    const up = vCross(right, forward);
    return { pos, forward, right, up };
  }

  function projectPoint(pt, basis, cam, w, h) {
    const d = vSub(pt, basis.pos);
    const z = vDot(d, basis.forward);
    if (z <= 0.5) return null;
    const x = vDot(d, basis.right);
    const y = vDot(d, basis.up);
    const f = (h * 0.5) / Math.tan((cam.fov * 0.5) * DEG);
    return {
      x: w * 0.5 + (x / z) * f,
      y: h * 0.5 - (y / z) * f,
      z,
      scale: f / z,
    };
  }

  /**
   * Unprojects a 2D canvas coordinate (sx, sy) onto a horizontal plane y = planeY.
   */
  function unprojectToPlaneY(sx, sy, cam, w, h, planeY) {
    const basis = getCameraBasis(cam);
    const f = (h * 0.5) / Math.tan((cam.fov * 0.5) * DEG);
    const nx = (sx - w * 0.5) / f;
    const ny = -(sy - h * 0.5) / f;
    const rayDir = vNorm(
      vAdd(vAdd(basis.forward, vScale(basis.right, nx)), vScale(basis.up, ny))
    );
    if (Math.abs(rayDir.y) < 1e-4) return null;
    const t = (planeY - basis.pos.y) / rayDir.y;
    if (t <= 0) return null;
    return vAdd(basis.pos, vScale(rayDir, t));
  }

  // Low-level wireframe drawing helpers
  function drawLine3D(ctx, a, b, basis, cam, w, h, stroke, width, dash) {
    const pa = projectPoint(a, basis, cam, w, h);
    const pb = projectPoint(b, basis, cam, w, h);
    if (!pa || !pb) return;
    ctx.save();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width || 1;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
    ctx.restore();
  }

  function drawPolyline3D(ctx, pts, basis, cam, w, h, stroke, width, close, fill, dash) {
    const proj = [];
    for (let i = 0; i < pts.length; i++) {
      const p = projectPoint(pts[i], basis, cam, w, h);
      if (!p) return null;
      proj.push(p);
    }
    if (proj.length < 2) return proj;
    ctx.save();
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(proj[0].x, proj[0].y);
    for (let i = 1; i < proj.length; i++) ctx.lineTo(proj[i].x, proj[i].y);
    if (close) ctx.closePath();
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width || 1;
      ctx.stroke();
    }
    ctx.restore();
    return proj;
  }

  function drawCircle3D(ctx, center, radius, axisA, axisB, segs, basis, cam, w, h, stroke, width, fill, dash) {
    const pts = [];
    for (let i = 0; i < segs; i++) {
      const a = (i / segs) * Math.PI * 2;
      const ca = Math.cos(a) * radius;
      const sa = Math.sin(a) * radius;
      pts.push({
        x: center.x + axisA.x * ca + axisB.x * sa,
        y: center.y + axisA.y * ca + axisB.y * sa,
        z: center.z + axisA.z * ca + axisB.z * sa,
      });
    }
    return drawPolyline3D(ctx, pts, basis, cam, w, h, stroke, width, true, fill, dash);
  }

  function drawWireBox(ctx, minX, maxX, minY, maxY, minZ, maxZ, basis, cam, w, h, stroke, width, fillTop) {
    const c = [
      { x: minX, y: minY, z: minZ },
      { x: maxX, y: minY, z: minZ },
      { x: maxX, y: minY, z: maxZ },
      { x: minX, y: minY, z: maxZ },
      { x: minX, y: maxY, z: minZ },
      { x: maxX, y: maxY, z: minZ },
      { x: maxX, y: maxY, z: maxZ },
      { x: minX, y: maxY, z: maxZ },
    ];
    if (fillTop) {
      drawPolyline3D(ctx, [c[4], c[5], c[6], c[7]], basis, cam, w, h, null, 0, true, fillTop);
    }
    drawPolyline3D(ctx, [c[0], c[1], c[2], c[3]], basis, cam, w, h, stroke, width, true);
    drawPolyline3D(ctx, [c[4], c[5], c[6], c[7]], basis, cam, w, h, stroke, width, true);
    for (let i = 0; i < 4; i++) {
      drawLine3D(ctx, c[i], c[i + 4], basis, cam, w, h, stroke, width);
    }
  }

  /**
   * Draws a 3D wireframe beam between p1 and p2 (4 longitudinal wireframe lines + cross rings).
   */
  function drawWireBeam(ctx, p1, p2, r1, r2, sideVec, basis, cam, w, h, stroke, fill, lineW) {
    const dir = vNorm(vSub(p2, p1));
    const n1 = vNorm(vSub(sideVec, vScale(dir, vDot(sideVec, dir))));
    const n2 = vNorm(vCross(dir, n1));

    const ring = (p, r) => [
      vAdd(p, vScale(n1, r)),
      vAdd(p, vScale(n2, r * 0.82)),
      vAdd(p, vScale(n1, -r)),
      vAdd(p, vScale(n2, -r * 0.82)),
    ];

    const a = ring(p1, r1);
    const b = ring(p2, r2);

    // Subtle wireframe face shading for depth readability
    if (fill) {
      drawPolyline3D(ctx, [a[0], b[0], b[2], a[2]], basis, cam, w, h, null, 0, true, fill);
    }
    // 4 longitudinal struts + end caps
    for (let i = 0; i < 4; i++) {
      drawLine3D(ctx, a[i], b[i], basis, cam, w, h, stroke, lineW || 1.25);
    }
    drawPolyline3D(ctx, a, basis, cam, w, h, stroke, (lineW || 1.25) * 0.85, true);
    drawPolyline3D(ctx, b, basis, cam, w, h, stroke, (lineW || 1.25) * 0.85, true);
  }

  /**
   * Draws a rotating 3D wireframe sphere (ball) with restrained monotone default & semantic active accent.
   */
  function drawWireBall(ctx, ball, basis, cam, w, h, opts) {
    const r = ball.radius || BALL_RADIUS;
    const isTarget = opts && opts.isTarget;
    const isGrasped = opts && opts.isGrasped;
    const isDelivered = ball.zone === 'BAY' || ball.zone === 'HANDED';
    const color = isGrasped || isDelivered
      ? PALETTE.pass
      : isTarget
        ? PALETTE.accent
        : (ball.color || '#B8C0CA');
    const coreFill = isGrasped || isDelivered
      ? PALETTE.passDim
      : isTarget
        ? PALETTE.accentDim
        : (ball.dimColor || 'rgba(184, 192, 202, 0.14)');

    // Ground shadow ring directly underneath the ball
    const groundY = ball.x <= LAYOUT.chute.topX && ball.x >= LAYOUT.chute.bottomX && Math.abs(ball.z) <= LAYOUT.chute.halfZ
      ? chuteSurfaceY(ball.x)
      : 0.15;
    if (ball.y > groundY + 0.4) {
      drawCircle3D(
        ctx,
        { x: ball.x, y: groundY, z: ball.z },
        r * clamp(1 - (ball.y - groundY) / 28, 0.45, 0.95),
        { x: 1, y: 0, z: 0 },
        { x: 0, y: 0, z: 1 },
        14,
        basis,
        cam,
        w,
        h,
        'rgba(180, 188, 196, 0.22)',
        1,
        'rgba(8, 10, 13, 0.35)'
      );
      drawLine3D(
        ctx,
        { x: ball.x, y: ball.y - r, z: ball.z },
        { x: ball.x, y: groundY, z: ball.z },
        basis,
        cam,
        w,
        h,
        isTarget ? 'rgba(214, 158, 114, 0.35)' : 'rgba(160, 168, 178, 0.22)',
        1,
        [3, 3]
      );
    }

    const pCenter = projectPoint(ball, basis, cam, w, h);
    if (!pCenter) return null;
    const screenR = Math.max(4, r * pCenter.scale);

    // Subtle shaded sphere core
    const grad = ctx.createRadialGradient(
      pCenter.x - screenR * 0.25,
      pCenter.y - screenR * 0.25,
      screenR * 0.1,
      pCenter.x,
      pCenter.y,
      screenR
    );
    grad.addColorStop(0, isTarget || isGrasped ? color : '#D8DCE2');
    grad.addColorStop(0.55, coreFill);
    grad.addColorStop(1, 'rgba(14, 16, 20, 0.16)');
    ctx.save();
    ctx.beginPath();
    ctx.arc(pCenter.x, pCenter.y, screenR, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = isTarget || isGrasped ? color : '#9AA3AE';
    ctx.lineWidth = isTarget || isGrasped ? 1.9 : 1.2;
    ctx.stroke();
    ctx.restore();

    // Rotating 3D wireframe meridian/equator rings so rolling spin is unmistakably visible
    const spinZ = ball.spinZ || 0;
    const spinX = ball.spinX || 0;
    const cz = Math.cos(spinZ);
    const sz = Math.sin(spinZ);
    const cx = Math.cos(spinX);
    const sx = Math.sin(spinX);

    const rot = (lx, ly, lz) => {
      const x1 = lx * cz - ly * sz;
      const y1 = lx * sz + ly * cz;
      const z1 = lz;
      const y2 = y1 * cx - z1 * sx;
      const z2 = y1 * sx + z1 * cx;
      return { x: ball.x + x1, y: ball.y + y2, z: ball.z + z2 };
    };

    const drawRing = (fn, col, lw) => {
      const pts = [];
      const segs = 18;
      for (let i = 0; i < segs; i++) {
        const a = (i / segs) * Math.PI * 2;
        pts.push(fn(a));
      }
      drawPolyline3D(ctx, pts, basis, cam, w, h, col, lw, true);
    };

    drawRing((a) => rot(Math.cos(a) * r, Math.sin(a) * r, 0), color, 1.1);
    drawRing((a) => rot(Math.cos(a) * r, 0, Math.sin(a) * r), 'rgba(220, 224, 230, 0.42)', 0.85);
    drawRing((a) => rot(0, Math.cos(a) * r, Math.sin(a) * r), color, 0.9);

    ctx.save();
    ctx.fillStyle = isTarget || isGrasped ? '#FFFFFF' : '#D0D6DC';
    ctx.beginPath();
    ctx.arc(pCenter.x, pCenter.y, Math.max(1.8, screenR * 0.14), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    return { cx: pCenter.x, cy: pCenter.y, r: screenR, z: pCenter.z };
  }

  /**
   * Renders the complete 3D Wireframe Robotic Hand in architectural monotone CAD lines.
   * Color appears ONLY on active grasping fingertips/tendons (amber/green) or emergency stop (red).
   */
  function drawWireframeHand(ctx, kin, stateTone, isEmergency, basis, cam, w, h, minimalMode) {
    const primary = isEmergency ? PALETTE.warn : '#DCE0E5';
    const secondary = isEmergency ? 'rgba(217, 83, 79, 0.65)' : '#868F9A';
    const accent = isEmergency ? PALETTE.warn : PALETTE.accent;
    const green = isEmergency ? PALETTE.warn : PALETTE.pass;

    const X = { x: 1, y: 0, z: 0 };
    const Z = { x: 0, y: 0, z: 1 };

    // 1. Base pedestal & 180° Pan Compass Ring on the table
    drawCircle3D(ctx, { x: 0, y: 0.15, z: 0 }, 5.2, X, Z, 16, basis, cam, w, h, 'rgba(130, 138, 148, 0.38)', 1.1, 'rgba(24, 27, 31, 0.5)');
    drawCircle3D(ctx, { x: 0, y: 0.2, z: 0 }, 7.0, X, Z, 24, basis, cam, w, h, 'rgba(130, 138, 148, 0.22)', 1, null, [3, 4]);

    const segs = 8;
    const baseBot = [];
    const baseTop = [];
    for (let i = 0; i < segs; i++) {
      const a = (i / segs) * Math.PI * 2 + (kin.pan * 0.015);
      baseBot.push({ x: Math.cos(a) * 4.4, y: 0.2, z: Math.sin(a) * 4.4 });
      baseTop.push({ x: Math.cos(a) * 3.1, y: LAYOUT.turretHeight - 0.6, z: Math.sin(a) * 3.1 });
    }
    drawPolyline3D(ctx, baseBot, basis, cam, w, h, '#59616B', 1.1, true, 'rgba(26, 30, 35, 0.55)');
    drawPolyline3D(ctx, baseTop, basis, cam, w, h, secondary, 1.2, true);
    for (let i = 0; i < segs; i++) {
      drawLine3D(ctx, baseBot[i], baseTop[i], basis, cam, w, h, 'rgba(118, 126, 136, 0.48)', 1.0);
    }

    // Pan direction needle on floor compass
    drawLine3D(ctx, { x: 0, y: 0.25, z: 0 }, vScale(kin.fwd, 7.6), basis, cam, w, h, '#B6BCC4', 1.4);

    // 2. Pan-Tilt Servo Gimbal Ring
    drawCircle3D(ctx, kin.turretOrigin, 2.3, kin.fwd, { x: 0, y: 1, z: 0 }, 16, basis, cam, w, h, primary, 1.35, 'rgba(220, 224, 229, 0.04)');
    drawCircle3D(ctx, kin.turretOrigin, 1.25, kin.right, { x: 0, y: 1, z: 0 }, 12, basis, cam, w, h, secondary, 1.0);

    // 3. Forearm Wireframe Truss Link
    drawWireBeam(
      ctx,
      kin.turretOrigin,
      kin.wrist,
      1.75,
      1.35,
      kin.right,
      basis,
      cam,
      w,
      h,
      primary,
      'rgba(220, 224, 229, 0.04)',
      1.3
    );
    drawLine3D(ctx, kin.turretOrigin, kin.wrist, basis, cam, w, h, 'rgba(154, 162, 172, 0.38)', 0.95, [4, 3]);

    // Wrist Joint Ring
    drawCircle3D(ctx, kin.wrist, 1.55, kin.palmRight, kin.palmFront, 14, basis, cam, w, h, primary, 1.3, 'rgba(28, 32, 37, 0.5)');

    // 4. 3D Wireframe Palm Cage
    drawPolyline3D(ctx, kin.palmBackPts, basis, cam, w, h, 'rgba(140, 148, 158, 0.65)', 1.1, true, 'rgba(22, 26, 31, 0.42)');
    drawPolyline3D(ctx, kin.palmFrontPts, basis, cam, w, h, primary, 1.45, true, 'rgba(220, 224, 229, 0.06)');
    for (let i = 0; i < kin.palmFrontPts.length; i++) {
      drawLine3D(ctx, kin.palmBackPts[i], kin.palmFrontPts[i], basis, cam, w, h, 'rgba(160, 168, 178, 0.48)', 1.0);
    }

    // Internal Tendon Wires: monotone silver at rest, subtle warm amber only under grip tension
    for (let i = 0; i < kin.fingerChains.length; i++) {
      const fc = kin.fingerChains[i];
      const tendonCol = fc.curl > 0.55 ? accent : 'rgba(138, 146, 156, 0.45)';
      drawLine3D(ctx, fc.tendonAnchor, fc.joints[0], basis, cam, w, h, tendonCol, fc.curl > 0.55 ? 1.4 : 0.95);
    }

    // Palm Status LED dot (minimal semantic dot)
    const pLed = projectPoint(kin.ledPos, basis, cam, w, h);
    if (pLed) {
      ctx.save();
      ctx.fillStyle = isEmergency ? PALETTE.warn : (stateTone === 'accent' ? accent : '#B6BCC4');
      ctx.beginPath();
      ctx.arc(pLed.x, pLed.y, Math.max(2.2, 0.42 * pLed.scale), 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // 5. 5 Articulated Wireframe Fingers (pure CAD bone-white; amber/green touch only when curled)
    const sortedFingers = kin.fingerChains.slice().sort((a, b) => {
      const pa = projectPoint(a.joints[1], basis, cam, w, h);
      const pb = projectPoint(b.joints[1], basis, cam, w, h);
      return (pb ? pb.z : 0) - (pa ? pa.z : 0);
    });

    for (const fc of sortedFingers) {
      const isCurled = fc.curl > 0.52;
      const boneStroke = isEmergency
        ? PALETTE.warn
        : isCurled
          ? '#F0E4D8'
          : '#D5DAE0';
      const boneFill = isCurled
        ? 'rgba(214, 158, 114, 0.11)'
        : 'rgba(215, 220, 226, 0.04)';

      for (let s = 0; s < 3; s++) {
        drawWireBeam(
          ctx,
          fc.joints[s],
          fc.joints[s + 1],
          fc.widths[s],
          fc.widths[s + 1],
          fc.side,
          basis,
          cam,
          w,
          h,
          boneStroke,
          boneFill,
          minimalMode ? 1.1 : 1.3
        );
      }

      for (let j = 0; j < 4; j++) {
        const pj = projectPoint(fc.joints[j], basis, cam, w, h);
        if (!pj) continue;
        const jr = Math.max(1.9, (j === 3 ? 0.46 : 0.38) * pj.scale);
        ctx.save();
        ctx.beginPath();
        ctx.arc(pj.x, pj.y, jr, 0, Math.PI * 2);
        if (j === 3) {
          ctx.fillStyle = isEmergency ? PALETTE.warn : isCurled ? green : '#C8CED6';
          ctx.fill();
          ctx.strokeStyle = '#181B1F';
          ctx.lineWidth = 0.9;
          ctx.stroke();
        } else {
          ctx.fillStyle = '#14171A';
          ctx.fill();
          ctx.strokeStyle = isCurled ? accent : '#8E97A2';
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    if (!minimalMode) {
      const pg = projectPoint(kin.gripCenter, basis, cam, w, h);
      if (pg) {
        ctx.save();
        ctx.strokeStyle = 'rgba(182, 188, 196, 0.40)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pg.x - 5, pg.y);
        ctx.lineTo(pg.x + 5, pg.y);
        ctx.moveTo(pg.x, pg.y - 5);
        ctx.lineTo(pg.x, pg.y + 5);
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  /**
   * Draws a wireframe virtual worker hand approaching the Receiving Bay during HANDOVER.
   */
  function drawWorkerHand3D(ctx, workerPos, openFactor, basis, cam, w, h) {
    if (!workerPos) return;
    const col = '#C8CED6';
    const wrist = { x: workerPos.x - 5.5, y: workerPos.y + 2.5, z: workerPos.z };
    const palm = { x: workerPos.x - 1.5, y: workerPos.y + 1.4, z: workerPos.z };

    drawLine3D(ctx, { x: wrist.x - 4, y: wrist.y + 1.5, z: wrist.z }, wrist, basis, cam, w, h, 'rgba(182, 188, 196, 0.45)', 1.8);
    drawLine3D(ctx, wrist, palm, basis, cam, w, h, col, 1.8);

    const offsets = [-1.4, -0.5, 0.4, 1.3];
    for (const dz of offsets) {
      const knuckle = { x: palm.x + 1.2, y: palm.y - 0.2, z: palm.z + dz };
      const mid = { x: knuckle.x + 1.6, y: knuckle.y - (1 - openFactor) * 1.1, z: knuckle.z };
      const tip = { x: mid.x + 1.1, y: mid.y - (1.2 - openFactor * 0.7), z: mid.z };
      drawPolyline3D(ctx, [wrist, knuckle, mid, tip], basis, cam, w, h, col, 1.3);
      const pt = projectPoint(tip, basis, cam, w, h);
      if (pt) {
        ctx.save();
        ctx.fillStyle = PALETTE.pass;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 2.3, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }
  }

  /**
   * Draws the shared 3D workstation environment (floor grid, tray, chute, receiving bay).
   */
  function drawWorkstationEnvironment(ctx, world, basis, cam, w, h, isVisionView) {
    // 1. Blueprint 3D Floor Grid
    const gridMinX = -42;
    const gridMaxX = 28;
    const gridMinZ = -18;
    const gridMaxZ = 18;
    const step = 4;

    for (let x = gridMinX; x <= gridMaxX; x += step) {
      const major = x % 12 === 0;
      drawLine3D(
        ctx,
        { x, y: 0, z: gridMinZ },
        { x, y: 0, z: gridMaxZ },
        basis,
        cam,
        w,
        h,
        major ? 'rgba(92, 104, 118, 0.28)' : 'rgba(64, 72, 82, 0.16)',
        major ? 1 : 0.75
      );
    }
    for (let z = gridMinZ; z <= gridMaxZ; z += step) {
      const major = z % 12 === 0;
      drawLine3D(
        ctx,
        { x: gridMinX, y: 0, z },
        { x: gridMaxX, y: 0, z },
        basis,
        cam,
        w,
        h,
        major ? 'rgba(92, 104, 118, 0.28)' : 'rgba(64, 72, 82, 0.16)',
        major ? 1 : 0.75
      );
    }

    // 2. Parts Tray (T1, T2, T3) on +X side
    const t = LAYOUT.tray;
    drawWireBox(
      ctx,
      t.minX,
      t.maxX,
      0,
      1.1,
      t.minZ,
      t.maxZ,
      basis,
      cam,
      w,
      h,
      'rgba(148, 158, 170, 0.65)',
      1.2,
      'rgba(32, 37, 44, 0.45)'
    );
    // Slot dividers & glowing target slot outline
    drawLine3D(ctx, { x: t.minX, y: 1.1, z: -3.2 }, { x: t.maxX, y: 1.1, z: -3.2 }, basis, cam, w, h, 'rgba(120,130,142,0.55)', 1);
    drawLine3D(ctx, { x: t.minX, y: 1.1, z: 3.2 }, { x: t.maxX, y: 1.1, z: 3.2 }, basis, cam, w, h, 'rgba(120,130,142,0.55)', 1);

    for (const part of PARTS) {
      const isSelected = world.targetPart === part.id && world.state !== 'IDLE';
      const stock = world.stock[part.id] != null ? world.stock[part.id] : 3;
      const ringCol = stock <= 0
        ? PALETTE.warn
        : isSelected
          ? PALETTE.accent
          : 'rgba(130, 138, 148, 0.38)';
      drawCircle3D(
        ctx,
        { x: part.slot.x, y: 0.25, z: part.slot.z },
        2.55,
        { x: 1, y: 0, z: 0 },
        { x: 0, y: 0, z: 1 },
        18,
        basis,
        cam,
        w,
        h,
        ringCol,
        isSelected ? 1.8 : 1.0,
        isSelected ? PALETTE.accentDim : null
      );

      if (!isVisionView) {
        const lblPt = projectPoint({ x: t.maxX + 1.2, y: 0.8, z: part.slot.z }, basis, cam, w, h);
        if (lblPt) {
          const col = stock <= 0 ? PALETTE.warn : (isSelected ? PALETTE.accent : '#9AA3AE');
          pendingLabels.push({
            x: lblPt.x + 4, y: lblPt.y, text: stock <= 0 ? `${part.name} · 비어 있음` : `${part.name} · ${stock}개`,
            priority: isSelected ? 1 : 3, prefer: 'right',
            opts: { font: isSelected ? LABEL_FONT_BOLD : LABEL_FONT, color: col, border: isSelected || stock <= 0 ? col : null, height: 19 },
          });
        }
      }
    }

    // 3. Inclined Chute (경사로 슈트) on -X side
    const ch = LAYOUT.chute;
    const chPts = [
      { x: ch.topX, y: ch.topY, z: -ch.halfZ },
      { x: ch.topX, y: ch.topY, z: ch.halfZ },
      { x: ch.bottomX, y: ch.bottomY, z: ch.halfZ },
      { x: ch.bottomX, y: ch.bottomY, z: -ch.halfZ },
    ];
    const isRolling = world.ballZone === 'CHUTE';
    const chuteStroke = isRolling ? PALETTE.accent : '#656D78';
    drawPolyline3D(
      ctx,
      chPts,
      basis,
      cam,
      w,
      h,
      chuteStroke,
      isRolling ? 1.7 : 1.2,
      true,
      isRolling ? 'rgba(214, 158, 114, 0.11)' : 'rgba(30, 34, 39, 0.45)'
    );

    // Chute side guard rails
    for (const sgn of [-1, 1]) {
      const z = sgn * ch.halfZ;
      const topA = { x: ch.topX, y: ch.topY + ch.wallH, z };
      const botA = { x: ch.bottomX, y: ch.bottomY + ch.wallH, z };
      drawLine3D(ctx, topA, botA, basis, cam, w, h, chuteStroke, 1.2);
      drawLine3D(ctx, { x: ch.topX, y: ch.topY, z }, topA, basis, cam, w, h, chuteStroke, 1.0);
      drawLine3D(ctx, { x: ch.bottomX, y: ch.bottomY, z }, botA, basis, cam, w, h, chuteStroke, 1.0);
      drawLine3D(ctx, { x: ch.topX - 0.8, y: ch.topY - 0.3, z }, { x: ch.topX - 0.8, y: 0, z }, basis, cam, w, h, 'rgba(98,106,116,0.5)', 1.1);
      drawLine3D(ctx, { x: ch.bottomX + 1.0, y: ch.bottomY + 0.3, z }, { x: ch.bottomX + 1.0, y: 0, z }, basis, cam, w, h, 'rgba(98,106,116,0.5)', 1.1);
    }

    // Animated gravity flow chevrons along chute surface
    for (let k = 0; k < 4; k++) {
      const phase = ((k / 4) + (isRolling ? (world.time * 0.9) % 1 : 0)) % 1;
      const cx = lerp(ch.topX - 1.5, ch.bottomX + 1.5, phase);
      const cy = chuteSurfaceY(cx) + 0.12;
      const col = isRolling ? 'rgba(214, 158, 114, 0.85)' : 'rgba(130, 138, 148, 0.24)';
      drawPolyline3D(
        ctx,
        [
          { x: cx + 1.0, y: cy + 0.4, z: -1.2 },
          { x: cx - 0.8, y: cy - 0.3, z: 0 },
          { x: cx + 1.0, y: cy + 0.4, z: 1.2 },
        ],
        basis,
        cam,
        w,
        h,
        col,
        isRolling ? 1.6 : 1.0
      );
    }

    // 4. Receiving Tray (수령 트레이) & Ultrasonic Sensor
    const b = LAYOUT.bay;
    const bayActive = world.state === 'HANDOVER' || world.ballZone === 'BAY';
    drawWireBox(
      ctx,
      b.minX,
      b.maxX,
      0,
      b.wallH,
      -b.halfZ,
      b.halfZ,
      basis,
      cam,
      w,
      h,
      bayActive ? PALETTE.pass : '#5C646E',
      bayActive ? 1.6 : 1.15,
      bayActive ? 'rgba(111, 191, 115, 0.09)' : 'rgba(22, 25, 30, 0.42)'
    );

    // Zone labels: tray / chute / receiving bay (the zone holding the part is boxed)
    if (!isVisionView) {
      const zone = activeZone(world);
      drawZoneLabel(ctx, projectPoint({ x: (t.minX + t.maxX) / 2, y: 0.4, z: t.maxZ + 3.2 }, basis, cam, w, h), '부품 트레이', zone === 'tray' ? PALETTE.accent : null);
      drawZoneLabel(ctx, projectPoint({ x: -18.5, y: 1.0, z: ch.halfZ + 3.0 }, basis, cam, w, h), '슈트', zone === 'chute' ? PALETTE.accent : null);
      drawZoneLabel(ctx, projectPoint({ x: b.center.x, y: 0.6, z: b.halfZ + 3.2 }, basis, cam, w, h), '수령 트레이', zone === 'bay' ? PALETTE.pass : null);
    }
  }

  /**
   * Draws the Proximity Grasp Zone Dome and live Distance Vector between the robotic hand and target ball.
   */
  function drawProximityOverlay3D(ctx, kin, world, basis, cam, w, h) {
    const threshold = world.graspThresholdCm || 8.5;
    const focusBall = world.getFocusBall();
    const showDome = world.proximityMode || world.state === 'PICK' || (focusBall && focusBall.distToHand < threshold + 10);

    if (showDome) {
      const dist = focusBall ? focusBall.distToHand : 99;
      const inRange = dist <= threshold;
      const approaching = dist <= threshold + 6;
      const domeCol = inRange
        ? 'rgba(111, 191, 115, 0.65)'
        : approaching
          ? 'rgba(214, 158, 114, 0.45)'
          : 'rgba(170, 178, 188, 0.24)';

      const X = { x: 1, y: 0, z: 0 };
      const Y = { x: 0, y: 1, z: 0 };
      const Z = { x: 0, y: 0, z: 1 };

      drawCircle3D(ctx, kin.gripCenter, threshold, X, Z, 24, basis, cam, w, h, domeCol, inRange ? 1.5 : 1.0, null, [4, 4]);
      drawCircle3D(ctx, kin.gripCenter, threshold, X, Y, 20, basis, cam, w, h, domeCol, 0.85, null, [4, 5]);
      drawCircle3D(ctx, kin.gripCenter, threshold, Z, Y, 20, basis, cam, w, h, domeCol, 0.85, null, [4, 5]);
    }

    if (focusBall && world.heldBallId !== focusBall.id) {
      const inRange = focusBall.distToHand <= threshold;
      const lineCol = inRange ? PALETTE.pass : (world.proximityMode || world.targetPart === focusBall.id ? PALETTE.accent : 'rgba(154, 162, 172, 0.45)');
      drawLine3D(ctx, kin.gripCenter, focusBall, basis, cam, w, h, lineCol, inRange ? 1.4 : 1.0, [4, 3]);
      const mid = {
        x: (kin.gripCenter.x + focusBall.x) * 0.5,
        y: (kin.gripCenter.y + focusBall.y) * 0.5 + 1.4,
        z: (kin.gripCenter.z + focusBall.z) * 0.5,
      };
      const pm = world.proximityMode ? projectPoint(mid, basis, cam, w, h) : null;
      if (pm) {
        pendingLabels.push({
          x: pm.x, y: pm.y, text: `${focusBall.distToHand.toFixed(1)} cm${inRange ? ' · 잡기' : ''}`, priority: 5,
          opts: { font: LABEL_FONT, color: inRange ? PALETTE.pass : '#DCE0E5', border: lineCol, height: 18 },
        });
      }
    }
  }

  /**
   * Draws the fading 3D motion trajectory trail of the active ball.
   */
  function drawBallTrajectory3D(ctx, trail, color, basis, cam, w, h) {
    if (!trail || trail.length < 2) return;
    for (let i = 1; i < trail.length; i++) {
      const t = i / trail.length;
      const p0 = projectPoint(trail[i - 1], basis, cam, w, h);
      const p1 = projectPoint(trail[i], basis, cam, w, h);
      if (!p0 || !p1) continue;
      ctx.save();
      ctx.strokeStyle = color || '#D4A588';
      ctx.globalAlpha = t * 0.85;
      ctx.lineWidth = 1 + t * 2.2;
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.lineTo(p1.x, p1.y);
      ctx.stroke();
      ctx.restore();
    }
  }



  // ---------------------------------------------------------------------
  // Label layout: places each label in the first free spot (no overlaps),
  // avoids reserved regions (status box, hand, balls, HUD bars) and draws a
  // thin leader line when a label had to move away from its anchor.
  // ---------------------------------------------------------------------
  const LABEL_FONT = '600 11px "IBM Plex Mono", Consolas, monospace';
  const LABEL_FONT_BOLD = '700 12px "IBM Plex Mono", Consolas, monospace';

  function createLabelLayout(w, h, margins) {
    const m = Object.assign({ top: 4, bottom: 4, left: 4, right: 4 }, margins || {});
    const rects = [];
    const GAP = 3;
    const hit = (r) => rects.some((q) => r.x < q.x + q.w + GAP && r.x + r.w + GAP > q.x && r.y < q.y + q.h + GAP && r.y + r.h + GAP > q.y);
    return {
      reserve(x, y, rw, rh) { rects.push({ x, y, w: rw, h: rh }); },
      /** anchor (ax, ay) is where the label prefers to sit (its bottom-center by default). */
      place(ax, ay, bw, bh, prefer) {
        const up = { dx: -bw / 2, dy: -bh };
        const order = prefer === 'right'
          ? [{ dx: 0, dy: -bh / 2 }, { dx: 0, dy: -bh * 1.5 - GAP }, { dx: 0, dy: bh / 2 + GAP }]
          : prefer === 'below'
          ? [{ dx: -bw / 2, dy: 2 }, up]
          : prefer === 'left-top'
            ? [{ dx: 0, dy: -bh }, { dx: 0, dy: 2 }]
            : [up, { dx: -bw / 2, dy: 2 }];
        const steps = [];
        for (const o of order) steps.push(o);
        for (let k = 1; k <= 4; k += 1) {
          steps.push({ dx: -bw / 2, dy: -bh - k * (bh + GAP) });
          steps.push({ dx: -bw / 2, dy: 2 + k * (bh + GAP) });
          steps.push({ dx: 8 + (k - 1) * 18, dy: -bh / 2 });
          steps.push({ dx: -bw - 8 - (k - 1) * 18, dy: -bh / 2 });
        }
        for (const st of steps) {
          const x = clamp(ax + st.dx, m.left, w - m.right - bw);
          const y = clamp(ay + st.dy, m.top, h - m.bottom - bh);
          const r = { x, y, w: bw, h: bh };
          if (!hit(r)) { rects.push(r); return r; }
        }
        return null;
      },
    };
  }

  /** Draw a plated text label through the layout. Returns false if no free spot. */
  function drawLayoutLabel(ctx, layout, ax, ay, text, opts) {
    const o = opts || {};
    ctx.save();
    ctx.font = o.font || LABEL_FONT;
    const padX = o.padX != null ? o.padX : 6;
    const bh = o.height || 18;
    const bw = ctx.measureText(text).width + padX * 2;
    const r = layout.place(ax, ay, bw, bh, o.prefer);
    if (!r) { ctx.restore(); return false; }
    const cx = r.x + bw / 2;
    const cy = r.y + bh / 2;
    // leader line when the label is not touching its anchor
    const nx = clamp(ax, r.x, r.x + bw);
    const ny = clamp(ay, r.y, r.y + bh);
    if (Math.hypot(nx - ax, ny - ay) > 5) {
      ctx.strokeStyle = o.leader || o.border || 'rgba(154, 162, 172, 0.8)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(nx, ny);
      ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fillRect(ax - 2, ay - 2, 4, 4);
    }
    ctx.fillStyle = o.plate || 'rgba(18, 21, 25, 0.9)';
    ctx.fillRect(r.x, r.y, bw, bh);
    if (o.border) {
      ctx.strokeStyle = o.border;
      ctx.lineWidth = o.borderWidth || 1;
      ctx.strokeRect(r.x + 0.5, r.y + 0.5, bw - 1, bh - 1);
    }
    ctx.fillStyle = o.color || '#B6BCC4';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, cx, cy + 0.5);
    ctx.restore();
    return true;
  }

  // Labels requested while drawing the environment; rendered last (on top, laid out).
  let pendingLabels = [];

  function activeZone(world) {
    if (!world || world.state === 'IDLE') return null;
    if (world.heldBallId) return 'hand';
    const b = world.getActiveBall ? world.getActiveBall() : null;
    if (!b) return 'tray';
    if (b.zone === 'CHUTE') return 'chute';
    if (b.zone === 'BAY' || b.zone === 'HANDED') return 'bay';
    if (b.zone === 'GRIP') return 'hand';
    return 'tray';
  }

  function drawZoneLabel(ctx, p, text, activeColor) {
    if (!p) return;
    pendingLabels.push({
      x: p.x, y: p.y + 9, text, priority: activeColor ? 2 : 4, prefer: 'below',
      opts: activeColor
        ? { font: LABEL_FONT_BOLD, color: activeColor, border: activeColor, borderWidth: 1.5, height: 22 }
        : { font: LABEL_FONT, color: '#9AA3AE', height: 20 },
    });
  }

  /** Dashed route the part will travel: tray slot -> chute -> receiving bay. */
  function drawRouteGuide(ctx, world, basis, cam, w, h) {
    if (!world || world.state === 'IDLE' || world.emergencyStopped || !world.targetPart) return;
    const part = PARTS.find((p) => p.id === world.targetPart);
    if (!part) return;
    const ch = LAYOUT.chute;
    const bay = LAYOUT.bay;
    const legs = [
      [{ x: part.slot.x, y: 2.2, z: part.slot.z }, { x: part.slot.x - 4, y: 14, z: part.slot.z * 0.5 }, { x: ch.topX + 1.5, y: ch.topY + 3.2, z: 0 }],
      [{ x: ch.topX, y: ch.topY + 1.2, z: 0 }, { x: ch.bottomX, y: ch.bottomY + 1.2, z: 0 }],
      [{ x: ch.bottomX - 0.5, y: ch.bottomY + 1.0, z: 0 }, { x: bay.center.x, y: bay.center.y + 0.8, z: 0 }],
    ];
    let current = 0;
    if (world.state === 'DELIVER') current = world.deliverSubstep === 'RELEASE' ? 1 : 0;
    const zone = activeZone(world);
    if (zone === 'chute') current = 1;
    if (zone === 'bay' || world.state === 'HANDOVER') current = 2;
    if (world.state === 'RETURN') return;

    legs.forEach((pts, i) => {
      const done = i < current;
      const active = i === current;
      const col = done ? 'rgba(154, 162, 172, 0.35)' : active ? PALETTE.accent : 'rgba(154, 162, 172, 0.55)';
      const scr = pts.map((pt) => projectPoint(pt, basis, cam, w, h));
      if (scr.some((q) => !q)) return;
      ctx.save();
      ctx.strokeStyle = col;
      ctx.lineWidth = active ? 2.2 : 1.3;
      ctx.setLineDash(active ? [7, 5] : [3, 5]);
      if (active) ctx.lineDashOffset = -((world.time || 0) * 24) % 24;
      ctx.beginPath();
      ctx.moveTo(scr[0].x, scr[0].y);
      if (scr.length === 3) ctx.quadraticCurveTo(scr[1].x, scr[1].y, scr[2].x, scr[2].y);
      else ctx.lineTo(scr[1].x, scr[1].y);
      ctx.stroke();
      // arrowhead at the end of the leg
      const end = scr[scr.length - 1];
      const prev = scr[scr.length - 2];
      const ang = Math.atan2(end.y - prev.y, end.x - prev.x);
      const L = active ? 11 : 8;
      ctx.setLineDash([]);
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.moveTo(end.x, end.y);
      ctx.lineTo(end.x - L * Math.cos(ang - 0.42), end.y - L * Math.sin(ang - 0.42));
      ctx.lineTo(end.x - L * Math.cos(ang + 0.42), end.y - L * Math.sin(ang + 0.42));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    });
  }

  /** Plain-language status box (replaces the numeric telemetry HUD). */
  function drawPlainStatus(ctx, world, kin) {
    const nameOf = (id) => {
      const p = PARTS.find((q) => q.id === id);
      return p ? p.name : `부품 ${id}`;
    };
    const part = world.targetPart ? nameOf(world.targetPart) : '';
    const grip = avgCurl(world.pose.fingers) > 0.55 ? '쥠' : '펼침';
    let title;
    let tone = 'idle';
    if (world.emergencyStopped) { title = '정지됨 · 관절 고정'; tone = 'warn'; }
    else if (world.state === 'IDLE') title = '대기 중';
    else if (world.state === 'REQUEST' || world.state === 'CONFIRM') { title = `${part} 가져갈 준비`; tone = 'accent'; }
    else if (world.state === 'PICK') { title = world.heldBallId ? `${part} 잡음` : `${part} 잡는 중`; tone = 'accent'; }
    else if (world.state === 'DELIVER') { title = world.deliverSubstep === 'RELEASE' ? `${part} 슈트에 놓는 중` : `${part} 슈트로 옮기는 중`; tone = 'accent'; }
    else if (world.state === 'HANDOVER') { title = '수령 트레이 도착 · 작업자 수령 대기'; tone = 'pass'; }
    else if (world.state === 'RETURN') { title = `${part} 트레이로 되돌리는 중`; tone = 'accent'; }
    else title = world.state;
    if (world.statusOverride) {                 // 실물 모드: 대시보드가 상태 문구를 직접 넘김
      title = world.statusOverride;
      tone = world.statusTone || tone;
    }
    const sub = world.standingHand
      ? `손 ${grip} · 손목 ${Math.round(world.pose.pan != null ? world.pose.pan : 90)}°`
      : `손 ${grip}`;

    const col = tone === 'warn' ? PALETTE.warn : tone === 'pass' ? PALETTE.pass : tone === 'accent' ? PALETTE.accent : '#DCE0E5';
    ctx.save();
    ctx.font = '700 14px "IBM Plex Mono", Consolas, monospace';
    const tw1 = ctx.measureText(title).width;
    ctx.font = '600 11px "IBM Plex Mono", Consolas, monospace';
    const tw2 = ctx.measureText(sub).width;
    const bw = Math.max(tw1, tw2) + 30;
    ctx.fillStyle = 'rgba(18, 21, 25, 0.92)';
    ctx.fillRect(10, 10, bw, 46);
    ctx.fillStyle = col;
    ctx.fillRect(10, 10, 4, 46);
    ctx.strokeStyle = 'rgba(154, 162, 172, 0.55)';
    ctx.lineWidth = 1;
    ctx.strokeRect(10.5, 10.5, bw - 1, 45);
    ctx.font = '700 14px "IBM Plex Mono", Consolas, monospace';
    ctx.fillStyle = col === '#DCE0E5' ? '#ECEEEB' : col;
    ctx.fillText(title, 22, 30);
    ctx.font = '600 11px "IBM Plex Mono", Consolas, monospace';
    ctx.fillStyle = '#9AA3AE';
    ctx.fillText(sub, 22, 47);
    ctx.restore();
    return { x: 10, y: 10, w: bw, h: 46 };
  }

  /**
   * Main entry point to render the 3D Wireframe Viewport.
   */
  /** 서 있는 손 (팔 없음): 바닥 + 받침 + 수직 손 + 손목 회전. 사람이 손바닥에 부품을 쥐어 준다. */
  function renderStandingScene(ctx, w, h, world, cam) {
    ctx.clearRect(0, 0, w, h);
    const bg = ctx.createRadialGradient(w * 0.5, h * 0.5, 30, w * 0.5, h * 0.5, Math.max(w, h) * 0.72);
    bg.addColorStop(0, '#1A1E24');
    bg.addColorStop(1, '#111418');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);
    const basis = getCameraBasis(cam);
    const kin = world.kinematics || computeHandKinematics(world.pose);

    // 바닥 격자 (손 주변만)
    for (let v = -16; v <= 16; v += 4) {
      const major = v % 8 === 0;
      const col = major ? 'rgba(92, 104, 118, 0.30)' : 'rgba(64, 72, 82, 0.18)';
      drawLine3D(ctx, { x: v, y: 0, z: -16 }, { x: v, y: 0, z: 16 }, basis, cam, w, h, col, major ? 1 : 0.75);
      drawLine3D(ctx, { x: -16, y: 0, z: v }, { x: 16, y: 0, z: v }, basis, cam, w, h, col, major ? 1 : 0.75);
    }
    // 받침대
    drawWireBox(ctx, -4, 4, 0, 1.2, -4, 4, basis, cam, w, h, 'rgba(160, 168, 178, 0.75)', 1.2, 'rgba(32, 37, 44, 0.45)');
    drawLine3D(ctx, { x: 0, y: 1.2, z: 0 }, { x: 0, y: LAYOUT.turretHeight - 0.6, z: 0 }, basis, cam, w, h, '#8B939E', 1.4);

    // 손목 방향 표시 (정면 90° 기준선 + 현재 방향)
    const yawNow = panToYawDeg(world.pose.pan != null ? world.pose.pan : 90) * DEG;
    const r = 6.5;
    drawLine3D(ctx, { x: 0, y: 0.05, z: 0 }, { x: 0, y: 0.05, z: r }, basis, cam, w, h, 'rgba(154, 162, 172, 0.45)', 1, [3, 4]);
    drawLine3D(ctx, { x: 0, y: 0.08, z: 0 }, { x: Math.sin(yawNow) * r, y: 0.08, z: Math.cos(yawNow) * r }, basis, cam, w, h, PALETTE.accent, 1.8);

    // 받기 단계: 손바닥 앞 '올려 주십시오' 자리
    if (world.receiving) {
      drawCircle3D(ctx, kin.gripCenter, 2.6, kin.palmRight, kin.palmUp, 20, basis, cam, w, h,
        'rgba(214, 158, 114, 0.85)', 1.4, 'rgba(214, 158, 114, 0.10)', [5, 4]);
    }

    drawWireframeHand(ctx, kin, world.tone, world.emergencyStopped, basis, cam, w, h, false);

    // 쥔 부품
    if (world.heldPart) {
      const c = kin.gripCenter;
      const ball = HELD_BALL[world.heldPart];
      if (ball) {
        drawCircle3D(ctx, c, 1.3, kin.palmRight, kin.palmFront, 20, basis, cam, w, h, ball.color, 1.8, ball.dim);
        drawCircle3D(ctx, c, 1.3, kin.palmRight, kin.palmUp, 20, basis, cam, w, h, ball.color, 1.2);
      } else {
        drawCircle3D(ctx, c, 1.3, kin.palmRight, kin.palmFront, 20, basis, cam, w, h, PALETTE.pass, 1.8, PALETTE.passDim);
      }
    }

    const layout = createLabelLayout(w, h, { top: 4, bottom: 4 });
    const statusRect = drawPlainStatus(ctx, world, kin);
    layout.reserve(statusRect.x, statusRect.y, statusRect.w, statusRect.h);
    const pPalm = projectPoint(kin.palmCenter, basis, cam, w, h);
    if (pPalm) layout.reserve(pPalm.x - 30, pPalm.y - 40, 60, 80);
    if (world.heldPart) {
      const pg = projectPoint(kin.gripCenter, basis, cam, w, h);
      if (pg) drawLayoutLabel(ctx, layout, pg.x, pg.y - 18, (HELD_BALL[world.heldPart] || { name: '공' }).name,
        { font: LABEL_FONT_BOLD, color: (HELD_BALL[world.heldPart] || {}).color || PALETTE.pass, border: (HELD_BALL[world.heldPart] || {}).color || PALETTE.pass, height: 20 });
    } else if (world.receiving) {
      const pg = projectPoint(kin.gripCenter, basis, cam, w, h);
      if (pg) drawLayoutLabel(ctx, layout, pg.x, pg.y + 30, '여기에 올려 주십시오',
        { font: LABEL_FONT_BOLD, color: PALETTE.accent, border: PALETTE.accent, height: 20, prefer: 'below' });
    }
    const p0 = projectPoint({ x: 0, y: 0.1, z: r + 1.2 }, basis, cam, w, h);
    if (p0) drawLayoutLabel(ctx, layout, p0.x, p0.y, '정면', { font: LABEL_FONT, color: '#9AA3AE', height: 18, prefer: 'below' });
    return { screenBalls: [], kin };
  }

  function renderWireframeScene(ctx, w, h, world, cam) {
    if (world.standingHand) return renderStandingScene(ctx, w, h, world, cam);
    ctx.clearRect(0, 0, w, h);
    pendingLabels = [];

    // Blueprint dark charcoal background
    const bg = ctx.createRadialGradient(w * 0.52, h * 0.48, 30, w * 0.52, h * 0.48, Math.max(w, h) * 0.72);
    bg.addColorStop(0, '#1A1E24');
    bg.addColorStop(1, '#111418');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    const basis = getCameraBasis(cam);
    const kin = world.kinematics || computeHandKinematics(world.pose);

    // 1. Environment (Floor Grid, Tray, Chute, Receiving Bay, Ultrasonic)
    drawWorkstationEnvironment(ctx, world, basis, cam, w, h, false);

    // 2. Trajectory ribbon (warm amber highlight for ball movement)
    if (world.showTrail && world.ballTrail.length > 1) {
      drawBallTrajectory3D(ctx, world.ballTrail, PALETTE.accent, basis, cam, w, h);
    }

    // 2b. Route guide (where the part goes next)
    drawRouteGuide(ctx, world, basis, cam, w, h);

    // 3. Proximity Dome & Distance Vector
    drawProximityOverlay3D(ctx, kin, world, basis, cam, w, h);

    // 4. Balls & Wireframe Hand
    const screenBalls = [];
    for (const ball of world.balls) {
      if (!ball.visible) continue;
      const isGrasped = world.heldBallId === ball.id;
      const isTarget = world.targetPart === ball.id || (world.proximityMode && ball.id === world.focusBallId);
      if (!isGrasped) {
        const sb = drawWireBall(ctx, ball, basis, cam, w, h, { isTarget, isGrasped: false });
        if (sb) screenBalls.push({ ball, ...sb });
      }
    }

    const heldBall = world.balls.find((b) => b.id === world.heldBallId && b.visible);
    if (heldBall) {
      const sb = drawWireBall(ctx, heldBall, basis, cam, w, h, { isTarget: true, isGrasped: true });
      if (sb) screenBalls.push({ ball: heldBall, ...sb });
    }

    // 5. Wireframe 5-Finger Hiwonder AiHand
    drawWireframeHand(ctx, kin, world.tone, world.emergencyStopped, basis, cam, w, h, false);

    // 6. Wireframe Worker Hand (during HANDOVER approach)
    if (world.workerHand.visible) {
      drawWorkerHand3D(ctx, world.workerHand.pos, world.workerHand.open, basis, cam, w, h);
    }

    // 7. Labels — status box first, then everything else laid out without overlap
    const layout = createLabelLayout(w, h, { top: 4, bottom: 4 });
    const statusRect = drawPlainStatus(ctx, world, kin);
    layout.reserve(statusRect.x, statusRect.y, statusRect.w, statusRect.h);
    for (const sb of screenBalls) layout.reserve(sb.cx - sb.r, sb.cy - sb.r, sb.r * 2, sb.r * 2);
    const pPalm = projectPoint(kin.palmCenter, basis, cam, w, h);
    if (pPalm) {
      const hr = Math.max(18, 4.2 * pPalm.scale);
      layout.reserve(pPalm.x - hr, pPalm.y - hr * 1.3, hr * 2, hr * 2.2);
    }
    // the part that left the tray gets a name tag
    for (const sb of screenBalls) {
      const b = sb.ball;
      if (b.zone === 'TRAY' && world.heldBallId !== b.id) continue;
      const col = world.heldBallId === b.id || b.zone === 'BAY' || b.zone === 'HANDED' ? PALETTE.pass : PALETTE.accent;
      pendingLabels.push({ x: sb.cx, y: sb.cy - sb.r - 3, text: b.name, priority: 0, opts: { font: LABEL_FONT_BOLD, color: col, border: col, height: 20 } });
    }
    pendingLabels.sort((a, b) => a.priority - b.priority);
    for (const L of pendingLabels) drawLayoutLabel(ctx, layout, L.x, L.y, L.text, Object.assign({ prefer: L.prefer }, L.opts));
    pendingLabels = [];

    return { screenBalls, kin };
  }

  function avgCurl(fingers) {
    if (!fingers || !fingers.length) return 0;
    let s = 0;
    for (let i = 0; i < fingers.length; i++) s += fingers[i];
    return s / fingers.length;
  }

  function drawAxisGizmo(ctx, basis, ox, oy) {
    const len = 18;
    const axes = [
      { v: { x: 1, y: 0, z: 0 }, col: '#9AA3AE', lbl: 'X' },
      { v: { x: 0, y: 1, z: 0 }, col: '#C8CED6', lbl: 'Y' },
      { v: { x: 0, y: 0, z: 1 }, col: '#78808B', lbl: 'Z' },
    ];
    ctx.save();
    ctx.font = '600 9px "IBM Plex Mono", Consolas, monospace';
    for (const ax of axes) {
      const dx = vDot(ax.v, basis.right) * len;
      const dy = -vDot(ax.v, basis.up) * len;
      ctx.strokeStyle = ax.col;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + dx, oy + dy);
      ctx.stroke();
      ctx.fillStyle = ax.col;
      ctx.fillText(ax.lbl, ox + dx * 1.25 - 3, oy + dy * 1.25 + 3);
    }
    ctx.restore();
  }

  const api = {
    PARTS,
    LAYOUT,
    BALL_RADIUS,
    CAM_PRESETS,
    clamp,
    lerp,
    vDist,
    chuteSurfaceY,
    panToYawDeg,
    yawDegToPan,
    computeHandKinematics,
    solvePoseForPoint,
    getCameraBasis,
    projectPoint,
    unprojectToPlaneY,
    drawLine3D,
    drawWireBox,
    drawPolyline3D,
    drawCircle3D,
    drawWireBall,
    drawWireframeHand,
    drawWorkstationEnvironment,
    drawBallTrajectory3D,
    renderWireframeScene,
    createLabelLayout,
    drawLayoutLabel,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    global.AiHand3D = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
