/**
 * Simulated VLM — 작업자의 글(한국어/영어)을 트레이 위 부품과 대조해 가져올 부품을 고른다.
 *
 * 실제 VLM 대신 카탈로그 속성(이름·번호·색·모양·크기)과 재고를 근거로 점수를 매긴다.
 * 결과 형식은 실제 모델로 교체해도 그대로 쓰도록 고정:
 *   { status: 'ok' | 'ambiguous' | 'empty' | 'unknown' | 'invalid',
 *     part, name, confidence, cues: [...], reply, candidates: [{part, score}] }
 */
(function (global) {
  'use strict';

  // dashboard/catalog.py 와 같은 부품 정의 (색·모양은 VLM 이 "보는" 속성)
  const CATALOG = [
    {
      id: 1, name: '캡 A', code: 'CAP-A', shape: '실린더', color: '테라코타', size: '표준',
      cues: {
        name: ['캡a', '캡에이', 'capa', 'cap-a', 'cap a'],
        number: ['1번', '첫번째', '첫째', '일번', 'number one', 'first', 'no.1', '#1'],
        color: ['테라코타', '주황', '오렌지', '갈색', '빨간', '빨강', '붉은', 'orange', 'brown', 'terracotta', 'red'],
        shape: ['실린더', '원통', '둥근', '동그란', '원기둥', 'cylinder', 'round'],
        size: ['표준', '기본', '일반', 'standard', 'basic'],
      },
    },
    {
      id: 2, name: '캡 B', code: 'CAP-B', shape: '육각', color: '슬레이트 블루', size: '중형',
      cues: {
        name: ['캡b', '캡비', 'capb', 'cap-b', 'cap b'],
        number: ['2번', '두번째', '둘째', 'number two', 'second', 'no.2', '#2'],
        color: ['파란', '파랑', '블루', '푸른', '남색', '슬레이트', 'blue', 'slate', 'navy'],
        shape: ['육각', '헥사', '6각', 'hex', 'hexagon'],
        size: ['중형', '중간', '보통', 'medium', 'mid'],
      },
    },
    {
      id: 3, name: '캡 C', code: 'CAP-C', shape: '락킹 엔드', color: '포레스트 그린', size: '잠금형',
      cues: {
        name: ['캡c', '캡씨', 'capc', 'cap-c', 'cap c'],
        number: ['3번', '세번째', '셋째', '삼번', 'number three', 'third', 'no.3', '#3'],
        color: ['초록', '녹색', '그린', '연두', '포레스트', 'green', 'forest'],
        shape: ['락킹', '잠금', '잠그는', '엔드', '마개', 'lock', 'locking', 'end cap'],
        size: ['잠금형'],
      },
    },
  ];

  const WEIGHTS = { name: 4, number: 3, color: 2, shape: 2, size: 1 };
  const CUE_KO = { name: '이름', number: '번호', color: '색상', shape: '모양', size: '크기' };
  const NEGATION = /(말고|빼고|아니고|아닌\s*거|제외하고|제외|대신|except|not|other than|instead of)/i;
  const ANY = /(아무거나|아무\s*거|아무\s*캡|아무\s*부품|남은\s*거|재고\s*많은|any(thing)?|whatever)/i;
  const UNKNOWN_OBJECT = /(나사|볼트|너트|드라이버|렌치|스패너|망치|와셔|screw|bolt|nut|driver|wrench|hammer|washer)/i;

  function normalize(text) {
    return String(text || '').toLowerCase().replace(/\s+/g, ' ').trim();
  }

  function scoreClause(clause) {
    const compact = clause.replace(/\s+/g, '');
    const out = {};
    for (const part of CATALOG) {
      let score = 0;
      const cues = [];
      for (const [kind, words] of Object.entries(part.cues)) {
        const hit = words.find((w) => {
          const wc = w.replace(/\s+/g, '');
          if (kind === 'name' && /^[a-c]$/.test(wc)) return false;
          return compact.includes(wc);
        });
        if (hit) {
          score += WEIGHTS[kind];
          cues.push(`${CUE_KO[kind]}: ${hit}`);
        }
      }
      // 단독 숫자 / 단독 코드 문자 ("2", "b 주세요")
      const digit = clause.match(/(^|[^0-9])([1-3])(?![0-9]|\s*(개|ea|pcs|piece))/);
      if (digit && Number(digit[2]) === part.id && !cues.some((c) => c.startsWith('번호'))) {
        score += WEIGHTS.number;
        cues.push(`번호: ${digit[2]}`);
      }
      const letter = clause.match(/(^|[^a-z])([abc])([^a-z]|$)/);
      if (letter && 'abc'.indexOf(letter[2]) + 1 === part.id && !cues.some((c) => c.startsWith('이름'))) {
        score += WEIGHTS.name;
        cues.push(`이름: ${letter[2].toUpperCase()}`);
      }
      out[part.id] = { score, cues };
    }
    return out;
  }

  /**
   * @param {string} text   작업자 요청 글
   * @param {{stock?: Object<number, number>}} scene  현재 트레이 상태
   */
  function analyze(text, scene) {
    const stock = (scene && scene.stock) || { 1: 3, 2: 3, 3: 3 };
    const raw = String(text || '').trim();
    const t = normalize(raw);
    const base = { part: null, name: '', confidence: 0, cues: [], candidates: [] };

    if (!t) {
      return { ...base, status: 'invalid', reply: '가져올 부품을 글로 적어주십시오.' };
    }
    if (/(^|[^0-9])([04-9]|[1-9][0-9]+)\s*번/.test(t)) {
      return { ...base, status: 'unknown', reply: '트레이에는 1·2·3번 부품만 있습니다.' };
    }

    // "파란 거 말고 초록" → 부정 앞쪽은 제외, 뒤쪽은 요청
    let include = t;
    let exclude = '';
    const neg = t.match(NEGATION);
    if (neg) {
      exclude = t.slice(0, neg.index);
      include = t.slice(neg.index + neg[0].length);
    }
    const plus = scoreClause(include);
    const minus = exclude ? scoreClause(exclude) : {};
    const excluded = new Set(Object.keys(minus).filter((k) => minus[k].score > 0).map(Number));

    let ranked = CATALOG.map((p) => ({
      part: p.id,
      score: excluded.has(p.id) ? -1 : plus[p.id].score,
      cues: plus[p.id].cues,
    })).sort((a, b) => b.score - a.score);

    const anyRequested = ANY.test(t) || (neg && ranked[0].score <= 0);
    if (ranked[0].score <= 0 && anyRequested) {
      // 조건 없는 요청: 제외되지 않은 부품 중 재고가 가장 많은 것
      const pool = CATALOG.filter((p) => !excluded.has(p.id) && (stock[p.id] || 0) > 0)
        .sort((a, b) => (stock[b.id] || 0) - (stock[a.id] || 0));
      if (!pool.length) {
        return { ...base, status: 'empty', reply: '가져올 수 있는 부품이 트레이에 없습니다. 보충해주십시오.' };
      }
      const pick = pool[0];
      const cues = [`재고 우선: ${pick.name} ${stock[pick.id]}개`];
      if (excluded.size) cues.unshift(`제외: ${[...excluded].map((i) => `${i}번`).join(', ')}`);
      return {
        ...base, status: 'ok', part: pick.id, name: pick.name, confidence: 0.74, cues,
        candidates: ranked.map(({ part, score }) => ({ part, score })),
        reply: `${pick.id}번 ${pick.name}을(를) 가져오겠습니다. (재고가 가장 많습니다)`,
      };
    }

    const [top, second] = ranked;
    const candidates = ranked.map(({ part, score }) => ({ part, score }));
    if (top.score <= 0) {
      if (UNKNOWN_OBJECT.test(t)) {
        return { ...base, status: 'unknown', candidates, reply: '그 물건은 트레이에서 보이지 않습니다. 캡 A·B·C 중에서 골라주십시오.' };
      }
      return { ...base, status: 'unknown', candidates, reply: '어떤 부품인지 알아듣지 못했습니다. 예: "파란 육각 캡", "2번", "초록색 거".' };
    }
    if (second && second.score === top.score) {
      const names = ranked.filter((r) => r.score === top.score).map((r) => `${r.part}번`).join(' / ');
      return {
        ...base, status: 'ambiguous', candidates, cues: top.cues,
        reply: `${names} 중 어느 것인지 헷갈립니다. 색이나 번호를 더 알려주십시오.`,
      };
    }

    const part = CATALOG.find((p) => p.id === top.part);
    const margin = top.score - Math.max(0, second ? second.score : 0);
    const confidence = Math.min(0.99, 0.6 + 0.07 * top.score + 0.03 * margin);
    const cues = excluded.size ? [`제외: ${[...excluded].map((i) => `${i}번`).join(', ')}`, ...top.cues] : top.cues;
    if ((stock[part.id] || 0) <= 0) {
      return {
        ...base, status: 'empty', part: part.id, name: part.name, confidence, cues, candidates,
        reply: `${part.id}번 ${part.name}을(를) 찾았지만 트레이가 비어 있습니다. 보충 후 다시 요청해주십시오.`,
      };
    }
    return {
      ...base, status: 'ok', part: part.id, name: part.name, confidence, cues, candidates,
      reply: `${part.id}번 ${part.name}(${part.color} · ${part.shape})을(를) 가져오겠습니다.`,
    };
  }

  global.SimVlm = { CATALOG, analyze };
})(typeof window !== 'undefined' ? window : globalThis);
