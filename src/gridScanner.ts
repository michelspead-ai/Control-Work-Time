// 100% local, zero-API perspective rectification, grid detection,
// ink-isolation passes, multi-pass OCR candidate comparison, and row verification
// for French restaurant attendance sheets ("Tableau de présence").

export interface Point2D {
  x: number; // normalized 0..1 when relative, or pixel coord
  y: number;
}

export interface PixelBuffer {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export type ImagePassType = 'normalized' | 'high_contrast' | 'blue_ink';

export interface OcrPassResult {
  pass: ImagePassType;
  rawText: string;
  confidence: number;
}

export interface ParsedCandidate {
  value: string; // normalized value e.g. "10:30" or "2026-11-01" or "8h30"
  minutes?: number;
  pass: ImagePassType;
  confidence: number;
  rawText: string;
}

export interface CellAnalysisResult {
  bestValue: string;
  bestMinutes: number | null;
  candidates: ParsedCandidate[];
  confidence: number;
  isAmbiguous: boolean;
  isMissing: boolean;
  reason: string;
  thumbDataUrl?: string;
}

export interface GridRowDetection {
  rowIndex: number;
  yStart: number;
  yEnd: number;
  dateCell: CellAnalysisResult;
  midiStartCell: CellAnalysisResult;
  midiEndCell: CellAnalysisResult;
  soirStartCell: CellAnalysisResult;
  soirEndCell: CellAnalysisResult;
  totalCell: CellAnalysisResult;
  computedTotalMinutes: number | null;
  midiMinutes: number | null;
  soirMinutes: number | null;
  discrepancyMinutes: number | null;
  discrepancyNotice: string;
  needsVerification: boolean;
  verificationReasons: string[];
}

export const MONTH_NAME_MAP: Record<string, number> = {
  janvier: 1,
  janv: 1,
  fevrier: 2,
  février: 2,
  fev: 2,
  mars: 3,
  avril: 4,
  avr: 4,
  mai: 5,
  juin: 6,
  juillet: 7,
  juil: 7,
  aout: 8,
  août: 8,
  septembre: 9,
  sept: 9,
  octobre: 10,
  oct: 10,
  novembre: 11,
  nov: 11,
  decembre: 12,
  décembre: 12,
  dec: 12,
};

export function pad2(n: number): string {
  return n < 10 ? '0' + String(n) : String(n);
}

export function formatMinsShort(mins: number): string {
  const m = Math.max(0, Math.round(mins));
  return Math.floor(m / 60) + 'h' + pad2(m % 60);
}

/**
 * Parses explicit month & year from a French title header (e.g., "Feuille de présence - Novembre 2026" or "11/2026").
 */
export function parseFrenchTitleMonthYear(titleText: string): {
  month: number | null;
  year: number | null;
} {
  if (!titleText || typeof titleText !== 'string') {
    return { month: null, year: null };
  }
  const lower = titleText.toLowerCase();
  let year: number | null = null;
  const yearMatch = lower.match(/\b(20[2-3]\d)\b/);
  if (yearMatch) {
    year = parseInt(yearMatch[1], 10);
  }

  let month: number | null = null;
  for (const [key, val] of Object.entries(MONTH_NAME_MAP)) {
    const rx = new RegExp('\\b' + key + '\\b', 'i');
    if (rx.test(lower)) {
      month = val;
      break;
    }
  }

  if (month === null) {
    const numMonthMatch = lower.match(/\b(0?[1-9]|1[0-2])\s*[\/\-]\s*(20[2-3]\d)\b/);
    if (numMonthMatch) {
      month = parseInt(numMonthMatch[1], 10);
      if (year === null) {
        year = parseInt(numMonthMatch[2], 10);
      }
    }
  }

  return { month, year };
}

/**
 * Solves an 8x8 linear system Ax = b via Gaussian elimination with partial pivoting.
 */
function solveLinear8(A: number[][], b: number[]): number[] | null {
  const n = 8;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      const val = Math.abs(M[r][col]);
      if (val > maxVal) {
        maxVal = val;
        maxRow = r;
      }
    }
    if (maxVal < 1e-10) return null;

    if (maxRow !== col) {
      const tmp = M[col];
      M[col] = M[maxRow];
      M[maxRow] = tmp;
    }

    const pivot = M[col][col];
    for (let r = col + 1; r < n; r++) {
      const factor = M[r][col] / pivot;
      for (let c = col; c <= n; c++) {
        M[r][c] -= factor * M[col][c];
      }
    }
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= M[i][j] * x[j];
    }
    x[i] = sum / M[i][i];
  }
  return x;
}

/**
 * Computes 3x3 homography matrix mapping dstPts -> srcPts (for inverse warping).
 * Points are in [TL, TR, BR, BL] order.
 */
export function computeHomography(srcPts: Point2D[], dstPts: Point2D[]): number[] | null {
  if (srcPts.length !== 4 || dstPts.length !== 4) return null;
  const A: number[][] = [];
  const b: number[] = [];

  for (let i = 0; i < 4; i++) {
    const x = dstPts[i].x;
    const y = dstPts[i].y;
    const u = srcPts[i].x;
    const v = srcPts[i].y;
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }

  const h = solveLinear8(A, b);
  if (!h) return null;
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

/**
 * Perspective unwarping using bilinear interpolation from srcCorners [TL, TR, BR, BL] in pixel coordinates
 * onto a rectangular output of size dstW x dstH.
 */
export function warpPerspectivePixels(
  src: PixelBuffer,
  srcCornersPx: Point2D[],
  dstW: number,
  dstH: number
): PixelBuffer {
  const dstPts: Point2D[] = [
    { x: 0, y: 0 },
    { x: dstW - 1, y: 0 },
    { x: dstW - 1, y: dstH - 1 },
    { x: 0, y: dstH - 1 },
  ];

  const H = computeHomography(srcCornersPx, dstPts);
  const out = new Uint8ClampedArray(dstW * dstH * 4);

  if (!H) {
    return { width: dstW, height: dstH, data: out };
  }

  const [h0, h1, h2, h3, h4, h5, h6, h7, h8] = H;
  const sw = src.width;
  const sh = src.height;
  const sdata = src.data;

  for (let dy = 0; dy < dstH; dy++) {
    for (let dx = 0; dx < dstW; dx++) {
      const denom = h6 * dx + h7 * dy + h8;
      if (Math.abs(denom) < 1e-8) continue;
      const sx = (h0 * dx + h1 * dy + h2) / denom;
      const sy = (h3 * dx + h4 * dy + h5) / denom;

      const dstIdx = (dy * dstW + dx) * 4;
      if (sx < 0 || sx >= sw - 1 || sy < 0 || sy >= sh - 1) {
        out[dstIdx] = 255;
        out[dstIdx + 1] = 255;
        out[dstIdx + 2] = 255;
        out[dstIdx + 3] = 255;
        continue;
      }

      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      const x1 = x0 + 1;
      const y1 = y0 + 1;

      const i00 = (y0 * sw + x0) * 4;
      const i10 = (y0 * sw + x1) * 4;
      const i01 = (y1 * sw + x0) * 4;
      const i11 = (y1 * sw + x1) * 4;

      const w00 = (1 - fx) * (1 - fy);
      const w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy;
      const w11 = fx * fy;

      out[dstIdx] = Math.round(sdata[i00] * w00 + sdata[i10] * w10 + sdata[i01] * w01 + sdata[i11] * w11);
      out[dstIdx + 1] = Math.round(
        sdata[i00 + 1] * w00 + sdata[i10 + 1] * w10 + sdata[i01 + 1] * w01 + sdata[i11 + 1] * w11
      );
      out[dstIdx + 2] = Math.round(
        sdata[i00 + 2] * w00 + sdata[i10 + 2] * w10 + sdata[i01 + 2] * w01 + sdata[i11 + 2] * w11
      );
      out[dstIdx + 3] = 255;
    }
  }

  return { width: dstW, height: dstH, data: out };
}

/**
 * Attempts lightweight automatic page/grid corner estimation using foreground paper color
 * matching + morphological opening + robust line fitting (safe against background objects & bottom cutoffs).
 */
export function estimateDocumentQuadNormalized(src: PixelBuffer): {
  corners: Point2D[];
  confident: boolean;
} {
  const { width: w, height: h, data } = src;
  const fallback: Point2D[] = [
    { x: 0.04, y: 0.04 },
    { x: 0.96, y: 0.04 },
    { x: 0.96, y: 0.96 },
    { x: 0.04, y: 0.96 },
  ];
  if (w < 60 || h < 60) return { corners: fallback, confident: false };

  const gw = 128;
  const gh = Math.max(48, Math.round(gw * (h / w)));
  const rG = new Float32Array(gw * gh);
  const gG = new Float32Array(gw * gh);
  const bG = new Float32Array(gw * gh);
  const lG = new Float32Array(gw * gh);

  for (let gy = 0; gy < gh; gy++) {
    const sy = Math.min(h - 1, Math.floor((gy + 0.5) * (h / gh)));
    for (let gx = 0; gx < gw; gx++) {
      const sx = Math.min(w - 1, Math.floor((gx + 0.5) * (w / gw)));
      const p = (sy * w + sx) * 4;
      rG[gy * gw + gx] = data[p];
      gG[gy * gw + gx] = data[p + 1];
      bG[gy * gw + gx] = data[p + 2];
      lG[gy * gw + gx] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
    }
  }

  // Check border darkness vs center brightness to verify it's a photographed sheet
  let borderDark = 0;
  let borderTotal = 0;
  const centerSamples: Array<{ l: number; r: number; g: number; b: number }> = [];
  for (let gy = 0; gy < gh; gy++) {
    for (let gx = 0; gx < gw; gx++) {
      const i = gy * gw + gx;
      const isBorder = gx < gw * 0.04 || gx > gw * 0.96 || gy < gh * 0.04 || gy > gh * 0.96;
      if (isBorder) {
        borderTotal++;
        if (lG[i] < 130) borderDark++;
      }
      if (gx > gw * 0.26 && gx < gw * 0.74 && gy > gh * 0.26 && gy < gh * 0.74) {
        centerSamples.push({ l: lG[i], r: rG[i], g: gG[i], b: bG[i] });
      }
    }
  }

  if (centerSamples.length === 0) return { corners: fallback, confident: false };
  centerSamples.sort((a, b) => a.l - b.l);
  const ref = centerSamples[Math.floor(centerSamples.length * 0.72)];
  const minLum = Math.max(105, ref.l - 56);

  const rawMask = new Uint8Array(gw * gh);
  for (let i = 0; i < gw * gh; i++) {
    const dist = Math.abs(rG[i] - ref.r) + Math.abs(gG[i] - ref.g) + Math.abs(bG[i] - ref.b);
    const isBluePen = bG[i] > rG[i] + 12;
    if (!isBluePen && lG[i] >= minLum && dist < 86) {
      rawMask[i] = 1;
    }
  }

  // Fill 1px grid lines inside paper
  const closed = new Uint8Array(rawMask);
  for (let gy = 1; gy < gh - 1; gy++) {
    for (let gx = 1; gx < gw - 1; gx++) {
      const i = gy * gw + gx;
      if (!rawMask[i] && rawMask[i - 1] + rawMask[i + 1] + rawMask[i - gw] + rawMask[i + gw] >= 3) {
        closed[i] = 1;
      }
    }
  }

  // Morphological erode + connected component to sever touching background papers
  const eroded = new Uint8Array(gw * gh);
  for (let gy = 1; gy < gh - 1; gy++) {
    for (let gx = 1; gx < gw - 1; gx++) {
      const i = gy * gw + gx;
      if (closed[i] && closed[i - 1] && closed[i + 1] && closed[i - gw] && closed[i + gw]) {
        eroded[i] = 1;
      }
    }
  }

  const visited = new Uint8Array(gw * gh);
  let bestComp: number[] = [];
  for (let i = 0; i < gw * gh; i++) {
    if (eroded[i] && !visited[i]) {
      const stack = [i];
      const comp: number[] = [];
      visited[i] = 1;
      while (stack.length > 0) {
        const cur = stack.pop()!;
        comp.push(cur);
        const cx = cur % gw;
        const cy = Math.floor(cur / gw);
        if (cx > 0 && eroded[cur - 1] && !visited[cur - 1]) { visited[cur - 1] = 1; stack.push(cur - 1); }
        if (cx + 1 < gw && eroded[cur + 1] && !visited[cur + 1]) { visited[cur + 1] = 1; stack.push(cur + 1); }
        if (cy > 0 && eroded[cur - gw] && !visited[cur - gw]) { visited[cur - gw] = 1; stack.push(cur - gw); }
        if (cy + 1 < gh && eroded[cur + gw] && !visited[cur + gw]) { visited[cur + gw] = 1; stack.push(cur + gw); }
      }
      if (comp.length > bestComp.length) bestComp = comp;
    }
  }

  const areaFrac = bestComp.length / (gw * gh);
  if (areaFrac < 0.2 || (borderTotal > 0 && borderDark / borderTotal < 0.15 && areaFrac > 0.85)) {
    return { corners: fallback, confident: false };
  }

  const compMask = new Uint8Array(gw * gh);
  for (const idx of bestComp) {
    compMask[idx] = 1;
    const cx = idx % gw;
    const cy = Math.floor(idx / gw);
    if (cx > 0 && closed[idx - 1]) compMask[idx - 1] = 1;
    if (cx + 1 < gw && closed[idx + 1]) compMask[idx + 1] = 1;
    if (cy > 0 && closed[idx - gw]) compMask[idx - gw] = 1;
    if (cy + 1 < gh && closed[idx + gw]) compMask[idx + gw] = 1;
  }

  let tl = { x: 0, y: 0, score: Infinity };
  let tr = { x: 1, y: 0, score: -Infinity };
  let br = { x: 1, y: 1, score: -Infinity };
  let bl = { x: 0, y: 1, score: Infinity };
  for (let i = 0; i < gw * gh; i++) {
    if (!compMask[i]) continue;
    const nx = (i % gw) / gw;
    const ny = Math.floor(i / gw) / gh;
    if (nx + ny < tl.score) tl = { x: nx, y: ny, score: nx + ny };
    if (nx - ny > tr.score) tr = { x: nx, y: ny, score: nx - ny };
    if (nx + ny > br.score) br = { x: nx, y: ny, score: nx + ny };
    if (nx - ny < bl.score) bl = { x: nx, y: ny, score: nx - ny };
  }

  return {
    corners: [
      { x: Math.max(0.01, Math.min(0.48, tl.x)), y: Math.max(0.01, Math.min(0.48, tl.y)) },
      { x: Math.max(0.52, Math.min(0.99, tr.x)), y: Math.max(0.01, Math.min(0.48, tr.y)) },
      { x: Math.max(0.52, Math.min(0.99, br.x)), y: Math.max(0.52, Math.min(0.99, br.y)) },
      { x: Math.max(0.01, Math.min(0.48, bl.x)), y: Math.max(0.52, Math.min(0.99, bl.y)) },
    ],
    confident: areaFrac >= 0.25 && areaFrac <= 0.88,
  };
}

/**
 * Estimates dominant horizontal table/text tilt angle (degrees) in [-24..+24]
 * using dark-stroke horizontal projection variance on bright paper regions.
 */
export function estimateTableSkewAngle(src: PixelBuffer): number {
  const { width: w, height: h, data } = src;
  if (w < 60 || h < 60) return 0;

  const scale = Math.min(1, 240 / Math.max(w, h));
  const dw = Math.max(40, Math.round(w * scale));
  const dh = Math.max(40, Math.round(h * scale));
  const lum = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(h - 1, Math.floor(y / scale));
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(w - 1, Math.floor(x / scale));
      const p = (sy * w + sx) * 4;
      lum[y * dw + x] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
    }
  }

  const stroke = new Float32Array(dw * dh);
  for (let y = 2; y < dh - 2; y++) {
    for (let x = 2; x < dw - 2; x++) {
      const p = (y * dw + x) * 4;
      const r = data[p];
      const b = data[p + 2];
      const bg = Math.max(lum[(y - 2) * dw + x], lum[(y + 2) * dw + x]);
      if (bg > 125 && bg < 248 && b - r <= 18) {
        const diff = bg - lum[y * dw + x];
        if (diff > 13) stroke[y * dw + x] = diff;
      }
    }
  }

  const cx = dw / 2;
  const cy = dh / 2;
  const projLen = dh * 3;
  const proj = new Float32Array(projLen);
  const cnt = new Float32Array(projLen);

  function scoreAngle(deg: number): number {
    proj.fill(0);
    cnt.fill(0);
    const rad = (deg * Math.PI) / 180;
    const sin = Math.sin(rad);
    const cos = Math.cos(rad);
    const y0 = Math.floor(dh * 0.1);
    const y1 = Math.floor(dh * 0.9);
    const x0 = Math.floor(dw * 0.12);
    const x1 = Math.floor(dw * 0.88);
    const minRowSpan = Math.max(18, (x1 - x0) * 0.35);
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const ry = Math.round((x - cx) * sin + (y - cy) * cos + dh * 1.5);
        if (ry >= 0 && ry < projLen) {
          proj[ry] += stroke[y * dw + x];
          cnt[ry]++;
        }
      }
    }
    let energy = 0;
    for (let i = 0; i < projLen; i++) {
      if (cnt[i] > minRowSpan) {
        const mean = proj[i] / cnt[i];
        energy += mean * mean;
      }
    }
    return energy;
  }

  let bestAngle = 0;
  let bestScore = -1;
  for (let a = -24; a <= 24; a += 1.5) {
    const s = scoreAngle(a);
    if (s > bestScore) {
      bestScore = s;
      bestAngle = a;
    }
  }
  const coarse = bestAngle;
  for (let a = coarse - 1.5; a <= coarse + 1.5; a += 0.25) {
    const s = scoreAngle(a);
    if (s > bestScore) {
      bestScore = s;
      bestAngle = a;
    }
  }
  return Math.abs(bestAngle) < 0.6 ? 0 : bestAngle;
}

/**
 * Estimates residual local horizontal line slope (dy/dx) within a vertical band of an already-rotated sheet.
 * Used to remove perspective trapezoidal fanning between top and bottom table rows.
 */
export function estimateBandLineSlope(src: PixelBuffer, yStartFrac: number, yEndFrac: number): number {
  const { width: w, height: h, data: d } = src;
  if (w < 60 || h < 60) return 0;
  const s = Math.min(0.35, 320 / Math.max(w, h));
  const dw = Math.max(60, Math.round(w * s));
  const dh = Math.max(60, Math.round(h * s));
  const stroke = new Float32Array(dw * dh);

  for (let y = 2; y < dh - 2; y++) {
    const sy = Math.min(h - 1, Math.floor(y / s));
    const syU = Math.max(0, sy - 4);
    const syD = Math.min(h - 1, sy + 4);
    for (let x = 2; x < dw - 2; x++) {
      const sx = Math.min(w - 1, Math.floor(x / s));
      const p = (sy * w + sx) * 4;
      const pU = (syU * w + sx) * 4;
      const pD = (syD * w + sx) * 4;
      const r = d[p];
      const b = d[p + 2];
      const l = 0.299 * r + 0.587 * d[p + 1] + 0.114 * b;
      const bg = Math.max(
        0.299 * d[pU] + 0.587 * d[pU + 1] + 0.114 * d[pU + 2],
        0.299 * d[pD] + 0.587 * d[pD + 1] + 0.114 * d[pD + 2]
      );
      if (bg > 125 && bg < 246 && bg - l > 13 && b - r <= 15) {
        stroke[y * dw + x] = bg - l;
      }
    }
  }

  const y0 = Math.max(0, Math.floor(dh * yStartFrac));
  const y1 = Math.min(dh, Math.floor(dh * yEndFrac));
  const x0 = Math.floor(dw * 0.15);
  const x1 = Math.floor(dw * 0.85);
  const cx = dw / 2;
  const proj = new Float32Array(dh * 2);
  let bestSlope = 0;
  let bestEnergy = -1;

  for (let sl = -0.15; sl <= 0.151; sl += 0.005) {
    proj.fill(0);
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const ry = Math.round(y - (x - cx) * sl + dh * 0.5);
        if (ry >= 0 && ry < proj.length) {
          proj[ry] += stroke[y * dw + x];
        }
      }
    }
    let e = 0;
    for (let i = 0; i < proj.length; i++) {
      e += proj[i] * proj[i];
    }
    if (e > bestEnergy) {
      bestEnergy = e;
      bestSlope = sl;
    }
  }
  return Math.abs(bestSlope) < 0.006 ? 0 : bestSlope;
}

/**
 * Un-fans perspective line convergence by interpolating vertical row slopes from yTop to yBot.
 */
export function unFanPerspectiveBuffer(
  src: PixelBuffer,
  yTop: number,
  slopeTop: number,
  yBot: number,
  slopeBot: number
): PixelBuffer {
  if (Math.abs(slopeTop) < 0.006 && Math.abs(slopeBot) < 0.006) return src;
  const { width: w, height: h, data: d } = src;
  const out = new Uint8ClampedArray(d.length).fill(255);
  const cx = w / 2;
  const spanY = Math.max(1, yBot - yTop);

  for (let y = 0; y < h; y++) {
    const t = Math.max(0, Math.min(1, (y - yTop) / spanY));
    const slope = slopeTop + (slopeBot - slopeTop) * t;
    for (let x = 0; x < w; x++) {
      const sy = y + (x - cx) * slope;
      if (sy >= 0 && sy < h - 1) {
        const y0 = Math.floor(sy);
        const fy = sy - y0;
        const i0 = (y0 * w + x) * 4;
        const i1 = ((y0 + 1) * w + x) * 4;
        const p = (y * w + x) * 4;
        out[p] = Math.round(d[i0] * (1 - fy) + d[i1] * fy);
        out[p + 1] = Math.round(d[i0 + 1] * (1 - fy) + d[i1 + 1] * fy);
        out[p + 2] = Math.round(d[i0 + 2] * (1 - fy) + d[i1 + 2] * fy);
        out[p + 3] = 255;
      }
    }
  }
  return { width: w, height: h, data: out };
}

/**
 * Rotates a PixelBuffer by angleDeg (matching estimateTableSkewAngle) onto an unclipped white canvas.
 */
export function rotatePixelBuffer(src: PixelBuffer, angleDeg: number, scale = 1.8): PixelBuffer {
  const sf = Math.max(1, Math.min(3.2, scale));
  const rad = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const absAngle = Math.abs(angleDeg);
  const padMulW = absAngle > 3 ? 1.15 : 1.02;
  const padMulH = absAngle > 3 ? 1.24 : 1.02;
  const w = Math.max(32, Math.round(src.width * sf * padMulW));
  const h = Math.max(32, Math.round(src.height * sf * padMulH));
  const out = new Uint8ClampedArray(w * h * 4).fill(255);
  const cx = src.width / 2;
  const cy = src.height / 2;
  const dcx = w / 2;
  const dcy = h / 2;
  const sw = src.width;
  const sh = src.height;
  const sdata = src.data;

  for (let y = 0; y < h; y++) {
    const ry = (y - dcy) / sf;
    for (let x = 0; x < w; x++) {
      const rx = (x - dcx) / sf;
      const sx = rx * cos + ry * sin + cx;
      const sy = -rx * sin + ry * cos + cy;
      if (sx >= 0 && sx < sw - 1 && sy >= 0 && sy < sh - 1) {
        const x0 = Math.floor(sx);
        const y0 = Math.floor(sy);
        const fx = sx - x0;
        const fy = sy - y0;
        const i00 = (y0 * sw + x0) * 4;
        const i10 = (y0 * sw + x0 + 1) * 4;
        const i01 = ((y0 + 1) * sw + x0) * 4;
        const i11 = ((y0 + 1) * sw + x0 + 1) * 4;
        const w00 = (1 - fx) * (1 - fy);
        const w10 = fx * (1 - fy);
        const w01 = (1 - fx) * fy;
        const w11 = fx * fy;
        const p = (y * w + x) * 4;
        out[p] = Math.round(sdata[i00] * w00 + sdata[i10] * w10 + sdata[i01] * w01 + sdata[i11] * w11);
        out[p + 1] = Math.round(sdata[i00 + 1] * w00 + sdata[i10 + 1] * w10 + sdata[i01 + 1] * w01 + sdata[i11 + 1] * w11);
        out[p + 2] = Math.round(sdata[i00 + 2] * w00 + sdata[i10 + 2] * w10 + sdata[i01 + 2] * w01 + sdata[i11 + 2] * w11);
        out[p + 3] = 255;
      }
    }
  }
  return { width: w, height: h, data: out };
}

/**
 * Checks whether a pixel buffer contains significant blue-pen strokes.
 */
export function hasSignificantBlueInk(src: PixelBuffer): boolean {
  const { width, height, data } = src;
  let bluePixels = 0;
  const total = width * height;
  if (total === 0) return false;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    if (b > r + 16 && b > g + 6 && b > 80) {
      bluePixels++;
    }
  }
  return bluePixels / total > 0.002;
}

/**
 * Smooth local-background-normalized grayscale pass with optional blue-ink emphasis.
 * Preserves anti-aliased edge smoothing ideal for Tesseract LSTM.
 */
export function applySmoothCleanPass(src: PixelBuffer, blueBoost = true): PixelBuffer {
  const { width, height, data } = src;
  const n = width * height;
  const out = new Uint8ClampedArray(n * 4).fill(255);
  if (n === 0) return { width, height, data: out };

  const lum = new Float32Array(n);
  const integral = new Float64Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y++) {
    let rs = 0;
    for (let x = 0; x < width; x++) {
      const p = (y * width + x) * 4;
      const l = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
      lum[y * width + x] = l;
      rs += l;
      integral[(y + 1) * (width + 1) + (x + 1)] = integral[y * (width + 1) + (x + 1)] + rs;
    }
  }

  const rad = Math.max(8, Math.min(26, Math.round(Math.min(width, height) / 14)));
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - rad);
    const y1 = Math.min(height - 1, y + rad);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - rad);
      const x1 = Math.min(width - 1, x + rad);
      const cnt = (x1 - x0 + 1) * (y1 - y0 + 1);
      const mean =
        (integral[(y1 + 1) * (width + 1) + (x1 + 1)] -
          integral[y0 * (width + 1) + (x1 + 1)] -
          integral[(y1 + 1) * (width + 1) + x0] +
          integral[y0 * (width + 1) + x0]) /
        cnt;
      const p = (y * width + x) * 4;
      const r = data[p];
      const g = data[p + 1];
      const b = data[p + 2];
      let ink = Math.max(0, mean - lum[y * width + x]);
      if (blueBoost) {
        const blueScore = Math.max(0, b - 0.6 * r - 0.4 * g + 14);
        ink = Math.max(ink, ink * 0.5 + blueScore * 1.3);
      }
      const val = ink <= 11 ? 255 : Math.max(0, Math.round(255 - (ink - 11) * 4.8));
      out[p] = val;
      out[p + 1] = val;
      out[p + 2] = val;
      out[p + 3] = 255;
    }
  }
  return { width, height, data: out };
}

/**
 * Cleans and upscales a single horizontal table row strip, automatically erasing
 * neutral-dark vertical divider bars while preserving blue ink & characters.
 */
export function cleanRowStripBuffer(src: PixelBuffer, blueBoost = true): PixelBuffer {
  const { width: w, height: h, data } = src;
  if (w < 8 || h < 6) return applySmoothCleanPass(src, blueBoost);

  const vNeutral = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let neutralDark = 0;
    for (let y = 0; y < h; y++) {
      const p = (y * w + x) * 4;
      const r = data[p];
      const g = data[p + 1];
      const b = data[p + 2];
      const l = 0.299 * r + 0.587 * g + 0.114 * b;
      if (l < 165 && b - r <= 12) neutralDark++;
    }
    vNeutral[x] = neutralDark / h;
  }

  const colBg = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    const vals: number[] = [];
    for (let y = 0; y < h; y++) {
      const p = (y * w + x) * 4;
      vals.push(0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]);
    }
    vals.sort((a, b) => a - b);
    colBg[x] = vals[Math.floor(vals.length * 0.85)] || 205;
  }

  const scale = Math.max(1.4, Math.min(2.8, 88 / Math.max(18, h)));
  const dw = Math.round(w * scale);
  const dh = Math.round(h * scale);
  const pad = 14;
  const ow = dw + pad * 2;
  const oh = dh + pad * 2;
  const out = new Uint8ClampedArray(ow * oh * 4).fill(255);

  for (let y = 0; y < dh; y++) {
    const sy = Math.min(h - 1.001, y / scale);
    const y0 = Math.floor(sy);
    const fy = sy - y0;
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(w - 1.001, x / scale);
      const x0 = Math.floor(sx);
      const fx = sx - x0;
      if (
        vNeutral[x0] > 0.68 ||
        (x0 > 0 && vNeutral[x0 - 1] > 0.72) ||
        (x0 + 1 < w && vNeutral[x0 + 1] > 0.72)
      ) {
        continue;
      }

      const i00 = (y0 * w + x0) * 4;
      const i10 = (y0 * w + x0 + 1) * 4;
      const i01 = ((y0 + 1) * w + x0) * 4;
      const i11 = ((y0 + 1) * w + x0 + 1) * 4;
      const w00 = (1 - fx) * (1 - fy);
      const w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy;
      const w11 = fx * fy;

      const r = data[i00] * w00 + data[i10] * w10 + data[i01] * w01 + data[i11] * w11;
      const g = data[i00 + 1] * w00 + data[i10 + 1] * w10 + data[i01 + 1] * w01 + data[i11 + 1] * w11;
      const b = data[i00 + 2] * w00 + data[i10 + 2] * w10 + data[i01 + 2] * w01 + data[i11 + 2] * w11;
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      let ink = Math.max(0, colBg[x0] - lum);
      if (blueBoost) {
        const blueScore = Math.max(0, b - 0.6 * r - 0.4 * g + 14);
        ink = Math.max(ink, ink * 0.5 + blueScore * 1.3);
      }
      const val = ink <= 11 ? 255 : Math.max(0, Math.round(255 - (ink - 11) * 4.8));
      const p = ((y + pad) * ow + (x + pad)) * 4;
      out[p] = val;
      out[p + 1] = val;
      out[p + 2] = val;
      out[p + 3] = 255;
    }
  }
  return { width: ow, height: oh, data: out };
}

/**
 * Applies a local preprocessing filter pass to improve OCR legibility on cell crops or sheets.
 */
export function applyImagePass(src: PixelBuffer, pass: ImagePassType): PixelBuffer {
  const { width, height, data } = src;
  const out = new Uint8ClampedArray(data.length);
  const n = width * height;
  if (n === 0) return { width, height, data: out };

  if (pass === 'blue_ink') {
    // Boost blue ink strokes relative to R/G against warm cream/beige paper
    const scores = new Float32Array(n);
    let maxScore = 10;
    for (let i = 0; i < n; i++) {
      const p = i * 4;
      const r = data[p];
      const g = data[p + 1];
      const b = data[p + 2];
      const blueExcess = Math.max(0, b - 0.6 * r - 0.4 * g + 12);
      scores[i] = blueExcess;
      if (blueExcess > maxScore) maxScore = blueExcess;
    }

    for (let i = 0; i < n; i++) {
      const p = i * 4;
      const norm = Math.min(1, scores[i] / Math.max(24, maxScore * 0.68));
      const v = norm > 0.22 ? Math.max(0, Math.round(255 * (1 - norm * 1.25))) : 255;
      out[p] = v;
      out[p + 1] = v;
      out[p + 2] = v;
      out[p + 3] = 255;
    }
    return { width, height, data: out };
  }

  // Compute grayscale luminance array
  const gray = new Uint8Array(n);
  const hist = new Int32Array(256);
  for (let i = 0; i < n; i++) {
    const p = i * 4;
    const lum = Math.max(0, Math.min(255, Math.round(0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2])));
    gray[i] = lum;
    hist[lum]++;
  }

  let cum = 0;
  let pLow = 0;
  let pHigh = 255;
  const lowTarget = n * 0.02;
  const highTarget = n * 0.98;
  for (let v = 0; v < 256; v++) {
    cum += hist[v];
    if (cum <= lowTarget) pLow = v;
    if (cum <= highTarget) pHigh = v;
  }
  const span = Math.max(20, pHigh - pLow);

  if (pass === 'normalized') {
    for (let i = 0; i < n; i++) {
      const stretched = Math.max(0, Math.min(255, Math.round(((gray[i] - pLow) / span) * 255)));
      const p = i * 4;
      out[p] = stretched;
      out[p + 1] = stretched;
      out[p + 2] = stretched;
      out[p + 3] = 255;
    }
    return { width, height, data: out };
  }

  return applySmoothCleanPass(src, true);
}

/**
 * Crops a rectangular sub-region from a PixelBuffer.
 */
export function cropPixelBuffer(src: PixelBuffer, x: number, y: number, w: number, h: number): PixelBuffer {
  const sx = Math.max(0, Math.floor(x));
  const sy = Math.max(0, Math.floor(y));
  const cw = Math.max(1, Math.min(src.width - sx, Math.floor(w)));
  const ch = Math.max(1, Math.min(src.height - sy, Math.floor(h)));
  const out = new Uint8ClampedArray(cw * ch * 4);

  for (let ry = 0; ry < ch; ry++) {
    const srcStart = ((sy + ry) * src.width + sx) * 4;
    const dstStart = ry * cw * 4;
    out.set(src.data.subarray(srcStart, srcStart + cw * 4), dstStart);
  }
  return { width: cw, height: ch, data: out };
}

export interface DetectedGridBounds {
  titleEndY: number;
  headerEndY: number;
  rowLinesY: number[];
  colLinesX: number[];
  autoDetected: boolean;
}

/**
 * Finds local maxima peaks separated by at least minGap.
 */
function findProjectionPeaks(profile: Float32Array, minVal: number, minGap: number): number[] {
  const peaks: number[] = [];
  let i = 0;
  while (i < profile.length) {
    if (profile[i] >= minVal) {
      let bestIdx = i;
      let bestVal = profile[i];
      let j = i + 1;
      while (j < profile.length && j - i <= minGap) {
        if (profile[j] > bestVal) {
          bestVal = profile[j];
          bestIdx = j;
        }
        j++;
      }
      peaks.push(bestIdx);
      i = bestIdx + minGap;
    } else {
      i++;
    }
  }
  return peaks;
}

/**
 * Detects horizontal grid rows using local background contrast (handles beige/shadowed paper)
 * and continuous vertical column dividers (avoiding false splits from stacked colons).
 */
export function detectGridBounds(
  rectified: PixelBuffer,
  expectedRowsFallback = 10,
  timePairCount = 2
): DetectedGridBounds {
  const { width: w, height: h, data } = rectified;
  const pairs = Math.max(1, Math.min(6, Math.round(timePairCount || 2)));
  const expectedDataCols = 1 + pairs * 2 + 1;
  const rowDark = new Float32Array(h);

  const xMin = Math.floor(w * 0.08);
  const xMax = Math.floor(w * 0.92);
  const yMin = Math.max(4, Math.floor(h * 0.06));
  const yMax = Math.min(h - 5, Math.floor(h * 0.98));

  for (let y = yMin; y < yMax; y++) {
    let darkCount = 0;
    for (let x = xMin; x < xMax; x++) {
      const p = (y * w + x) * 4;
      const pU = ((y - 4) * w + x) * 4;
      const pD = ((y + 4) * w + x) * 4;
      const lum = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
      const bg = Math.max(
        0.299 * data[pU] + 0.587 * data[pU + 1] + 0.114 * data[pU + 2],
        0.299 * data[pD] + 0.587 * data[pD + 1] + 0.114 * data[pD + 2]
      );
      if ((bg > 130 && bg - lum > 14) || lum < 128) {
        darkCount++;
      }
    }
    rowDark[y] = darkCount / Math.max(1, xMax - xMin);
  }

  const minRowGap = Math.max(16, Math.floor(h / 42));
  const rowPeaks = findProjectionPeaks(rowDark, 0.28, minRowGap);

  let titleEndY = Math.floor(h * 0.14);
  let headerEndY = Math.floor(h * 0.24);
  let rowLinesY: number[] = [];
  let autoDetected = false;

  if (rowPeaks.length >= 4) {
    autoDetected = true;
    titleEndY = Math.max(Math.floor(h * 0.05), rowPeaks[0]);
    headerEndY = rowPeaks[1];
    rowLinesY = rowPeaks.slice(1);
  } else {
    const startY = headerEndY;
    const endY = Math.floor(h * 0.96);
    const stepY = (endY - startY) / expectedRowsFallback;
    for (let r = 0; r <= expectedRowsFallback; r++) {
      rowLinesY.push(Math.round(startY + r * stepY));
    }
  }

  // Vertical column detection using neutral dark continuity (ignores stacked blue ':' digits)
  const colNeutral = new Float32Array(w);
  const colY0 = Math.max(4, Math.min(h - 10, titleEndY));
  const colY1 = Math.max(colY0 + 10, rowLinesY[rowLinesY.length - 1] || Math.floor(h * 0.95));
  for (let x = 4; x < w - 4; x++) {
    let dark = 0;
    for (let y = colY0; y < colY1; y++) {
      const p = (y * w + x) * 4;
      const r = data[p];
      const g = data[p + 1];
      const b = data[p + 2];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      const pL = (y * w + x - 4) * 4;
      const pR = (y * w + x + 4) * 4;
      const bg = Math.max(
        0.299 * data[pL] + 0.587 * data[pL + 1] + 0.114 * data[pL + 2],
        0.299 * data[pR] + 0.587 * data[pR + 1] + 0.114 * data[pR + 2]
      );
      if (b - r <= 12 && bg - lum > 14) {
        dark++;
      }
    }
    colNeutral[x] = dark / Math.max(1, colY1 - colY0);
  }

  const minColGap = Math.max(24, Math.floor(w / (expectedDataCols * 2.5 + 4)));
  const colPeaks = findProjectionPeaks(colNeutral, 0.42, minColGap);

  let colLinesX: number[] = [];
  if (colPeaks.length >= expectedDataCols + 1 && colPeaks.length <= expectedDataCols + 4) {
    colLinesX = colPeaks.slice(0, expectedDataCols + 1);
  } else {
    const dateW = 0.13;
    const totalW = 0.13;
    const startX = 0.02;
    const dataEndX = 0.82;
    const slotsSpan = Math.max(0.2, dataEndX - startX - dateW - totalW);
    const slotColW = slotsSpan / (pairs * 2);

    colLinesX.push(Math.round(startX * w));
    colLinesX.push(Math.round((startX + dateW) * w));
    for (let k = 1; k <= pairs * 2; k++) {
      colLinesX.push(Math.round((startX + dateW + k * slotColW) * w));
    }
    colLinesX.push(Math.round(dataEndX * w));
  }

  return {
    titleEndY,
    headerEndY,
    rowLinesY,
    colLinesX,
    autoDetected,
  };
}

/**
 * Parses a single raw string into a normalized HH:MM time (24h).
 */
export function parseStrictTimeString(raw: string): string | null {
  if (!raw || typeof raw !== 'string') return null;
  const cleaned = raw
    .trim()
    .replace(/[|!lI]/g, '1')
    .replace(/[oOQ]/g, '0')
    .replace(/\s+/g, ' ');

  const m =
    cleaned.match(/\b(2[0-3]|[01]?[0-9])\s*[hH:.\-]\s*([0-5][0-9])\b/) ||
    cleaned.match(/\b(2[0-3]|[01]?[0-9])\s+([0-5][0-9])\b/) ||
    cleaned.match(/\b(2[0-3]|[01]?[0-9])\s*[hH]\b/);

  if (!m) return null;
  const h = parseInt(m[1], 10);
  const min = m[2] ? parseInt(m[2], 10) : 0;
  if (h < 0 || h > 23 || min < 0 || min > 59) return null;
  return pad2(h) + ':' + pad2(min);
}

/**
 * Recovers a plausible HH:MM time from noisy OCR digit pairs (e.g. serif '1' misread as '4' or '7': "40:30" -> "10:30", "45:00" -> "15:00", "48:30" -> "18:30").
 * Marks recovered times with corrected=true so they ALWAYS require human review.
 */
export function recoverPlausibleOcrTime(rawToken: string): {
  time: string;
  corrected: boolean;
  original: string;
} | null {
  if (!rawToken || typeof rawToken !== 'string') return null;
  const cleaned = rawToken
    .trim()
    .replace(/[|!lI]/g, '1')
    .replace(/[oOQ]/g, '0');

  const strict = parseStrictTimeString(cleaned);
  if (strict) {
    return { time: strict, corrected: false, original: rawToken.trim() };
  }

  // Match two-digit hour > 23 with minute (e.g. 40:30, 45:00, 48:30, 75:00, 25:00)
  const badHm = cleaned.match(/^([3-9][0-9]|2[4-9])\s*[:.hH\-]\s*([0-5][0-9])$/);
  if (badHm) {
    const hStr = badHm[1];
    const mStr = badHm[2];
    const h1 = parseInt(hStr[0], 10);
    const h2 = parseInt(hStr[1], 10);
    // Serif/handwritten '1' often OCRs as '4', '7', or '2' in tens place (e.g. 40->10, 45->15, 48->18, 25->15)
    if (h1 === 4 || h1 === 7 || h1 === 2) {
      const recoveredHour = 10 + h2;
      if (recoveredHour >= 10 && recoveredHour <= 23) {
        return {
          time: pad2(recoveredHour) + ':' + mStr,
          corrected: true,
          original: rawToken.trim(),
        };
      }
    }
  }
  return null;
}

/**
 * Extracts structured shift items from a single row string emitted by row-strip OCR.
 */
export function parseSmartTimesheetRowTokens(
  rawRowText: string,
  titleMonth: number | null,
  titleYear: number | null,
  employeeName: string
): Array<{
  employeeName: string;
  date: string;
  serviceLabel: string;
  startTime: string;
  endTime: string;
  explicitNetMinutes: number | null;
  writtenTotalDisplay: string;
  needsVerification: boolean;
  verificationReason: string;
}> {
  if (!rawRowText || typeof rawRowText !== 'string') return [];
  const norm = rawRowText
    .replace(/[|!l]/g, '1')
    .replace(/[oOQ]/g, '0')
    .replace(/,/g, '.')
    .replace(/\s+/g, ' ')
    .trim();

  if (!norm || norm.length < 4) return [];

  // Extract date (accepts missing leading zero, leading slash, or border-digit prefix e.g. "01/11", "1402/11")
  let isoDate = '';
  let dateRecoveredNote = '';
  let remainder = norm;
  const dateMatch = norm.match(/(?:^|\s|\/)(\d{1,4})\s*[\/\-.]\s*(0?[1-9]|1[0-2])(?:\s*[\/\-.]\s*(20\d\d|\d\d))?(?=\s|$|\/|\.)/);
  if (dateMatch) {
    let day = parseInt(dateMatch[1], 10);
    if (day > 31 && dateMatch[1].length >= 3) {
      const tailTwo = parseInt(dateMatch[1].slice(-2), 10);
      if (tailTwo >= 1 && tailTwo <= 31) {
        day = tailTwo;
        dateRecoveredNote = 'Date redressée (' + dateMatch[1] + '→' + pad2(day) + ')';
      }
    }
    const mon = parseInt(dateMatch[2], 10);
    const yr = dateMatch[3]
      ? dateMatch[3].length === 2
        ? 2000 + parseInt(dateMatch[3], 10)
        : parseInt(dateMatch[3], 10)
      : titleYear || new Date().getFullYear();
    const dtCheck = new Date(yr, mon - 1, day);
    if (
      day >= 1 &&
      day <= 31 &&
      dtCheck.getFullYear() === yr &&
      dtCheck.getMonth() === mon - 1 &&
      dtCheck.getDate() === day
    ) {
      isoDate = String(yr) + '-' + pad2(mon) + '-' + pad2(day);
    }
    remainder = (norm.slice(0, dateMatch.index || 0) + ' ' + norm.slice((dateMatch.index || 0) + dateMatch[0].length)).trim();
  }

  // Extract all XX:YY time tokens (including OCR-recoverable 40:30, 45:00, 48:30)
  const timeMatches = remainder.match(/\b\d{1,2}\s*[:.hH\-]\s*\d{2}\b/g) || [];
  const parsedTimes: Array<{ time: string; corrected: boolean; original: string }> = [];
  for (const tok of timeMatches) {
    const rec = recoverPlausibleOcrTime(tok);
    if (rec) parsedTimes.push(rec);
  }

  // Extract trailing written total (if any)
  const durParsed = parseWrittenDurationString(remainder);

  if (parsedTimes.length < 2 && !durParsed) {
    return [];
  }

  const results: Array<{
    employeeName: string;
    date: string;
    serviceLabel: string;
    startTime: string;
    endTime: string;
    explicitNetMinutes: number | null;
    writtenTotalDisplay: string;
    needsVerification: boolean;
    verificationReason: string;
  }> = [];

  if (parsedTimes.length >= 2) {
    const pairCount = Math.floor(parsedTimes.length / 2);
    for (let k = 0; k < pairCount; k++) {
      const s = parsedTimes[k * 2];
      const e = parsedTimes[k * 2 + 1];
      const anyCorr = s.corrected || e.corrected;
      const reasons: string[] = [];
      if (!isoDate) reasons.push('Date manquante');
      else if (dateRecoveredNote) reasons.push(dateRecoveredNote + ' — à vérifier');
      if (anyCorr) {
        const notes: string[] = [];
        if (s.corrected) notes.push(s.original + '→' + s.time);
        if (e.corrected) notes.push(e.original + '→' + e.time);
        reasons.push('OCR auto (' + notes.join(', ') + ') — à vérifier');
      }
      if (!employeeName || employeeName === 'À vérifier') {
        reasons.push('Salarié à confirmer');
      }

      results.push({
        employeeName: employeeName || 'À vérifier',
        date: isoDate,
        serviceLabel: k === 0 ? 'Service du midi' : k === 1 ? 'Service du soir' : 'Créneau ' + (k + 1),
        startTime: s.time,
        endTime: e.time,
        explicitNetMinutes: null,
        writtenTotalDisplay: durParsed ? durParsed.display : '',
        needsVerification: !isoDate || Boolean(dateRecoveredNote) || anyCorr || !employeeName || employeeName === 'À vérifier',
        verificationReason: reasons.join(' · ') || 'Extrait localement (OCR ligne)',
      });
    }
  } else if (durParsed && isoDate) {
    results.push({
      employeeName: employeeName || 'À vérifier',
      date: isoDate,
      serviceLabel: 'Service',
      startTime: '',
      endTime: '',
      explicitNetMinutes: durParsed.minutes,
      writtenTotalDisplay: durParsed.display,
      needsVerification: true,
      verificationReason: 'Horaires exacts illisibles — total jour détecté (' + durParsed.display + ')',
    });
  }

  return results;
}

/**
 * Parses a single raw cell string as a written duration ("8,5 h", "8h30", "9 h", "9:00").
 */
export function parseWrittenDurationString(raw: string): { display: string; minutes: number } | null {
  if (!raw || typeof raw !== 'string') return null;
  const cleaned = raw.trim().replace(/[oO]/g, '0');

  const decMatch = cleaned.match(/\b(\d{1,2})\s*[,.]\s*(\d{1,2})\s*h?\b/i);
  if (decMatch) {
    const whole = parseInt(decMatch[1], 10);
    const fracStr = decMatch[2];
    const decimalVal = parseFloat(whole + '.' + fracStr);
    if (!isNaN(decimalVal) && decimalVal >= 0 && decimalVal <= 24) {
      const mins = Math.round(decimalVal * 60);
      return { display: formatMinsShort(mins), minutes: mins };
    }
  }

  const hmMatch = cleaned.match(/\b(\d{1,2})\s*[hH:]\s*([0-5]\d)\b/);
  if (hmMatch) {
    const h = parseInt(hmMatch[1], 10);
    const m = parseInt(hmMatch[2], 10);
    if (h >= 0 && h <= 24 && m >= 0 && m < 60) {
      const mins = h * 60 + m;
      return { display: formatMinsShort(mins), minutes: mins };
    }
  }

  const wholeMatch = cleaned.match(/\b(\d{1,2})\s*h\b/i);
  if (wholeMatch) {
    const h = parseInt(wholeMatch[1], 10);
    if (h > 0 && h <= 24) {
      const mins = h * 60;
      return { display: formatMinsShort(mins), minutes: mins };
    }
  }

  return null;
}

/**
 * Parses a date cell string ("01/11", "01/11/2026") using explicit title month & year.
 * NEVER falls back to today's date when empty or illegible.
 */
export function parseCellDateString(
  raw: string,
  titleMonth: number | null,
  titleYear: number | null
): { iso: string; day: number; month: number; year: number; yearExplicit: boolean } | null {
  if (!raw || typeof raw !== 'string') return null;
  const cleaned = raw.trim().replace(/[oO]/g, '0').replace(/[|!]/g, '1');
  const m = cleaned.match(/\b(0?[1-9]|[12]\d|3[01])\s*[\/\-.]\s*(0?[1-9]|1[0-2])(?:\s*[\/\-.]\s*(20\d\d|\d\d))?\b/);
  if (!m) return null;

  const day = parseInt(m[1], 10);
  const month = parseInt(m[2], 10);
  let yearExplicit = false;
  let year: number | null = null;

  if (m[3]) {
    yearExplicit = true;
    year = m[3].length === 2 ? 2000 + parseInt(m[3], 10) : parseInt(m[3], 10);
  } else if (titleYear !== null) {
    year = titleYear;
  }

  if (year === null) {
    return null;
  }

  const dtCheck = new Date(year, month - 1, day);
  if (dtCheck.getFullYear() !== year || dtCheck.getMonth() !== month - 1 || dtCheck.getDate() !== day) {
    return null;
  }

  return {
    iso: String(year) + '-' + pad2(month) + '-' + pad2(day),
    day,
    month,
    year,
    yearExplicit,
  };
}

/**
 * Evaluates multi-pass OCR outputs for a Date cell and surfaces candidates if passes disagree or confidence is low.
 */
export function analyzeDatePasses(
  passes: OcrPassResult[],
  titleMonth: number | null,
  titleYear: number | null,
  thumbDataUrl?: string
): CellAnalysisResult {
  const candidatesMap = new Map<string, ParsedCandidate>();
  let anyNonEmptyText = false;

  for (const pr of passes) {
    if (pr.rawText && pr.rawText.trim()) anyNonEmptyText = true;
    const parsed = parseCellDateString(pr.rawText, titleMonth, titleYear);
    if (parsed) {
      const existing = candidatesMap.get(parsed.iso);
      if (!existing || pr.confidence > existing.confidence) {
        candidatesMap.set(parsed.iso, {
          value: parsed.iso,
          pass: pr.pass,
          confidence: pr.confidence,
          rawText: pr.rawText.trim(),
        });
      }
    }
  }

  const candidates = Array.from(candidatesMap.values()).sort((a, b) => b.confidence - a.confidence);
  if (candidates.length === 0) {
    return {
      bestValue: '',
      bestMinutes: null,
      candidates: [],
      confidence: 0,
      isAmbiguous: anyNonEmptyText,
      isMissing: true,
      reason: anyNonEmptyText ? 'Date manuscrite illisible' : 'Date manquante',
      thumbDataUrl,
    };
  }

  const best = candidates[0];
  const monthMismatch = titleMonth !== null && parseInt(best.value.slice(5, 7), 10) !== titleMonth;
  const isAmbiguous = candidates.length > 1 || best.confidence < 72 || monthMismatch;
  let reason = '';
  if (candidates.length > 1) reason = 'Plusieurs dates candidates détectées';
  else if (monthMismatch) reason = 'Mois différent du titre';
  else if (best.confidence < 72) reason = 'Date peu fiable (confiance ' + Math.round(best.confidence) + '%)';

  return {
    bestValue: best.value,
    bestMinutes: null,
    candidates,
    confidence: best.confidence,
    isAmbiguous,
    isMissing: false,
    reason,
    thumbDataUrl,
  };
}

/**
 * Evaluates multi-pass OCR outputs for a Time cell (Arrivée / Départ).
 */
export function analyzeTimePasses(passes: OcrPassResult[], thumbDataUrl?: string): CellAnalysisResult {
  const candidatesMap = new Map<string, ParsedCandidate>();
  let anyNonEmptyText = false;

  for (const pr of passes) {
    const trimmed = (pr.rawText || '').trim();
    if (trimmed) anyNonEmptyText = true;
    const rec = recoverPlausibleOcrTime(trimmed);
    if (rec) {
      const existing = candidatesMap.get(rec.time);
      const effConf = rec.corrected ? Math.min(62, pr.confidence) : pr.confidence;
      if (!existing || effConf > existing.confidence) {
        candidatesMap.set(rec.time, {
          value: rec.time,
          pass: pr.pass,
          confidence: effConf,
          rawText: trimmed,
        });
      }
    }
  }

  const candidates = Array.from(candidatesMap.values()).sort((a, b) => b.confidence - a.confidence);
  if (candidates.length === 0) {
    return {
      bestValue: '',
      bestMinutes: null,
      candidates: [],
      confidence: 0,
      isAmbiguous: anyNonEmptyText,
      isMissing: true,
      reason: anyNonEmptyText ? 'Horaire manuscrit illisible' : '',
      thumbDataUrl,
    };
  }

  const best = candidates[0];
  const isAmbiguous = candidates.length > 1 || best.confidence < 74;
  let reason = '';
  if (candidates.length > 1) reason = 'Heures concurrentes (' + candidates.map((c) => c.value).join(' / ') + ')';
  else if (best.confidence < 74) reason = 'Chiffres incertains (' + Math.round(best.confidence) + '%)';

  return {
    bestValue: best.value,
    bestMinutes: null,
    candidates,
    confidence: best.confidence,
    isAmbiguous,
    isMissing: false,
    reason,
    thumbDataUrl,
  };
}

/**
 * Evaluates multi-pass OCR outputs for a Written Total Hours cell ("8,5 h", "8h30", "9 h").
 */
export function analyzeTotalPasses(passes: OcrPassResult[], thumbDataUrl?: string): CellAnalysisResult {
  const candidatesMap = new Map<string, ParsedCandidate>();
  let anyNonEmptyText = false;

  for (const pr of passes) {
    const trimmed = (pr.rawText || '').trim();
    if (trimmed) anyNonEmptyText = true;
    const parsed = parseWrittenDurationString(trimmed);
    if (parsed) {
      const existing = candidatesMap.get(parsed.display);
      if (!existing || pr.confidence > existing.confidence) {
        candidatesMap.set(parsed.display, {
          value: parsed.display,
          minutes: parsed.minutes,
          pass: pr.pass,
          confidence: pr.confidence,
          rawText: trimmed,
        });
      }
    }
  }

  const candidates = Array.from(candidatesMap.values()).sort((a, b) => b.confidence - a.confidence);
  if (candidates.length === 0) {
    return {
      bestValue: '',
      bestMinutes: null,
      candidates: [],
      confidence: 0,
      isAmbiguous: anyNonEmptyText,
      isMissing: true,
      reason: anyNonEmptyText ? 'Total écrit illisible' : '',
      thumbDataUrl,
    };
  }

  const best = candidates[0];
  const isAmbiguous = candidates.length > 1 || best.confidence < 72;
  const reason = candidates.length > 1 ? 'Totaux concurrents (' + candidates.map((c) => c.value).join(' / ') + ')' : '';

  return {
    bestValue: best.value,
    bestMinutes: best.minutes ?? null,
    candidates,
    confidence: best.confidence,
    isAmbiguous,
    isMissing: false,
    reason,
    thumbDataUrl,
  };
}

function slotDurationMinutes(startStr: string, endStr: string): number | null {
  if (!startStr || !endStr) return null;
  const sm = startStr.match(/^(\d{2}):(\d{2})$/);
  const em = endStr.match(/^(\d{2}):(\d{2})$/);
  if (!sm || !em) return null;
  const s = parseInt(sm[1], 10) * 60 + parseInt(sm[2], 10);
  const e = parseInt(em[1], 10) * 60 + parseInt(em[2], 10);
  if (s === e) return null;
  return e > s ? e - s : e + 1440 - s;
}

export function evaluateGridRow(params: {
  rowIndex: number;
  yStart: number;
  yEnd: number;
  dateCell: CellAnalysisResult;
  midiStartCell: CellAnalysisResult;
  midiEndCell: CellAnalysisResult;
  soirStartCell: CellAnalysisResult;
  soirEndCell: CellAnalysisResult;
  totalCell: CellAnalysisResult;
}): GridRowDetection {
  const {
    rowIndex,
    yStart,
    yEnd,
    dateCell,
    midiStartCell,
    midiEndCell,
    soirStartCell,
    soirEndCell,
    totalCell,
  } = params;

  const midiMinutes = slotDurationMinutes(midiStartCell.bestValue, midiEndCell.bestValue);
  const soirMinutes = slotDurationMinutes(soirStartCell.bestValue, soirEndCell.bestValue);

  let computedTotalMinutes: number | null = null;
  if (midiMinutes !== null || soirMinutes !== null) {
    computedTotalMinutes = (midiMinutes || 0) + (soirMinutes || 0);
  }

  let discrepancyMinutes: number | null = null;
  let discrepancyNotice = '';
  const reasons: string[] = [];

  if (dateCell.isMissing || !dateCell.bestValue) {
    reasons.push(dateCell.reason || 'Date manquante');
  } else if (dateCell.isAmbiguous) {
    reasons.push(dateCell.reason || 'Date incertaine');
  }

  if ((midiStartCell.bestValue && !midiEndCell.bestValue) || (!midiStartCell.bestValue && midiEndCell.bestValue)) {
    reasons.push('Créneau matin/midi incomplet');
  }
  if ((soirStartCell.bestValue && !soirEndCell.bestValue) || (!soirStartCell.bestValue && soirEndCell.bestValue)) {
    reasons.push('Créneau soir incomplet');
  }

  if (midiStartCell.isAmbiguous) reasons.push('Midi arrivée : ' + (midiStartCell.reason || 'incertain'));
  if (midiEndCell.isAmbiguous) reasons.push('Midi départ : ' + (midiEndCell.reason || 'incertain'));
  if (soirStartCell.isAmbiguous) reasons.push('Soir arrivée : ' + (soirStartCell.reason || 'incertain'));
  if (soirEndCell.isAmbiguous) reasons.push('Soir départ : ' + (soirEndCell.reason || 'incertain'));

  if (
    computedTotalMinutes !== null &&
    totalCell.bestMinutes !== null &&
    computedTotalMinutes !== totalCell.bestMinutes
  ) {
    discrepancyMinutes = Math.abs(computedTotalMinutes - totalCell.bestMinutes);
    discrepancyNotice =
      'Écart ' +
      discrepancyMinutes +
      ' min (calculé ' +
      formatMinsShort(computedTotalMinutes) +
      ' vs écrit ' +
      formatMinsShort(totalCell.bestMinutes) +
      ' — aucune pause déduite automatiquement)';
    reasons.push(discrepancyNotice);
  }

  const needsVerification = reasons.length > 0;

  return {
    rowIndex,
    yStart,
    yEnd,
    dateCell,
    midiStartCell,
    midiEndCell,
    soirStartCell,
    soirEndCell,
    totalCell,
    computedTotalMinutes,
    midiMinutes,
    soirMinutes,
    discrepancyMinutes,
    discrepancyNotice,
    needsVerification,
    verificationReasons: reasons,
  };
}

export interface FlexibleGridSlot {
  slotIndex: number;
  label: string;
  startCell: CellAnalysisResult;
  endCell: CellAnalysisResult;
  minutes: number | null;
}

export interface FlexibleGridRowDetection {
  rowIndex: number;
  yStart: number;
  yEnd: number;
  dateCell: CellAnalysisResult;
  slots: FlexibleGridSlot[];
  totalCell: CellAnalysisResult;
  computedTotalMinutes: number | null;
  discrepancyMinutes: number | null;
  discrepancyNotice: string;
  needsVerification: boolean;
  verificationReasons: string[];
}

export function evaluateFlexibleGridRow(params: {
  rowIndex: number;
  yStart: number;
  yEnd: number;
  dateCell: CellAnalysisResult;
  slotPairs: Array<{ label: string; startCell: CellAnalysisResult; endCell: CellAnalysisResult }>;
  totalCell: CellAnalysisResult;
}): FlexibleGridRowDetection {
  const { rowIndex, yStart, yEnd, dateCell, slotPairs, totalCell } = params;
  const reasons: string[] = [];

  if (dateCell.isMissing || !dateCell.bestValue) {
    reasons.push(dateCell.reason || 'Date manquante');
  } else if (dateCell.isAmbiguous) {
    reasons.push(dateCell.reason || 'Date incertaine');
  }

  let computedSum = 0;
  let hasAnyValidSlot = false;

  const slots: FlexibleGridSlot[] = slotPairs.map((pair, idx) => {
    const mins = slotDurationMinutes(pair.startCell.bestValue, pair.endCell.bestValue);
    if (mins !== null) {
      hasAnyValidSlot = true;
      computedSum += mins;
    }
    if ((pair.startCell.bestValue && !pair.endCell.bestValue) || (!pair.startCell.bestValue && pair.endCell.bestValue)) {
      reasons.push(pair.label + ' incomplet');
    }
    if (pair.startCell.isAmbiguous) reasons.push(pair.label + ' arrivée : ' + (pair.startCell.reason || 'incertain'));
    if (pair.endCell.isAmbiguous) reasons.push(pair.label + ' départ : ' + (pair.endCell.reason || 'incertain'));

    return {
      slotIndex: idx,
      label: pair.label,
      startCell: pair.startCell,
      endCell: pair.endCell,
      minutes: mins,
    };
  });

  const computedTotalMinutes = hasAnyValidSlot ? computedSum : null;

  let discrepancyMinutes: number | null = null;
  let discrepancyNotice = '';
  if (
    computedTotalMinutes !== null &&
    totalCell.bestMinutes !== null &&
    computedTotalMinutes !== totalCell.bestMinutes
  ) {
    discrepancyMinutes = Math.abs(computedTotalMinutes - totalCell.bestMinutes);
    discrepancyNotice =
      'Écart ' +
      discrepancyMinutes +
      ' min (calculé ' +
      formatMinsShort(computedTotalMinutes) +
      ' vs écrit ' +
      formatMinsShort(totalCell.bestMinutes) +
      ' — aucune pause déduite automatiquement)';
    reasons.push(discrepancyNotice);
  }

  return {
    rowIndex,
    yStart,
    yEnd,
    dateCell,
    slots,
    totalCell,
    computedTotalMinutes,
    discrepancyMinutes,
    discrepancyNotice,
    needsVerification: reasons.length > 0,
    verificationReasons: reasons,
  };
}


