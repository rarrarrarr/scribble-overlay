// ==UserScript==
// @name         Scribble overlay (experiment)
// @namespace    https://github.com/rarrarrarr/scribble-overlay
// @version      0.1.0
// @description  Draws the video you're watching as a live pencil-and-ink sketch. Alt+S toggles it.
// @match        *://*/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==

// A WebGL2 port of scripts/scribble.py in https://github.com/rarrarrarr/afha. Each video frame goes through:
//   CLAHE gray -> pencil (colour dodge) + ink (Canny edges) + cross-hatching -> paper -> wobble -> tint
// The Python version is the reference for the look; tune here, then carry settings back.

(() => {
  'use strict';
  if (window.scribbleOverlay) return;

  const MAX_HEIGHT = 1080; // larger videos are processed at 1080p
  const STORE_KEY = 'scribble-overlay-params';

  // key, label, min, max, step, default (defaults reproduce scribble.py)
  const PARAMS = [
    ['split', 'Before / after', 0, 1, 0.01, 0],
    ['gamma', 'Pencil darkness', 0.5, 3, 0.05, 1.6],
    ['ink', 'Ink', 0, 1, 0.01, 0.85],
    ['edgeLow', 'Edge threshold (lower = more lines)', 10, 120, 1, 40],
    ['hatch', 'Hatching', 0, 1, 0.01, 0.75],
    ['shadow', 'Shadow reach', -0.3, 0.3, 0.01, 0],
    ['paper', 'Paper grain', 0, 0.15, 0.005, 0.035],
    ['wobble', 'Wobble', 0, 3, 0.05, 0.6],
  ].map(([key, label, min, max, step, def]) => ({ key, label, min, max, step, def }));
  const EDGE_HIGH_RATIO = 110 / 40; // Canny high threshold = low * this

  const defaults = () => Object.fromEntries(PARAMS.map((p) => [p.key, p.def]));
  const params = defaults();
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
    for (const p of PARAMS) if (typeof saved[p.key] === 'number') params[p.key] = saved[p.key];
  } catch { /* storage blocked or corrupt: use defaults */ }
  const saveParams = () => { try { localStorage.setItem(STORE_KEY, JSON.stringify(params)); } catch { } };

  // ---------------------------------------------------------------- shaders
  // All intermediate textures are stored top row first (like the video), so a
  // pass's texel (x, y) is image pixel (x, y). Only the final pass flips to the screen.

  const VS = `#version 300 es
void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

  const HEAD = `#version 300 es
precision highp float; precision highp int; out vec4 o;
`;

  const FS = {
    // Gray + CLAHE (clip 2.0, 8x8 tiles), using per-tile lookup tables computed on the CPU.
    clahe: HEAD + `
uniform sampler2D uVideo, uLut; uniform vec2 uSize;
float lut(float v, vec2 t) { return texelFetch(uLut, ivec2(int(v), int(t.y) * 8 + int(t.x)), 0).r * 255.0; }
void main() {
  vec2 px = gl_FragCoord.xy - 0.5;
  vec3 c = texture(uVideo, (px + 0.5) / uSize).rgb;
  float v = floor(dot(c, vec3(0.299, 0.587, 0.114)) * 255.0 + 0.5);
  vec2 tf = px / ceil(uSize / 8.0) - 0.5;
  vec2 t1 = floor(tf), a = tf - t1;
  vec2 t2 = min(t1 + 1.0, 7.0); t1 = max(t1, 0.0);
  float r = mix(mix(lut(v, t1), lut(v, vec2(t2.x, t1.y)), a.x),
                mix(lut(v, vec2(t1.x, t2.y)), lut(v, t2), a.x), a.y);
  r = floor(r + 0.5) / 255.0;
  o = vec4(r, r, 0.0, 1.0);
}`,

    // Separable Gaussian, a different sigma per channel (R: pencil blur, G: hatch tone).
    blur: HEAD + `
uniform sampler2D uSrc; uniform vec2 uDir, uSigma;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy), mx = textureSize(uSrc, 0) - 1, d = ivec2(uDir);
  int rad = int(ceil(3.0 * max(uSigma.x, uSigma.y)));
  vec2 acc = vec2(0.0), ws = vec2(0.0), k = -0.5 / (uSigma * uSigma);
  for (int i = -rad; i <= rad; i++) {
    vec2 w = exp(float(i * i) * k);
    acc += texelFetch(uSrc, clamp(p + d * i, ivec2(0), mx), 0).rg * w; ws += w;
  }
  o = vec4(acc / ws, 0.0, 1.0);
}`,

    // cv2.bilateralFilter(gray, 7, 50, 7)
    bilateral: HEAD + `
uniform sampler2D uSrc;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy), mx = textureSize(uSrc, 0) - 1;
  float c = texelFetch(uSrc, p, 0).r * 255.0, acc = 0.0, ws = 0.0;
  for (int y = -3; y <= 3; y++) for (int x = -3; x <= 3; x++) {
    if (x * x + y * y > 9) continue;
    float s = texelFetch(uSrc, clamp(p + ivec2(x, y), ivec2(0), mx), 0).r * 255.0;
    float w = exp(-float(x * x + y * y) / 98.0 - (s - c) * (s - c) / 5000.0);
    acc += s * w; ws += w;
  }
  o = vec4(floor(acc / ws + 0.5) / 255.0, 0.0, 0.0, 1.0);
}`,

    // Canny: Sobel (L1 magnitude) + non-maximum suppression + double threshold.
    // Output 1 = strong edge, 0.5 = weak edge, 0 = none.
    canny: HEAD + `
uniform sampler2D uSrc; uniform vec2 uThresh;
ivec2 mx;
float px(ivec2 q) { return texelFetch(uSrc, clamp(q, ivec2(0), mx), 0).r * 255.0; }
vec2 grad(ivec2 p) {
  float a = px(p + ivec2(-1, -1)), b = px(p + ivec2(0, -1)), c = px(p + ivec2(1, -1));
  float d = px(p + ivec2(-1, 0)), f = px(p + ivec2(1, 0));
  float g = px(p + ivec2(-1, 1)), h = px(p + ivec2(0, 1)), i = px(p + ivec2(1, 1));
  return vec2(c + 2.0 * f + i - a - 2.0 * d - g, g + 2.0 * h + i - a - 2.0 * b - c);
}
float mag(ivec2 p) { vec2 g = grad(p); return abs(g.x) + abs(g.y); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); mx = textureSize(uSrc, 0) - 1;
  vec2 g = grad(p); float m = abs(g.x) + abs(g.y), res = 0.0;
  if (m > uThresh.x) {
    float ax = abs(g.x), ay = abs(g.y);
    ivec2 d1, d2;
    if (ay < ax * 0.41421356) { d1 = ivec2(-1, 0); d2 = ivec2(1, 0); }
    else if (ay > ax * 2.41421356) { d1 = ivec2(0, -1); d2 = ivec2(0, 1); }
    else { int s = g.x * g.y < 0.0 ? -1 : 1; d1 = ivec2(-s, -1); d2 = ivec2(s, 1); }
    if (m > mag(p + d1) && m >= mag(p + d2)) res = m > uThresh.y ? 1.0 : 0.5;
  }
  o = vec4(res, 0.0, 0.0, 1.0);
}`,

    // One step of hysteresis: a weak edge touching a strong one becomes strong.
    hyst: HEAD + `
uniform sampler2D uSrc;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy), mx = textureSize(uSrc, 0) - 1;
  float c = texelFetch(uSrc, p, 0).r;
  if (c > 0.25 && c < 0.75)
    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++)
      if (texelFetch(uSrc, clamp(p + ivec2(x, y), ivec2(0), mx), 0).r > 0.75) c = 1.0;
  o = vec4(c, 0.0, 0.0, 1.0);
}`,

    // Keep strong edges, dilated with a 2x2 kernel (cv2.dilate anchor: up and left).
    dilate: HEAD + `
uniform sampler2D uSrc;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float e = 0.0;
  for (int y = -1; y <= 0; y++) for (int x = -1; x <= 0; x++)
    e = max(e, step(0.75, texelFetch(uSrc, max(p + ivec2(x, y), ivec2(0)), 0).r));
  o = vec4(e, 0.0, 0.0, 1.0);
}`,

    // Compose pencil * ink * hatching * paper, with the slow hand wobble and the warm tint.
    final: HEAD + `
uniform sampler2D uVideo, uGray, uBlur, uEdge, uTex;
uniform vec2 uSize;
uniform float uTime, uGamma, uInk, uHatch, uShadow, uPaper, uWobble, uSplit;
vec4 at(sampler2D s, vec2 p) { return texture(s, (p + 0.5) / uSize); }
void main() {
  vec2 px = vec2(gl_FragCoord.x - 0.5, uSize.y - gl_FragCoord.y - 0.5);
  if (px.x < uSplit * uSize.x) { o = vec4(at(uVideo, px).rgb, 1.0); return; }
  // cv2.warpAffine with M = [[1, a, 0], [b, 1, 0]] samples the source at M^-1 * dst
  float k = sin(uTime * 0.7) * uWobble, a = 0.004 * k, b = 0.003 * k;
  vec2 p = vec2(px.x - a * px.y, px.y - b * px.x) / (1.0 - a * b);
  float g = at(uGray, p).r;
  vec2 bl = at(uBlur, p).rg;
  float pencil = pow(clamp(g / max(bl.x, 1e-3), 0.0, 1.0), uGamma);
  float e = 0.0, ws = 0.0;
  for (int y = -2; y <= 2; y++) for (int x = -2; x <= 2; x++) {
    float w = exp(-float(x * x + y * y) / 0.98);
    e += at(uEdge, p + vec2(x, y)).r * w; ws += w;
  }
  float ink = 1.0 - e / ws * uInk;
  vec4 t = at(uTex, p);
  vec3 mask = clamp((vec3(0.55, 0.35, 0.18) + uShadow - bl.y) / 0.12, 0.0, 1.0);
  vec3 sh = 1.0 - mask * (1.0 - t.rgb) * uHatch;
  float v = pencil * ink * min(min(sh.x, sh.y), sh.z) * (1.0 - uPaper * t.a);
  o = vec4(clamp(vec3(v, v * 0.96 + 0.025, v * 0.93 + 0.04), 0.0, 1.0), 1.0);
}`,
  };

  // ---------------------------------------------------------------- textures made once per size

  function rng(seed) { // mulberry32
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const randInt = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1)); // inclusive

  // Parallel jittered lines at `angle` degrees, every `spacing` px (as Sketcher._hatch).
  function hatch(W, H, angle, spacing, seed) {
    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, W, H);
    ctx.translate(W / 2, H / 2);
    ctx.rotate(-angle * Math.PI / 180); // OpenCV's positive angle is counter-clockwise
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    const r = rng(seed), D = W + H;
    ctx.beginPath();
    for (let y = -D; y < D; y += spacing) {
      const off = randInt(r, -1, 1), jitter = randInt(r, -3, 3);
      ctx.moveTo(-D, y + off + 0.5);
      ctx.lineTo(D, y + off + jitter + 0.5);
    }
    ctx.stroke();
    return ctx.getImageData(0, 0, W, H).data;
  }

  // RGB: three hatch layers. A: paper grain (blurred noise, 0..1).
  function makeTextureData(W, H) {
    const out = new Uint8Array(W * H * 4);
    [[45, 6, 1], [-45, 6, 2], [0, 5, 3]].forEach(([angle, spacing, seed], ch) => {
      const px = hatch(W, H, angle, spacing, seed);
      for (let i = 0; i < W * H; i++) out[i * 4 + ch] = px[i * 4];
    });
    const r = rng(7), n = new Float32Array(W * H), tmp = new Float32Array(W * H);
    for (let i = 0; i < n.length; i++) { // Box-Muller
      n[i] = Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
    }
    const k = [-2, -1, 0, 1, 2].map((i) => Math.exp(-i * i / (2 * 0.8 * 0.8)));
    const ks = k.reduce((s, v) => s + v);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let s = 0;
      for (let i = -2; i <= 2; i++) s += n[y * W + Math.min(W - 1, Math.max(0, x + i))] * k[i + 2];
      tmp[y * W + x] = s / ks;
    }
    let lo = Infinity, hi = -Infinity;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let s = 0;
      for (let i = -2; i <= 2; i++) s += tmp[Math.min(H - 1, Math.max(0, y + i)) * W + x] * k[i + 2];
      n[y * W + x] = s / ks;
      lo = Math.min(lo, n[y * W + x]); hi = Math.max(hi, n[y * W + x]);
    }
    for (let i = 0; i < n.length; i++) out[i * 4 + 3] = Math.round(255 * (n[i] - lo) / (hi - lo));
    return out;
  }

  // ---------------------------------------------------------------- the WebGL pipeline

  class Engine {
    constructor(canvas) {
      const gl = canvas.getContext('webgl2', {
        preserveDrawingBuffer: true, antialias: false, alpha: false, premultipliedAlpha: false,
      });
      if (!gl) throw new Error('WebGL2 is not available in this browser.');
      this.gl = gl;
      this.canvas = canvas;
      this.vao = gl.createVertexArray();
      this.progs = Object.fromEntries(Object.entries(FS).map(([k, src]) => [k, this.program(k, src)]));
      this.videoTex = this.texture(gl.LINEAR, gl.CLAMP_TO_EDGE);
      this.lutTex = this.texture(gl.NEAREST, gl.CLAMP_TO_EDGE);
      this.lut = new Uint8Array(256 * 64);
      this.hist = new Uint32Array(256 * 64);
      this.sample = document.createElement('canvas');
      this.sampleCtx = this.sample.getContext('2d', { willReadFrequently: true });
      this.W = this.H = 0;
      this.targets = null;
    }

    program(name, fs) {
      const gl = this.gl;
      const compile = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`${name} shader: ${gl.getShaderInfoLog(s)}`);
        return s;
      };
      const p = gl.createProgram();
      gl.attachShader(p, compile(gl.VERTEX_SHADER, VS));
      gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`${name} program: ${gl.getProgramInfoLog(p)}`);
      p.locs = {};
      return p;
    }

    texture(filter, wrap) {
      const gl = this.gl, t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
      return t;
    }

    target(W, H) {
      const gl = this.gl;
      // mirrored repeat = OpenCV's BORDER_REFLECT for the wobble warp
      const tex = this.texture(gl.LINEAR, gl.MIRRORED_REPEAT);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      return { tex, fb };
    }

    resize(W, H) {
      if (W === this.W && H === this.H) return;
      const gl = this.gl;
      if (this.targets) {
        for (const t of Object.values(this.targets)) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); }
        gl.deleteTexture(this.hatchTex);
      }
      this.W = W; this.H = H;
      this.canvas.width = W; this.canvas.height = H;
      this.targets = Object.fromEntries(['gray', 'tmp', 'blur', 'bil', 'ea', 'eb', 'edge'].map((k) => [k, this.target(W, H)]));
      this.hatchTex = this.texture(gl.LINEAR, gl.MIRRORED_REPEAT);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, makeTextureData(W, H));
      // CLAHE histograms come from a small copy of the frame, sized in whole tiles
      const s = Math.min(1, 384 / W);
      this.sample.width = Math.max(8, 8 * Math.round(W * s / 8));
      this.sample.height = Math.max(8, 8 * Math.round(H * s / 8));
    }

    // Per-tile CLAHE lookup tables, as OpenCV builds them (clip limit 2.0, 8x8 tiles).
    updateLut(video) {
      const { sample, sampleCtx: ctx, hist, lut } = this;
      const sw = sample.width, sh = sample.height, tw = sw / 8, th = sh / 8, area = tw * th;
      ctx.drawImage(video, 0, 0, sw, sh);
      const d = ctx.getImageData(0, 0, sw, sh).data; // throws SecurityError for cross-origin video
      hist.fill(0);
      for (let y = 0; y < sh; y++) {
        const row = Math.floor(y / th) * 8;
        for (let x = 0; x < sw; x++) {
          const i = (y * sw + x) * 4;
          const g = Math.round(d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114);
          hist[(row + Math.floor(x / tw)) * 256 + g]++;
        }
      }
      const limit = Math.max(1, Math.floor(2.0 * area / 256)), scale = 255 / area;
      for (let t = 0; t < 64; t++) {
        const h = hist.subarray(t * 256, t * 256 + 256);
        let clipped = 0;
        for (let i = 0; i < 256; i++) if (h[i] > limit) { clipped += h[i] - limit; h[i] = limit; }
        const batch = Math.floor(clipped / 256);
        let residual = clipped - batch * 256;
        for (let i = 0; i < 256; i++) h[i] += batch;
        const step = Math.max(Math.floor(256 / Math.max(residual, 1)), 1);
        for (let i = 0; i < 256 && residual > 0; i += step, residual--) h[i]++;
        let sum = 0;
        for (let i = 0; i < 256; i++) { sum += h[i]; lut[t * 256 + i] = Math.min(255, Math.round(sum * scale)); }
      }
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 256, 64, 0, gl.RED, gl.UNSIGNED_BYTE, lut);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    }

    pass(name, target, textures, uniforms = {}) {
      const gl = this.gl, p = this.progs[name];
      const loc = (n) => (n in p.locs ? p.locs[n] : (p.locs[n] = gl.getUniformLocation(p, n)));
      gl.useProgram(p);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fb : null);
      gl.viewport(0, 0, this.W, this.H);
      let unit = 0;
      for (const [n, t] of Object.entries(textures)) {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.uniform1i(loc(n), unit++);
      }
      for (const [n, v] of Object.entries(uniforms)) {
        const l = loc(n);
        if (l === null) continue;
        if (Array.isArray(v)) gl.uniform2fv(l, v); else gl.uniform1f(l, v);
      }
      gl.bindVertexArray(this.vao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    render(video, time, P) {
      const gl = this.gl, T = this.targets, size = [this.W, this.H];
      gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
      this.updateLut(video);
      this.pass('clahe', T.gray, { uVideo: this.videoTex, uLut: this.lutTex }, { uSize: size });
      this.pass('blur', T.tmp, { uSrc: T.gray.tex }, { uDir: [1, 0], uSigma: [6, 3] });
      this.pass('blur', T.blur, { uSrc: T.tmp.tex }, { uDir: [0, 1], uSigma: [6, 3] });
      this.pass('bilateral', T.bil, { uSrc: T.gray.tex });
      this.pass('canny', T.ea, { uSrc: T.bil.tex }, { uThresh: [P.edgeLow, P.edgeLow * EDGE_HIGH_RATIO] });
      let a = T.ea, b = T.eb;
      for (let i = 0; i < 4; i++) { this.pass('hyst', b, { uSrc: a.tex }); [a, b] = [b, a]; }
      this.pass('dilate', T.edge, { uSrc: a.tex });
      this.pass('final', null, {
        uVideo: this.videoTex, uGray: T.gray.tex, uBlur: T.blur.tex, uEdge: T.edge.tex, uTex: this.hatchTex,
      }, {
        uSize: size, uTime: time, uGamma: P.gamma, uInk: P.ink, uHatch: P.hatch,
        uShadow: P.shadow, uPaper: P.paper, uWobble: P.wobble, uSplit: P.split,
      });
    }
  }

  // ---------------------------------------------------------------- overlay on the page

  let engine = null, canvas = null, video = null, panel = null, statusEl = null;
  let on = false, dirty = true, mediaTime = 0, lastTime = -1, gen = 0, failed = false;
  let sizeText = '';
  // Performance readout, refreshed once a second: frames the video presented vs frames
  // we sketched, CPU time per sketch, and frames the browser dropped (too slow to keep up).
  const stats = { start: 0, presented: 0, drawn: 0, cpuMs: 0, dropped: 0 };

  function el(tag, props = {}, ...kids) {
    const e = document.createElement(tag); // DOM APIs only: YouTube enforces Trusted Types
    for (const [k, v] of Object.entries(props)) {
      if (k === 'style') Object.assign(e.style, v);
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e[k] = v;
    }
    e.append(...kids);
    return e;
  }

  function setStatus(text) { if (statusEl) statusEl.textContent = text; }

  // The largest video on screen that has loaded.
  function pickVideo() {
    let best = null, bestArea = 0;
    for (const v of document.querySelectorAll('video')) {
      const r = v.getBoundingClientRect();
      const area = Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0)) *
        Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0));
      if (v.readyState >= 1 && area > bestArea) { best = v; bestArea = area; }
    }
    return best;
  }

  function attach(v) {
    video = v;
    v.after(canvas);
    const g = ++gen;
    dirty = true;
    failed = false;
    resetStats();
    if (v.mediaKeys) setStatus('This video is DRM-protected; the browser shows its frames as black.');
    if ('requestVideoFrameCallback' in v) {
      const onFrame = (_now, meta) => {
        if (g !== gen) return;
        mediaTime = meta.mediaTime;
        stats.presented++;
        dirty = true;
        v.requestVideoFrameCallback(onFrame);
      };
      v.requestVideoFrameCallback(onFrame);
    }
  }

  // Cover the area where the video's picture is actually drawn.
  function layout() {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;
    const r = video.getBoundingClientRect();
    const fit = getComputedStyle(video).objectFit;
    let w = r.width, h = r.height;
    if (fit !== 'fill') {
      const s = fit === 'cover' ? Math.max(r.width / vw, r.height / vh) : Math.min(r.width / vw, r.height / vh);
      w = vw * s; h = vh * s;
    }
    let ox = -scrollX, oy = -scrollY;
    const op = canvas.offsetParent;
    if (op && !(op === document.body && getComputedStyle(op).position === 'static')) {
      const pr = op.getBoundingClientRect();
      ox = pr.left + op.clientLeft - op.scrollLeft;
      oy = pr.top + op.clientTop - op.scrollTop;
    }
    const s = canvas.style;
    s.left = `${r.left + (r.width - w) / 2 - ox}px`;
    s.top = `${r.top + (r.height - h) / 2 - oy}px`;
    s.width = `${w}px`;
    s.height = `${h}px`;
  }

  function draw() {
    let W = video.videoWidth, H = video.videoHeight;
    if (!W || !H || video.readyState < 2) return;
    if (H > MAX_HEIGHT) { W = Math.round(W * MAX_HEIGHT / H); H = MAX_HEIGHT; }
    try {
      const t0 = performance.now();
      engine.resize(W, H);
      engine.render(video, mediaTime || video.currentTime, params);
      stats.cpuMs += performance.now() - t0;
      stats.drawn++;
      sizeText = `${W}×${H}${H < video.videoHeight ? ` (from ${video.videoHeight}p)` : ''}`;
    } catch (e) {
      failed = true; // don't retry every frame
      setStatus(e.name === 'SecurityError'
        ? "This video comes from another site without permission to read its frames, so it can't be sketched."
        : `Error: ${e.message}`);
      console.error('[scribble]', e);
    }
  }

  function tick() {
    if (!on) return;
    requestAnimationFrame(tick);
    if (!video || !video.isConnected) {
      const v = pickVideo();
      if (!v) return;
      attach(v);
    }
    if (!canvas.isConnected) video.after(canvas);
    layout();
    if (!('requestVideoFrameCallback' in video) && video.currentTime !== lastTime) {
      lastTime = video.currentTime;
      dirty = true;
    }
    if (dirty && !failed) { dirty = false; draw(); }
    reportStats();
  }

  function droppedFrames() {
    const q = video.getVideoPlaybackQuality && video.getVideoPlaybackQuality();
    return q ? q.droppedVideoFrames : 0;
  }

  function resetStats() {
    Object.assign(stats, { start: performance.now(), presented: 0, drawn: 0, cpuMs: 0, dropped: droppedFrames() });
  }

  function reportStats() {
    const now = performance.now(), secs = (now - stats.start) / 1000;
    if (secs < 1) return;
    if (!failed && !video.mediaKeys && sizeText) {
      const perf = video.paused
        ? 'paused'
        : `${Math.round(stats.drawn / secs)} fps sketched of ${Math.round(stats.presented / secs)} · ` +
          `${stats.drawn ? (stats.cpuMs / stats.drawn).toFixed(1) : '–'} ms CPU · ` +
          `${droppedFrames() - stats.dropped} dropped`;
      setStatus(`${sizeText} · ${perf}`);
    }
    resetStats();
  }

  function enable(v) {
    if (on) return;
    v = v || pickVideo();
    if (!panel) panel = buildPanel();
    placePanel();
    if (!v) { setStatus('No playing video found on this page.'); return; }
    if (!canvas) {
      canvas = el('canvas', {
        style: { position: 'absolute', pointerEvents: 'none', display: 'block', margin: '0', padding: '0' },
      });
      try {
        engine = new Engine(canvas);
      } catch (e) {
        canvas = null;
        setStatus(e.message);
        console.error('[scribble]', e);
        return;
      }
    }
    on = true;
    attach(v);
    requestAnimationFrame(tick);
  }

  function disable() {
    on = false;
    gen++;
    if (canvas) canvas.remove();
    if (panel) panel.remove();
    video = null;
  }

  const toggle = () => (on || (panel && panel.isConnected) ? disable() : enable());

  // In fullscreen only the fullscreen element's subtree is shown, so the panel moves into it.
  function placePanel() {
    if (!panel) return;
    const host = document.fullscreenElement && !(document.fullscreenElement instanceof HTMLVideoElement)
      ? document.fullscreenElement : document.body;
    if (panel.parentNode !== host) host.append(panel);
  }
  document.addEventListener('fullscreenchange', () => { if (panel && panel.isConnected) placePanel(); });

  function saveSnapshot() {
    if (!canvas || !engine || !engine.W) return;
    const t = video ? video.currentTime : 0;
    canvas.toBlob((blob) => {
      const a = el('a', { href: URL.createObjectURL(blob), download: `scribble-${Math.floor(t / 60)}m${(t % 60).toFixed(2)}s.png` });
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    }, 'image/png');
  }

  function buildPanel() {
    const inputs = {};
    const fmt = (p, v) => (p.step >= 1 ? String(v) : v.toFixed(p.step < 0.01 ? 3 : 2));
    const rows = PARAMS.map((p) => {
      const value = el('span', { style: { float: 'right', opacity: '0.7' } }, fmt(p, params[p.key]));
      const input = el('input', {
        type: 'range', min: p.min, max: p.max, step: p.step, value: params[p.key],
        style: { width: '100%', margin: '2px 0 6px' },
        oninput: () => {
          params[p.key] = Number(input.value);
          value.textContent = fmt(p, params[p.key]);
          saveParams();
          dirty = true;
        },
      });
      inputs[p.key] = { input, value, p };
      return el('label', { style: { display: 'block' } }, p.label, value, input);
    });
    const button = (text, onclick) => el('button', {
      onclick,
      style: {
        font: 'inherit', padding: '3px 8px', marginRight: '4px', border: '1px solid #bbb',
        borderRadius: '4px', background: '#fff', color: '#222', cursor: 'pointer',
      },
    }, text);
    statusEl = el('div', { style: { margin: '2px 0 8px', opacity: '0.7', minHeight: '1em' } });
    const box = el('div', {
      style: {
        position: 'fixed', top: '12px', right: '12px', zIndex: '2147483647', width: '230px',
        padding: '10px 12px', background: 'rgba(250, 247, 240, 0.96)', color: '#222',
        font: '12px/1.35 system-ui, sans-serif', borderRadius: '8px', boxShadow: '0 4px 18px rgba(0,0,0,0.3)',
        textAlign: 'left',
      },
      // keep slider keys from reaching the site's own shortcuts (e.g. YouTube's arrow-key seek)
      onkeydown: (e) => e.stopPropagation(),
    },
    el('div', { style: { fontWeight: '600', fontSize: '13px' } }, 'scribble',
      el('span', { style: { float: 'right', cursor: 'pointer', opacity: '0.6' }, title: 'Close (Alt+S)', onclick: disable }, '✕')),
    statusEl,
    ...rows,
    el('div', { style: { marginTop: '4px' } },
      button('Save PNG', saveSnapshot),
      button('Copy settings', () => navigator.clipboard.writeText(JSON.stringify(params)).then(() => setStatus('Settings copied.'))),
      button('Reset', () => {
        Object.assign(params, defaults());
        for (const { input, value, p } of Object.values(inputs)) { input.value = params[p.key]; value.textContent = fmt(p, params[p.key]); }
        saveParams();
        dirty = true;
      })),
    el('div', { style: { marginTop: '6px', opacity: '0.55' } }, 'Alt+S toggles the sketch.'));
    return box;
  }

  window.addEventListener('keydown', (e) => {
    if (e.altKey && !e.ctrlKey && !e.metaKey && e.code === 'KeyS') {
      e.preventDefault();
      e.stopPropagation();
      toggle();
    }
  }, true);

  // For the test page and the console.
  window.scribbleOverlay = {
    enable, disable, toggle, params,
    render: () => { if (on) { failed = false; draw(); } },
    get canvas() { return canvas; },
  };
})();
