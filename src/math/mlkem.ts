export type Poly = number[];
export type PresetId = "512" | "768" | "1024" | "lab";
export type SecretDistribution = "cbd" | "ternary";

export interface Parameters {
  n: number;
  k: number;
  q: number;
  eta1: number;
  eta2: number;
  keyErrorEta: number;
  du: number;
  dv: number;
  secretDistribution: SecretDistribution;
}

export const PRESETS: Record<PresetId, Parameters> = {
  "512": { n: 256, k: 2, q: 3329, eta1: 3, eta2: 2, keyErrorEta: 3, du: 10, dv: 4, secretDistribution: "cbd" },
  "768": { n: 256, k: 3, q: 3329, eta1: 2, eta2: 2, keyErrorEta: 2, du: 10, dv: 4, secretDistribution: "cbd" },
  "1024": { n: 256, k: 4, q: 3329, eta1: 2, eta2: 2, keyErrorEta: 2, du: 11, dv: 5, secretDistribution: "cbd" },
  lab: { n: 4, k: 1, q: 97, eta1: 2, eta2: 2, keyErrorEta: 2, du: 6, dv: 4, secretDistribution: "cbd" },
};

export function standardName(p: Parameters): string | null {
  for (const id of ["512", "768", "1024"] as const) {
    const preset = PRESETS[id];
    if ((Object.keys(preset) as (keyof Parameters)[]).every((key) => p[key] === preset[key])) {
      return `ML-KEM-${id}`;
    }
  }
  return null;
}

export interface CipherState {
  r: Poly[];
  e1: Poly[];
  e2: Poly;
  bits: number[];
  mu: Poly;
  u: Poly[];
  v: Poly;
  cu: Poly[];
  cv: Poly;
  uDecoded: Poly[];
  vDecoded: Poly;
  wExact: Poly;
  w: Poly;
  expectedNoise: Poly;
  compressionDelta: Poly;
  recoveredBits: number[];
  bitErrors: number;
}

export interface Instance {
  p: Parameters;
  seed: number;
  A: Poly[][];
  s: Poly[];
  e: Poly[];
  exact: Poly[];
  t: Poly[];
  cipher: CipherState;
  keyIdentity: boolean;
  cipherIdentity: boolean;
  generationMs: number;
}

export interface PublicData {
  p: Parameters;
  A: Poly[][];
  t: Poly[];
}

export function mod(value: number, q: number): number {
  return ((value % q) + q) % q;
}

export function centered(value: number, q: number): number {
  const residue = mod(value, q);
  return residue > Math.floor(q / 2) ? residue - q : residue;
}

export function makeRng(seed: number, domain: string): () => number {
  let state = 2166136261;
  const input = `${seed}:${domain}`;
  for (let i = 0; i < input.length; i++) {
    state = Math.imul(state ^ input.charCodeAt(i), 16777619);
  }
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return (value ^ (value >>> 14)) >>> 0;
  };
}

export function uniformInt(next: () => number, range: number): number {
  const limit = Math.floor(4294967296 / range) * range;
  let value = next();
  while (value >= limit) value = next();
  return value % range;
}

export function sampleCBD(next: () => number, eta: number): number {
  let value = 0;
  for (let i = 0; i < eta; i++) value += next() & 1;
  for (let i = 0; i < eta; i++) value -= next() & 1;
  return value;
}

export function multiply(a: Poly, b: Poly, q: number): Poly {
  const n = a.length;
  const output = Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    if (a[i] === 0) continue;
    for (let j = 0; j < n; j++) {
      if (b[j] === 0) continue;
      const index = i + j;
      if (index < n) output[index] += a[i] * b[j];
      else output[index - n] -= a[i] * b[j];
    }
  }
  return output.map((value) => mod(value, q));
}

export function addPolys(a: Poly, b: Poly, q: number): Poly {
  return a.map((value, i) => mod(value + b[i], q));
}

export function subtractPolys(a: Poly, b: Poly, q: number): Poly {
  return a.map((value, i) => mod(value - b[i], q));
}

export function dotPolys(a: Poly[], b: Poly[], q: number): Poly {
  let output = Array<number>(a[0].length).fill(0);
  for (let i = 0; i < a.length; i++) output = addPolys(output, multiply(a[i], b[i], q), q);
  return output;
}

export function matrixVector(A: Poly[][], vector: Poly[], q: number, transpose = false): Poly[] {
  const k = vector.length;
  const output: Poly[] = [];
  for (let i = 0; i < k; i++) {
    let row = Array<number>(vector[0].length).fill(0);
    for (let j = 0; j < k; j++) {
      row = addPolys(row, multiply(transpose ? A[j][i] : A[i][j], vector[j], q), q);
    }
    output.push(row);
  }
  return output;
}

export function compress(value: number, bits: number, q: number): number {
  const size = 2 ** bits;
  return Math.floor((size * mod(value, q)) / q + 0.5) % size;
}

export function decompress(value: number, bits: number, q: number): number {
  return Math.floor((q * value) / 2 ** bits + 0.5);
}

function samplePoly(n: number, next: () => number, eta: number): Poly {
  return Array.from({ length: n }, () => sampleCBD(next, eta));
}

export function generateInstance(p: Parameters, seed: number): Instance {
  const started = performance.now();
  const matrixRng = makeRng(seed, `matrix:${p.n}:${p.k}:${p.q}`);
  const secretRng = makeRng(seed, `secret:${p.n}:${p.k}`);
  const errorRng = makeRng(seed, `key-error:${p.n}:${p.k}`);
  const ephemeralRng = makeRng(seed, `ephemeral:${p.n}:${p.k}`);
  const encryptionRng = makeRng(seed, `encryption-error:${p.n}:${p.k}`);
  const messageRng = makeRng(seed, `message:${p.n}`);

  const A = Array.from({ length: p.k }, () =>
    Array.from({ length: p.k }, () =>
      Array.from({ length: p.n }, () => uniformInt(matrixRng, p.q)),
    ),
  );
  const s = Array.from({ length: p.k }, () =>
    p.secretDistribution === "cbd"
      ? samplePoly(p.n, secretRng, p.eta1)
      : Array.from({ length: p.n }, () => uniformInt(secretRng, 3) - 1),
  );
  const e = Array.from({ length: p.k }, () => samplePoly(p.n, errorRng, p.keyErrorEta));
  const exact = matrixVector(A, s, p.q);
  const t = exact.map((poly, i) => addPolys(poly, e[i], p.q));

  const r = Array.from({ length: p.k }, () => samplePoly(p.n, ephemeralRng, p.eta1));
  const e1 = Array.from({ length: p.k }, () => samplePoly(p.n, encryptionRng, p.eta2));
  const e2 = samplePoly(p.n, encryptionRng, p.eta2);
  const bits = Array.from({ length: p.n }, () => messageRng() & 1);
  const mu = bits.map((bit) => decompress(bit, 1, p.q));
  const u = matrixVector(A, r, p.q, true).map((poly, i) => addPolys(poly, e1[i], p.q));
  const v = addPolys(addPolys(dotPolys(t, r, p.q), e2, p.q), mu, p.q);
  const cu = u.map((poly) => poly.map((value) => compress(value, p.du, p.q)));
  const cv = v.map((value) => compress(value, p.dv, p.q));
  const uDecoded = cu.map((poly) => poly.map((value) => decompress(value, p.du, p.q)));
  const vDecoded = cv.map((value) => decompress(value, p.dv, p.q));
  const wExact = subtractPolys(v, dotPolys(s, u, p.q), p.q);
  const w = subtractPolys(vDecoded, dotPolys(s, uDecoded, p.q), p.q);
  const expectedNoise = subtractPolys(
    addPolys(dotPolys(e, r, p.q), e2, p.q),
    dotPolys(s, e1, p.q),
    p.q,
  );
  const compressionDelta = w.map((value, i) => centered(value - wExact[i], p.q));
  const recoveredBits = w.map((value) => compress(value, 1, p.q));

  return {
    p, seed, A, s, e, exact, t,
    cipher: {
      r, e1, e2, bits, mu, u, v, cu, cv, uDecoded, vDecoded,
      wExact, w, expectedNoise, compressionDelta, recoveredBits,
      bitErrors: recoveredBits.reduce((count, bit, i) => count + Number(bit !== bits[i]), 0),
    },
    keyIdentity: t.every((poly, i) => poly.every((value, j) => value === mod(exact[i][j] + e[i][j], p.q))),
    cipherIdentity: wExact.every((value, i) => value === mod(mu[i] + expectedNoise[i], p.q)),
    generationMs: performance.now() - started,
  };
}

export function flatten(polys: Poly[]): number[] {
  return polys.flat();
}

export function coefficientTerms(instance: Instance, row: number, column: number, coefficient: number) {
  const { p, A, s } = instance;
  const terms = Array.from({ length: p.n }, (_, aIndex) => {
    const sIndex = (coefficient - aIndex + p.n) % p.n;
    const sign = aIndex <= coefficient ? 1 : -1;
    return { aIndex, sIndex, sign, a: A[row][column][aIndex], s: s[column][sIndex], value: sign * A[row][column][aIndex] * s[column][sIndex] };
  });
  return { terms, rawSum: terms.reduce((sum, term) => sum + term.value, 0), product: multiply(A[row][column], s[column], p.q)[coefficient] };
}