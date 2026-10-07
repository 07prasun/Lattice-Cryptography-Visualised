import { centered, flatten, makeRng, matrixVector, uniformInt, type Instance, type PublicData } from "./mlkem";

export type Construction = "embedding" | "primal";
export type Vector = number[];

export interface AttackLattice {
  construction: Construction;
  m: number;
  d: number;
  D: number;
  tau: number;
  rows: number[];
  source: Vector[];
  labels: string[];
  fullBasis: boolean;
  relationIndex: number;
  determinantLog2: number;
  publicTarget: Vector;
  exactObservation: Vector;
  errorComponent: Vector;
  plantedPoint: Vector;
  witness: Vector;
  witnessVerified: boolean;
}

export interface SamplePoint {
  id: string;
  coeffs: number[];
  vector: Vector;
}

export interface GramSchmidt {
  orthogonal: Vector[];
  mu: number[][];
  squared: number[];
  lengths: number[];
  log2Covolume: number;
}

export interface BKZResult {
  start: number;
  size: number;
  tested: number;
  before: number;
  best: number;
  improved: boolean;
  coefficients: number[] | null;
  preparationSteps: number;
}

export interface BasisState {
  vectors: Vector[];
  provenance: number[][];
  k: number;
  done: boolean;
  steps: number;
  swaps: number;
  reductions: number;
  operation: string;
  blockIndex: number;
  tours: number;
  bkzChecks: number;
  lastBKZ: BKZResult | null;
}

export interface SieveVector {
  vector: Vector;
  coeffs: number[];
}

export interface SieveState {
  list: SieveVector[];
  pairChecks: number;
  steps: number;
  lastPair: [number, number] | null;
  lastSource: Vector | null;
  lastReducer: Vector | null;
  lastDifference: Vector | null;
  lastBefore: number;
  lastAfter: number;
  exhausted: boolean;
}

export interface CandidateResult {
  admissible: boolean;
  reason: string;
  candidate: number[] | null;
  residual: number[] | null;
  maxResidual: number | null;
  smallSecret: boolean;
  smallError: boolean;
}

export function dot(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

export function norm(a: Vector): number {
  return Math.sqrt(dot(a, a));
}

export function addScaled(target: Vector, source: Vector, amount: number): Vector {
  return target.map((value, i) => value + amount * source[i]);
}

export function linearCombination(basis: Vector[], coefficients: number[]): Vector {
  const result = Array<number>(basis[0].length).fill(0);
  coefficients.forEach((coefficient, index) => {
    if (!coefficient) return;
    for (let j = 0; j < result.length; j++) result[j] += coefficient * basis[index][j];
  });
  return result;
}

// Coefficient matrix of multiplication by the actual public module matrix A.
export function matrixCoefficient(publicData: PublicData, observation: number, unknown: number): number {
  const { n, q } = publicData.p;
  const row = Math.floor(observation / n);
  const degree = observation % n;
  const column = Math.floor(unknown / n);
  const power = unknown % n;
  const a = publicData.A[row][column][(degree - power + n) % n];
  return centered((degree >= power ? 1 : -1) * a, q);
}

export function constructAttack(instance: Instance, samples: number, construction: Construction, tau: number): AttackLattice {
  const { n, k, q } = instance.p;
  const d = n * k;
  const m = Math.max(2, Math.min(samples, d));
  const rows = Array.from({ length: m }, (_, index) => Math.floor((index * d) / m));
  const embedded = construction === "embedding";
  const D = m + d + Number(embedded);
  const source: Vector[] = [];
  const labels: string[] = [];
  const fullBasis = D <= 17;

  const qColumns = fullBasis ? Array.from({ length: m }, (_, i) => i) : Array.from({ length: Math.min(3, m) }, (_, i) => i);
  const relationColumns = fullBasis ? Array.from({ length: d }, (_, i) => i) : [0, 1];
  for (const i of qColumns) {
    const vector = Array<number>(D).fill(0);
    vector[i] = q;
    source.push(vector);
    labels.push(`q e[obs ${rows[i]}]`);
  }
  const relationIndex = source.length;
  for (const j of relationColumns) {
    const vector = Array<number>(D).fill(0);
    for (let i = 0; i < m; i++) vector[i] = matrixCoefficient(instance, rows[i], j);
    vector[m + j] = 1;
    source.push(vector);
    labels.push(`(M[:,${j}], e[secret ${j}])`);
  }

  const tFlat = flatten(instance.t);
  const sFlat = flatten(instance.s);
  const eFlat = flatten(instance.e);
  const publicTarget = Array<number>(D).fill(0);
  const exactObservation = Array<number>(D).fill(0);
  const errorComponent = Array<number>(D).fill(0);
  const plantedPoint = Array<number>(D).fill(0);
  const witness = Array<number>(D).fill(0);
  let witnessVerified = true;
  for (let i = 0; i < m; i++) {
    const observed = centered(tFlat[rows[i]], q);
    const clean = observed - eFlat[rows[i]];
    publicTarget[i] = observed;
    exactObservation[i] = clean;
    errorComponent[i] = eFlat[rows[i]];
    plantedPoint[i] = clean;
    witness[i] = -eFlat[rows[i]];

    let liftedProduct = 0;
    for (let j = 0; j < d; j++) {
      if (sFlat[j]) liftedProduct += matrixCoefficient(instance, rows[i], j) * sFlat[j];
    }
    if ((clean - liftedProduct) % q !== 0) witnessVerified = false;
  }
  for (let j = 0; j < d; j++) {
    plantedPoint[m + j] = sFlat[j];
    witness[m + j] = sFlat[j];
  }
  if (embedded) {
    const embeddingColumn = Array<number>(D).fill(0);
    for (let i = 0; i < m; i++) embeddingColumn[i] = publicTarget[i];
    embeddingColumn[D - 1] = tau;
    source.push(embeddingColumn);
    labels.push(`(t, 0, ${tau})`);
    witness[D - 1] = -tau;
  }

  return {
    construction, m, d, D, tau, rows, source, labels, fullBasis, relationIndex,
    determinantLog2: m * Math.log2(q) + (embedded ? Math.log2(tau) : 0),
    publicTarget, exactObservation, errorComponent, plantedPoint, witness, witnessVerified,
  };
}

export function sampleLatticePoints(attack: AttackLattice): SamplePoint[] {
  const samples: SamplePoint[] = [];
  for (let a = -3; a <= 3; a++) {
    for (let b = -3; b <= 3; b++) {
      for (let c = -1; c <= 1; c++) {
        const coeffs = Array<number>(attack.source.length).fill(0);
        coeffs[0] = a;
        coeffs[1] = b;
        coeffs[attack.relationIndex] = c;
        samples.push({ id: `${a},${b},${c}`, coeffs, vector: linearCombination(attack.source, coeffs) });
      }
    }
  }
  return samples;
}

export function gramSchmidt(basis: Vector[]): GramSchmidt {
  const orthogonal: Vector[] = [];
  const mu: number[][] = [];
  const squared: number[] = [];
  for (let i = 0; i < basis.length; i++) {
    const orth = [...basis[i]];
    mu[i] = Array<number>(basis.length).fill(0);
    for (let j = 0; j < i; j++) {
      const coefficient = squared[j] > 1e-18 ? dot(basis[i], orthogonal[j]) / squared[j] : 0;
      mu[i][j] = coefficient;
      for (let x = 0; x < orth.length; x++) orth[x] -= coefficient * orthogonal[j][x];
    }
    orthogonal.push(orth);
    squared.push(dot(orth, orth));
  }
  return {
    orthogonal,
    mu,
    squared,
    lengths: squared.map((value) => Math.sqrt(Math.max(0, value))),
    log2Covolume: squared.reduce((sum, value) => sum + Math.log2(Math.max(value, 1e-30)) / 2, 0),
  };
}

function identity(size: number): number[][] {
  return Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => Number(i === j)));
}

export function initialBasis(attack: AttackLattice, skewed = false): BasisState {
  const vectors = attack.source.map((vector) => [...vector]);
  const provenance = identity(vectors.length);
  if (skewed) {
    for (let i = 0; i < Math.min(vectors.length - 1, 5); i++) {
      const amount = i % 2 === 0 ? 3 : -2;
      vectors[i] = addScaled(vectors[i], vectors[i + 1], amount);
      provenance[i] = addScaled(provenance[i], provenance[i + 1], amount);
    }
  }
  return {
    vectors, provenance, k: 1, done: vectors.length <= 1, steps: 0, swaps: 0,
    reductions: 0, operation: skewed ? "Unimodular shears applied; lattice unchanged." : "Public basis constructed.",
    blockIndex: 0, tours: 0, bkzChecks: 0, lastBKZ: null,
  };
}

export function verifyBasisLineage(attack: AttackLattice, state: BasisState): boolean {
  return state.vectors.every((vector, i) => {
    const reconstructed = linearCombination(attack.source, state.provenance[i]);
    return vector.every((value, j) => value === reconstructed[j]);
  });
}

export function provenanceDeterminant(state: BasisState): bigint {
  const matrix = state.provenance.map((row) => row.map((value) => BigInt(value)));
  const size = matrix.length;
  let sign = 1n;
  let previous = 1n;
  for (let k = 0; k < size - 1; k++) {
    if (matrix[k][k] === 0n) {
      const pivotRow = matrix.findIndex((row, index) => index > k && row[k] !== 0n);
      if (pivotRow === -1) return 0n;
      [matrix[k], matrix[pivotRow]] = [matrix[pivotRow], matrix[k]];
      sign = -sign;
    }
    const pivot = matrix[k][k];
    for (let i = k + 1; i < size; i++) {
      for (let j = k + 1; j < size; j++) {
        matrix[i][j] = (matrix[i][j] * pivot - matrix[i][k] * matrix[k][j]) / previous;
      }
      matrix[i][k] = 0n;
    }
    previous = pivot;
  }
  return sign * matrix[size - 1][size - 1];
}

export function lllStep(state: BasisState): BasisState {
  if (state.done) return state;
  const { vectors, provenance } = state;
  const k = state.k;
  if (k >= vectors.length) return { ...state, done: true, operation: "LLL conditions satisfied for this represented basis." };
  const gs = gramSchmidt(vectors);
  for (let j = k - 1; j >= 0; j--) {
    if (Math.abs(gs.mu[k][j]) > 0.50000001) {
      const factor = Math.round(gs.mu[k][j]);
      const nextVectors = [...vectors];
      const nextProvenance = [...provenance];
      nextVectors[k] = addScaled(vectors[k], vectors[j], -factor);
      nextProvenance[k] = addScaled(provenance[k], provenance[j], -factor);
      return {
        ...state, vectors: nextVectors, provenance: nextProvenance,
        steps: state.steps + 1, reductions: state.reductions + 1,
        operation: `Size reduce b${k + 1} <- b${k + 1} - (${factor}) b${j + 1}.`,
      };
    }
  }
  const delta = 0.99;
  const satisfiesLovasz = gs.squared[k] + 1e-8 >= (delta - gs.mu[k][k - 1] ** 2) * gs.squared[k - 1];
  if (satisfiesLovasz) {
    const nextK = k + 1;
    return {
      ...state, k: nextK, done: nextK >= vectors.length, steps: state.steps + 1,
      operation: `Lovasz condition holds at b${k + 1}; advance${nextK >= vectors.length ? " (complete)" : ""}.`,
    };
  }
  const nextVectors = [...vectors];
  const nextProvenance = [...provenance];
  [nextVectors[k - 1], nextVectors[k]] = [nextVectors[k], nextVectors[k - 1]];
  [nextProvenance[k - 1], nextProvenance[k]] = [nextProvenance[k], nextProvenance[k - 1]];
  return {
    ...state, vectors: nextVectors, provenance: nextProvenance,
    k: Math.max(1, k - 1), steps: state.steps + 1, swaps: state.swaps + 1,
    operation: `Lovasz test fails; swap b${k} and b${k + 1}.`,
  };
}

export function finishLLL(state: BasisState, limit = 4000): BasisState {
  let current = state;
  for (let step = 0; step < limit && !current.done; step++) current = lllStep(current);
  if (!current.done) return { ...current, operation: `Stopped after ${limit} LLL operations; continue manually.` };
  return current;
}

function gcd(a: number, b: number): number {
  while (b !== 0) [a, b] = [b, a % b];
  return Math.abs(a);
}

function extendedGcd(a: number, b: number): [number, number, number] {
  if (b === 0) return [Math.abs(a), Math.sign(a) || 1, 0];
  const [g, x, y] = extendedGcd(b, a % b);
  return [g, y, x - Math.trunc(a / b) * y];
}

// First row equals a primitive coefficient vector; every row operation has determinant 1.
function primitiveCompletion(coeffs: number[]): number[][] {
  const U = identity(coeffs.length);
  const current = [...coeffs];
  for (let j = 1; j < coeffs.length; j++) {
    const a = current[0];
    const b = current[j];
    if (b === 0) continue;
    const [g, x, y] = extendedGcd(a, b);
    const first = [...U[0]];
    const other = [...U[j]];
    U[0] = first.map((value, index) => (a / g) * value + (b / g) * other[index]);
    U[j] = first.map((value, index) => -y * value + x * other[index]);
    current[0] = g;
    current[j] = 0;
  }
  if (current[0] === -1) U[0] = U[0].map((value) => -value);
  return U;
}

function projectedNormSq(vector: Vector, gs: GramSchmidt, prefix: number): number {
  const residual = [...vector];
  for (let i = 0; i < prefix; i++) {
    const scalar = dot(vector, gs.orthogonal[i]) / gs.squared[i];
    for (let j = 0; j < residual.length; j++) residual[j] -= scalar * gs.orthogonal[i][j];
  }
  return dot(residual, residual);
}

export function bkzBlockStep(state: BasisState, beta: number): BasisState {
  const prepared = state.done ? state : finishLLL(state);
  const rank = prepared.vectors.length;
  const start = prepared.blockIndex % rank;
  const size = Math.min(beta, rank - start);
  const gs = gramSchmidt(prepared.vectors);
  const beforeSq = projectedNormSq(prepared.vectors[start], gs, start);
  let bestSq = beforeSq;
  let bestCoeffs: number[] | null = null;
  let tested = 0;
  const coeffs = Array<number>(size).fill(0);
  const bound = size <= 5 ? 2 : 1;

  function enumerate(index: number) {
    if (index === size) {
      if (coeffs.every((value) => value === 0)) return;
      let common = 0;
      for (const value of coeffs) common = gcd(common, value);
      if (common !== 1) return;
      tested++;
      const candidate = linearCombination(prepared.vectors.slice(start, start + size), coeffs);
      const candidateSq = projectedNormSq(candidate, gs, start);
      if (candidateSq + 1e-6 < bestSq) {
        bestSq = candidateSq;
        bestCoeffs = [...coeffs];
      }
      return;
    }
    for (let value = -bound; value <= bound; value++) {
      coeffs[index] = value;
      enumerate(index + 1);
    }
  }
  enumerate(0);

  let next = prepared;
  if (bestCoeffs !== null) {
    const U = primitiveCompletion(bestCoeffs as number[]);
    const oldVectors = prepared.vectors.slice(start, start + size);
    const oldProvenance = prepared.provenance.slice(start, start + size);
    const vectors = [...prepared.vectors];
    const provenance = [...prepared.provenance];
    for (let i = 0; i < size; i++) {
      vectors[start + i] = linearCombination(oldVectors, U[i]);
      provenance[start + i] = linearCombination(oldProvenance, U[i]);
    }
    next = finishLLL({ ...prepared, vectors, provenance, k: 1, done: false });
  }
  const nextIndex = start + 1 >= rank ? 0 : start + 1;
  return {
    ...next,
    blockIndex: nextIndex,
    tours: prepared.tours + Number(nextIndex === 0),
    bkzChecks: prepared.bkzChecks + tested,
    lastBKZ: {
      start, size, tested, before: Math.sqrt(beforeSq), best: Math.sqrt(bestSq),
      improved: bestCoeffs !== null, coefficients: bestCoeffs,
      preparationSteps: prepared.steps - state.steps,
    },
    operation: bestCoeffs !== null
      ? `Block ${start + 1}..${start + size}: inserted a shorter projected primitive vector; LLL cleaned up.`
      : `Block ${start + 1}..${start + size}: no shorter vector in the bounded search.`,
  };
}

export function bkzTour(state: BasisState, beta: number): BasisState {
  let next = state;
  const rank = state.vectors.length;
  for (let i = 0; i < rank; i++) next = bkzBlockStep(next, beta);
  return next;
}

export function initialSieve(basis: Vector[], seed: number): SieveState {
  const next = makeRng(seed, "sieve-list");
  const list: SieveVector[] = [];
  const used = new Set<string>();
  const active = Math.min(4, basis.length);
  while (list.length < 16) {
    const coeffs = Array<number>(basis.length).fill(0);
    for (let i = 0; i < active; i++) coeffs[i] = uniformInt(next, 5) - 2;
    const key = coeffs.join(",");
    if (used.has(key) || coeffs.every((value) => value === 0)) continue;
    used.add(key);
    list.push({ coeffs, vector: linearCombination(basis, coeffs) });
  }
  return { list, pairChecks: 0, steps: 0, lastPair: null, lastSource: null, lastReducer: null, lastDifference: null, lastBefore: 0, lastAfter: 0, exhausted: false };
}

export function sieveStep(state: SieveState): SieveState {
  let best: { i: number; j: number; before: number; after: number; difference: Vector } | null = null;
  let checks = 0;
  for (let i = 0; i < state.list.length; i++) {
    for (let j = 0; j < state.list.length; j++) {
      if (i === j) continue;
      checks++;
      const before = norm(state.list[i].vector);
      const difference = addScaled(state.list[i].vector, state.list[j].vector, -1);
      const after = norm(difference);
      if (after + 1e-7 < before && (!best || before - after > best.before - best.after)) {
        best = { i, j, before, after, difference };
      }
    }
  }
  if (!best) return { ...state, pairChecks: state.pairChecks + checks, exhausted: true, lastPair: null };
  const { i, j, before, after, difference } = best;
  const list = [...state.list];
  list[i] = {
    vector: difference,
    coeffs: addScaled(state.list[i].coeffs, state.list[j].coeffs, -1),
  };
  return {
    list, pairChecks: state.pairChecks + checks, steps: state.steps + 1,
    lastPair: [i, j], lastSource: state.list[i].vector, lastReducer: state.list[j].vector,
    lastDifference: difference, lastBefore: before, lastAfter: after, exhausted: false,
  };
}

// Only A and t enter this test; the generated private s and e are never consulted.
export function verifyPublicCandidate(publicData: PublicData, attack: AttackLattice, vector: Vector): CandidateResult {
  if (attack.construction !== "embedding") {
    return { admissible: false, reason: "Primal form is a CVP target, not an embedding candidate.", candidate: null, residual: null, maxResidual: null, smallSecret: false, smallError: false };
  }
  const last = vector[attack.D - 1];
  if (Math.abs(last) !== attack.tau) {
    return { admissible: false, reason: `Last coordinate must be +/-${attack.tau}.`, candidate: null, residual: null, maxResidual: null, smallSecret: false, smallError: false };
  }
  const sign = last < 0 ? 1 : -1;
  const candidate = vector.slice(attack.m, attack.m + attack.d).map((value) => sign * value);
  const { p } = publicData;
  const secretBound = p.secretDistribution === "ternary" ? 1 : p.eta1;
  const smallSecret = candidate.every((value) => Number.isInteger(value) && Math.abs(value) <= secretBound);
  if (!smallSecret) {
    return { admissible: true, reason: "Candidate secret is outside the selected small-secret support.", candidate, residual: null, maxResidual: null, smallSecret, smallError: false };
  }
  const candidatePolys = Array.from({ length: p.k }, (_, i) => candidate.slice(i * p.n, (i + 1) * p.n));
  const predicted = flatten(matrixVector(publicData.A, candidatePolys, p.q));
  const observed = flatten(publicData.t);
  const residual = observed.map((value, i) => centered(value - predicted[i], p.q));
  const maxResidual = Math.max(...residual.map(Math.abs));
  const smallError = residual.every((value) => Math.abs(value) <= p.keyErrorEta);
  return {
    admissible: true, candidate, residual, maxResidual, smallSecret, smallError,
    reason: smallError
      ? "Passes small-secret and all public-key coefficient checks; this is a compatible candidate, not a proof of uniqueness."
      : `Fails public check: max |t - M s'|_q = ${maxResidual}, support bound ${p.keyErrorEta}.`,
  };
}