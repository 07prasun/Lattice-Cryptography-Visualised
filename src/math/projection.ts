import { dot, gramSchmidt, norm, type AttackLattice, type Vector } from "./lattice";

export type ProjectionMode = "coordinates" | "gram-schmidt" | "witness";

export interface ProjectionSettings {
  mode: ProjectionMode;
  x: number;
  y: number;
  angle: number;
  zoom: number;
  panX: number;
  panY: number;
  showWitness: boolean;
  showNoise: boolean;
  showComponents: boolean;
}

export interface Projection {
  rowX: Vector;
  rowY: Vector;
  descriptionX: string;
  descriptionY: string;
  degenerate: boolean;
  apply: (vector: Vector) => [number, number];
}

export function coordinateName(attack: AttackLattice, index: number): string {
  if (index < attack.m) return `observation ${index} (t[${attack.rows[index]}])`;
  if (index < attack.m + attack.d) return `secret coefficient ${index - attack.m}`;
  return "embedding coordinate";
}

function unit(vector: Vector): Vector {
  const length = norm(vector);
  return length < 1e-12 ? [...vector] : vector.map((value) => value / length);
}

function coordinateAxis(dimension: number, index: number): Vector {
  const axis = Array<number>(dimension).fill(0);
  axis[Math.min(dimension - 1, Math.max(0, index))] = 1;
  return axis;
}

export function makeProjection(attack: AttackLattice, settings: ProjectionSettings, attackerView: boolean): Projection {
  const mode = attackerView && settings.mode === "witness" ? "coordinates" : settings.mode;
  let x: Vector;
  let y: Vector;
  let descriptionX: string;
  let descriptionY: string;
  if (mode === "gram-schmidt") {
    const gs = gramSchmidt(attack.source);
    const xIndex = Math.min(attack.source.length - 1, Math.max(0, settings.x));
    const yIndex = Math.min(attack.source.length - 1, Math.max(0, settings.y));
    x = unit(gs.orthogonal[xIndex]);
    y = unit(gs.orthogonal[yIndex]);
    descriptionX = `fixed public basis b*${xIndex + 1}`;
    descriptionY = `fixed public basis b*${yIndex + 1}`;
  } else if (mode === "witness") {
    x = unit(attack.witness);
    const seed = attack.source[0];
    const component = dot(seed, x);
    y = unit(seed.map((value, i) => value - component * x[i]));
    descriptionX = "private embedded witness direction";
    descriptionY = "public q-generator orthogonalized to witness";
  } else {
    const xIndex = Math.min(attack.D - 1, Math.max(0, settings.x));
    const yIndex = Math.min(attack.D - 1, Math.max(0, settings.y));
    x = coordinateAxis(attack.D, xIndex);
    y = coordinateAxis(attack.D, yIndex);
    descriptionX = coordinateName(attack, xIndex);
    descriptionY = coordinateName(attack, yIndex);
  }
  const radians = (settings.angle * Math.PI) / 180;
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  const rowX = x.map((value, i) => c * value - s * y[i]);
  const rowY = x.map((value, i) => s * value + c * y[i]);
  return {
    rowX, rowY, descriptionX, descriptionY,
    degenerate: Math.abs(dot(x, y)) > 0.999999 || norm(x) < 1e-8 || norm(y) < 1e-8,
    apply: (vector: Vector) => [dot(vector, rowX), dot(vector, rowY)],
  };
}