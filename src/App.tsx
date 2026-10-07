import { useEffect, useMemo, useState } from "react";
import AlgebraView from "./components/AlgebraView";
import CostView from "./components/CostView";
import ProjectionFigure, { type Selection } from "./components/ProjectionFigure";
import {
  bkzBlockStep, bkzTour, constructAttack, finishLLL, gramSchmidt, initialBasis,
  initialSieve, lllStep, matrixCoefficient, norm, provenanceDeterminant, sampleLatticePoints, sieveStep,
  verifyBasisLineage, verifyPublicCandidate, type AttackLattice, type BasisState, type CandidateResult,
  type SamplePoint, type SieveState, type Vector,
} from "./math/lattice";
import { makeProjection, type Projection, type ProjectionSettings } from "./math/projection";
import { centered, generateInstance, PRESETS, standardName, type Instance, type Parameters, type PresetId, type PublicData } from "./math/mlkem";

type Tab = "algebra" | "projection" | "reduction" | "sieve" | "attack" | "scaling";

const TABS: { id: Tab; number: string; label: string }[] = [
  { id: "algebra", number: "01", label: "Construction" },
  { id: "projection", number: "02", label: "Projection" },
  { id: "reduction", number: "03", label: "Reduction" },
  { id: "sieve", number: "04", label: "Sieving" },
  { id: "attack", number: "05", label: "Attacker" },
  { id: "scaling", number: "06", label: "Scaling" },
];

const START_PROJECTION: ProjectionSettings = {
  mode: "coordinates", x: 0, y: 1, angle: -12, zoom: 1, panX: 0, panY: 0,
  showWitness: true, showNoise: true, showComponents: false,
};

function displayNumber(value: number): string {
  if (!Number.isFinite(value)) return "n/a";
  if (Math.abs(value) >= 100000) return value.toExponential(2);
  if (Math.abs(value) >= 100) return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return value.toFixed(value < 10 && value !== Math.round(value) ? 2 : 1);
}

function SelectNumber({ label, value, choices, onChange, hint }: { label: string; value: number; choices: number[]; onChange: (value: number) => void; hint: string }) {
  return <label className="advanced-field"><span>{label}</span><select value={value} onChange={(event) => onChange(Number(event.target.value))}>{choices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}</select><small>{hint}</small></label>;
}

function ParameterRail({ preset, setPreset, params, setParams, seed, setSeed, samples, setSamples, construction, setConstruction, tau, setTau, attack }: {
  preset: PresetId;
  setPreset: (value: PresetId) => void;
  params: Parameters;
  setParams: (value: Parameters) => void;
  seed: number;
  setSeed: (value: number) => void;
  samples: number;
  setSamples: (value: number) => void;
  construction: "embedding" | "primal";
  setConstruction: (value: "embedding" | "primal") => void;
  tau: number;
  setTau: (value: number) => void;
  attack: AttackLattice;
}) {
  const standard = standardName(params);
  const d = params.n * params.k;

  function change<K extends keyof Parameters>(key: K, value: Parameters[K]) {
    const next = { ...params, [key]: value };
    if (key === "eta1" && params.keyErrorEta === params.eta1) next.keyErrorEta = value as number;
    if (key === "q") {
      const limit = Math.ceil(Math.log2(value as number));
      next.du = Math.min(next.du, limit);
      next.dv = Math.min(next.dv, limit);
    }
    setParams(next);
  }

  function choosePreset(value: PresetId) {
    setPreset(value);
    setParams({ ...PRESETS[value] });
    setSamples(PRESETS[value].n * PRESETS[value].k);
  }

  return <aside className="parameter-rail">
    <div className="rail-top"><span className="micro-label">CONTROL DESK / 01</span><h2>The instance</h2></div>
    <label className="field-label">PARAMETER SET
      <select className="large-select" value={preset} onChange={(event) => choosePreset(event.target.value as PresetId)}>
        <option value="512">ML-KEM-512</option><option value="768">ML-KEM-768</option><option value="1024">ML-KEM-1024</option><option value="lab">Small laboratory</option>
      </select>
    </label>
    <div className={`status-line ${standard ? "" : "modified"}`}><span className="status-dot" />{standard ? `${standard} parameter values` : "Modified construction / not standardized"}</div>
    <p className="rail-note">Arithmetic follows the coefficient-ring form of Module-LWE. This seeded demonstrator is not FIPS byte-compatible or cryptographically secure software.</p>

    <div className="rail-divider"><span>FIPS INPUTS / RING + MODULE</span></div>
    <dl className="parameter-list">
      <div><dt>polynomial degree <i>n</i></dt><dd>{params.n}</dd></div>
      <div><dt>module rank <i>k</i></dt><dd>{params.k}</dd></div>
      <div><dt>modulus <i>q</i></dt><dd>{params.q}</dd></div>
      <div><dt>secret / key error</dt><dd>{params.secretDistribution === "cbd" ? `CBD(${params.eta1})` : "ternary"} / CBD({params.keyErrorEta})</dd></div>
      <div><dt>enc. errors <i>e</i><sub>1</sub>, <i>e</i><sub>2</sub></dt><dd>CBD({params.eta2})</dd></div>
      <div><dt>cipher bits <i>d</i><sub>u</sub> / <i>d</i><sub>v</sub></dt><dd>{params.du} / {params.dv}</dd></div>
    </dl>
    <details className="advanced-details">
      <summary>Modify the construction <span>+</span></summary>
      <p>Edits regenerate A, s, e, t and ciphertext where relevant. Any off-preset values are not ML-KEM.</p>
      <SelectNumber label="n / ring degree" value={params.n} choices={[4, 8, 16, 256]} onChange={(v) => { change("n", v); setSamples(Math.min(samples, v * params.k)); }} hint="Coefficients per polynomial; standard fixes 256." />
      <SelectNumber label="k / module rank" value={params.k} choices={[1, 2, 3, 4]} onChange={(v) => { change("k", v); setSamples(Math.min(samples, params.n * v)); }} hint="Polynomials per vector; not attack dimension." />
      <SelectNumber label="q / modulus" value={params.q} choices={[17, 97, 257, 3329]} onChange={(v) => change("q", v)} hint="Changes every modular product and compression." />
      <SelectNumber label="eta1 / s and r" value={params.eta1} choices={[1, 2, 3, 4]} onChange={(v) => change("eta1", v)} hint="CBD secret and ephemeral support in FIPS sets." />
      <SelectNumber label="etaE / key error" value={params.keyErrorEta} choices={[0, 1, 2, 3, 4]} onChange={(v) => change("keyErrorEta", v)} hint="Independent override; FIPS requires etaE = eta1." />
      <SelectNumber label="eta2 / e1 and e2" value={params.eta2} choices={[0, 1, 2, 3, 4]} onChange={(v) => change("eta2", v)} hint="Ciphertext noise; affects correctness too." />
      <label className="advanced-field"><span>secret distribution</span><select value={params.secretDistribution} onChange={(event) => change("secretDistribution", event.target.value as Parameters["secretDistribution"])}><option value="cbd">CBD(eta1)</option><option value="ternary">uniform ternary</option></select><small>Ternary is a modified construction.</small></label>
      <SelectNumber label="du / u bits" value={params.du} choices={Array.from({ length: Math.max(1, Math.ceil(Math.log2(params.q)) - 1) }, (_, i) => i + 2)} onChange={(v) => change("du", v)} hint="Quantization of the vector u." />
      <SelectNumber label="dv / v bits" value={params.dv} choices={Array.from({ length: Math.max(1, Math.ceil(Math.log2(params.q)) - 1) }, (_, i) => i + 2)} onChange={(v) => change("dv", v)} hint="Quantization of the message-bearing v." />
    </details>

    <div className="rail-divider"><span>GENERATED INSTANCE</span></div>
    <label className="field-label seed-field">DETERMINISTIC DEMO SEED
      <span><input type="number" min="0" value={seed} onChange={(event) => setSeed(Math.max(0, Math.trunc(Number(event.target.value) || 0)))} /><button title="Generate another instance" onClick={() => setSeed(seed + 1)}>next &rarr;</button></span>
    </label>
    <p className="rail-note">Independent seeded streams keep A and s fixed when only an error parameter changes. The generator is for reproducible mathematics, not key generation.</p>

    <div className="rail-divider"><span>ATTACK MODEL / DERIVED</span></div>
    <label className="field-label range-field">PUBLIC COEFFICIENT ROWS <strong>m = {attack.m} / {d}</strong>
      <input type="range" min="2" max={d} step="1" value={Math.min(samples, d)} onChange={(event) => setSamples(Number(event.target.value))} />
    </label>
    <p className="rail-note">Selects m evenly spaced coefficient equations from the one public t. They are related by the same module matrix, not fresh independent LWE samples.</p>
    <label className="field-label">LATTICE CONSTRUCTION
      <select value={construction} onChange={(event) => setConstruction(event.target.value as "embedding" | "primal")}><option value="embedding">Kannan embedding / SVP</option><option value="primal">Primal q-ary / CVP</option></select>
    </label>
    {construction === "embedding" && <label className="field-label">EMBEDDING SCALE <i>tau</i>
      <select value={tau} onChange={(event) => setTau(Number(event.target.value))}><option value="1">1</option><option value="2">2</option><option value="4">4</option><option value="8">8</option></select>
    </label>}
    <div className="rail-bottom-note">Attack ambient D = m + kn{construction === "embedding" ? " + 1" : ""} = <b>{attack.D}</b>. This is neither the ring degree n nor the module rank k.</div>
  </aside>;
}

function CoordinateList({ vector, attack, label = "INTEGER AMBIENT COORDINATES" }: { vector: Vector; attack: AttackLattice; label?: string }) {
  const [page, setPage] = useState(0);
  const pageSize = 32;
  const pages = Math.ceil(vector.length / pageSize);
  const start = page * pageSize;
  const nonzero = vector.filter((value) => Math.abs(value) > 1e-10).length;
  return <div className="coordinate-inspector">
    <div className="coordinate-head"><span>{label} / {nonzero} NONZERO</span><span><button disabled={page === 0} onClick={() => setPage(page - 1)}>&larr;</button> {page + 1}/{pages} <button disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>&rarr;</button></span></div>
    <div className="coordinate-grid">{vector.slice(start, start + pageSize).map((value, i) => <div key={start + i}><small>{start + i < attack.m ? "o" : start + i < attack.m + attack.d ? "s" : "t"}{start + i}</small><b>{Number.isInteger(value) ? value : Math.abs(value) < 1e-10 ? 0 : value.toFixed(4)}</b></div>)}</div>
    <p>o = observation coordinates; s = secret-coordinate block; t = embedding coordinate. Page through all {attack.D} entries.</p>
  </div>;
}

function ObjectInspector({ attack, basis, points, sieve, projection, selected, attacker }: {
  attack: AttackLattice; basis: BasisState; points: SamplePoint[]; sieve: SieveState;
  projection: Projection; selected: Selection; attacker: boolean;
}) {
  const gs = useMemo(() => gramSchmidt(basis.vectors), [basis.vectors]);
  let vector: Vector | null = null;
  let title = "Projection operator";
  let description = "A linear map from the actual ambient coordinate space to this two-dimensional screen plane.";
  let algebra = "P(v) = (row_x dot v, row_y dot v).";
  if (selected.kind === "basis") {
    const i = Math.min(selected.index, basis.vectors.length - 1);
    vector = basis.vectors[i];
    title = `Basis vector b${i + 1}`;
    const isOriginal = basis.provenance[i].every((value, j) => value === Number(i === j));
    description = isOriginal ? `Original public generator: ${attack.labels[i]}.` : "Integer combination of the fixed public generators.";
    const provenance = basis.provenance[i].map((value, j) => value ? `${value} g${j + 1}` : "").filter(Boolean).join(" + ");
    algebra = `b${i + 1} = ${provenance || "0"}. Gram-Schmidt length = ${displayNumber(gs.lengths[i])}.`;
  } else if (selected.kind === "point") {
    const point = points[selected.index];
    if (point) {
      vector = point.vector;
      title = `Sampled lattice point (${point.id})`;
      description = "A fixed integer combination of three public generators; selection does not enumerate the lattice.";
      algebra = `v = ${point.coeffs[0]} g1 + ${point.coeffs[1]} g2 + ${point.coeffs[attack.relationIndex]} g${attack.relationIndex + 1}.`;
    }
  } else if (selected.kind === "sieve") {
    const item = sieve.list[selected.index];
    if (item) {
      vector = item.vector;
      title = `Sieve list element ${selected.index + 1}`;
      description = "Exact integer combination of the current basis vectors.";
      algebra = item.coeffs.map((value, i) => value ? `${value} b${i + 1}` : "").filter(Boolean).join(" + ") || "0";
    }
  } else if (selected.kind === "target") {
    vector = attack.publicTarget;
    title = "Public target (t, 0)";
    description = "Centered integer lifts of selected public-key coefficients; the attacker can compute all of these entries.";
    algebra = "T = (lift(t_rows), 0). In primal form it is a CVP target; the augmented (t, 0, tau) is an embedding column.";
  } else if (!attacker && selected.kind === "exact") {
    vector = attack.exactObservation;
    title = "Exact relation, observation block";
    description = "Private explanatory lens: the public target minus the sampled key error in each chosen equation.";
    algebra = "C = (lift(t_rows) - e_rows, 0). Its observation part agrees with M s modulo q.";
  } else if (!attacker && selected.kind === "noise") {
    vector = attack.errorComponent;
    title = "Key error component";
    description = "The actual CBD error in the selected public equations, not arbitrary point displacement.";
    algebra = "T - C = (e_rows, 0). It is not by itself asserted to be a lattice vector.";
  } else if (!attacker && selected.kind === "witness") {
    vector = attack.witness;
    title = attack.construction === "embedding" ? "Private short witness" : "Private relation residual";
    description = "Shown only in the explanatory view. The attacker and the reduction routines do not receive this vector.";
    algebra = attack.construction === "embedding" ? "W = (-e_rows, s, -tau) is an exact full-lattice vector." : "(-e_rows, s) is the difference between the target relation and its lifted counterpart.";
  }

  if (!vector) return <div className="object-inspector">
    <span className="micro-label">INSPECTOR / LINEAR MAP</span><h3>{title}</h3><p>{description}</p>
    <div className="inspection-facts"><div><span>SOURCE SPACE</span><strong>R<sup>{attack.D}</sup></strong></div><div><span>DISPLAY</span><strong>R<sup>2</sup></strong></div></div>
    <p className="inspector-equation">{algebra}</p>
    <div className="operator-rows"><div><b>row x</b><span>{projection.rowX.map((value, i) => Math.abs(value) > 1e-8 ? `${i}:${value.toFixed(3)}` : "").filter(Boolean).slice(0, 9).join("  ") || "all zero"}</span></div><div><b>row y</b><span>{projection.rowY.map((value, i) => Math.abs(value) > 1e-8 ? `${i}:${value.toFixed(3)}` : "").filter(Boolean).slice(0, 9).join("  ") || "all zero"}</span></div></div>
    <p className="inspector-note">Rows are the actual projection transformation after rotation; the first nine nonzero entries are shown above. X: {projection.descriptionX}. Y: {projection.descriptionY}.</p>
    <details className="matrix-detail"><summary>All entries of row x</summary><CoordinateList vector={projection.rowX} attack={attack} label="PROJECTION ROW X" /></details>
    <details className="matrix-detail"><summary>All entries of row y</summary><CoordinateList vector={projection.rowY} attack={attack} label="PROJECTION ROW Y" /></details>
  </div>;

  const [px, py] = projection.apply(vector);
  return <div className="object-inspector">
    <span className="micro-label">INSPECTOR / {selected.kind.toUpperCase()}</span><h3>{title}</h3><p>{description}</p>
    <div className="inspection-facts"><div><span>FULL NORM</span><strong>{displayNumber(norm(vector))}</strong></div><div><span>PROJECTED</span><strong>({displayNumber(px)}, {displayNumber(py)})</strong></div></div>
    <p className="inspector-equation">{algebra}</p>
    {selected.kind === "basis" && selected.index > 0 && <p className="inspector-note">Angle to preceding basis vector: {(Math.acos(Math.max(-1, Math.min(1, basis.vectors[selected.index].reduce((sum, v, i) => sum + v * basis.vectors[selected.index - 1][i], 0) / (norm(basis.vectors[selected.index]) * norm(basis.vectors[selected.index - 1]))))) * 180 / Math.PI).toFixed(1)} degrees. A basis vector need not be a shortest lattice vector.</p>}
    <CoordinateList key={`${selected.kind}:${"index" in selected ? selected.index : "fixed"}`} vector={vector} attack={attack} />
  </div>;
}

function BasisTable({ attack, basis, beta, selected, onSelect }: { attack: AttackLattice; basis: BasisState; beta: number; selected: Selection; onSelect: (value: Selection) => void }) {
  const gs = gramSchmidt(basis.vectors);
  const shortest = Math.min(...basis.vectors.map(norm));
  return <div className="below-figure">
    <div className="section-rule"><span>BASIS LEDGER / FULL AMBIENT NORMS</span></div>
    <div className="table-heading"><p>Each row is an integer combination of the original public basis. LLL and the bounded block procedure apply unimodular operations, so the represented lattice does not move.</p><span>rank {basis.vectors.length} / ambient {attack.D}</span></div>
    <div className="math-table-wrap"><table className="math-table basis-table"><thead><tr><th>vector</th><th>||b<sub>i</sub>||</th><th>||b*<sub>i</sub>||</th><th>&mu;<sub>i,i-1</sub></th><th>status</th></tr></thead><tbody>
      {basis.vectors.map((vector, i) => <tr key={i} className={selected.kind === "basis" && selected.index === i ? "active-row" : ""} onClick={() => onSelect({ kind: "basis", index: i })}>
        <td>b<sub>{i + 1}</sub> <small>{basis.provenance[i].every((value, j) => value === Number(i === j)) ? attack.labels[i] : "integer combination"}</small></td>
        <td>{displayNumber(norm(vector))}</td><td>{displayNumber(gs.lengths[i])}</td><td>{i ? gs.mu[i][i - 1].toFixed(3) : "-"}</td>
        <td>{i >= basis.blockIndex && i < basis.blockIndex + beta ? "next block" : Math.abs(norm(vector) - shortest) < 1e-7 ? "shortest in basis" : ""}</td>
      </tr>)}
    </tbody></table></div>
    <p className="secondary-caption">Gram-Schmidt is computed in R<sup>{attack.D}</sup>, not from the 2D picture. The shortest listed basis vector is not necessarily a shortest vector of the lattice. Rank-{basis.vectors.length} covolume: 2<sup>{gs.log2Covolume.toFixed(2)}</sup>; full attack-lattice determinant: q<sup>{attack.m}</sup>{attack.construction === "embedding" ? " tau" : ""} = 2<sup>{attack.determinantLog2.toFixed(2)}</sup>.</p>
  </div>;
}

function Derivation({ attack, instance, onAttack }: { attack: AttackLattice; instance: Instance; onAttack: () => void }) {
  const firstRow = attack.rows[0];
  const lifted = centered(instance.t[Math.floor(firstRow / instance.p.n)][firstRow % instance.p.n], instance.p.q);
  return <div className="below-figure derivation">
    <div className="section-rule"><span>DERIVATION / ONE CONTINUOUS OBJECT</span></div>
    <div className="derivation-steps">
      <div><small>01 / MODULE EQUATION</small><strong><i>t</i> = <i>A s</i> + <i>e</i> &nbsp;(mod <i>q</i>)</strong><p>Polynomial multiplication uses X<sup>{instance.p.n}</sup> = -1. Selected public row {firstRow} has centered lift {lifted}.</p></div>
      <div><small>02 / COEFFICIENT MATRIX</small><strong><i>t</i><sub>rows</sub> = <i>M s</i> + <i>e</i><sub>rows</sub> &nbsp;(mod <i>q</i>)</strong><p><i>M</i> is the signed negacyclic convolution matrix of that same A; m = {attack.m}, unknown coefficients d = kn = {attack.d}.</p></div>
      <div><small>03 / PUBLIC q-ARY LATTICE</small><strong>&Lambda; = &#123;(<i>M x</i> + <i>qz</i>, <i>x</i>)&#125;</strong><p>Ambient dimension m + d = {attack.m + attack.d}; target T = (lift(t), 0) is public. One planted lattice point is L = (lift(t)-e, s), with T-L = (e,-s); L need not be the unique closest point.</p></div>
      <div><small>04 / {attack.construction === "embedding" ? "KANNAN EMBEDDING" : "CLOSEST VECTOR FORM"}</small><strong>{attack.construction === "embedding" ? "B = [ qI  M  t ; 0  I  0 ; 0  0  tau ]" : "B = [ qI  M ; 0  I ]"}</strong><p>{attack.construction === "embedding" ? `Ambient D = ${attack.D}; a private vector (-e, s, -${attack.tau}) belongs to the full lattice. Its appearance in 2D is only a projection.` : `Ambient D = ${attack.D}; the attacker seeks a lattice point near the public target, not an arbitrary dot.`}</p></div>
    </div>
    <div className="derivation-footer"><span>CHECK / key equation {instance.keyIdentity ? "holds" : "FAILED"} &nbsp; / &nbsp; lattice witness congruences {attack.witnessVerified ? "hold" : "FAILED"}</span><button className="plain-link" onClick={onAttack}>View public attack &rarr;</button></div>
  </div>;
}

function AttackTrace({ attack, publicData }: { attack: AttackLattice; publicData: PublicData }) {
  return <div className="below-figure attack-trace">
    <div className="section-rule"><span>PUBLIC TRANSCRIPT / WHAT THE ADVERSARY HAS</span></div>
    <div className="public-steps">
      <div><span>01</span><strong>Public module relation</strong><p>FIPS ek contains a seed for A and NTT-encoded t. Here A and t are stored explicitly in the equivalent coefficient-ring representation.</p></div>
      <div><span>02</span><strong>Expand negacyclic rows</strong><p>M[0,0] = {matrixCoefficient(publicData, attack.rows[0], 0)}; M[0,1] = {matrixCoefficient(publicData, attack.rows[0], 1)}. Only {attack.m} of {attack.d} available public coefficient rows are selected.</p></div>
      <div><span>03</span><strong>Build attack basis</strong><p>{attack.construction === "embedding" ? `B = [qI_m, M, lift(t); 0, I_d, 0; 0, 0, ${attack.tau}].` : "B = [qI_m, M; 0, I_d], with target (lift(t),0)."} D = {attack.D}; shown basis rank {attack.source.length}.</p></div>
      <div><span>04</span><strong>Reduce, search, verify</strong><p>Try short vectors with LLL, bounded block enumeration and differences. A proposed small secret is checked against all {attack.d} public-key coefficients, not against the hidden secret.</p></div>
    </div>
    <p className="secondary-caption">{attack.fullBasis ? "This small modified instance permits a full attack-lattice basis demonstration." : `For this instance the live algorithms operate only on a rank-${attack.source.length} public sublattice of the D=${attack.D} attack lattice. They cannot be interpreted as full ML-KEM attacks.`} Ciphertext values are public too, but this view constructs a key-recovery primal attack from the public key relation.</p>
  </div>;
}

function SieveLedger({ sieve, attack, onSelect }: { sieve: SieveState; attack: AttackLattice; onSelect: (value: Selection) => void }) {
  return <div className="below-figure">
    <div className="section-rule"><span>LIST / EXACT DIFFERENCES</span></div>
    <p className="ledger-intro">A list element is an integer combination of current public generators. If ||v<sub>i</sub> - v<sub>j</sub>|| &lt; ||v<sub>i</sub>|| in the full {attack.D}-coordinate norm, replace v<sub>i</sub>. The plotted list contains 16 elements only.</p>
    <div className="sieve-list">{sieve.list.slice(0, 16).map((item, index) => <button key={index} onClick={() => onSelect({ kind: "sieve", index })}><span>v{String(index + 1).padStart(2, "0")}</span><strong>{displayNumber(norm(item.vector))}</strong><small>||v|| in R<sup>{attack.D}</sup></small></button>)}</div>
  </div>;
}

function ReductionControls({ attack, basis, setBasis, running, setRunning, beta, setBeta, selected, onSelect }: {
  attack: AttackLattice; basis: BasisState; setBasis: (update: (previous: BasisState) => BasisState) => void;
  running: boolean; setRunning: (value: boolean) => void; beta: number; setBeta: (value: number) => void;
  selected: Selection; onSelect: (selection: Selection) => void;
}) {
  const gs = gramSchmidt(basis.vectors);
  const sourceCovolume = gramSchmidt(attack.source).log2Covolume;
  const lineageValid = verifyBasisLineage(attack, basis);
  const determinant = provenanceDeterminant(basis);
  return <div className="algorithm-controls">
    <span className="micro-label">REDUCTION / LIVE INTEGER OPERATIONS</span><h3>The basis changes.<br />The lattice does not.</h3>
    <p className="inspector-note">{attack.fullBasis ? `Full D=${attack.D} basis in this small modified example.` : `Only a rank-${basis.vectors.length} sublattice in ambient D=${attack.D}; no full-scale reduction is performed.`}</p>
    <div className="operation-readout"><span>LAST OPERATION</span><p>{basis.operation}</p></div>
    <div className="action-grid"><button className="primary-action" disabled={basis.done} onClick={() => setBasis(lllStep)}>Step LLL</button><button disabled={basis.done && !running} onClick={() => setRunning(!running)}>{running ? "Pause" : "Run continuously"}</button><button onClick={() => { setRunning(false); setBasis((previous) => finishLLL(previous)); }}>Finish LLL</button><button onClick={() => { setRunning(false); setBasis(() => initialBasis(attack, true)); }}>Skew basis</button><button onClick={() => { setRunning(false); setBasis(() => initialBasis(attack, false)); onSelect({ kind: "basis", index: 0 }); }}>Public basis</button></div>
    <p className="small-proof">LLL uses &delta; = 0.99, integer size reductions and Lovasz swaps. Floating-point Gram-Schmidt guides the decisions; integer basis updates and lineage are exact. This is not certified reduction software or an SVP solver.</p>
    <dl className="readout-list"><div><dt>LLL operations / swaps</dt><dd>{basis.steps} / {basis.swaps}</dd></div><div><dt>size reductions</dt><dd>{basis.reductions}</dd></div><div><dt>shortest listed ||b||</dt><dd>{displayNumber(Math.min(...basis.vectors.map(norm)))}</dd></div><div><dt>rank-{basis.vectors.length} log2 covolume</dt><dd>{gs.log2Covolume.toFixed(2)}</dd></div><div><dt>integer lineage / det U</dt><dd>{lineageValid ? "exact" : "FAILED"} / {determinant.toString()}</dd></div><div><dt>numeric volume check</dt><dd>{Math.abs(gs.log2Covolume - sourceCovolume) < 0.01 ? "preserved" : "numeric drift"}</dd></div></dl>

    <div className="rail-divider"><span>BLOCK REDUCTION / BOUNDED BKZ-STYLE</span></div>
    <label className="field-label range-field">LOCAL BLOCK DIMENSION <strong>&beta; = {beta}</strong><input type="range" min="2" max={Math.min(5, basis.vectors.length)} value={beta} onChange={(event) => setBeta(Number(event.target.value))} /></label>
    <p className="inspector-note">Next block: b{basis.blockIndex + 1}...b{Math.min(basis.vectors.length, basis.blockIndex + beta)} in a rank-{basis.vectors.length} basis; ambient attack dimension D = {attack.D}. The algorithm enumerates primitive combinations with coefficients in [-2, 2] in each projected block, inserts an improvement unimodularly, then LLL-cleans. It is a bounded BKZ-style demonstration, not production BKZ.</p>
    <div className="action-grid"><button className="primary-action" onClick={() => setBasis((previous) => bkzBlockStep(previous, beta))}>Next block</button><button onClick={() => setBasis((previous) => bkzTour(previous, beta))}>One tour</button></div>
    {basis.lastBKZ && <div className="block-result"><span>LAST BLOCK / b{basis.lastBKZ.start + 1}...b{basis.lastBKZ.start + basis.lastBKZ.size}</span><strong>{displayNumber(basis.lastBKZ.before)} &rarr; {displayNumber(basis.lastBKZ.best)}</strong><small>{basis.lastBKZ.tested} primitive candidates tested; {basis.lastBKZ.improved ? "shorter projected vector inserted" : "no improvement in bounded window"}.</small></div>}
    <p className="small-proof">{basis.bkzChecks.toLocaleString()} local combinations tested across {basis.tours} completed tour{basis.tours === 1 ? "" : "s"}. A larger block searches more combinations here but does not guarantee a shorter first vector in one tour.</p>
    {selected.kind === "basis" && <p className="inspector-note">Selected b{selected.index + 1}: click any arrow or basis-ledger row to inspect all coordinates.</p>}
  </div>;
}

function SieveControls({ attack, sieve, setSieve, basis, reset }: { attack: AttackLattice; sieve: SieveState; setSieve: (update: (previous: SieveState) => SieveState) => void; basis: BasisState; reset: () => void }) {
  const minNorm = Math.min(...sieve.list.map((item) => norm(item.vector)));
  return <div className="algorithm-controls"><span className="micro-label">SIEVING / LIST EXPERIMENT</span><h3>Subtract to shorten.</h3>
    <p className="inspector-note">Actual lattice vectors, actual differences, and comparisons in the full {attack.D}-coordinate norm. This 16-element list is deliberately tiny.</p>
    <div className="action-grid"><button className="primary-action" disabled={sieve.exhausted} onClick={() => setSieve(sieveStep)}>One difference</button><button disabled={sieve.exhausted} onClick={() => setSieve((previous) => { let next = previous; for (let i = 0; i < 8 && !next.exhausted; i++) next = sieveStep(next); return next; })}>Run 8 steps</button><button onClick={reset}>Resample list</button></div>
    <dl className="readout-list"><div><dt>sampled elements</dt><dd>{sieve.list.length}</dd></div><div><dt>pair comparisons</dt><dd>{sieve.pairChecks.toLocaleString()}</dd></div><div><dt>successful differences</dt><dd>{sieve.steps}</dd></div><div><dt>shortest list element</dt><dd>{displayNumber(minNorm)}</dd></div></dl>
    {sieve.lastPair && <div className="block-result"><span>LAST PAIR / v{sieve.lastPair[0] + 1} - v{sieve.lastPair[1] + 1}</span><strong>{displayNumber(sieve.lastBefore)} &rarr; {displayNumber(sieve.lastAfter)}</strong><small>Exact norm reduction in R<sup>{attack.D}</sup>, not merely a shorter line on screen.</small></div>}
    <div className="rail-divider"><span>SCALING / NOT THIS DISPLAY</span></div>
    <p className="inspector-note">Heuristic high-dimensional classical SVP sieves use about 2<sup>0.208d+o(d)</sup> list storage and 2<sup>0.292d+o(d)</sup> time. A real block sieve would use dimension d = &beta;; neither 16 arrows nor browser runtime represents its work.</p>
    <p className="small-proof">The list is generated from {Math.min(4, basis.vectors.length)} current basis vectors. Difference closure preserves membership in the same represented lattice.</p>
  </div>;
}

function AttackControls({ attack, basis, setBasis, publicData, sieve, selected, verification, setVerification, setTab }: {
  attack: AttackLattice; basis: BasisState; setBasis: (update: (previous: BasisState) => BasisState) => void;
  publicData: PublicData; sieve: SieveState; selected: Selection;
  verification: CandidateResult | null; setVerification: (value: CandidateResult | null) => void; setTab: (value: Tab) => void;
}) {
  const basisCandidates = basis.vectors.map((vector, index) => ({ vector, label: `b${index + 1}` }));
  const sieveCandidates = sieve.list.map((item, index) => ({ vector: item.vector, label: `list v${index + 1}` }));
  const eligible = [...basisCandidates, ...sieveCandidates].filter(({ vector }) => attack.construction === "embedding" && Math.abs(vector[attack.D - 1]) === attack.tau).sort((a, b) => norm(a.vector) - norm(b.vector));
  const selectedVector = selected.kind === "basis" ? basis.vectors[selected.index] : selected.kind === "sieve" ? sieve.list[selected.index]?.vector : null;
  const tested = selectedVector && Math.abs(selectedVector[attack.D - 1]) === attack.tau ? { vector: selectedVector, label: `selected ${selected.kind}` } : eligible[0];

  function test() {
    if (tested) setVerification(verifyPublicCandidate(publicData, attack, tested.vector));
  }

  return <div className="algorithm-controls attacker-panel"><span className="micro-label">ATTACKER / PUBLIC INFORMATION ONLY</span><h3>No secret is given.</h3>
    <p className="inspector-note">Input: A, t, parameter values and optional ciphertext. No generated s, e, private witness, or secret-aligned projection is passed to this recovery procedure.</p>
    <dl className="readout-list"><div><dt>selected public equations</dt><dd>{attack.m}</dd></div><div><dt>unknown coefficients kn</dt><dd>{attack.d}</dd></div><div><dt>attack ambient D</dt><dd>{attack.D}</dd></div><div><dt>live reduction rank</dt><dd>{basis.vectors.length}</dd></div></dl>
    {!attack.fullBasis && <p className="inline-warning">Live reduction is a rank-{basis.vectors.length} slice. Increasing its resources does not simulate reducing the full D={attack.D} lattice.</p>}
    <div className="rail-divider"><span>ALLOCATE LOCAL WORK</span></div>
    <div className="action-grid"><button className="primary-action" disabled={basis.done} onClick={() => setBasis(lllStep)}>Step LLL</button><button onClick={() => setBasis((previous) => finishLLL(previous))}>Finish LLL</button><button onClick={() => setBasis((previous) => bkzBlockStep(previous, Math.min(4, previous.vectors.length)))}>Block search</button><button onClick={() => setTab("reduction")}>Reduction details &rarr;</button></div>
    <p className="inspector-note">Current shortest represented basis norm: {displayNumber(Math.min(...basis.vectors.map(norm)))}. This is not a success probability.</p>
    <div className="rail-divider"><span>PUBLIC CANDIDATE TEST</span></div>
    {attack.construction === "embedding" ? <><p className="inspector-note">Take {tested?.label || "a short vector"} with last coordinate &plusmn;{attack.tau}, read its candidate secret block, and compute all residuals t - A s' mod q.</p><button className="wide-action primary-action" disabled={!tested} onClick={test}>Verify candidate against t &rarr;</button></> : <p className="inspector-note">The primal form is a closest-vector problem about T=(t,0). Switch to embedding to read secret candidates from short lattice vectors.</p>}
    {verification && <div className={`verification-result ${verification.smallError ? "passed" : ""}`}><span>RESULT / PUBLIC EQUATIONS</span><strong>{verification.smallError && verification.smallSecret ? "Compatible small solution" : "Not verified"}</strong><p>{verification.reason}</p>{verification.candidate && <small>candidate s'[0..7] = [{verification.candidate.slice(0, 8).join(", ")}]{verification.maxResidual !== null ? `; max centered public residual = ${verification.maxResidual}` : ""}</small>}</div>}
    <p className="small-proof">A passing small solution need not be unique. A failed candidate says nothing about full ML-KEM security. Attack success probability is not inferred from this demonstration.</p>
  </div>;
}

function AlgebraInspector({ instance, row, column, coefficient, attack, setTab }: { instance: Instance; row: number; column: number; coefficient: number; attack: AttackLattice; setTab: (value: Tab) => void }) {
  const { p, cipher } = instance;
  const term = matrixCoefficient(instance, row * p.n + coefficient, column * p.n);
  return <div className="algorithm-controls algebra-inspector"><span className="micro-label">INSPECTOR / COEFFICIENT</span><h3>[X<sup>{coefficient}</sup>] of row {row}</h3>
    <p className="inspector-note">The selected algebra is computed in Z<sub>{p.q}</sub>[X]/(X<sup>{p.n}</sup>+1), not inferred from a drawing.</p>
    <dl className="readout-list"><div><dt>selected A[{row},{column}][0]</dt><dd>{instance.A[row][column][0]}</dd></div><div><dt>selected s[{column}][0]</dt><dd>{instance.s[column][0]}</dd></div><div><dt>coefficient M[{row * p.n + coefficient},{column * p.n}]</dt><dd>{term}</dd></div><div><dt>exact (As)[{row},{coefficient}]</dt><dd>{instance.exact[row][coefficient]}</dd></div><div><dt>key error e[{row},{coefficient}]</dt><dd>{instance.e[row][coefficient]}</dd></div><div><dt>public t[{row},{coefficient}]</dt><dd>{instance.t[row][coefficient]}</dd></div></dl>
    <div className="rail-divider"><span>CIPHERTEXT / SAME COEFFICIENT</span></div>
    <dl className="readout-list"><div><dt>u[{row},{coefficient}] exact / packed / decoded</dt><dd>{cipher.u[row][coefficient]} / {cipher.cu[row][coefficient]} / {cipher.uDecoded[row][coefficient]}</dd></div><div><dt>v[{coefficient}] exact / packed / decoded</dt><dd>{cipher.v[coefficient]} / {cipher.cv[coefficient]} / {cipher.vDecoded[coefficient]}</dd></div><div><dt>e1[{row},{coefficient}] / e2[{coefficient}]</dt><dd>{cipher.e1[row][coefficient]} / {cipher.e2[coefficient]}</dd></div><div><dt>message bit / recovered</dt><dd>{cipher.bits[coefficient]} / {cipher.recoveredBits[coefficient]}</dd></div><div><dt>raw theoretical noise</dt><dd>{centered(cipher.expectedNoise[coefficient], p.q)}</dd></div><div><dt>compression contribution</dt><dd>{cipher.compressionDelta[coefficient]}</dd></div><div><dt>final w</dt><dd>{cipher.w[coefficient]}</dd></div></dl>
    <p className="small-proof">Across all coefficients: key identity {instance.keyIdentity ? "exactly holds" : "FAILED"}; ciphertext identity {instance.cipherIdentity ? "exactly holds" : "FAILED"}; {cipher.bitErrors} / {p.n} bit mismatches for this one ciphertext.</p>
    <div className="rail-divider"><span>NEXT / THE SAME INSTANCE</span></div><p className="inspector-note">Expand A into its signed convolution matrix M, then add q-multiple columns to obtain a public lattice of dimension {attack.D}.</p><button className="wide-action primary-action" onClick={() => setTab("projection")}>Open the projection &rarr;</button>
  </div>;
}

function ScalingInspector({ instance, attack, trials, setTrials, busy, setBusy }: {
  instance: Instance; attack: AttackLattice; trials: { count: number; failures: number; bitErrors: number } | null;
  setTrials: (value: { count: number; failures: number; bitErrors: number } | null) => void;
  busy: boolean; setBusy: (value: boolean) => void;
}) {
  function runTrials() {
    if (busy) return;
    setBusy(true);
    setTrials(null);
    let count = 0;
    let failures = 0;
    let bitErrors = 0;
    const step = () => {
      if (count >= 24) {
        setTrials({ count, failures, bitErrors });
        setBusy(false);
        return;
      }
      const trial = generateInstance(instance.p, instance.seed + 100001 + count);
      failures += Number(trial.cipher.bitErrors > 0);
      bitErrors += trial.cipher.bitErrors;
      count++;
      setTimeout(step, 0);
    };
    setTimeout(step, 0);
  }
  let interval: [number, number] | null = null;
  if (trials) {
    const z = 1.96;
    const observed = trials.failures / trials.count;
    const denom = 1 + z * z / trials.count;
    const center = (observed + z * z / (2 * trials.count)) / denom;
    const half = z * Math.sqrt((observed * (1 - observed) + z * z / (4 * trials.count)) / trials.count) / denom;
    interval = [Math.max(0, center - half), Math.min(1, center + half)];
  }
  return <div className="algorithm-controls"><span className="micro-label">INTERPRETATION / FOUR DIFFERENT QUESTIONS</span><h3>Not one security meter.</h3>
    <p className="inspector-note"><b>Hardness.</b> The Module-LWE distribution and attack construction determine the problem. This workbench does not output security bits.</p>
    <p className="inspector-note"><b>Reduction cost.</b> The plot gives leading heuristic costs for a beta-dimensional SVP oracle, not a concrete lattice-estimator result.</p>
    <p className="inspector-note"><b>Attack success.</b> Unknown here without a calibrated attack/success model; the attacker view only verifies actual public candidates.</p>
    <p className="inspector-note"><b>Correctness.</b> Compression and both error sources act together. Raising error can worsen decryption even if it makes some observations noisier.</p>
    <div className="rail-divider"><span>FINITE CORRECTNESS EXPERIMENT</span></div>
    <p className="inspector-note">Generate 24 independently seeded demo key/ciphertext instances with these same parameters. Count any decoded message mismatch; this is not a rare-event failure bound for ML-KEM.</p>
    <button className="wide-action primary-action" disabled={busy} onClick={runTrials}>{busy ? "Computing 24 instances..." : "Run 24 demo trials"}</button>
    {trials && <div className="verification-result"><span>OBSERVED CIPHERTEXT FAILURES</span><strong>{trials.failures} / {trials.count}</strong><p>{trials.bitErrors} wrong bits total. Illustrative Wilson 95% interval for a Bernoulli failure rate: [{(interval![0] * 100).toFixed(1)}%, {(interval![1] * 100).toFixed(1)}%]. Pseudorandom finite trials are not a certified FIPS probability estimate.</p></div>}
    <div className="rail-divider"><span>DIMENSION ACCOUNTING</span></div>
    <dl className="readout-list"><div><dt>ring degree n</dt><dd>{instance.p.n}</dd></div><div><dt>module rank k</dt><dd>{instance.p.k}</dd></div><div><dt>secret coefficients kn</dt><dd>{attack.d}</dd></div><div><dt>attack rows m</dt><dd>{attack.m}</dd></div><div><dt>attack ambient D</dt><dd>{attack.D}</dd></div><div><dt>screen projection</dt><dd>2</dd></div></dl>
  </div>;
}

function Session({ instance, attack, tab, setTab }: { instance: Instance; attack: AttackLattice; tab: Tab; setTab: (value: Tab) => void }) {
  const [basis, setBasisState] = useState<BasisState>(() => initialBasis(attack, true));
  const [sieve, setSieveState] = useState<SieveState>(() => initialSieve(basis.vectors, instance.seed));
  const [settings, setSettings] = useState<ProjectionSettings>(START_PROJECTION);
  const [selected, setSelected] = useState<Selection>({ kind: "basis", index: 0 });
  const [row, setRow] = useState(0);
  const [column, setColumn] = useState(0);
  const [coefficient, setCoefficient] = useState(0);
  const [running, setRunning] = useState(false);
  const [beta, setBeta] = useState(Math.min(4, attack.source.length));
  const [modelBeta, setModelBeta] = useState(Math.min(96, attack.D));
  const [tours, setTours] = useState(1);
  const [verification, setVerification] = useState<CandidateResult | null>(null);
  const [trials, setTrials] = useState<{ count: number; failures: number; bitErrors: number } | null>(null);
  const [trialBusy, setTrialBusy] = useState(false);

  function setBasis(update: (previous: BasisState) => BasisState) {
    setBasisState(update);
    setVerification(null);
  }
  function setSieve(update: (previous: SieveState) => SieveState) {
    setSieveState(update);
    setVerification(null);
  }
  useEffect(() => { setSieveState(initialSieve(basis.vectors, instance.seed)); }, [basis.vectors, instance.seed]);
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      setBasisState((previous) => lllStep(previous));
      setVerification(null);
    }, 160);
    return () => window.clearInterval(timer);
  }, [running]);
  useEffect(() => { if (basis.done) setRunning(false); }, [basis.done]);
  useEffect(() => { setRunning(false); if (tab === "attack" && ["exact", "noise", "witness"].includes(selected.kind)) setSelected({ kind: "target" }); }, [tab]);

  const points = useMemo(() => sampleLatticePoints(attack), [attack]);
  const publicData = useMemo<PublicData>(() => ({ p: instance.p, A: instance.A, t: instance.t }), [instance]);
  const redactedAttack = useMemo<AttackLattice>(() => ({ ...attack, witness: [], exactObservation: [], errorComponent: [], plantedPoint: [], witnessVerified: false }), [attack]);
  const projection = useMemo(() => makeProjection(tab === "attack" ? redactedAttack : attack, settings, tab === "attack"), [attack, redactedAttack, settings.mode, settings.x, settings.y, settings.angle, tab]);
  const graphView = tab === "projection" || tab === "reduction" || tab === "sieve" || tab === "attack";

  return <>
    <section className="primary-column">
      {tab === "algebra" && <AlgebraView instance={instance} row={row} column={column} coefficient={coefficient} setRow={setRow} setColumn={setColumn} setCoefficient={setCoefficient} />}
      {graphView && <ProjectionFigure attack={tab === "attack" ? redactedAttack : attack} basis={basis} points={points} sieve={sieve} projection={projection} settings={settings} setSettings={setSettings} selected={selected} onSelect={setSelected} view={tab} blockSize={beta} />}
      {tab === "projection" && <Derivation attack={attack} instance={instance} onAttack={() => setTab("attack")} />}
      {tab === "reduction" && <BasisTable attack={attack} basis={basis} beta={beta} selected={selected} onSelect={setSelected} />}
      {tab === "sieve" && <SieveLedger sieve={sieve} attack={attack} onSelect={setSelected} />}
      {tab === "attack" && <AttackTrace attack={redactedAttack} publicData={publicData} />}
      {tab === "scaling" && <CostView attack={attack} instance={instance} beta={modelBeta} setBeta={setModelBeta} tours={tours} setTours={setTours} />}
    </section>
    <aside className="inspection-rail">
      <div className="rail-top"><span className="micro-label">READOUT / 02</span><h2>{tab === "algebra" ? "The calculation" : tab === "reduction" ? "The algorithm" : tab === "sieve" ? "The search" : tab === "attack" ? "The adversary" : tab === "scaling" ? "The limits" : "Inspect an object"}</h2></div>
      {tab === "algebra" && <AlgebraInspector instance={instance} row={row} column={column} coefficient={coefficient} attack={attack} setTab={setTab} />}
      {tab === "reduction" && <><ReductionControls attack={attack} basis={basis} setBasis={setBasis} running={running} setRunning={setRunning} beta={beta} setBeta={setBeta} selected={selected} onSelect={setSelected} /><div className="inspector-separator" /><ObjectInspector attack={attack} basis={basis} points={points} sieve={sieve} projection={projection} selected={selected} attacker={false} /></>}
      {tab === "sieve" && <><SieveControls attack={attack} sieve={sieve} setSieve={setSieve} basis={basis} reset={() => setSieveState(initialSieve(basis.vectors, instance.seed + sieve.steps + 1))} /><div className="inspector-separator" /><ObjectInspector attack={attack} basis={basis} points={points} sieve={sieve} projection={projection} selected={selected} attacker={false} /></>}
      {tab === "attack" && <><AttackControls attack={redactedAttack} basis={basis} setBasis={setBasis} publicData={publicData} sieve={sieve} selected={selected} verification={verification} setVerification={setVerification} setTab={setTab} /><div className="inspector-separator" /><ObjectInspector attack={redactedAttack} basis={basis} points={points} sieve={sieve} projection={projection} selected={selected} attacker /></>}
      {tab === "scaling" && <ScalingInspector instance={instance} attack={attack} trials={trials} setTrials={setTrials} busy={trialBusy} setBusy={setTrialBusy} />}
      {tab === "projection" && <ObjectInspector attack={attack} basis={basis} points={points} sieve={sieve} projection={projection} selected={selected} attacker={false} />}
    </aside>
  </>;
}

export default function App() {
  const [preset, setPreset] = useState<PresetId>("768");
  const [params, setParams] = useState<Parameters>({ ...PRESETS["768"] });
  const [seed, setSeed] = useState(31415);
  const [samples, setSamples] = useState(PRESETS["768"].n * PRESETS["768"].k);
  const [construction, setConstruction] = useState<"embedding" | "primal">("embedding");
  const [tau, setTau] = useState(2);
  const [tab, setTab] = useState<Tab>("algebra");
  const instance = useMemo(() => generateInstance(params, seed), [params, seed]);
  const attack = useMemo(() => constructAttack(instance, samples, construction, tau), [instance, samples, construction, tau]);
  const sessionKey = `${seed}:${JSON.stringify(params)}:${attack.m}:${construction}:${tau}`;
  const standard = standardName(params);

  return <div className="app-shell">
    <header className="site-header">
      <div className="brand"><span className="brand-monogram">M<span>/</span>L</span><div><strong>ML-KEM</strong><small>GEOMETRIC INSTRUMENT</small></div></div>
      <div className="header-center">MODULE-LWE &nbsp; / &nbsp; LATTICE REDUCTION &nbsp; / &nbsp; POST-QUANTUM CRYPTOGRAPHY</div>
      <a href="https://csrc.nist.gov/pubs/fips/203/final" target="_blank" rel="noreferrer" className="header-source">FIPS 203 <span>&nearr;</span></a>
    </header>
    <div className="page-intro">
      <div><span className="micro-label">AN INTERACTIVE MATHEMATICAL NOTEBOOK &nbsp; / &nbsp; NO. 001</span><h1>The geometry of <em>ML-KEM.</em></h1><p>Follow one Module-LWE instance from polynomial arithmetic to its public attack lattice. The picture is a projection; the equations are the object.</p></div>
      <div className="intro-equation"><span>THE DEFINING RELATION</span><strong><i>t</i> = <i>A s</i> + <i>e</i> &nbsp;(mod <i>q</i>)</strong><small><i>R</i><sub>q</sub> = Z<sub>q</sub>[X] / (X<sup>n</sup> + 1)</small></div>
    </div>
    <nav className="chapter-nav" aria-label="Mathematical views">{TABS.map((item) => <button key={item.id} className={tab === item.id ? "active" : ""} onClick={() => setTab(item.id)}><span>{item.number}</span> {item.label}</button>)}</nav>
    <div className="dimension-ribbon"><span>{standard ? "FIPS 203 PARAMETER VALUES" : "MODIFIED CONSTRUCTION"}</span><span>RING DEGREE <b>n = {params.n}</b></span><span>MODULE RANK <b>k = {params.k}</b></span><span>SECRET COEFFICIENTS <b>kn = {attack.d}</b></span><span>ATTACK AMBIENT <b>D = {attack.D}</b></span><span>DISPLAY <b>{tab === "algebra" ? "ONE ALGEBRAIC COEFFICIENT" : tab === "scaling" ? "HEURISTIC COST PLOT" : "2D PROJECTION"}</b></span></div>
    <main className="workspace">
      <ParameterRail preset={preset} setPreset={setPreset} params={params} setParams={setParams} seed={seed} setSeed={setSeed} samples={samples} setSamples={setSamples} construction={construction} setConstruction={setConstruction} tau={tau} setTau={setTau} attack={attack} />
      <Session key={sessionKey} instance={instance} attack={attack} tab={tab} setTab={setTab} />
    </main>
    <footer className="site-footer"><div><strong>ML-KEM / GEOMETRIC INSTRUMENT</strong><span>Exact arithmetic in the chosen ring. Public attack construction. Explicitly labeled projections and heuristics.</span></div><div className="footer-links"><a href="https://csrc.nist.gov/pubs/fips/203/final" target="_blank" rel="noreferrer">FIPS 203 &nearr;</a><a href="https://simons.berkeley.edu/sites/default/files/docs/14988/20200120-lwe-latticebootcamp.pdf" target="_blank" rel="noreferrer">Lattice reduction notes &nearr;</a><a href="https://arxiv.org/abs/2105.05608" target="_blank" rel="noreferrer">Quantum sieve reference &nearr;</a></div></footer>
  </div>;
}
