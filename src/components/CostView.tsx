import { type AttackLattice } from "../math/lattice";
import { standardName, type Instance } from "../math/mlkem";

interface Props {
  attack: AttackLattice;
  instance: Instance;
  beta: number;
  setBeta: (value: number) => void;
  tours: number;
  setTours: (value: number) => void;
}

const classical = (beta: number) => 0.292 * beta;
const quantum = (beta: number) => 0.2653 * beta;
const memory = (beta: number) => 0.208 * beta;

export default function CostView({ attack, instance, beta, setBeta, tours, setTours }: Props) {
  const upper = Math.min(attack.D, 768);
  const safeBeta = Math.min(beta, upper);
  const callFactor = Math.log2(attack.D * tours);
  const maxY = Math.ceil((classical(upper) + callFactor + 8) / 25) * 25;
  const left = 62;
  const right = 734;
  const top = 34;
  const bottom = 315;
  const sx = (value: number) => left + ((value - 2) / Math.max(1, upper - 2)) * (right - left);
  const sy = (value: number) => bottom - (value / maxY) * (bottom - top);
  const path = (fn: (value: number) => number) => Array.from({ length: 90 }, (_, i) => {
    const x = 2 + (i / 89) * (upper - 2);
    return `${i === 0 ? "M" : "L"}${sx(x).toFixed(2)} ${sy(fn(x)).toFixed(2)}`;
  }).join(" ");
  const standard = standardName(instance.p);
  const p = instance.p;
  const publicBytes = 384 * p.k + 32;
  const cipherBytes = (p.n * (p.du * p.k + p.dv)) / 8;
  const axisTicks = Array.from({ length: 5 }, (_, i) => Math.round((maxY * i) / 4));
  const xTicks = Array.from(new Set([2, Math.round(upper / 4), Math.round(upper / 2), Math.round((3 * upper) / 4), upper]));

  return (
    <div className="cost-view">
      <div className="figure-head">
        <div><span className="micro-label">FIG. 06 / COST MODELS, NOT A SECURITY SCORE</span><h2>Dimension changes the bill.</h2></div>
        <span className="figure-head-end">log<sub>2</sub> scale / heuristic leading terms</span>
      </div>
      <p className="view-intro">These are reference asymptotic costs for a <em>beta-dimensional SVP subroutine</em>. They are not calibrated ML-KEM attack costs or success probabilities. The full attack lattice here has dimension {attack.D}; the chosen block has dimension &beta; = {safeBeta}.</p>
      {safeBeta < 40 && <p className="inline-warning">At beta = {safeBeta}, the asymptotic leading exponents are not predictive. The curves show formulas only; do not interpret these values as a real attack estimate.</p>}

      <div className="model-controls">
        <label>HYPOTHETICAL BLOCK SIZE <strong>&beta; = {safeBeta}</strong>
          <input type="range" min="2" max={upper} value={safeBeta} onChange={(event) => setBeta(Number(event.target.value))} />
          <small>2 &le; &beta; &le; D = {attack.D}. Changing this does not run a full-dimensional attack.</small>
        </label>
        <label>ILLUSTRATIVE TOURS <strong>{tours}</strong>
          <input type="range" min="1" max="20" value={tours} onChange={(event) => setTours(Number(event.target.value))} />
          <small>Only affects the explicit D &times; tours oracle-call count below.</small>
        </label>
      </div>

      <div className="cost-chart-wrap">
        <svg viewBox="0 0 780 360" className="cost-chart" role="img" aria-label="Log base two cost of classical and quantum sieving subroutines by block dimension">
          <rect x="0" y="0" width="780" height="360" fill="#f7f7f3" />
          {axisTicks.map((tick) => <g key={tick}><line x1={left} x2={right} y1={sy(tick)} y2={sy(tick)} stroke="#dedfd9" strokeWidth="1" /><text x={left - 14} y={sy(tick) + 4} textAnchor="end" className="chart-tick">{tick}</text></g>)}
          {xTicks.map((tick) => <g key={tick}><line x1={sx(tick)} x2={sx(tick)} y1={bottom} y2={bottom + 5} stroke="#777973" /><text x={sx(tick)} y={bottom + 22} textAnchor="middle" className="chart-tick">{tick}</text></g>)}
          {upper >= 40 && <rect x={left} y={top} width={sx(40) - left} height={bottom - top} fill="#eaeae4" opacity="0.65" />}
          <path d={path((x) => classical(x) + callFactor)} fill="none" stroke="#8a8d85" strokeWidth="1.5" strokeDasharray="5 4" />
          <path d={path(classical)} fill="none" stroke="#252623" strokeWidth="2.3" />
          <path d={path(quantum)} fill="none" stroke="#a54838" strokeWidth="2.3" />
          <path d={path(memory)} fill="none" stroke="#8e938b" strokeWidth="1.7" />
          <line x1={sx(safeBeta)} x2={sx(safeBeta)} y1={top} y2={bottom} stroke="#8e9189" strokeDasharray="3 4" />
          <circle cx={sx(safeBeta)} cy={sy(classical(safeBeta))} r="4" fill="#252623" />
          <circle cx={sx(safeBeta)} cy={sy(quantum(safeBeta))} r="4" fill="#a54838" />
          <text x="17" y="19" className="chart-axis-title">log2 work / storage</text>
          <text x={right} y="348" textAnchor="end" className="chart-axis-title">block dimension beta</text>
          {upper >= 40 && <text x={left + 7} y={top + 14} className="chart-tick">asymptotic regime not meaningful here</text>}
        </svg>
      </div>
      <div className="chart-legend">
        <span><i className="legend-line black" /> classical sieve time / 0.292&beta;</span>
        <span><i className="legend-line red" /> quantum reference / 0.2653&beta;</span>
        <span><i className="legend-line gray" /> classical sieve memory / 0.208&beta;</span>
        <span><i className="legend-line dashed" /> D &times; tours &times; classical oracle proxy</span>
      </div>

      <div className="model-values">
        <div><small>CLASSICAL SVP SIEVE</small><strong>2<sup>{classical(safeBeta).toFixed(1)}</sup></strong><span>heuristic time, block only</span></div>
        <div><small>QUANTUM SVP REFERENCE</small><strong>2<sup>{quantum(safeBeta).toFixed(1)}</sup></strong><span>heuristic time, not polynomial</span></div>
        <div><small>CLASSICAL LIST MEMORY</small><strong>2<sup>{memory(safeBeta).toFixed(1)}</sup></strong><span>heuristic entries</span></div>
        <div><small>REDUCTION CALL PROXY</small><strong>2<sup>{(classical(safeBeta) + callFactor).toFixed(1)}</sup></strong><span>D &times; {tours} oracle calls, no overhead</span></div>
      </div>

      <div className="section-rule"><span>SEPARATE CONSEQUENCES</span></div>
      <div className="consequence-list">
        <div><span>01 / UNDERLYING HARDNESS</span><p>Depends on ring/module structure, distributions, modulus and attack selection. No hardness or security-bit claim is computed by this plot.</p></div>
        <div><span>02 / ATTACK SUCCESS</span><p>Unestimated. A block-SVP oracle cost alone does not give a secret-recovery probability; only a candidate checked against all public equations can be reported here.</p></div>
        <div><span>03 / CORRECTNESS</span><p>Current generated ciphertext: {instance.cipher.bitErrors} wrong bit{instance.cipher.bitErrors === 1 ? "" : "s"} / {p.n}. This exact observation is not a decryption-failure probability.</p></div>
        <div><span>04 / LOCAL PERFORMANCE</span><p>This browser built one algebraic instance in {instance.generationMs.toFixed(1)} ms using direct O(k<sup>2</sup>n<sup>2</sup>) multiplication. Projection/render time is not benchmarked. Neither is an attack cost; production ML-KEM uses the NTT.</p></div>
      </div>
      <p className="model-footnote">{standard ? `${standard} specifies n=${p.n}, k=${p.k}, q=${p.q}, eta1=${p.eta1}, eta2=${p.eta2}, du=${p.du}, dv=${p.dv}; encapsulation key ${publicBytes} bytes, ciphertext ${cipherBytes} bytes.` : `Modified construction: n=${p.n}, k=${p.k}, q=${p.q}; unpadded coefficient payload is ${cipherBytes.toFixed(1)} bytes, not an ML-KEM wire format.`} The attack dimension D={attack.D} and block size &beta; arise from a separately chosen attack model, not from FIPS 203.</p>
      <p className="model-footnote">Model: BDGL-style heuristic SVP sieve time 2<sup>0.292&beta;+o(&beta;)</sup>, memory 2<sup>0.208&beta;+o(&beta;)</sup>. The plotted quantum reference uses a Laarhoven-style 2<sup>0.2653&beta;+o(&beta;)</sup> sieve; later quantum-random-walk work reports 0.2570 with additional resource assumptions. Quantum search speeds a near-neighbor subroutine; it does not try all secrets at once. No efficient (polynomial-time) quantum algorithm is known for the underlying Module-LWE problem. Polynomial factors, memory access, preprocessing, attack optimization and success modeling are omitted.</p>
    </div>
  );
}