import { coefficientTerms, centered, mod, type Instance } from "../math/mlkem";

interface Props {
  instance: Instance;
  row: number;
  column: number;
  coefficient: number;
  setRow: (value: number) => void;
  setColumn: (value: number) => void;
  setCoefficient: (value: number) => void;
}

export default function AlgebraView({ instance, row, column, coefficient, setRow, setColumn, setCoefficient }: Props) {
  const { p, cipher } = instance;
  const terms = Array.from({ length: p.k }, (_, index) => coefficientTerms(instance, row, index, coefficient));
  const expanded = terms[column].terms.filter((term) => term.s !== 0);
  const page = Math.floor(coefficient / 16);
  const first = page * 16;
  const coefficientChoices = Array.from({ length: Math.min(16, p.n - first) }, (_, i) => first + i);
  const exactValue = instance.exact[row][coefficient];
  const error = instance.e[row][coefficient];
  const observed = instance.t[row][coefficient];
  const w = cipher.w[coefficient];
  const rawNoise = centered(cipher.expectedNoise[coefficient], p.q);
  const binomial = (n: number, k: number) => {
    let value = 1;
    for (let i = 1; i <= k; i++) value = (value * (n - i + 1)) / i;
    return value;
  };
  const errorLaw = Array.from({ length: 2 * p.keyErrorEta + 1 }, (_, index) => {
    const value = index - p.keyErrorEta;
    return { value, probability: binomial(2 * p.keyErrorEta, index) / 2 ** (2 * p.keyErrorEta) };
  });
  const maxProbability = Math.max(...errorLaw.map((item) => item.probability));

  return (
    <div className="algebra-view">
      <div className="figure-head">
        <div>
          <span className="micro-label">FIG. 01 / THE CONSTRUCTION</span>
          <h2>Begin with the noisy equation.</h2>
        </div>
        <span className="figure-head-end">All values from this generated instance</span>
      </div>
      <p className="lead-math"><i>R</i><sub>q</sub> = Z<sub>{p.q}</sub>[X] / (X<sup>{p.n}</sup> + 1) <span className="math-spacer">/</span> <i>A</i> &isin; <i>R</i><sub>q</sub><sup>{p.k}&times;{p.k}</sup>, &nbsp; <i>s</i>, <i>e</i>, <i>t</i> &isin; <i>R</i><sub>q</sub><sup>{p.k}</sup></p>

      <div className="coefficient-picker">
        <label>MODULE ROW
          <select value={row} onChange={(event) => setRow(Number(event.target.value))}>
            {Array.from({ length: p.k }, (_, i) => <option key={i} value={i}>i = {i}</option>)}
          </select>
        </label>
        <label>COEFFICIENT
          <input type="number" value={coefficient} min={0} max={p.n - 1} onChange={(event) => setCoefficient(Math.max(0, Math.min(p.n - 1, Number(event.target.value) || 0)))} />
        </label>
        <span className="picker-description">Examine [X<sup>{coefficient}</sup>] in polynomial row {row}.</span>
      </div>

      <div className="equation-exhibit">
        <div className="equation-item">
          <span className="equation-label">01 / EXACT MODULE PRODUCT</span>
          <span className="equation-symbol">[<i>A s</i>]<sub>{row},{coefficient}</sub></span>
          <strong>{exactValue.toLocaleString()}</strong>
          <small>sum of {p.k} negacyclic products, mod {p.q}</small>
        </div>
        <span className="equation-operator">+</span>
        <div className="equation-item noise-term">
          <span className="equation-label">02 / SAMPLED KEY ERROR</span>
          <span className="equation-symbol"><i>e</i><sub>{row},{coefficient}</sub></span>
          <strong>{error >= 0 ? `+${error}` : error}</strong>
          <small>CBD({p.keyErrorEta}) coefficient</small>
        </div>
        <span className="equation-operator">=</span>
        <div className="equation-item observed-term">
          <span className="equation-label">03 / PUBLIC OBSERVATION</span>
          <span className="equation-symbol"><i>t</i><sub>{row},{coefficient}</sub></span>
          <strong>{observed.toLocaleString()}</strong>
          <small>({exactValue} {error < 0 ? "-" : "+"} {Math.abs(error)}) mod {p.q}</small>
        </div>
      </div>

      <div className="noise-law">
        <div><span className="micro-label">THE ACTUAL KEY-ERROR LAW / CBD({p.keyErrorEta})</span><p>P(e = j) = C({2 * p.keyErrorEta}, {p.keyErrorEta} + j) / 2<sup>{2 * p.keyErrorEta}</sup>. The marked coefficient is e<sub>{row},{coefficient}</sub> = {error}.</p></div>
        <div className="noise-bars" aria-label="Exact centered binomial error probabilities">
          {errorLaw.map((item) => <div key={item.value} title={`P(e=${item.value}) = ${item.probability.toFixed(5)}`}><span style={{ height: `${Math.max(3, (item.probability / maxProbability) * 38)}px`, background: item.value === error ? "#a54838" : "#9da098" }} /><small>{item.value}</small></div>)}
        </div>
      </div>

      <div className="trace-subhead">
        <div><span className="micro-label">COEFFICIENT INDEX / [X^j]</span><h3>Walk along the public relation</h3></div>
        <div className="pager-buttons"><button disabled={page === 0} onClick={() => setCoefficient(Math.max(0, first - 16))}>prev 16</button><button disabled={first + 16 >= p.n} onClick={() => setCoefficient(first + 16)}>next 16</button></div>
      </div>
      <div className="coefficient-strip">
        {coefficientChoices.map((index) => <button key={index} className={coefficient === index ? "active" : ""} onClick={() => setCoefficient(index)} title={`t[${row},${index}] = ${instance.t[row][index]}, error = ${instance.e[row][index]}`}>
          <span>{String(index).padStart(2, "0")}</span><b>{instance.t[row][index]}</b><small>{instance.e[row][index] >= 0 ? "+" : ""}{instance.e[row][index]}</small>
        </button>)}
      </div>

      <div className="trace-subhead tighter">
        <div><span className="micro-label">NEGACYCLIC MULTIPLICATION / X^n = -1</span><h3>Where that coefficient came from</h3></div>
        <span className="right-note">Select a polynomial contribution</span>
      </div>
      <div className="math-table-wrap">
        <table className="math-table">
          <thead><tr><th>term</th><th>input coefficients [0..3]</th><th>raw [X<sup>{coefficient}</sup>]</th><th>mod {p.q}</th></tr></thead>
          <tbody>
            {terms.map((term, index) => <tr key={index} className={column === index ? "active-row" : ""} onClick={() => setColumn(index)}>
              <td><i>A</i><sub>{row},{index}</sub> &middot; <i>s</i><sub>{index}</sub></td>
              <td className="mono-cell">A: {instance.A[row][index].slice(0, 4).join(", ")} &nbsp; / &nbsp; s: {instance.s[index].slice(0, 4).join(", ")}</td>
              <td>{term.rawSum.toLocaleString()}</td>
              <td>{term.product.toLocaleString()}</td>
            </tr>)}
          </tbody>
        </table>
      </div>
      <p className="math-explanation">[X<sup>{coefficient}</sup>](<i>A</i><sub>{row},{column}</sub> <i>s</i><sub>{column}</sub>) = &Sigma;<sub>h=0</sub><sup>{p.n - 1}</sup> (&plusmn; A<sub>{row},{column},h</sub> s<sub>{column},({coefficient}-h) mod {p.n}</sub>); terms crossing degree {p.n} change sign. {expanded.length} of {p.n} terms have a nonzero secret coefficient.</p>
      <div className="term-sample">
        {expanded.slice(0, 7).map((term) => <span key={term.aIndex}>{term.sign < 0 ? "-" : "+"} {term.a} &times; {term.s} <small>(h={term.aIndex})</small></span>)}
        {expanded.length > 7 && <span className="muted">+ {expanded.length - 7} further terms</span>}
        {expanded.length === 0 && <span className="muted">All secret coefficients in this product are zero.</span>}
      </div>

      <div className="cipher-trace">
        <div className="trace-subhead"><div><span className="micro-label">THE SAME INSTANCE / K-PKE CIPHERTEXT</span><h3>Reconciliation is arithmetic, too.</h3></div></div>
        <p><i>u</i> = <i>A</i><sup>T</sup><i>r</i> + <i>e</i><sub>1</sub>, &nbsp; <i>v</i> = <i>t</i><sup>T</sup><i>r</i> + <i>e</i><sub>2</sub> + &mu;. Ciphertext stores compressed (<i>u</i>, <i>v</i>); this view also computes their decompression.</p>
        <div className="cipher-columns">
          <div><small>UNCOMPRESSED IDENTITY</small><strong><i>w</i><sub>0</sub> = {cipher.wExact[coefficient]}</strong><span>&mu; + e<sup>T</sup>r + e<sub>2</sub> - s<sup>T</sup>e<sub>1</sub> = {mod(cipher.mu[coefficient] + cipher.expectedNoise[coefficient], p.q)}</span></div>
          <div><small>COMPRESSION CHANGES IT</small><strong>&Delta;<sub>comp</sub> = {cipher.compressionDelta[coefficient] >= 0 ? "+" : ""}{cipher.compressionDelta[coefficient]}</strong><span>key/encryption noise {rawNoise >= 0 ? "+" : ""}{rawNoise}; all differences centered mod {p.q}</span></div>
          <div><small>DECODER OUTPUT</small><strong>{cipher.bits[coefficient]} &rarr; {cipher.recoveredBits[coefficient]}</strong><span><i>w</i> = {w}; Compress<sub>1</sub>(<i>w</i>) = {cipher.recoveredBits[coefficient]}</span></div>
        </div>
        <p className="secondary-caption">This is the exact K-PKE arithmetic of the displayed instance, including coefficient compression. ML-KEM also hashes, re-encrypts, compares ciphertexts, and uses implicit rejection during KEM decapsulation; those CCA wrapper steps are not simulated here.</p>
      </div>
    </div>
  );
}