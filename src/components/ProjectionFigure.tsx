import { useRef, type Dispatch, type PointerEvent, type SetStateAction, type WheelEvent } from "react";
import { norm, type AttackLattice, type BasisState, type SamplePoint, type SieveState, type Vector } from "../math/lattice";
import { type Projection, type ProjectionSettings } from "../math/projection";

export type Selection =
  | { kind: "point"; index: number }
  | { kind: "basis"; index: number }
  | { kind: "sieve"; index: number }
  | { kind: "target" | "exact" | "noise" | "witness" | "projection" };

interface Props {
  attack: AttackLattice;
  basis: BasisState;
  points: SamplePoint[];
  sieve: SieveState;
  projection: Projection;
  settings: ProjectionSettings;
  setSettings: Dispatch<SetStateAction<ProjectionSettings>>;
  selected: Selection;
  onSelect: (selection: Selection) => void;
  view: "projection" | "reduction" | "sieve" | "attack";
  blockSize: number;
}

const W = 800;
const H = 500;

function rounded(value: number): string {
  if (Math.abs(value) >= 10000) return value.toExponential(2);
  if (Math.abs(value) >= 100) return Math.round(value).toLocaleString();
  return value.toFixed(value === 0 ? 0 : 1);
}

export default function ProjectionFigure({ attack, basis, points, sieve, projection, settings, setSettings, selected, onSelect, view, blockSize }: Props) {
  const drag = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const plotted = points.map((point) => projection.apply(point.vector));
  const maxAbs = Math.max(1, ...plotted.flat().map(Math.abs));
  const baseScale = 211 / maxAbs;
  const scale = baseScale * settings.zoom;
  const originX = W / 2 + settings.panX;
  const originY = H / 2 + settings.panY;
  const screen = (vector: Vector): [number, number] => {
    const [x, y] = projection.apply(vector);
    return [originX + x * scale, originY - y * scale];
  };
  const screenProjected = ([x, y]: [number, number]): [number, number] => [originX + x * scale, originY - y * scale];
  const unique = new Set(plotted.map(([x, y]) => `${Math.round(x * 1e6)}:${Math.round(y * 1e6)}`)).size;
  const attacker = view === "attack";
  const showPrivate = !attacker;
  const pointSelected = selected.kind === "point" ? points[selected.index]?.vector : null;
  const basisSelected = selected.kind === "basis" ? basis.vectors[selected.index] : null;
  const sieveSelected = selected.kind === "sieve" ? sieve.list[selected.index]?.vector : null;
  const selectedVector = pointSelected || basisSelected || sieveSelected ||
    (selected.kind === "target" ? attack.publicTarget : null) ||
    (selected.kind === "witness" && showPrivate ? attack.witness : null);
  const observedPart = selectedVector ? selectedVector.map((value, i) => (i < attack.m ? value : 0)) : null;
  const secretPart = selectedVector ? selectedVector.map((value, i) => (i >= attack.m && i < attack.m + attack.d ? value : 0)) : null;
  const decomposed = observedPart && secretPart ? observedPart.map((value, i) => value + secretPart[i]) : null;
  const emphasized = new Set<number>(view === "reduction"
    ? Array.from({ length: Math.min(blockSize, basis.vectors.length - basis.blockIndex) }, (_, i) => basis.blockIndex + i)
    : [Math.max(0, basis.k - 1)]);
  const shownIndices = Array.from(new Set([
    ...Array.from({ length: Math.min(4, basis.vectors.length) }, (_, i) => i),
    ...emphasized,
    ...(selected.kind === "basis" ? [selected.index] : []),
  ])).filter((index) => index >= 0 && index < basis.vectors.length).slice(0, 9);

  function setPlane(mode: ProjectionSettings["mode"], x: number, y: number) {
    setSettings((previous) => ({ ...previous, mode, x, y, zoom: 1, panX: 0, panY: 0 }));
  }

  function focusNoise() {
    const [dx, dy] = projection.apply(attack.errorComponent);
    const length = Math.hypot(dx, dy);
    if (length < 1e-10) return;
    const zoom = Math.min(100000, Math.max(1, 135 / (length * baseScale)));
    const [cx, cy] = projection.apply(attack.exactObservation);
    setSettings((previous) => ({ ...previous, zoom, panX: -(cx + dx / 2) * baseScale * zoom, panY: (cy + dy / 2) * baseScale * zoom }));
  }

  function onPointerDown(event: PointerEvent<SVGSVGElement>) {
    drag.current = { x: event.clientX, y: event.clientY, panX: settings.panX, panY: settings.panY };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    if (!drag.current) return;
    const box = event.currentTarget.getBoundingClientRect();
    setSettings((previous) => ({
      ...previous,
      panX: drag.current!.panX + ((event.clientX - drag.current!.x) / box.width) * W,
      panY: drag.current!.panY + ((event.clientY - drag.current!.y) / box.height) * H,
    }));
  }

  function onWheel(event: WheelEvent<SVGSVGElement>) {
    event.preventDefault();
    const factor = event.deltaY > 0 ? 0.83 : 1.2;
    setSettings((previous) => ({ ...previous, zoom: Math.max(0.08, Math.min(100000, previous.zoom * factor)) }));
  }

  const axisMax = settings.mode === "gram-schmidt" ? attack.source.length - 1 : attack.D - 1;
  const target = screen(attack.publicTarget);
  const exact = showPrivate ? screen(attack.exactObservation) : target;
  const witness = showPrivate ? screen(attack.witness) : [0, 0];
  const noiseLengthPixels = showPrivate ? Math.hypot(target[0] - exact[0], target[1] - exact[1]) : 0;

  return (
    <div className="projection-figure">
      <div className="figure-head">
        <div>
          <span className="micro-label">FIG. 02 / PUBLIC ATTACK LATTICE</span>
          <h2>{view === "sieve" ? "Short vectors by difference" : view === "reduction" ? "One lattice, many bases" : view === "attack" ? "Only the public coordinates" : "The same object, another shadow"}</h2>
        </div>
        <button className="plain-link" onClick={() => onSelect({ kind: "projection" })}>Inspect projection &rarr;</button>
      </div>

      <div className="plane-controls" aria-label="Projection controls">
        <span className="control-caption">PLANE</span>
        <div className="mini-switches">
          <button className={settings.mode === "coordinates" && settings.x === 0 && settings.y === 1 ? "selected" : ""} onClick={() => setPlane("coordinates", 0, 1)}>obs / obs</button>
          <button className={settings.mode === "coordinates" && settings.x === attack.m ? "selected" : ""} onClick={() => setPlane("coordinates", attack.m, attack.m + 1)}>secret / secret</button>
          <button onClick={() => setPlane("coordinates", 0, attack.m)}>mixed</button>
          <button className={settings.mode === "gram-schmidt" ? "selected" : ""} onClick={() => setPlane("gram-schmidt", 0, attack.relationIndex)}>G-S</button>
          {!attacker && <button className={settings.mode === "witness" ? "selected" : ""} onClick={() => setPlane("witness", 0, 1)}>witness</button>}
        </div>
        <div className="axis-inputs">
          <label>x <input aria-label="First projection source dimension" type="number" min="0" max={axisMax} value={Math.min(settings.x, axisMax)} disabled={settings.mode === "witness"} onChange={(event) => setSettings((previous) => ({ ...previous, x: Math.max(0, Math.min(axisMax, Number(event.target.value) || 0)) }))} /></label>
          <label>y <input aria-label="Second projection source dimension" type="number" min="0" max={axisMax} value={Math.min(settings.y, axisMax)} disabled={settings.mode === "witness"} onChange={(event) => setSettings((previous) => ({ ...previous, y: Math.max(0, Math.min(axisMax, Number(event.target.value) || 0)) }))} /></label>
        </div>
      </div>

      <div className="plot-frame">
        <svg className="lattice-plot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Two-dimensional projection of a ${attack.D}-dimensional attack lattice`} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }} onWheel={onWheel}>
          <defs>
            <clipPath id="plot-clip"><rect width={W} height={H} /></clipPath>
            <marker id="arrow-ink" markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="none" stroke="#272724" strokeWidth="1.2" /></marker>
            <marker id="arrow-red" markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="none" stroke="#a54838" strokeWidth="1.2" /></marker>
          </defs>
          <rect width={W} height={H} fill="#f7f7f3" />
          <g clipPath="url(#plot-clip)">
            <line x1="0" y1={originY} x2={W} y2={originY} stroke="#d9dad4" strokeWidth="1" />
            <line x1={originX} y1="0" x2={originX} y2={H} stroke="#d9dad4" strokeWidth="1" />
            <circle cx={originX} cy={originY} r="3" fill="#777973" />

            {points.map((point, index) => {
              const [cx, cy] = screenProjected(plotted[index]);
              const active = selected.kind === "point" && selected.index === index;
              return <circle key={point.id} cx={cx} cy={cy} r={active ? 5.5 : 2.7} fill={active ? "#262623" : "#aeb1aa"} opacity={active ? 1 : view === "sieve" ? 0.28 : 0.72} className="plot-click" onClick={(event) => { event.stopPropagation(); onSelect({ kind: "point", index }); }} />;
            })}

            {settings.showComponents && observedPart && secretPart && decomposed && (
              <g>
                <line x1={originX} y1={originY} x2={screen(observedPart)[0]} y2={screen(observedPart)[1]} stroke="#878a82" strokeWidth="2" strokeDasharray="5 4" />
                <line x1={screen(observedPart)[0]} y1={screen(observedPart)[1]} x2={screen(decomposed)[0]} y2={screen(decomposed)[1]} stroke="#a54838" strokeWidth="2" strokeDasharray="5 4" />
              </g>
            )}

            {view !== "sieve" && shownIndices.map((index) => {
              const [x, y] = screen(basis.vectors[index]);
              const active = selected.kind === "basis" && selected.index === index;
              const highlighted = emphasized.has(index);
              return <g key={`basis-${index}`} className="plot-click" onClick={(event) => { event.stopPropagation(); onSelect({ kind: "basis", index }); }}>
                <line x1={originX} y1={originY} x2={x} y2={y} stroke={highlighted ? "#282825" : "#7e817a"} strokeWidth={active ? 2.5 : highlighted ? 1.8 : 1.25} opacity={highlighted ? 1 : 0.68} markerEnd="url(#arrow-ink)" />
                <circle cx={x} cy={y} r={active ? 6 : 4} fill={highlighted ? "#282825" : "#777973"} />
                {Math.abs(x - originX) + Math.abs(y - originY) > 17 && <text x={x + 7} y={y - 7} className="svg-vector-label">b{index + 1}</text>}
              </g>;
            })}

            {view === "sieve" && sieve.list.map((item, index) => {
              const [x, y] = screen(item.vector);
              const active = selected.kind === "sieve" && selected.index === index;
              return <g key={`sieve-${index}`} className="plot-click" onClick={(event) => { event.stopPropagation(); onSelect({ kind: "sieve", index }); }}>
                <line x1={originX} y1={originY} x2={x} y2={y} stroke={active ? "#272724" : "#999c95"} strokeWidth={active ? 2 : 1} opacity="0.65" />
                <circle cx={x} cy={y} r={active ? 5.5 : 3.5} fill={active ? "#272724" : "#777a73"} />
              </g>;
            })}
            {view === "sieve" && sieve.lastSource && sieve.lastReducer && sieve.lastDifference && (() => {
              const source = screen(sieve.lastSource!);
              const reducer = screen(sieve.lastReducer!);
              const difference = screen(sieve.lastDifference!);
              return <g>
                <line x1={reducer[0]} y1={reducer[1]} x2={source[0]} y2={source[1]} stroke="#a54838" strokeWidth="2.3" markerEnd="url(#arrow-red)" />
                <line x1={originX} y1={originY} x2={difference[0]} y2={difference[1]} stroke="#a54838" strokeWidth="1.4" strokeDasharray="5 4" />
                <circle cx={source[0]} cy={source[1]} r="4" fill="#a54838" />
                <circle cx={reducer[0]} cy={reducer[1]} r="4" fill="#a54838" />
              </g>;
            })()}

            {view !== "sieve" && <g className="plot-click" onClick={(event) => { event.stopPropagation(); onSelect({ kind: "target" }); }}>
              <rect x={target[0] - 5} y={target[1] - 5} width="10" height="10" fill="#f7f7f3" stroke="#343531" strokeWidth="1.6" />
            </g>}
            {showPrivate && settings.showNoise && view !== "sieve" && <g>
              <line x1={exact[0]} y1={exact[1]} x2={target[0]} y2={target[1]} stroke="#a54838" strokeWidth="1.5" />
              <circle cx={exact[0]} cy={exact[1]} r="4" fill="none" stroke="#a54838" strokeWidth="1.5" className="plot-click" onClick={(event) => { event.stopPropagation(); onSelect({ kind: "exact" }); }} />
            </g>}
            {showPrivate && settings.showWitness && attack.construction === "embedding" && view !== "sieve" && <g className="plot-click" onClick={(event) => { event.stopPropagation(); onSelect({ kind: "witness" }); }}>
              <line x1={originX} y1={originY} x2={witness[0]} y2={witness[1]} stroke="#a54838" strokeWidth="1.7" markerEnd="url(#arrow-red)" />
              <circle cx={witness[0]} cy={witness[1]} r="5" fill="#a54838" />
            </g>}
          </g>
          <text x="18" y="27" className="svg-corner-label">P : R^{attack.D} -&gt; R^2</text>
          <text x={W - 18} y="27" textAnchor="end" className="svg-corner-label">ROT {settings.angle} DEG / ZOOM {settings.zoom < 10 ? settings.zoom.toFixed(1) : settings.zoom.toFixed(0)}x</text>
          <text x="18" y={H - 17} className="svg-corner-label">X: {projection.descriptionX}</text>
          <text x={W - 18} y={H - 17} textAnchor="end" className="svg-corner-label">Y: {projection.descriptionY}</text>
        </svg>
        <div className="plot-tools" aria-label="View tools">
          <button title="Zoom in" onClick={() => setSettings((s) => ({ ...s, zoom: Math.min(100000, s.zoom * 1.45) }))}>+</button>
          <button title="Zoom out" onClick={() => setSettings((s) => ({ ...s, zoom: Math.max(0.08, s.zoom / 1.45) }))}>&minus;</button>
          <button title="Fit sampled vectors" onClick={() => setSettings((s) => ({ ...s, zoom: 1, panX: 0, panY: 0 }))}>fit</button>
        </div>
      </div>

      <div className="rotation-row">
        <label htmlFor="plane-rotation">ROTATE PLANE <span>{settings.angle}&deg;</span></label>
        <input id="plane-rotation" type="range" min="-180" max="180" step="1" value={settings.angle} onChange={(event) => setSettings((s) => ({ ...s, angle: Number(event.target.value) }))} />
        {showPrivate && <button className="text-button" disabled={norm(projection.apply(attack.errorComponent)) < 1e-10} onClick={focusNoise}>focus residual</button>}
      </div>

      <div className="figure-options">
        {!attacker && <>
          <label><input type="checkbox" checked={settings.showWitness} onChange={(event) => setSettings((s) => ({ ...s, showWitness: event.target.checked }))} /> secret witness</label>
          <label><input type="checkbox" checked={settings.showNoise} onChange={(event) => setSettings((s) => ({ ...s, showNoise: event.target.checked }))} /> exact / error</label>
          <button className="text-button" onClick={() => onSelect({ kind: "witness" })}>inspect witness</button>
          <button className="text-button" onClick={() => onSelect({ kind: "noise" })}>inspect error</button>
        </>}
        <label><input type="checkbox" checked={settings.showComponents} onChange={(event) => setSettings((s) => ({ ...s, showComponents: event.target.checked }))} /> selected vector components</label>
      </div>

      <p className="figure-caption">
        <strong>Projection, not the lattice.</strong> {points.length} fixed integer combinations of three public generators are drawn in 2D from an ambient {attack.D}-coordinate object; {unique} distinct images at 10<sup>-6</sup> resolution. {attack.fullBasis ? `The available basis spans the full ${attack.D}-dimensional attack lattice.` : `Reduction acts on a rank-${attack.source.length} sublattice, not on the full attack lattice.`} {showPrivate && settings.showNoise && noiseLengthPixels < 1.5 && "The true error separation is below 1.5 pixels at this zoom."}
      </p>
      {settings.showComponents && <p className="secondary-caption">Dashed gray is the selected vector's observation-coordinate component; dashed red is its secret-coordinate component. An embedding coordinate, if present, is separate.</p>}
      {projection.degenerate && <p className="inline-warning">These source directions coincide. This 2D display has rank at most one.</p>}
      {view === "reduction" && <p className="secondary-caption">Arrow lengths in this plane are projected lengths. The reported norms in the basis table are full {attack.D}-coordinate norms.</p>}
      {view === "sieve" && <p className="secondary-caption">The line joining two endpoints is their vector difference. Only 16 sampled list elements are drawn; this is not the list size of a real cryptanalytic sieve.</p>}
      <span className="sr-only">Scale is {rounded(scale)} screen units per projected coordinate.</span>
    </div>
  );
}