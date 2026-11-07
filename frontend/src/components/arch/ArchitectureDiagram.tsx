import { useEffect, useMemo, useRef, useState } from "react";
import { Maximize2, Minus, Plus, Route } from "lucide-react";
import type { ArchitectureReport, LayerSpec } from "../../api/types";
import { classNames, LAYER_SHORT, shape } from "../../lib/format";

const W = 138;
const H = 66;
const GX = 24;
const GY = 58;
const LEFT = 96;
const TOP = 26;

interface Node {
  id: string;
  label: string;
  sub: string;
  shape: number[];
  kind: "io" | "conv" | "norm" | "act" | "pool" | "drop" | "flat" | "dense";
  valid: boolean;
  row: number;
  col: number;
  spec?: LayerSpec;
}

function kindOf(s: LayerSpec): Node["kind"] {
  if (s.name.includes("conv") || s.name.startsWith("separable")) return "conv";
  if (s.name.startsWith("bn")) return "norm";
  if (s.name.startsWith("act")) return "act";
  if (s.name.startsWith("pool")) return "pool";
  if (s.name.startsWith("drop")) return "drop";
  if (s.name === "flatten") return "flat";
  return "dense";
}

function subtitle(s: LayerSpec): string {
  if (s.kernel) return `${s.type.replace(" (depthwise)", " dw").replace(" (1x1)", "")} ${s.kernel.join("×")}${s.groups && s.groups > 1 ? ` g${s.groups}` : ""}`;
  return s.type;
}

const fills: Record<Node["kind"], string> = {
  io: "fill-card stroke-primary",
  conv: "fill-primary-soft stroke-line-strong",
  norm: "fill-subtle stroke-line",
  act: "fill-subtle stroke-line",
  pool: "fill-subtle stroke-line",
  drop: "fill-card stroke-line-strong",
  flat: "fill-subtle stroke-line",
  dense: "fill-primary-soft stroke-line-strong",
};

export function ArchitectureDiagram({
  report,
  selected,
  onSelect,
  height = 420,
}: {
  report: ArchitectureReport;
  selected: string | null;
  onSelect: (name: string | null) => void;
  height?: number;
}) {
  const [trace, setTrace] = useState(true);
  const [view, setView] = useState({ k: 1, x: 0, y: 0 });
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);

  const nodes = useMemo<Node[]>(() => {
    const out: Node[] = [{ id: "input", label: "Input", sub: report.input_convention.replace("[batch, ", "[").replace("time_samples", "T"), shape: report.input_shape, kind: "io", valid: true, row: 0, col: 0 }];
    const rowOf: Record<string, number> = { "Block 1": 0, "Block 2": 1, Classifier: 2 };
    const counts = [1, 0, 0];
    for (const s of report.layers) {
      const r = rowOf[s.block] ?? 2;
      out.push({ id: s.name, label: LAYER_SHORT[s.name] ?? s.label, sub: subtitle(s), shape: s.out_shape, kind: kindOf(s), valid: s.valid, row: r, col: counts[r]++, spec: s });
    }
    out.push({ id: "softmax", label: "Probabilities", sub: "softmax", shape: [report.config.n_classes], kind: "io", valid: report.valid, row: 2, col: counts[2]++ });
    return out;
  }, [report]);

  const maxCols = Math.max(...[0, 1, 2].map((r) => nodes.filter((n) => n.row === r).length));
  const vbW = LEFT + maxCols * (W + GX) + 10;
  const vbH = TOP + 3 * (H + GY) - GY + 30;
  const pos = (n: Node) => ({ x: LEFT + n.col * (W + GX), y: TOP + n.row * (H + GY) });
  const selIdx = selected ? nodes.findIndex((n) => n.id === selected) : -1;

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const sx = ((e.clientX - rect.left) / rect.width) * vbW;
      const sy = ((e.clientY - rect.top) / rect.height) * vbH;
      setView((v) => {
        const k = Math.min(4, Math.max(0.4, v.k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
        return { k, x: sx - ((sx - v.x) * k) / v.k, y: sy - ((sy - v.y) * k) / v.k };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [vbW, vbH]);

  const zoom = (f: number) => setView((v) => ({ k: Math.min(4, Math.max(0.4, v.k * f)), x: v.x, y: v.y }));

  return (
    <div>
      <div className="mb-2 ml-auto flex w-fit gap-1 rounded-full border border-line bg-card p-1 shadow-sm">
        <button type="button" aria-pressed={trace} title="Trace tensor shapes along the edges" onClick={() => setTrace((t) => !t)} className={classNames("grid size-8 place-items-center rounded-full", trace ? "bg-primary-soft text-link" : "text-ink-subtle hover:text-ink")}>
          <Route className="size-4" />
        </button>
        <button type="button" aria-label="Zoom in" onClick={() => zoom(1.2)} className="grid size-8 place-items-center rounded-full text-ink-subtle hover:text-ink"><Plus className="size-4" /></button>
        <button type="button" aria-label="Zoom out" onClick={() => zoom(1 / 1.2)} className="grid size-8 place-items-center rounded-full text-ink-subtle hover:text-ink"><Minus className="size-4" /></button>
        <button type="button" aria-label="Reset view" onClick={() => setView({ k: 1, x: 0, y: 0 })} className="grid size-8 place-items-center rounded-full text-ink-subtle hover:text-ink"><Maximize2 className="size-4" /></button>
      </div>
      <svg
        ref={svgRef}
        role="img"
        aria-label="EEGNet architecture diagram"
        viewBox={`0 0 ${vbW} ${vbH}`}
        preserveAspectRatio="xMinYMin meet"
        className="w-full cursor-grab touch-none select-none active:cursor-grabbing"
        style={{ height }}
        onPointerDown={(e) => {
          (e.target as Element).setPointerCapture?.(e.pointerId);
          drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
        }}
        onPointerMove={(e) => {
          if (!drag.current || !svgRef.current) return;
          const rect = svgRef.current.getBoundingClientRect();
          const scale = vbW / rect.width;
          setView((v) => ({ ...v, x: drag.current!.vx + (e.clientX - drag.current!.x) * scale, y: drag.current!.vy + (e.clientY - drag.current!.y) * scale }));
        }}
        onPointerUp={() => (drag.current = null)}
      >
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" className="fill-ink-subtle" />
          </marker>
        </defs>
        <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
          {["Block 1", "Block 2", "Classifier"].map((b, r) => (
            <text key={b} x={8} y={TOP + r * (H + GY) + H / 2 + 4} className="fill-ink-subtle" style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.04em" }}>
              {b.toUpperCase()}
            </text>
          ))}
          {nodes.slice(1).map((n, i) => {
            const a = nodes[i];
            const pa = pos(a);
            const pb = pos(n);
            const onPath = trace && selIdx >= 0 && i + 1 <= selIdx;
            const cls = onPath ? "stroke-primary" : "stroke-ink-subtle";
            let d: string;
            let lx: number;
            let ly: number;
            if (a.row === n.row) {
              d = `M ${pa.x + W} ${pa.y + H / 2} L ${pb.x - 2} ${pb.y + H / 2}`;
              lx = pa.x + W + GX / 2;
              ly = pa.y + H / 2 - 6;
            } else {
              const midY = pa.y + H + GY / 2;
              d = `M ${pa.x + W / 2} ${pa.y + H} L ${pa.x + W / 2} ${midY} L ${pb.x + W / 2} ${midY} L ${pb.x + W / 2} ${pb.y - 2}`;
              lx = (pa.x + pb.x + W) / 2;
              ly = midY - 6;
            }
            return (
              <g key={`e-${n.id}`}>
                <path d={d} fill="none" className={cls} strokeWidth={onPath ? 2 : 1.2} markerEnd="url(#arrow)" />
                {trace && a.row !== n.row && (
                  <text x={lx} y={ly} textAnchor="middle" className="fill-ink-muted" style={{ fontSize: 10.5, fontFamily: "var(--font-mono)" }}>
                    {shape(a.shape)}
                  </text>
                )}
              </g>
            );
          })}
          {nodes.map((n) => {
            const p = pos(n);
            const isSel = selected === n.id;
            return (
              <g
                key={n.id}
                transform={`translate(${p.x},${p.y})`}
                role="button"
                tabIndex={0}
                aria-label={`${n.label}, output ${shape(n.shape)}`}
                aria-pressed={isSel}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={() => onSelect(isSel ? null : n.id)}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(isSel ? null : n.id)}
                className="cursor-pointer outline-none"
              >
                <rect
                  width={W}
                  height={H}
                  rx={14}
                  className={classNames(!n.valid ? "fill-danger-soft stroke-danger" : fills[n.kind], isSel && "stroke-primary")}
                  strokeWidth={isSel ? 2.4 : 1}
                  strokeDasharray={n.kind === "drop" ? "4 3" : undefined}
                />
                <text x={12} y={21} className="fill-ink" style={{ fontSize: 12.5, fontWeight: 700 }}>{n.label}</text>
                <text x={12} y={37} className="fill-ink-subtle" style={{ fontSize: 10.5 }}>{n.sub.length > 22 ? n.sub.slice(0, 21) + "…" : n.sub}</text>
                <text x={12} y={55} className={n.valid ? "fill-link" : "fill-danger"} style={{ fontSize: 11, fontFamily: "var(--font-mono)" }}>
                  {n.valid ? shape(n.shape) : "invalid"}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
      <p className="mt-1 text-[12px] text-ink-subtle">Scroll to zoom, drag to pan, click a layer to inspect it. Shapes exclude the batch dimension.</p>
    </div>
  );
}
