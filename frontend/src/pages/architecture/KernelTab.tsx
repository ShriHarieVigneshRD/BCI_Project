import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, XCircle } from "lucide-react";
import { api, qs } from "../../api/client";
import { useTrials } from "../../api/hooks";
import type { ArenaResult, ArchitectureReport, ConvolutionResult, Experiment, Kernels } from "../../api/types";
import { TrialPicker } from "../../components/TrialPicker";
import { Bars, Heatmap, LinePlot } from "../../components/charts/Charts";
import { Card, CardBody, CardHeader, Stat } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Chip";
import { NumberField, Segmented, SelectField } from "../../components/ui/Field";
import { Alert, ErrorState, KeyValue, Loading, Mono } from "../../components/ui/Misc";
import { classNames, fmtInt, shape } from "../../lib/format";

type Section = "conv" | "gallery" | "arena";

export function KernelTab({ exp }: { exp: Experiment }) {
  const [params] = useSearchParams();
  const run = exp.active_run!;
  const nFolds = run.folds.length;
  const [fold, setFold] = useState(Math.min(Number(params.get("fold") ?? 0), nFolds - 1));
  const [section, setSection] = useState<Section>("conv");
  const { data: tl } = useTrials(exp.id, fold, true);
  const [trial, setTrial] = useState<string | null>(params.get("trial"));
  const { data: report } = useQuery({
    queryKey: ["validate-snapshot", run.snapshot.architecture_fingerprint],
    queryFn: () => api.post<ArchitectureReport>("/api/models/validate", { config: run.snapshot.architecture_config }),
  });
  useEffect(() => {
    if (tl && (!trial || !tl.trials.some((t) => t.trial_id === trial))) setTrial((tl.trials.find((t) => t.partition === "test") ?? tl.trials[0])?.trial_id ?? null);
  }, [tl, trial]);
  const ckpt = run.folds[fold]?.checkpoint_id;

  return (
    <div className="space-y-6">
      <Alert tone="info" title="Learned weights from the saved checkpoint">
        Every value below is read from checkpoint <Mono>{ckpt}</Mono> (fold {fold + 1}, architecture {run.snapshot.architecture_fingerprint}) and computed in evaluation mode on real preprocessed trials.
      </Alert>
      <Card>
        <CardBody className="grid gap-5 pt-5 lg:grid-cols-[auto_auto_minmax(0,1fr)]">
          <div>
            <div className="mb-1.5 text-[13px] font-bold text-ink-muted">View</div>
            <Segmented<Section> ariaLabel="Playground section" value={section} onChange={setSection} options={[{ value: "conv", label: "Convolution breakdown" }, { value: "gallery", label: "Kernel gallery" }, { value: "arena", label: "Convolution Arena" }]} />
          </div>
          {nFolds > 1 && (
            <div className="w-36"><SelectField label="Checkpoint" value={String(fold)} onChange={(v) => setFold(Number(v))} options={run.folds.map((f) => ({ value: String(f.fold), label: `Fold ${f.fold + 1}` }))} /></div>
          )}
          <div className="min-w-0">{tl && <TrialPicker trials={tl.trials} value={trial} onChange={setTrial} />}</div>
        </CardBody>
      </Card>
      {section === "conv" && report && trial && tl && <ConvolutionSection exp={exp} fold={fold} trial={trial} report={report} trials={tl.trials} initialLayer={params.get("layer")} />}
      {section === "gallery" && <KernelGallery exp={exp} fold={fold} />}
      {section === "arena" && trial && <ConvolutionArena exp={exp} fold={fold} trial={trial} />}
    </div>
  );
}

function ConvolutionSection({ exp, fold, trial, report, trials, initialLayer }: { exp: Experiment; fold: number; trial: string; report: ArchitectureReport; trials: { trial_id: string; label: string; group: string | null; session: string | null; partition: "train" | "validation" | "test" | "unused" | "unknown" }[]; initialLayer: string | null }) {
  const [layer, setLayer] = useState(initialLayer && report.layers.some((l) => l.name === initialLayer) ? initialLayer : "temporal_conv");
  const spec = report.layers.find((l) => l.name === layer)!;
  const out = layer === "flatten" ? spec.in_shape : spec.out_shape;
  const [channel, setChannel] = useState(0);
  const [row, setRow] = useState(0);
  const [position, setPosition] = useState(Math.floor((out[2] ?? 1) / 2));
  const [compare, setCompare] = useState<string | null>(null);
  const clamp = (v: number, n: number) => Math.max(0, Math.min(n - 1, v));
  const ch = clamp(channel, out[0] ?? 1);
  const rw = clamp(row, out[1] ?? 1);
  const ps = clamp(position, out[2] ?? 1);
  const { data, error, isFetching } = useQuery({
    queryKey: ["conv", exp.id, fold, layer, trial, ch, rw, ps, compare],
    queryFn: () => api.post<ConvolutionResult>(`/api/experiments/${exp.id}/convolution`, { layer, trial_id: trial, channel: ch, row: rw, position: ps, fold, compare_trial_id: compare }),
    placeholderData: (p) => p,
  });
  const is3d = out.length === 3;
  return (
    <div className="space-y-6">
      <Card>
        <CardBody className="grid gap-4 pt-5 sm:grid-cols-2 xl:grid-cols-6">
          <div className="sm:col-span-2">
            <SelectField label="Layer" value={layer} onChange={(v) => { setLayer(v); setPosition(0); }} options={report.layers.map((l) => ({ value: l.name, label: `${l.label} · ${shape(l.out_shape)}` }))} />
          </div>
          {is3d ? (
            <>
              <NumberField label={`${layer.includes("pointwise") || layer.includes("conv") || layer.startsWith("separable") ? "Output map / kernel" : "Feature map"} (0–${out[0] - 1})`} value={ch} min={0} max={out[0] - 1} onChange={(v) => setChannel(v ?? 0)} />
              <NumberField label={`Row (0–${out[1] - 1})`} value={rw} min={0} max={out[1] - 1} onChange={(v) => setRow(v ?? 0)} hint={layer === "temporal_conv" || layer === "bn1" ? "Rows are EEG channels before the spatial filter." : "After the depthwise spatial filter each map has one row (when the kernel spans all channels)."} />
              <div className="sm:col-span-2">
                <label className="mb-1.5 block text-[13px] font-bold text-ink-muted">Position / time step: {ps} of {out[2] - 1}</label>
                <input type="range" aria-label="Convolution position" min={0} max={out[2] - 1} value={ps} onChange={(e) => setPosition(Number(e.target.value))} className="w-full accent-[var(--c-primary)]" />
              </div>
            </>
          ) : (
            <NumberField label={`${layer === "classifier" ? "Class" : "Index"} (0–${out[0] - 1})`} value={ch} min={0} max={out[0] - 1} onChange={(v) => setChannel(v ?? 0)} />
          )}
          <div className="sm:col-span-2 xl:col-span-6">
            <SelectField label="Compare with a second trial (optional)" value={compare ?? ""} onChange={(v) => setCompare(v || null)} options={[{ value: "", label: "No comparison" }, ...trials.filter((t) => t.trial_id !== trial).slice(0, 800).map((t) => ({ value: t.trial_id, label: `${t.trial_id} · ${t.label} · ${t.partition}` }))]} />
          </div>
        </CardBody>
      </Card>
      {error && <ErrorState error={error} />}
      {!data ? <Loading label="Computing…" /> : <ConvolutionView res={data} updating={isFetching} />}
    </div>
  );
}

function Verified({ res }: { res: ConvolutionResult }) {
  return (
    <div className={classNames("flex flex-wrap items-center gap-x-4 gap-y-1 rounded-2xl border px-4 py-3", res.verified ? "border-success-line bg-success-soft" : "border-danger-line bg-danger-soft")}>
      {res.verified ? <CheckCircle2 className="size-5 text-success" /> : <XCircle className="size-5 text-danger" />}
      <span className="text-[14px] text-ink">Recomputed by hand: <Mono className="font-bold">{res.computed.toPrecision(7)}</Mono></span>
      <span className="text-[14px] text-ink">Activation from the forward hook: <Mono className="font-bold">{res.actual.toPrecision(7)}</Mono></span>
      <span className="text-[13px] text-ink-muted">|Δ| = {res.abs_diff.toExponential(1)}</span>
    </div>
  );
}

function ConvolutionView({ res, updating }: { res: ConvolutionResult; updating: boolean }) {
  const sel = res.selected;
  const w = res.window;
  const idx = (n: number) => Array.from({ length: n }, (_, i) => i);
  return (
    <Card>
      <CardHeader
        title={<span className="flex flex-wrap items-center gap-2">{res.label} <Badge>{res.type}</Badge> {updating && <Badge tone="warning">updating…</Badge>}</span>}
        subtitle={res.operation}
        actions={<Badge tone="primary">{res.partition} trial</Badge>}
      />
      <CardBody className="space-y-5">
        <Verified res={res} />
        <KeyValue items={[
          ["Shapes", <Mono key="s">{shape(res.in_shape)} → {shape(res.out_shape)}</Mono>],
          ["Kernel · stride · padding · groups", <Mono key="k">{res.kernel?.join("×") ?? "—"} · {res.stride?.join("×") ?? "—"} · {res.padding ? `${res.padding[0]}|${res.padding[1]}` : "—"} · {res.groups ?? "—"}</Mono>],
          ["Selected element", <Mono key="e">{Object.entries(sel).filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => `${k}=${v}`).join("  ")}</Mono>],
        ]} />
        {res.input_row && w?.indices && (
          <div className="grid gap-4 xl:grid-cols-2">
            <div>
              <div className="mb-1 text-[13px] font-bold text-ink-muted">Input row{sel.row_name ? ` (${sel.row_name})` : ""} with the kernel window</div>
              <LinePlot x={idx(res.input_row.length)} series={[{ name: "input", data: res.input_row }]} legend={false} xName="sample" markAreaX={[w.indices[0], w.indices[w.indices.length - 1]]} height={210} ariaLabel="Input row with window" />
            </div>
            <div>
              <div className="mb-1 text-[13px] font-bold text-ink-muted">Output row; the selected position is marked</div>
              {res.output_row && <LinePlot x={idx(res.output_row.length)} series={[{ name: "output", data: res.output_row }]} legend={false} xName="position" markLineX={Number(sel.position)} height={210} ariaLabel="Output row" />}
            </div>
          </div>
        )}
        {res.weights && res.products && (
          <div className="grid gap-4 xl:grid-cols-2">
            <div>
              <div className="mb-1 text-[13px] font-bold text-ink-muted">Learned kernel weights ({res.weights.length})</div>
              <Bars labels={w?.channels ?? idx(res.weights.length)} series={[{ name: "weight", data: res.weights }]} height={200} ariaLabel="Kernel weights" valueDigits={5} />
            </div>
            <div>
              <div className="mb-1 text-[13px] font-bold text-ink-muted">Element-wise products (input × weight), summed to the output</div>
              <Bars labels={w?.channels ?? idx(res.products.length)} series={[{ name: "product", data: res.products }]} height={200} ariaLabel="Products" valueDigits={5} />
            </div>
          </div>
        )}
        {w?.padded?.some(Boolean) && <p className="text-[13px] text-warning">{w.padded.filter(Boolean).length} window sample(s) fall in the zero padding (shown as 0).</p>}
        {res.input_map && (
          <div>
            <div className="mb-1 text-[13px] font-bold text-ink-muted">Input feature map read by this kernel (highlighted column = selected position)</div>
            <Heatmap matrix={res.input_map} diverging highlightCol={Number(sel.position)} yLabels={w?.channels && w.channels.length === res.input_map.length ? w.channels : undefined} height={Math.max(200, 13 * res.input_map.length + 80)} xName="time" ariaLabel="Input feature map" />
          </div>
        )}
        {res.params && <KeyValue items={Object.entries(res.params).map(([k, v]) => [k, <Mono key={k}>{v.toPrecision(6)}</Mono>])} />}
        {res.input_value !== undefined && <p className="text-[14px] text-ink">Input value <Mono>{res.input_value.toPrecision(6)}</Mono> → output <Mono>{res.actual.toPrecision(6)}</Mono></p>}
        {res.curve && <LinePlot x={res.curve.x} series={[{ name: "activation", data: res.curve.y }]} legend={false} xName="input" yName="output" markLineX={res.input_value ?? null} height={200} ariaLabel="Activation function" />}
        {res.top_contributions && (
          <div className="grid gap-4 xl:grid-cols-2">
            <div>
              <div className="mb-1 text-[13px] font-bold text-ink-muted">Largest contributions to the {String(sel.class_name)} logit (bias {res.bias?.toFixed(4)})</div>
              <div className="overflow-x-auto rounded-xl border border-line">
                <table className="w-full text-[13px] tabular">
                  <thead className="bg-subtle text-left text-ink-subtle"><tr><th className="px-3 py-1.5">index</th><th>[f, h, w]</th><th>feature</th><th>weight</th><th className="pr-3">product</th></tr></thead>
                  <tbody>{res.top_contributions.map((c) => <tr key={c.index} className="border-t border-line"><td className="px-3 py-1">{c.index}</td><td><Mono>[{c.fhw.join(", ")}]</Mono></td><td>{c.feature.toFixed(4)}</td><td>{c.weight.toFixed(4)}</td><td className={classNames("pr-3 font-bold", c.product >= 0 ? "text-success" : "text-danger")}>{c.product.toFixed(4)}</td></tr>)}</tbody>
                </table>
              </div>
            </div>
            {res.products_map && (
              <div>
                <div className="mb-1 text-[13px] font-bold text-ink-muted">Products per feature map and time step</div>
                <Heatmap matrix={res.products_map.flatMap((f) => f)} diverging yLabels={res.products_map.flatMap((f, i) => f.map((_, h) => (f.length > 1 ? `f${i}·h${h}` : `f${i}`)))} height={Math.max(220, 13 * res.products_map.length * res.products_map[0].length + 80)} xName="time step" ariaLabel="Classifier products" />
              </div>
            )}
          </div>
        )}
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <div className="mb-1 text-[13px] font-bold text-ink-muted">Model prediction: {res.prediction.predicted} (true: {res.prediction.true})</div>
            <Bars labels={res.prediction.class_names} series={[{ name: "probability", data: res.prediction.probabilities }]} yMax={1} height={170} ariaLabel="Prediction" />
          </div>
          {res.comparison && <Comparison c={res.comparison} />}
        </div>
      </CardBody>
    </Card>
  );
}

function Comparison({ c }: { c: NonNullable<ConvolutionResult["comparison"]> }) {
  const oneD = !Array.isArray(c.output_a[0]);
  return (
    <div className="space-y-3 rounded-2xl border border-line bg-surface p-4">
      <div className="text-[13px] font-bold text-ink-muted">Trial {c.trial_a} vs {c.trial_b} ({c.label_b}) at this layer</div>
      <div className="grid grid-cols-2 gap-3">
        <Stat label="Input RMS diff" value={c.input_difference_rms.toPrecision(3)} />
        <Stat label="Activation RMS diff" value={c.activation_difference_rms.toPrecision(3)} />
      </div>
      {oneD ? (
        <LinePlot x={(c.output_a as number[]).map((_, i) => i)} series={[{ name: c.trial_a, data: c.output_a as number[] }, { name: c.trial_b, data: c.output_b as number[] }, { name: "difference", data: c.difference as number[], dashed: true }]} height={200} ariaLabel="Activation comparison" />
      ) : (
        <Heatmap matrix={c.difference as number[][]} diverging height={200} xName="time" ariaLabel="Activation difference" />
      )}
      <Bars labels={c.probabilities_a.map((_, i) => String(i))} series={[{ name: c.trial_a, data: c.probabilities_a }, { name: c.trial_b, data: c.probabilities_b }]} yMax={1} height={170} ariaLabel="Probabilities comparison" />
      <p className="text-[13px] text-ink-muted">Second trial predicted as <b className="text-ink">{c.predicted_b}</b>.</p>
    </div>
  );
}

function KernelGallery({ exp, fold }: { exp: Experiment; fold: number }) {
  const { data: k, error } = useQuery({ queryKey: ["kernels", exp.id, fold, exp.active_run_id], queryFn: () => api.get<Kernels>(`/api/experiments/${exp.id}/kernels${qs({ fold })}`) });
  const [f, setF] = useState(0);
  const msAxis = useMemo(() => (k ? k.temporal.weights[0].map((_, i) => +((i / k.temporal.sfreq) * 1000).toFixed(1)) : []), [k]);
  if (error) return <ErrorState error={error} />;
  if (!k) return <Loading label="Reading learned weights…" />;
  const F1 = k.temporal.weights.length;
  return (
    <div className="space-y-6">
      <p className="text-[13px] text-ink-subtle">{k.source} · sha256 <Mono>{k.sha256.slice(0, 16)}…</Mono></p>
      <Card>
        <CardHeader title="Temporal kernels" subtitle={`F1 = ${F1} filters of ${k.temporal.weights[0].length} samples (${k.temporal.kernel_ms.toFixed(0)} ms). The frequency response is the FFT of each learned kernel.`} actions={<div className="w-40"><NumberField label={`Filter (0–${F1 - 1})`} value={f} min={0} max={F1 - 1} onChange={(v) => setF(Math.max(0, Math.min(F1 - 1, v ?? 0)))} /></div>} />
        <CardBody className="grid gap-4 xl:grid-cols-2">
          <div>
            <div className="mb-1 text-[13px] font-bold text-ink-muted">Kernel {f} weights</div>
            <LinePlot x={msAxis} series={[{ name: `kernel ${f}`, data: k.temporal.weights[f] }]} legend={false} xName="ms" height={220} ariaLabel={`Temporal kernel ${f}`} />
          </div>
          <div>
            <div className="mb-1 text-[13px] font-bold text-ink-muted">Kernel {f} frequency response (peak {k.temporal.peak_hz[f]} Hz)</div>
            <LinePlot x={k.temporal.freqs} series={[{ name: "gain (dB)", data: k.temporal.response_db[f].slice(0, k.temporal.freqs.length) }]} legend={false} xName="Hz" yName="dB" yMin={-40} yMax={0} height={220} ariaLabel={`Frequency response ${f}`} />
          </div>
          <div className="xl:col-span-2">
            <Heatmap matrix={k.temporal.weights} yLabels={k.temporal.peak_hz.map((p, i) => `k${i} · ${p} Hz`)} xLabels={msAxis} diverging height={Math.max(180, 18 * F1 + 70)} xName="ms" ariaLabel="All temporal kernels" highlightRow={f} />
          </div>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Depthwise spatial filters" subtitle={`${k.depthwise.weights.length} filters (D = ${k.depthwise.D} per temporal filter, groups = ${k.depthwise.groups}); max-norm ${k.depthwise.max_norm ?? "off"}.`} />
        <CardBody>
          <Heatmap matrix={k.depthwise.weights} xLabels={k.depthwise.rows} yLabels={k.depthwise.group_of_output.map((g, o) => `s${o} ← t${g}`)} diverging height={Math.max(220, 16 * k.depthwise.weights.length + 80)} xName="channel" ariaLabel="Depthwise spatial weights" />
        </CardBody>
      </Card>
      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader title="Separable: depthwise temporal kernels" subtitle={`shape ${shape(k.separable_depthwise.shape)}`} />
          <CardBody><Heatmap matrix={k.separable_depthwise.weights} diverging height={Math.max(200, 14 * k.separable_depthwise.weights.length + 80)} xName="tap" ariaLabel="Separable depthwise weights" /></CardBody>
        </Card>
        <Card>
          <CardHeader title="Separable: pointwise mixing matrix" subtitle={`shape ${shape(k.pointwise.shape)} (output map × input map)`} />
          <CardBody><Heatmap matrix={k.pointwise.weights} diverging height={Math.max(200, 14 * k.pointwise.weights.length + 80)} xName="input map" yName="output map" ariaLabel="Pointwise weights" /></CardBody>
        </Card>
      </div>
      <Card>
        <CardHeader title="Classifier weights" subtitle={`shape ${shape(k.classifier.shape)}; features are the flattened [${k.classifier.feature_shape.join(", ")}] tensor.`} />
        <CardBody className="space-y-3">
          <Heatmap matrix={k.classifier.weights} yLabels={k.classifier.class_names} diverging height={Math.max(160, 26 * k.classifier.weights.length + 80)} xName="feature index" ariaLabel="Classifier weights" />
          <p className="text-[13px] text-ink-muted">Biases: {k.classifier.class_names.map((c, i) => `${c} ${k.classifier.bias[i].toFixed(4)}`).join(" · ")}</p>
        </CardBody>
      </Card>
      {Object.keys(k.batchnorm).length > 0 && (
        <Card>
          <CardHeader title="Batch normalisation parameters" subtitle="Learned scale γ / shift β and running statistics used in evaluation mode." />
          <CardBody className="grid gap-4 lg:grid-cols-3">
            {Object.entries(k.batchnorm).map(([name, p]) => (
              <div key={name}>
                <div className="mb-1 text-[13px] font-bold text-ink-muted">{name}</div>
                <Bars labels={p.weight.map((_, i) => i)} series={[{ name: "γ", data: p.weight }, { name: "β", data: p.bias }]} height={180} ariaLabel={`${name} parameters`} />
              </div>
            ))}
          </CardBody>
        </Card>
      )}
    </div>
  );
}

function GroupMap({ F1, D, selected }: { F1: number; D: number; selected: number }) {
  const FD = F1 * D;
  const h = Math.max(F1, FD) * 22 + 20;
  const yIn = (i: number) => 10 + ((i + 0.5) * (h - 20)) / F1;
  const yOut = (o: number) => 10 + ((o + 0.5) * (h - 20)) / FD;
  return (
    <svg viewBox={`0 0 320 ${h}`} className="w-full max-w-sm" role="img" aria-label="Depthwise group mapping">
      {Array.from({ length: FD }, (_, o) => (
        <line key={`l${o}`} x1={70} y1={yIn(Math.floor(o / D))} x2={250} y2={yOut(o)} className={o === selected ? "stroke-primary" : "stroke-line-strong"} strokeWidth={o === selected ? 2.5 : 1} />
      ))}
      {Array.from({ length: F1 }, (_, i) => (
        <g key={`i${i}`}>
          <circle cx={60} cy={yIn(i)} r={8} className={Math.floor(selected / D) === i ? "fill-primary" : "fill-primary-soft stroke-line-strong"} />
          <text x={40} y={yIn(i) + 4} textAnchor="end" className="fill-ink-subtle" style={{ fontSize: 10 }}>t{i}</text>
        </g>
      ))}
      {Array.from({ length: FD }, (_, o) => (
        <g key={`o${o}`}>
          <circle cx={260} cy={yOut(o)} r={7} className={o === selected ? "fill-primary" : "fill-card stroke-line-strong"} />
          <text x={278} y={yOut(o) + 4} className="fill-ink-subtle" style={{ fontSize: 10 }}>s{o}</text>
        </g>
      ))}
    </svg>
  );
}

function ConvolutionArena({ exp, fold, trial }: { exp: Experiment; fold: number; trial: string }) {
  const arch = exp.active_run!.snapshot.architecture_config;
  const FD = arch.F1 * arch.D;
  const [map, setMap] = useState(0);
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState<"same" | "valid">(arch.padding);
  const { data: a, error } = useQuery({
    queryKey: ["arena", exp.id, fold, trial, map, stride, padding],
    queryFn: () => api.post<ArenaResult>(`/api/experiments/${exp.id}/arena`, { trial_id: trial, fold, map_index: map, demo_stride: stride, demo_padding: padding }),
    placeholderData: (p) => p,
  });
  if (error) return <ErrorState error={error} />;
  if (!a) return <Loading />;
  const dw = a.depthwise;
  const sp = a.separable;
  const idx = (n: number) => Array.from({ length: n }, (_, i) => i);
  const k2 = sp.depthwise.kernel[1];
  return (
    <div className="space-y-6">
      <Alert tone="info" title="Depthwise vs separable convolution in this trained EEGNet">
        Both operations below are the model's own layers with their learned weights and the real activations of trial <Mono>{trial}</Mono>. EEGNet uses a depthwise convolution for spatial filtering and a separable convolution (depthwise temporal + pointwise mixing) in block 2.
      </Alert>
      <div className="w-64"><NumberField label={`Feature map j (0–${FD - 1})`} value={map} min={0} max={FD - 1} onChange={(v) => setMap(Math.max(0, Math.min(FD - 1, v ?? 0)))} /></div>
      <Card>
        <CardHeader title="Depthwise spatial convolution" subtitle={dw.explanation} actions={<Badge tone="primary">{fmtInt(dw.params)} params</Badge>} />
        <CardBody className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
          <div>
            <div className="mb-2 text-[13px] font-bold text-ink-muted">Group mapping (F1 = {dw.groups} → F1·D = {dw.mapping.length})</div>
            <GroupMap F1={dw.groups} D={dw.D} selected={map} />
            <KeyValue items={[["Kernel", <Mono key="k">{dw.kernel.join("×")}</Mono>], ["Shape", <Mono key="s">{shape(dw.in_shape)} → {shape(dw.out_shape)}</Mono>], ["Parameters", `${dw.mapping.length} × ${dw.kernel[0]} = ${fmtInt(dw.params)}`]]} />
          </div>
          <div className="space-y-4">
            <div>
              <div className="mb-1 text-[13px] font-bold text-ink-muted">Spatial weights of filter s{map} (reads temporal map t{Math.floor(map / dw.D)})</div>
              <Bars labels={dw.ch_names.slice(0, dw.weights[map].length)} series={[{ name: "weight", data: dw.weights[map] }]} height={190} ariaLabel="Selected spatial filter" valueDigits={4} />
            </div>
            <div className="grid gap-4 2xl:grid-cols-2">
              <div>
                <div className="mb-1 text-[13px] font-bold text-ink-muted">Input: temporal map t{Math.floor(map / dw.D)} (channels × time)</div>
                <Heatmap matrix={dw.selected_input_map} yLabels={dw.selected_input_map.length === dw.ch_names.length ? dw.ch_names : undefined} diverging height={Math.max(200, 12 * dw.selected_input_map.length + 70)} xName="time" ariaLabel="Depthwise input map" />
              </div>
              <div>
                <div className="mb-1 text-[13px] font-bold text-ink-muted">Output: spatially filtered map s{map}</div>
                {dw.selected_output.length === 1 ? (
                  <LinePlot x={idx(dw.selected_output[0].length)} series={[{ name: `s${map}`, data: dw.selected_output[0] }]} legend={false} xName="time" height={200} ariaLabel="Depthwise output" />
                ) : (
                  <Heatmap matrix={dw.selected_output} diverging height={200} xName="time" ariaLabel="Depthwise output" />
                )}
              </div>
            </div>
          </div>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Separable convolution" subtitle={sp.explanation} actions={<Badge tone="primary">{fmtInt(sp.total_params)} params</Badge>} />
        <CardBody className="space-y-6">
          <div className="flex flex-wrap items-center gap-2 font-mono text-[13px] text-ink">
            <span className="rounded-lg bg-subtle px-2 py-1">{shape(sp.depthwise.in_shape)}</span>→
            <span className="rounded-lg bg-primary-soft px-2 py-1 text-link">depthwise 1×{k2}, groups {sp.depthwise.groups} · {fmtInt(sp.depthwise.params)} params</span>→
            <span className="rounded-lg bg-subtle px-2 py-1">{shape(sp.depthwise.out_shape)}</span>→
            <span className="rounded-lg bg-primary-soft px-2 py-1 text-link">pointwise 1×1 · {fmtInt(sp.pointwise.params)} params</span>→
            <span className="rounded-lg bg-subtle px-2 py-1">{shape(sp.pointwise.out_shape)}</span>
          </div>
          <div className="grid gap-4 xl:grid-cols-3">
            <div>
              <div className="mb-1 text-[13px] font-bold text-ink-muted">Step 1 · kernel of map {map}</div>
              <Bars labels={idx(k2)} series={[{ name: "weight", data: sp.depthwise.weights[map] }]} height={190} ariaLabel="Separable depthwise kernel" valueDigits={4} />
            </div>
            <div className="xl:col-span-2">
              <div className="mb-1 text-[13px] font-bold text-ink-muted">Step 1 · map {map} before and after temporal filtering</div>
              <LinePlot x={idx(sp.depthwise.selected_input.length)} series={[{ name: "input", data: sp.depthwise.selected_input }, { name: "depthwise output", data: sp.depthwise.selected_output }]} xName="time step" height={190} ariaLabel="Separable depthwise input and output" />
            </div>
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            <div>
              <div className="mb-1 text-[13px] font-bold text-ink-muted">Step 2 · pointwise weights (F2 × F1·D); column {map} is highlighted</div>
              <Heatmap matrix={sp.pointwise.weights} diverging highlightCol={map} height={Math.max(200, 14 * sp.pointwise.weights.length + 80)} xName="input map" yName="output map" ariaLabel="Pointwise weights" />
            </div>
            <div>
              <div className="mb-1 text-[13px] font-bold text-ink-muted">Step 2 · mixed outputs (F2 maps × time)</div>
              <Heatmap matrix={sp.pointwise.outputs} diverging height={Math.max(200, 14 * sp.pointwise.outputs.length + 80)} xName="time step" ariaLabel="Pointwise outputs" />
            </div>
          </div>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Stride and padding" subtitle="Isolated educational demonstration: the learned separable kernel applied with a different stride/padding. This is not the trained model's computation." actions={<Badge tone="warning">demonstration</Badge>} />
        <CardBody className="space-y-4">
          <div className="flex flex-wrap items-end gap-4">
            <div className="w-40"><NumberField label="Stride" value={stride} min={1} max={8} onChange={(v) => setStride(Math.max(1, Math.min(8, v ?? 1)))} /></div>
            <Segmented ariaLabel="Padding" value={padding} onChange={setPadding} options={[{ value: "same", label: "same" }, { value: "valid", label: "valid" }]} />
          </div>
          {a.demo ? (
            <>
              <p className="font-mono text-[13.5px] text-ink">output length = {a.demo.formula} (padding {a.demo.pad[0]} | {a.demo.pad[1]})</p>
              <LinePlot x={idx(a.demo.output.length).map((i) => i * a.demo!.stride)} series={[{ name: `stride ${a.demo.stride}, ${a.demo.padding}`, data: a.demo.output }]} xName="input position" height={200} ariaLabel="Stride demonstration" />
            </>
          ) : (
            <Alert tone="warning">The kernel is longer than the input with valid padding; no output.</Alert>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
