import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { api, errorText } from "../api/client";
import { useArchitecture, useCurrentExperiment } from "../api/hooks";
import type { ArchitectureConfig, ArchitectureReport, CoordinateMapping, LayerSpec } from "../api/types";
import { ArchForm, presetValues } from "../components/arch/ArchForm";
import { ReportSummary, useValidation } from "./architecture/ConfigTab";
import { Card, CardBody, CardHeader } from "../components/ui/Card";
import { Badge } from "../components/ui/Chip";
import { NumberField, Segmented, SelectField } from "../components/ui/Field";
import { Alert, KeyValue, Loading, Mono, PageHeader } from "../components/ui/Misc";
import { classNames, fmtInt, shape } from "../lib/format";

type Source = "experiment" | "trained" | "custom";

const DEFAULT_CUSTOM: ArchitectureConfig = {
  preset: "eegnet-8,2", n_channels: 22, n_times: 256, n_classes: 4, sfreq: 128, F1: 8, D: 2, F2: 16, temporal_kernel: 64,
  temporal_stride: 1, spatial_kernel: null, pool1_kernel: 4, pool1_stride: null, separable_kernel: 16, pool2_kernel: 8,
  pool2_stride: null, pool_type: "avg", padding: "same", dropout: 0.5, dropout_type: "dropout", activation: "elu",
  batch_norm: true, bn_momentum: 0.01, bn_eps: 0.001, depthwise_max_norm: 1, classifier_max_norm: 0.25,
};

function formula(s: LayerSpec): string {
  const inW = s.in_shape[s.in_shape.length - 1];
  const outW = s.out_shape[s.out_shape.length - 1];
  if (s.name === "depthwise_conv") return `H = C − Ks + 1 = ${s.in_shape[1]} − ${s.kernel?.[0]} + 1 = ${s.out_shape[1]}`;
  if (s.kernel && s.stride && s.padding && (s.name.includes("conv") || s.name.startsWith("separable")) && s.kernel[1] > 1) {
    return `W = ⌊(${inW} + ${s.padding[0]} + ${s.padding[1]} − ${s.kernel[1]}) / ${s.stride[1]}⌋ + 1 = ${outW}`;
  }
  if (s.name.startsWith("pool") && s.kernel && s.stride) return `W = ⌊(${inW} − ${s.kernel[1]}) / ${s.stride[1]}⌋ + 1 = ${outW}`;
  if (s.name === "separable_pointwise") return `channels ${s.in_shape[0]} → ${s.out_shape[0]} (1×1 mixing), W unchanged`;
  if (s.name === "flatten") return `${s.in_shape.join(" × ")} = ${s.out_shape[0]} features`;
  if (s.name === "classifier") return `${s.in_shape[0]} features × ${s.out_shape[0]} classes + ${s.out_shape[0]} biases`;
  return "element-wise: shape unchanged";
}

export function TensorDetectivePage() {
  const { id: current, data: exp } = useCurrentExperiment();
  const { data: archState } = useArchitecture(exp?.id, !!exp?.preprocessing_summary);
  const dims = archState?.data_dims;
  const trainedCfg = exp?.capabilities.can_inspect_learned ? exp.active_run?.snapshot.architecture_config ?? null : null;
  const [source, setSource] = useState<Source>("experiment");
  const [custom, setCustom] = useState<ArchitectureConfig>(DEFAULT_CUSTOM);
  useEffect(() => {
    if (exp === undefined && current) return; // still loading
    if (source === "experiment" && !exp?.architecture) setSource(trainedCfg ? "trained" : "custom");
  }, [exp, current, source, trainedCfg]);
  const cfg = source === "trained" ? trainedCfg : source === "experiment" ? exp?.architecture ?? null : custom;
  const { data: report, error } = useValidation(cfg);

  return (
    <>
      <PageHeader
        eyebrow="Separate area"
        title="Tensor"
        accent="detective."
        lead="Follow the exact shape of the data through every layer, map any output coordinate back to the input samples it depends on, and see how flattening indexes the classifier input. Shapes come from verified shape propagation, cross-checked against a real PyTorch forward pass."
      />
      <Card className="mb-6">
        <CardBody className="flex flex-wrap items-center gap-4 pt-5">
          <Segmented<Source>
            ariaLabel="Architecture source"
            value={source}
            onChange={setSource}
            options={[
              { value: "experiment", label: "Current experiment", disabled: !exp?.architecture },
              { value: "trained", label: "Trained checkpoint", disabled: !trainedCfg },
              { value: "custom", label: "Custom configuration" },
            ]}
          />
          {source === "trained" && exp?.active_run && (
            <span className="text-[13.5px] text-ink-muted">
              Checkpoint <Mono>{exp.active_run.folds[0]?.checkpoint_id}</Mono> · architecture <Mono>{exp.active_run.snapshot.architecture_fingerprint}</Mono> (v{exp.active_run.snapshot.architecture_version})
            </span>
          )}
          {source === "experiment" && exp && <span className="text-[13.5px] text-ink-muted">{exp.name} · pre-training validation</span>}
          {source !== "custom" && dims && (
            <span className="text-[13.5px] text-ink-muted">
              Data: {dims.datasets?.join(" + ")} → [1, {dims.n_channels}, {dims.n_times}] at {dims.sfreq} Hz ({dims.epoch_s?.toFixed(2)} s {dims.epoching_mode === "windows" ? "windows" : "epochs"}), {dims.n_classes} classes
            </span>
          )}
        </CardBody>
      </Card>
      <div className={classNames("grid gap-6", source === "custom" && "xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]")}>
        {source === "custom" && (
          <Card>
            <CardHeader title="Custom configuration" subtitle="Any input size and architecture; no data or training needed." actions={<button type="button" className="text-[13px] font-bold text-link" onClick={() => setCustom({ ...DEFAULT_CUSTOM, ...presetValues("eegnet-8,2", custom.sfreq) })}>reset</button>} />
            <CardBody><ArchForm value={custom} onChange={setCustom} editableDims /></CardBody>
          </Card>
        )}
        <div className="min-w-0 space-y-6">
          {error && <Alert tone="danger">{errorText(error)}</Alert>}
          {!cfg ? <Alert tone="info">Select a source.</Alert> : !report ? <Loading /> : <DetectiveBody report={report} />}
        </div>
      </div>
    </>
  );
}

function DetectiveBody({ report }: { report: ArchitectureReport }) {
  const C = report.input_shape[1];
  const T = report.input_shape[2];
  return (
    <>
      <ReportSummary report={report} />
      {!report.valid && report.issues.filter((i) => i.level === "error").map((i) => <Alert key={i.message} tone="danger" title={i.message}>{i.fix}</Alert>)}
      <Card>
        <CardHeader title="Step-by-step shape trace" subtitle={`Convention ${report.input_convention}; shapes below exclude the batch dimension.`} />
        <CardBody>
          <ol className="relative space-y-0 border-l-2 border-line pl-6">
            <li className="relative pb-4">
              <span className="absolute top-1 -left-[31px] size-3 rounded-full bg-primary" />
              <div className="text-[14.5px] font-bold text-ink">Input</div>
              <Mono className="text-link">[1, {C}, {T}]</Mono> <span className="text-[13px] text-ink-subtle">· {fmtInt(C * T)} elements</span>
            </li>
            {report.layers.map((s) => (
              <li key={s.name} className="relative pb-4">
                <span className={classNames("absolute top-1 -left-[31px] size-3 rounded-full", s.valid ? "bg-line-strong" : "bg-danger")} />
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[14.5px] font-bold text-ink">{s.label}</span>
                  <Badge>{s.type}</Badge>
                  {s.params > 0 && <span className="text-[12.5px] text-ink-subtle">{fmtInt(s.params)} params</span>}
                </div>
                <div className="mt-0.5 font-mono text-[13px] text-ink">
                  {shape(s.in_shape)} → <span className={s.valid ? "text-link" : "text-danger"}>{shape(s.out_shape)}</span>
                  <span className="ml-2 text-ink-subtle">({fmtInt(s.out_shape.reduce((a, b) => a * b, 1))} elements)</span>
                </div>
                <div className="mt-0.5 text-[12.5px] text-ink-muted">
                  {formula(s)}
                  {s.kernel && <> · kernel {s.kernel.join("×")}, stride {s.stride?.join("×")}, padding {s.padding?.join("|")}, dilation {s.dilation?.join("×") ?? "1×1"}, groups {s.groups ?? 1}</>}
                </div>
                {s.message && <div className="mt-1 text-[13px] font-bold text-danger">{s.message}</div>}
              </li>
            ))}
          </ol>
        </CardBody>
      </Card>
      {report.valid && <CoordinateExplorer report={report} />}
    </>
  );
}

function InputGrid({ C, T, ch, tr, padded }: { C: number; T: number; ch: number[]; tr: number[]; padded: number }) {
  const w = 600;
  const h = Math.min(220, Math.max(80, C * 9));
  const x = (t: number) => (t / T) * w;
  const y = (c: number) => (c / C) * h;
  return (
    <svg viewBox={`-4 -4 ${w + 8} ${h + 30}`} className="w-full" role="img" aria-label="Input region the coordinate depends on">
      <rect x={0} y={0} width={w} height={h} className="fill-subtle stroke-line-strong" />
      {Array.from({ length: Math.min(C, 64) }, (_, c) => <line key={c} x1={0} x2={w} y1={y(c)} y2={y(c)} className="stroke-line" strokeWidth={0.5} />)}
      <rect x={x(tr[0])} y={y(ch[0])} width={Math.max(2, x(tr[1] + 1) - x(tr[0]))} height={Math.max(2, y(ch[1] + 1) - y(ch[0]))} className="fill-primary stroke-primary" fillOpacity={0.35} />
      <text x={0} y={h + 18} className="fill-ink-subtle" style={{ fontSize: 12 }}>sample 0</text>
      <text x={w} y={h + 18} textAnchor="end" className="fill-ink-subtle" style={{ fontSize: 12 }}>sample {T - 1}</text>
      <text x={w / 2} y={h + 18} textAnchor="middle" className="fill-link" style={{ fontSize: 12, fontWeight: 700 }}>
        samples {tr[0]}–{tr[1]} · channels {ch[0]}–{ch[1]}{padded ? ` · ${padded} padded` : ""}
      </text>
    </svg>
  );
}

function CoordinateExplorer({ report }: { report: ArchitectureReport }) {
  const layers = report.layers.filter((l) => l.out_shape.length === 3 || l.name === "flatten" || l.name === "classifier");
  const [layer, setLayer] = useState("pool2");
  const spec = report.layers.find((l) => l.name === layer) ?? layers[0];
  const [f, setF] = useState(0);
  const [hh, setH] = useState(0);
  const [w, setW] = useState(0);
  const [flat, setFlat] = useState(0);
  const is3 = spec.out_shape.length === 3;
  const coord = is3 ? [Math.min(f, spec.out_shape[0] - 1), Math.min(hh, spec.out_shape[1] - 1), Math.min(w, spec.out_shape[2] - 1)] : [Math.min(flat, spec.out_shape[0] - 1)];
  const { data: m, error } = useQuery({
    queryKey: ["coord", report.fingerprint, report.config.n_channels, report.config.n_times, layer, coord.join(",")],
    queryFn: () => api.post<CoordinateMapping>("/api/models/coordinate", { config: report.config, layer, coord }),
    placeholderData: (p) => p,
  });
  const flatSpec = report.layers.find((l) => l.name === "flatten")!;
  const [Fm, H, W] = flatSpec.in_shape;
  const sample = useMemo(() => Array.from({ length: Math.min(12, Fm * H * W) }, (_, i) => Math.floor((i * (Fm * H * W - 1)) / 11)), [Fm, H, W]);
  return (
    <Card>
      <CardHeader icon={<Search className="size-5" />} title="Coordinate detective" subtitle="Pick an output element; see which input samples and channels determine it (receptive field) and where it lands in the classifier input." />
      <CardBody className="space-y-5">
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <div className="sm:col-span-2"><SelectField label="Layer" value={layer} onChange={setLayer} options={layers.map((l) => ({ value: l.name, label: `${l.label} · ${shape(l.out_shape)}` }))} /></div>
          {is3 ? (
            <>
              <NumberField label={`map f (0–${spec.out_shape[0] - 1})`} value={coord[0]} min={0} max={spec.out_shape[0] - 1} onChange={(v) => setF(Math.max(0, v ?? 0))} />
              <NumberField label={`row h (0–${spec.out_shape[1] - 1})`} value={coord[1]} min={0} max={spec.out_shape[1] - 1} onChange={(v) => setH(Math.max(0, v ?? 0))} />
              <NumberField label={`time w (0–${spec.out_shape[2] - 1})`} value={coord[2]} min={0} max={spec.out_shape[2] - 1} onChange={(v) => setW(Math.max(0, v ?? 0))} />
            </>
          ) : (
            <NumberField label={`${layer === "classifier" ? "class" : "flat index"} (0–${spec.out_shape[0] - 1})`} value={coord[0]} min={0} max={spec.out_shape[0] - 1} onChange={(v) => setFlat(Math.max(0, v ?? 0))} />
          )}
        </div>
        {error && <Alert tone="danger">{errorText(error)}</Alert>}
        {m && (
          <>
            <InputGrid C={report.input_shape[1]} T={report.input_shape[2]} ch={m.channel_range} tr={m.time_range} padded={m.padded_samples ?? 0} />
            <KeyValue items={[
              ["Depends on input samples", <Mono key="t">{m.time_range[0]} … {m.time_range[1]}{m.time_range_unclipped && (m.time_range_unclipped[0] < 0 || m.time_range_unclipped[1] >= report.input_shape[2]) ? ` (unclipped ${m.time_range_unclipped[0]} … ${m.time_range_unclipped[1]})` : ""}</Mono>],
              ["Channels", <Mono key="c">{m.channel_range[0]} … {m.channel_range[1]}</Mono>],
              ["Receptive field", m.receptive_field_samples ? `${m.receptive_field_samples} samples${report.config.sfreq ? ` (${((m.receptive_field_samples / report.config.sfreq) * 1000).toFixed(0)} ms)` : ""}; one step = ${m.jump} samples` : m.depends_on ?? "—"],
              ["Flattened index", m.flat_index !== null && m.flat_index !== undefined ? <Mono key="f">{m.flat_index}{m.unflattened ? ` = [${m.unflattened.join(", ")}]` : ""}</Mono> : "— (before the final pooling)"],
            ]} />
          </>
        )}
        <div>
          <div className="mb-2 text-[13px] font-bold text-ink-muted">Flattening map: classifier input index ↔ [f, h, w] of the [{Fm}, {H}, {W}] feature tensor (index = f·{H * W} + h·{W} + w)</div>
          <div className="flex flex-wrap gap-2">
            {sample.map((i) => {
              const ff = Math.floor(i / (H * W));
              const r = i % (H * W);
              return <span key={i} className="rounded-lg border border-line bg-surface px-2 py-1 font-mono text-[12px]">{i} ↔ [{ff}, {Math.floor(r / W)}, {r % W}]</span>;
            })}
          </div>
          <p className="mt-2 text-[12.5px] text-ink-subtle">Classifier input dimension: {fmtInt(report.classifier_in_features)} = F2 {Fm} × H {H} × W {W}.</p>
        </div>
      </CardBody>
    </Card>
  );
}
