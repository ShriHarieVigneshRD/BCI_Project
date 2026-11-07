import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, BookmarkPlus, CheckCircle2, Copy, Lock, RotateCcw, Save } from "lucide-react";
import { api, errorText } from "../../api/client";
import { useArchitecture, useRefreshExperiment } from "../../api/hooks";
import type { ArchitectureConfig, ArchitectureReport, ArchitectureVersion, Experiment } from "../../api/types";
import { ArchForm } from "../../components/arch/ArchForm";
import { ArchitectureDiagram } from "../../components/arch/ArchitectureDiagram";
import { LayerTable } from "../../components/arch/LayerTable";
import { Button } from "../../components/ui/Button";
import { Card, CardBody, CardHeader, Stat } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Chip";
import { TextField } from "../../components/ui/Field";
import { Alert, KeyValue, Loading, Mono } from "../../components/ui/Misc";
import { bytes, dateTime, fmtInt, shape, stableEqual } from "../../lib/format";

export function useValidation(cfg: ArchitectureConfig | null) {
  const [debounced, setDebounced] = useState(cfg);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(cfg), 350);
    return () => clearTimeout(t);
  }, [cfg]);
  return useQuery({
    queryKey: ["validate", JSON.stringify(debounced)],
    queryFn: () => api.post<ArchitectureReport>("/api/models/validate", { config: debounced }),
    enabled: !!debounced,
    placeholderData: (p) => p,
    retry: false,
  });
}

export function ReportSummary({ report }: { report: ArchitectureReport }) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
      <Stat label="Parameters" value={fmtInt(report.total_params)} hint={`${fmtInt(report.trainable_params)} trainable · ${fmtInt(report.buffers)} buffers`} />
      <Stat label="MACs / trial" value={fmtInt(report.total_macs)} hint="conv + dense" />
      <Stat label="Activations" value={bytes(report.activation_bytes_per_sample)} hint="per trial, float32" />
      <Stat label="Receptive field" value={report.receptive_field_samples ?? "—"} hint={report.receptive_field_seconds ? `${(report.receptive_field_seconds * 1000).toFixed(0)} ms of signal` : "samples"} />
      <Stat label="Classifier input" value={fmtInt(report.classifier_in_features)} hint="flattened features" />
      <Stat label="Torch check" value={report.torch_verified ? "verified" : "—"} tone={report.torch_verified ? "success" : undefined} hint={report.torch_verified ? "shapes match a forward pass" : (report.torch_message ?? "")} />
    </div>
  );
}

export function LayerDetail({ report, name }: { report: ArchitectureReport; name: string | null }) {
  const s = report.layers.find((l) => l.name === name);
  if (!s) return <p className="text-[14px] text-ink-subtle">Select a layer in the diagram or table to see its details.</p>;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="text-[17px] font-bold text-ink">{s.label}</h4>
        <Badge>{s.type}</Badge>
        <Badge tone="primary">{s.block}</Badge>
        {!s.valid && <Badge tone="danger">invalid</Badge>}
      </div>
      <p className="text-[14px] leading-relaxed text-ink-muted">{s.purpose}</p>
      {s.message && <Alert tone="danger">{s.message}</Alert>}
      <KeyValue items={[
        ["Input → output", <Mono key="io">{shape(s.in_shape)} → {shape(s.out_shape)}</Mono>],
        ["Kernel / stride", <Mono key="k">{s.kernel ? s.kernel.join("×") : "—"} / {s.stride ? s.stride.join("×") : "—"}</Mono>],
        ["Padding (left | right)", <Mono key="p">{s.padding ? `${s.padding[0]} | ${s.padding[1]}` : "—"}</Mono>],
        ["Groups / dilation", `${s.groups ?? "—"} / ${s.dilation ? s.dilation.join("×") : "—"}`],
        ["Parameters", `${fmtInt(s.params)} (${fmtInt(s.trainable)} trainable${s.buffers ? `, ${s.buffers} running statistics` : ""})`],
        ["Weight tensors", Object.entries(s.param_shapes).map(([k, v]) => `${k} ${shape(v)}`).join(", ") || "none"],
        ["Receptive field", `${s.rf_time} input samples × ${s.rf_channels} channel(s); time step ${s.jump_time}`],
      ]} />
    </div>
  );
}

export function ConfigTab({ exp }: { exp: Experiment }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const refresh = useRefreshExperiment();
  const { data: arch, isLoading } = useArchitecture(exp.id);
  const [cfg, setCfg] = useState<ArchitectureConfig | null>(null);
  const [selected, setSelected] = useState<string | null>("temporal_conv");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [libName, setLibName] = useState("");
  const { data: library } = useQuery({ queryKey: ["arch-library"], queryFn: () => api.get<ArchitectureVersion[]>("/api/models/configs") });

  useEffect(() => {
    if (arch?.config && !cfg) setCfg(arch.config);
  }, [arch, cfg]);
  const validation = useValidation(cfg);
  const report = validation.data;
  const current = useMemo(() => arch?.versions.find((v) => v.id === arch.current_version_id) ?? null, [arch]);

  if (isLoading || !arch || !cfg) return <Loading label="Loading architecture…" />;
  const dims = arch.data_dims;
  const dimsMismatch = dims && (cfg.n_channels !== dims.n_channels || cfg.n_times !== dims.n_times || cfg.n_classes !== dims.n_classes);
  const savedSame = arch.confirmed && current && stableEqual({ ...current.config, preset: "" }, { ...cfg, preset: "" });
  const willVersion = current?.locked && !savedSame;

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.put(`/api/experiments/${exp.id}/architecture`, { config: cfg, name: current?.name ?? "Architecture" });
      refresh(exp.id);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  const saveToLibrary = async () => {
    try {
      await api.post("/api/models/configs", { name: libName || `${cfg.preset} ${new Date().toLocaleDateString()}`, config: cfg });
      setLibName("");
      qc.invalidateQueries({ queryKey: ["arch-library"] });
    } catch (e) {
      setError(errorText(e));
    }
  };
  const duplicateLib = async (id: string) => {
    await api.post(`/api/models/configs/${id}/duplicate`);
    qc.invalidateQueries({ queryKey: ["arch-library"] });
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title="Architecture diagram"
          subtitle={<>Rendered from the shared architecture specification; every shape is cross-checked against a real PyTorch forward pass. Input convention <Mono>{report?.input_convention ?? "[batch, 1, channels, time_samples]"}</Mono>.</>}
          actions={report && <Badge tone={report.valid ? "success" : "danger"}>{report.valid ? "valid" : "invalid"}</Badge>}
        />
        <CardBody>
          {report ? <ArchitectureDiagram report={report} selected={selected} onSelect={setSelected} /> : <Loading />}
        </CardBody>
      </Card>
      {report && <ReportSummary report={report} />}
      <div className="grid gap-6 xl:grid-cols-[minmax(0,440px)_minmax(0,1fr)]">
        <div className="space-y-6">
          <Card>
            <CardHeader title="Configure" subtitle="Structural parameters change the graph; training-time settings only affect optimisation. No setting is claimed to improve accuracy without a trained comparison." />
            <CardBody>
              <ArchForm value={cfg} onChange={setCfg} base={arch.confirmed ? current?.config : null} disabled={!exp.capabilities.can_configure_architecture} />
            </CardBody>
          </Card>
          {dims && (
            <Alert tone="info" title="Input derived from the processed dataset">
              {dims.datasets?.join(" + ")}: {dims.n_epochs} {dims.epoching_mode === "windows" ? "windows" : "epochs"} of {dims.n_channels} channels × {dims.n_times} samples
              ({dims.epoch_s?.toFixed(2)} s at {dims.sfreq} Hz), {dims.n_classes} classes ({dims.classes.join(", ")}).
            </Alert>
          )}
          {dimsMismatch && <Alert tone="danger" title="Input does not match the data">The preprocessed data are [{dims!.n_channels} channels × {dims!.n_times} samples], {dims!.n_classes} classes.</Alert>}
          {willVersion && <Alert tone="warning" title="Saving creates a new version">The current version was used by a training run and is immutable. Saving creates v{(current?.version ?? 1) + 1} and marks the earlier results stale.</Alert>}
          {error && <Alert tone="danger">{error}</Alert>}
          <div className="flex flex-wrap gap-3">
            <Button icon={savedSame ? <CheckCircle2 className="size-4" /> : <Save className="size-4" />} onClick={save} loading={busy} disabled={!report?.valid || !!savedSame || !!dimsMismatch || !exp.capabilities.can_configure_architecture}>
              {savedSame ? "Architecture saved" : "Validate & save architecture"}
            </Button>
            <Button variant="secondary" icon={<RotateCcw className="size-4" />} onClick={() => setCfg(current?.config ?? arch.default ?? cfg)}>Reset</Button>
          </div>
          {savedSame && (
            <div className="flex flex-wrap gap-2">
              <Button variant="ghost" icon={<ArrowRight className="size-4" />} onClick={() => navigate("/architecture/journey")}>Follow a trial (Signal Journey)</Button>
              <Button variant="ghost" icon={<ArrowRight className="size-4" />} onClick={() => navigate("/architecture/training")}>Go to training</Button>
            </div>
          )}
        </div>
        <div className="min-w-0 space-y-6">
          <Card>
            <CardHeader title="Selected layer" />
            <CardBody>{report && <LayerDetail report={report} name={selected} />}</CardBody>
          </Card>
          <Card>
            <CardHeader title="Layer table" subtitle="Click a row to select it in the diagram." />
            <CardBody>{report && <LayerTable report={report} selected={selected} onSelect={setSelected} />}</CardBody>
          </Card>
          {report && report.issues.length > 0 && (
            <Card>
              <CardHeader title="Validation messages" />
              <CardBody className="space-y-2">
                {report.issues.map((i) => (
                  <Alert key={i.code + i.message} tone={i.level === "error" ? "danger" : i.level === "warning" ? "warning" : "info"} title={i.message}>
                    {i.fix}
                  </Alert>
                ))}
              </CardBody>
            </Card>
          )}
          <Card>
            <CardHeader title="Versions & saved configurations" subtitle="Trained versions are locked; edits after training create a new version." />
            <CardBody className="space-y-4">
              <ul className="divide-y divide-line rounded-xl border border-line">
                {arch.versions.map((v) => (
                  <li key={v.id} className="flex flex-wrap items-center gap-3 px-4 py-2.5 text-[14px]">
                    <span className="font-bold text-ink">v{v.version}</span>
                    <span className="text-ink-muted">{v.name}</span>
                    {v.id === arch.current_version_id && <Badge tone="primary">current</Badge>}
                    {v.locked && <Badge icon={<Lock className="size-3" />}>trained · immutable</Badge>}
                    <Mono className="text-ink-subtle">{v.fingerprint}</Mono>
                    <span className="ml-auto text-[12.5px] text-ink-subtle">{dateTime(v.created_at)}</span>
                    {v.id !== arch.current_version_id && <Button size="sm" variant="ghost" onClick={() => setCfg(v.config)}>Restore</Button>}
                  </li>
                ))}
                {!arch.versions.length && <li className="px-4 py-3 text-[14px] text-ink-subtle">Not saved yet.</li>}
              </ul>
              <div className="flex flex-wrap items-end gap-3">
                <div className="min-w-56 flex-1"><TextField label="Save current settings to the library as" value={libName} onChange={setLibName} placeholder="name" /></div>
                <Button variant="secondary" icon={<BookmarkPlus className="size-4" />} onClick={saveToLibrary} disabled={!report?.valid}>Save to library</Button>
              </div>
              {!!library?.length && (
                <ul className="divide-y divide-line rounded-xl border border-line">
                  {library.map((v) => (
                    <li key={v.id} className="flex flex-wrap items-center gap-3 px-4 py-2 text-[14px]">
                      <span className="font-bold text-ink">{v.name}</span>
                      <span className="text-ink-subtle">F1 {v.config.F1} · D {v.config.D} · F2 {v.config.F2} · K {v.config.temporal_kernel}</span>
                      <span className="ml-auto flex gap-1">
                        <Button size="sm" variant="ghost" onClick={() => setCfg({ ...v.config, n_channels: cfg.n_channels, n_times: cfg.n_times, n_classes: cfg.n_classes, sfreq: cfg.sfreq })}>Restore</Button>
                        <Button size="sm" variant="ghost" icon={<Copy className="size-3.5" />} onClick={() => duplicateLib(v.id)}>Duplicate</Button>
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </CardBody>
          </Card>
        </div>
      </div>
    </div>
  );
}
