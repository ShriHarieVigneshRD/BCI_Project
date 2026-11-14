import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, ChevronDown, Play, Save, Sparkles } from "lucide-react";
import { api, errorText } from "../api/client";
import { useDataset, useRefreshExperiment } from "../api/hooks";
import type { Experiment, IcaView, PreprocessingConfig, PreprocessingDefaults, PreprocessingResult, PreviewResponse } from "../api/types";
import { RequireExperiment } from "../components/layout/RequireExperiment";
import { LifecycleBanner } from "../components/layout/LifecycleBanner";
import { Button } from "../components/ui/Button";
import { Card, CardBody, CardHeader, Stat } from "../components/ui/Card";
import { Badge } from "../components/ui/Chip";
import { Checkbox, NumberField, Segmented, SelectField, Toggle } from "../components/ui/Field";
import { Alert, Checklist, KeyValue, Loading, LockedState, Mono, PageHeader } from "../components/ui/Misc";
import { Bars, Heatmap, LinePlot, StackedSignals } from "../components/charts/Charts";
import { bytes, classNames, fmtInt, shape, stableEqual } from "../lib/format";

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

/** Keep the baseline if it fits the epoch; otherwise the pre-event part of the epoch, or its first 0.2 s. */
export function baselineInside(b: { tmin: number; tmax: number }, [t0, t1]: [number, number]): { tmin: number; tmax: number } {
  if (b.tmin >= t0 && b.tmax <= t1 && b.tmin < b.tmax) return { tmin: b.tmin, tmax: b.tmax };
  if (t0 < 0 && t1 > 0) return { tmin: t0, tmax: 0 };
  return { tmin: t0, tmax: Math.round((t0 + Math.min(0.2, (t1 - t0) / 2)) * 1000) / 1000 };
}

export function PreprocessingPage() {
  return (
    <>
      <PageHeader
        eyebrow="Step 2 of the workflow"
        title="Preprocessing"
        accent="studio."
        lead="Shape the signal stage by stage and preview every change. Learned stages (ICA, normalisation) are fitted on training data only — and refitted inside every cross-validation fold."
      />
      <RequireExperiment>
        {(exp) =>
          exp.capabilities.can_configure_preprocessing ? (
            <PreprocessingBody exp={exp} />
          ) : (
            <>
              <LifecycleBanner exp={exp} here="preprocessing" />
              <LockedState title="Confirm the dataset and split first" reason="Preprocessing needs a validated dataset and a confirmed evaluation protocol, because learned stages are fitted on the training partition." />
            </>
          )
        }
      </RequireExperiment>
    </>
  );
}

function Stage({ n, title, children, enabled, summary, defaultOpen }: { n: number; title: string; children: ReactNode; enabled?: boolean; summary?: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(!!defaultOpen);
  return (
    <div className="border-t border-line first:border-t-0">
      <button type="button" aria-expanded={open} onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-3 px-5 py-3.5 text-left hover:bg-surface">
        <span className={classNames("grid size-7 shrink-0 place-items-center rounded-full text-[13px] font-bold", enabled === false ? "bg-subtle text-ink-subtle" : "bg-primary-soft text-link")}>{n}</span>
        <span className="min-w-0 flex-1">
          <span className="block text-[15px] font-bold text-ink">{title}</span>
          {summary && <span className="block truncate text-[12.5px] text-ink-subtle">{summary}</span>}
        </span>
        <ChevronDown className={classNames("size-4 text-ink-subtle transition-transform", open && "rotate-180")} />
      </button>
      {open && <div className="space-y-4 px-5 pb-5">{children}</div>}
    </div>
  );
}

function PreprocessingBody({ exp }: { exp: Experiment }) {
  const navigate = useNavigate();
  const refresh = useRefreshExperiment();
  const { data: ds } = useDataset(exp.dataset_id);
  const [cfg, setCfg] = useState<PreprocessingConfig | null>(exp.preprocessing_config);
  const [trialId, setTrialId] = useState<string | null>(null);
  const [channels, setChannels] = useState<string[] | null>(null);
  const [busy, setBusy] = useState<"save" | "run" | "ica" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runError, setRunError] = useState<{ message: string; result?: PreprocessingResult } | null>(null);
  const [ica, setIca] = useState<IcaView | null>(null);
  // Suggested settings derived from the dataset manifest (channel types, metadata, event structure).
  const { data: defaults } = useQuery({
    queryKey: ["prep-defaults", exp.id, exp.dataset_id, exp.test_dataset_id],
    queryFn: () => api.get<PreprocessingDefaults>(`/api/experiments/${exp.id}/preprocessing/defaults`),
  });

  useEffect(() => {
    if (!cfg && ds && (exp.preprocessing_config || defaults)) setCfg(exp.preprocessing_config ?? defaults!.config);
  }, [ds, cfg, exp.preprocessing_config, defaults]);
  useEffect(() => {
    if (exp.ica_reference && !ica) api.get<IcaView>(`/api/experiments/${exp.id}/preprocessing/ica`).then(setIca).catch(() => undefined);
  }, [exp.ica_reference, exp.id, ica]);

  const debounced = useDebounced(cfg, 700);
  const preview = useQuery({
    queryKey: ["prep-preview", exp.id, JSON.stringify(debounced), trialId, channels?.join(",")],
    queryFn: () => api.post<PreviewResponse>(`/api/experiments/${exp.id}/preprocessing/preview`, { config: debounced, trial_id: trialId, channels }),
    enabled: !!debounced,
    placeholderData: (prev) => prev,
  });

  if (!cfg || !ds) return <Loading label="Loading dataset…" />;
  const s = ds.summary;
  const saved = !!exp.preprocessing_config && stableEqual(exp.preprocessing_config, cfg);
  const completed = saved && !!exp.preprocessing_summary && exp.capabilities.can_configure_architecture;
  const set = <K extends keyof PreprocessingConfig>(k: K, patch: Partial<PreprocessingConfig[K]>) => setCfg((c) => (c ? { ...c, [k]: { ...c[k], ...patch } } : c));
  const allCh = s.ch_names;
  const selected = cfg.channels.selected ?? allCh;
  const classes = s.classes;
  // same rule as the backend's _epoch_window: the baseline must lie inside this window
  const ep = cfg.epoching;
  const epochWin: [number, number] = ep.mode === "windows" ? [0, ep.window_s]
    : s.structure === "continuous" || ep.mode === "crop" ? [ep.tmin, ep.tmax]
    : [s.tmin ?? 0, (s.tmin ?? 0) + (s.epoch_duration_s ?? 0)];
  const modeLabel = ep.mode === "windows" ? `Sliding windows: ${ep.window_s} s long, every ${ep.step_s} s within each event/segment`
    : ep.mode === "events" ? `Fixed window: ${ep.tmin} to ${ep.tmax} s from each event (one epoch per event)`
    : ep.mode === "crop" ? `Crop existing epochs to ${ep.tmin}–${ep.tmax} s` : "Keep epochs as provided";

  const save = async () => {
    setBusy("save");
    setError(null);
    try {
      await api.put(`/api/experiments/${exp.id}/preprocessing`, cfg);
      refresh(exp.id);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  };
  const run = async () => {
    setBusy("run");
    setRunError(null);
    try {
      if (!saved) await api.put(`/api/experiments/${exp.id}/preprocessing`, cfg);
      await api.post(`/api/experiments/${exp.id}/preprocessing/run`);
      refresh(exp.id);
    } catch (e) {
      const details = (e as { details?: { result?: PreprocessingResult } }).details;
      setRunError({ message: errorText(e), result: details?.result });
      refresh(exp.id);
    } finally {
      setBusy(null);
    }
  };
  const fitIca = async () => {
    setBusy("ica");
    setError(null);
    try {
      const v = await api.post<IcaView>(`/api/experiments/${exp.id}/preprocessing/ica`, { ...cfg, ica: { ...cfg.ica, exclude: [] } });
      setIca(v);
      set("ica", { exclude: [] });
      refresh(exp.id);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  };

  const pv = preview.data;
  const validation = pv?.validation;
  return (
    <>
      <LifecycleBanner exp={exp} here="preprocessing" />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,430px)_minmax(0,1fr)]">
        <div className="space-y-6">
          {defaults && defaults.notes.length > 0 && (
            <Card>
              <CardHeader
                icon={<Sparkles className="size-5" />}
                title="Suggested from the dataset"
                subtitle={`${ds.name} · ${s.n_participants || "?"} participant(s) · ${s.n_recordings ?? s.n_epochs} ${s.structure === "continuous" ? "recordings" : "epochs"}`}
                actions={<Button size="sm" variant="soft" disabled={stableEqual(cfg, defaults.config)} onClick={() => setCfg(defaults.config)}>Apply suggestions</Button>}
              />
              <CardBody>
                <ul className="list-disc space-y-1 pl-5 text-[13.5px] text-ink-muted">{defaults.notes.map((n) => <li key={n}>{n}</li>)}</ul>
              </CardBody>
            </Card>
          )}
          <Card>
            <CardHeader title="Pipeline" subtitle={s.structure === "continuous" ? `Continuous data from ${s.n_recordings} recording(s): filter → resample → ${cfg.epoching.mode === "windows" ? "sliding windows" : "fixed window at each event"}.` : "Epoched data: filter → resample → keep or crop epochs."} />
            <div className="border-t border-line">
              <Stage n={1} title="Channels" summary={`${selected.filter((c) => !cfg.channels.bad_channels.includes(c)).length} of ${allCh.length} used`} defaultOpen>
                <div className="max-h-64 overflow-y-auto rounded-xl border border-line scrollbar-thin">
                  <table className="w-full text-[13px]">
                    <thead className="sticky top-0 z-10 bg-subtle text-left text-ink-subtle">
                      <tr>
                        <th className="px-3 py-2 font-semibold">Channel</th>
                        <th className="px-3 py-2 font-semibold">Use</th>
                        <th className="px-3 py-2 font-semibold">Bad</th>
                        <th className="px-3 py-2 pr-4 font-semibold">Map to</th>
                      </tr>
                    </thead>
                    <tbody>
                      {allCh.map((c) => (
                        <tr key={c} className="border-t border-line transition-colors hover:bg-surface/50">
                          <td className="px-3 py-1.5 font-mono">{c}</td>
                          <td className="px-3 py-1.5"><input type="checkbox" aria-label={`use ${c}`} className="accent-[var(--c-primary)]" checked={selected.includes(c)} onChange={(e) => set("channels", { selected: e.target.checked ? allCh.filter((x) => selected.includes(x) || x === c) : selected.filter((x) => x !== c) })} /></td>
                          <td className="px-3 py-1.5"><input type="checkbox" aria-label={`mark ${c} bad`} className="accent-[var(--c-danger)]" checked={cfg.channels.bad_channels.includes(c)} onChange={(e) => set("channels", { bad_channels: e.target.checked ? [...cfg.channels.bad_channels, c] : cfg.channels.bad_channels.filter((x) => x !== c) })} /></td>
                          <td className="px-3 py-1.5 pr-4"><input aria-label={`rename ${c}`} className="h-7 w-24 rounded-md border border-line bg-card px-1.5 font-mono text-[12px] focus:border-primary focus:outline-none" placeholder={c} value={cfg.channels.rename[c] ?? ""} onChange={(e) => { const r = { ...cfg.channels.rename }; if (e.target.value) r[c] = e.target.value; else delete r[c]; set("channels", { rename: r }); }} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-[12.5px] text-ink-subtle">Bad channels are excluded from the model input and recorded in the experiment metadata.</p>
              </Stage>
              <Stage n={2} title="Filtering" enabled={cfg.filter.enabled || cfg.filter.notch_enabled} summary={cfg.filter.enabled ? `${cfg.filter.kind} ${cfg.filter.method.toUpperCase()} ${cfg.filter.l_freq ?? ""}–${cfg.filter.h_freq ?? ""} Hz` : "off"}>
                <Toggle label="Band filter" checked={cfg.filter.enabled} onChange={(v) => set("filter", { enabled: v })} />
                {cfg.filter.enabled && (
                  <div className="grid grid-cols-2 gap-3">
                    <SelectField label="Type" value={cfg.filter.kind} onChange={(v) => set("filter", { kind: v })} options={[{ value: "bandpass", label: "Band-pass" }, { value: "highpass", label: "High-pass" }, { value: "lowpass", label: "Low-pass" }]} />
                    <SelectField label="Design" value={cfg.filter.method} onChange={(v) => set("filter", { method: v, phase: v === "fir" ? (cfg.filter.phase === "causal" ? "minimum" : cfg.filter.phase) : cfg.filter.phase === "minimum" ? "causal" : cfg.filter.phase })} options={[{ value: "fir", label: "FIR (firwin)" }, { value: "iir", label: "IIR (Butterworth)" }]} />
                    {cfg.filter.kind !== "lowpass" && <NumberField label="High-pass cutoff" suffix="Hz" step={0.5} value={cfg.filter.l_freq} nullable onChange={(v) => set("filter", { l_freq: v })} />}
                    {cfg.filter.kind !== "highpass" && <NumberField label="Low-pass cutoff" suffix="Hz" step={1} value={cfg.filter.h_freq} nullable onChange={(v) => set("filter", { h_freq: v })} hint={`Must stay below Nyquist (${s.sfreq / 2} Hz).`} />}
                    <SelectField label="Phase" value={cfg.filter.phase} onChange={(v) => set("filter", { phase: v })} hint="Zero-phase preserves timing; causal/minimum phase delays signals." options={cfg.filter.method === "fir" ? [{ value: "zero", label: "Zero-phase" }, { value: "minimum", label: "Minimum-phase (causal)" }] : [{ value: "zero", label: "Zero-phase (forward-backward)" }, { value: "causal", label: "Causal (forward only)" }]} />
                    {cfg.filter.method === "iir" && <NumberField label="IIR order" value={cfg.filter.iir_order} min={1} max={10} onChange={(v) => set("filter", { iir_order: v ?? 4 })} />}
                  </div>
                )}
                <Toggle label="Notch (line noise)" checked={cfg.filter.notch_enabled} onChange={(v) => set("filter", { notch_enabled: v })} />
                {cfg.filter.notch_enabled && (
                  <SelectField label="Line-noise frequency" value={String(cfg.filter.notch_freqs[0] ?? 50)} onChange={(v) => set("filter", { notch_freqs: [Number(v)] })} options={[{ value: "50", label: "50 Hz (Europe, Asia)" }, { value: "60", label: "60 Hz (Americas)" }]} />
                )}
              </Stage>
              <Stage n={3} title="Artifact handling" summary={[cfg.artifacts.amplitude_enabled && `±${cfg.artifacts.amplitude_threshold_uv} µV`, cfg.artifacts.flat_enabled && "flat", cfg.ica.enabled && "ICA"].filter(Boolean).join(" · ") || "none"}>
                {s.structure === "continuous" && <Toggle label="Reject epochs overlapping BAD annotations" checked={cfg.artifacts.reject_bad_segments} onChange={(v) => set("artifacts", { reject_bad_segments: v })} />}
                <Toggle label="Amplitude rejection" checked={cfg.artifacts.amplitude_enabled} onChange={(v) => set("artifacts", { amplitude_enabled: v })} description="Reject an epoch if any channel's peak-to-peak exceeds the threshold." />
                {cfg.artifacts.amplitude_enabled && <NumberField label="Peak-to-peak threshold" suffix="µV" value={cfg.artifacts.amplitude_threshold_uv} onChange={(v) => set("artifacts", { amplitude_threshold_uv: v ?? 150 })} />}
                <Toggle label="Flat-signal rejection" checked={cfg.artifacts.flat_enabled} onChange={(v) => set("artifacts", { flat_enabled: v })} />
                {cfg.artifacts.flat_enabled && <NumberField label="Minimum peak-to-peak" suffix="µV" step={0.1} value={cfg.artifacts.flat_threshold_uv} onChange={(v) => set("artifacts", { flat_threshold_uv: v ?? 0.5 })} />}
                <div className="rounded-xl border border-line bg-surface p-3.5">
                  <Toggle label="ICA (inspection & component rejection)" checked={cfg.ica.enabled} onChange={(v) => set("ica", { enabled: v, exclude: [] })} description="Fitted on the training partition only, never on test trials." />
                  {cfg.ica.enabled && (
                    <div className="mt-3 grid grid-cols-2 gap-3">
                      <NumberField label="Components" value={cfg.ica.n_components} nullable placeholder="all" step={1} onChange={(v) => set("ica", { n_components: v, exclude: [] })} hint="An integer keeps that many PCA components before ICA; a fraction between 0 and 1 keeps enough components to explain that share of variance. Fewer components = coarser decomposition." />
                      <SelectField label="Algorithm" value={cfg.ica.method} onChange={(v) => set("ica", { method: v, exclude: [] })} options={[{ value: "fastica", label: "FastICA" }, { value: "infomax", label: "Infomax" }]} />
                      <div className="col-span-2">
                        <Button size="sm" variant="soft" icon={<Sparkles className="size-4" />} loading={busy === "ica"} onClick={fitIca}>Fit ICA on training data</Button>
                      </div>
                    </div>
                  )}
                </div>
              </Stage>
              <Stage n={4} title="Resampling" enabled={cfg.resample.enabled} summary={cfg.resample.enabled ? `${s.sfreq} → ${cfg.resample.target_sfreq} Hz` : `keep ${s.sfreq} Hz`}>
                <Toggle label="Resample" checked={cfg.resample.enabled} onChange={(v) => set("resample", { enabled: v })} description="Polyphase resampling with an anti-aliasing FIR filter; time axes are updated." />
                {cfg.resample.enabled && <NumberField label="Target rate" suffix="Hz" value={cfg.resample.target_sfreq} onChange={(v) => set("resample", { target_sfreq: v ?? 128 })} />}
              </Stage>
              <Stage n={5} title="Epoching & classes" defaultOpen={cfg.epoching.mode === "windows"} summary={`${ep.mode === "windows" ? `Sliding ${ep.window_s} s windows, ${ep.step_s} s step` : ep.mode === "events" ? `Fixed window ${ep.tmin}–${ep.tmax} s per event` : ep.mode} · ${ep.classes?.length ?? classes.length} classes`}>
                {s.structure === "continuous" ? (
                  <>
                    <Segmented
                      ariaLabel="Epoching mode"
                      value={cfg.epoching.mode === "windows" ? "windows" : "events"}
                      onChange={(v) => set("epoching", { mode: v })}
                      options={[{ value: "events", label: "Fixed window at each event" }, { value: "windows", label: "Sliding windows in segments" }]}
                    />
                    <p className="text-[12.5px] text-ink-subtle">
                      {cfg.epoching.mode === "windows"
                        ? "Each start/end segment (block) is cut into overlapping windows. All windows of a segment stay in that segment's split partition, so they cannot leak between training and test."
                        : "One epoch per event (or segment start), from start to end relative to the onset. Trial boundaries are never invented."}
                    </p>
                    {!!s.n_segments && cfg.epoching.mode !== "windows" && <Alert tone="info">This dataset contains {s.n_segments} start/end segments; sliding windows use their full duration.</Alert>}
                  </>
                ) : (
                  <SelectField label="Existing epochs" value={cfg.epoching.mode} onChange={(v) => set("epoching", { mode: v })} options={[{ value: "keep", label: "Keep as provided" }, { value: "crop", label: "Crop to a window" }, { value: "events", label: "Rebuild from events (needs continuous data)", disabled: true }]} />
                )}
                <Alert tone={ep.mode === "windows" ? "warning" : "info"} title="Applied now">
                  {modeLabel}.
                  {pv?.report && <> Gives <b>{fmtInt(pv.report.output.n_epochs)}</b> {ep.mode === "windows" ? "windows" : "epochs"} of {pv.report.output.shape.slice(2).join(" × ")} ({bytes(pv.report.output.shape.reduce((a, b) => a * b, 1) * 4)}){pv.report.scope?.subset ? " in the preview sample" : ""}.</>}
                  {ep.mode === "windows" && " Overlapping windows multiply the data: use a step equal to the window, or Max windows per segment, to keep it small."}
                </Alert>
                {cfg.epoching.mode === "windows" && (
                  <div className="grid grid-cols-2 gap-3">
                    <NumberField label="Window length" suffix="s" step={0.25} value={cfg.epoching.window_s} onChange={(v) => set("epoching", { window_s: v ?? 2 })} />
                    <NumberField label="Step" suffix="s" step={0.25} value={cfg.epoching.step_s} onChange={(v) => set("epoching", { step_s: v ?? 1 })} hint="Step < window gives overlapping windows." />
                    <NumberField label="Skip at segment start" suffix="s" step={0.25} value={cfg.epoching.segment_offset_s} onChange={(v) => set("epoching", { segment_offset_s: v ?? 0 })} hint="Ignore the transition right after a block starts." />
                    <NumberField label="Max windows per segment" value={cfg.epoching.max_windows} nullable placeholder="all" onChange={(v) => set("epoching", { max_windows: v })} hint="Caps long blocks so classes stay balanced." />
                  </div>
                )}
                {(cfg.epoching.mode === "events" || cfg.epoching.mode === "crop") && (
                  <div className="grid grid-cols-2 gap-3">
                    <NumberField label="Start (s)" step={0.05} value={cfg.epoching.tmin} onChange={(v) => set("epoching", { tmin: v ?? 0 })} />
                    <NumberField label="End (s)" step={0.05} value={cfg.epoching.tmax} onChange={(v) => set("epoching", { tmax: v ?? 1 })} />
                  </div>
                )}
                <div>
                  <div className="mb-1.5 text-[13px] font-bold text-ink-muted">Event / class mapping</div>
                  <div className="flex flex-wrap gap-3">
                    {classes.map((c) => {
                      const sel = cfg.epoching.classes ?? classes;
                      return <Checkbox key={c} checked={sel.includes(c)} label={`${c} (${s.class_distribution[c]}${s.segment_seconds?.[c] ? ` · ${Math.round(s.segment_seconds[c])} s` : ""})`} onChange={(v) => { const next = v ? [...sel, c] : sel.filter((x) => x !== c); set("epoching", { classes: next.length === classes.length ? null : next }); }} />;
                    })}
                  </div>
                </div>
              </Stage>
              <Stage n={6} title="Baseline correction" enabled={cfg.baseline.enabled} summary={cfg.baseline.enabled ? `${cfg.baseline.tmin} to ${cfg.baseline.tmax} s` : "off"}>
                <Toggle label="Subtract baseline mean" checked={cfg.baseline.enabled} onChange={(v) => set("baseline", v ? { enabled: true, ...baselineInside(cfg.baseline, epochWin) } : { enabled: false })} />
                {cfg.baseline.enabled && (
                  <>
                    <div className="grid grid-cols-2 gap-3">
                      <NumberField label="From (s)" step={0.05} value={cfg.baseline.tmin} onChange={(v) => set("baseline", { tmin: v ?? 0 })} />
                      <NumberField label="To (s)" step={0.05} value={cfg.baseline.tmax} onChange={(v) => set("baseline", { tmax: v ?? 0 })} />
                    </div>
                    <p className="text-[12.5px] text-ink-subtle">
                      Must lie within the epoch, {epochWin[0]} to {epochWin[1]} s.
                      {epochWin[0] >= 0 && " This epoch has no pre-event period, so the baseline is measured inside it (e.g. its first 0.2 s)."}
                    </p>
                  </>
                )}
              </Stage>
              <Stage n={7} title="Normalisation" summary={cfg.normalization.method}>
                <SelectField label="Method" value={cfg.normalization.method} onChange={(v) => set("normalization", { method: v })} options={[
                  { value: "channel_zscore", label: "Per-channel z-score (fit on training)" },
                  { value: "robust_channel", label: "Robust per-channel (median / IQR)" },
                  { value: "global_zscore", label: "Global z-score (fit on training)" },
                  { value: "trial_zscore", label: "Per-trial z-score (no fitting)" },
                  { value: "none", label: "None" },
                ]} />
                <p className="text-[12.5px] text-ink-subtle">Fitted parameters are estimated on each fold's training subset and applied unchanged to validation and test data.</p>
              </Stage>
              <Stage n={8} title="Final tensor validation" summary={pv?.report ? shape(pv.report.output.shape) : "—"}>
                {pv?.report && (
                  <KeyValue items={[
                    ["Model input", <Mono key="s">[N, 1, C, T] = [{pv.report.output.shape.join(", ")}]</Mono>],
                    ["Duration", `${pv.report.output.duration_s.toFixed(3)} s at ${pv.report.output.sfreq} Hz`],
                    ["Classes", pv.report.output.classes.join(", ")],
                  ]} />
                )}
              </Stage>
            </div>
          </Card>
          {validation && validation.errors.length > 0 && (
            <Alert tone="danger" title="Invalid configuration">
              <ul className="list-disc space-y-0.5 pl-5">{validation.errors.map((e) => <li key={e}>{e}</li>)}</ul>
            </Alert>
          )}
          {validation?.warnings.map((w) => <Alert key={w} tone="warning">{w}</Alert>)}
          {validation?.info.map((w) => <Alert key={w} tone="info">{w}</Alert>)}
          {error && <Alert tone="danger">{error}</Alert>}
          <div className="flex flex-wrap gap-3">
            <Button variant="secondary" icon={<Save className="size-4" />} onClick={save} loading={busy === "save"} disabled={saved || !!validation?.errors.length || !exp.capabilities.can_configure_preprocessing}>
              {saved ? "Configuration saved" : "Save configuration"}
            </Button>
            <Button icon={<Play className="size-4" />} onClick={run} loading={busy === "run"} disabled={!!validation?.errors.length || completed || exp.capabilities.busy}>
              {completed ? "Pipeline validated" : "Run & validate pipeline"}
            </Button>
          </div>
          {completed && (
            <Button variant="ghost" icon={<ArrowRight className="size-4" />} onClick={() => navigate("/architecture/config")}>Continue to Architecture Explorer</Button>
          )}
        </div>
        <div className="min-w-0 space-y-6">
          <PreviewPanel pv={pv} loading={preview.isFetching} allChannels={pv?.all_channels ?? []} channels={channels} setChannels={setChannels} trialId={trialId} setTrialId={setTrialId} />
          {cfg.ica.enabled && ica && (
            <IcaPanel ica={ica} exclude={cfg.ica.exclude} onToggle={(k) => set("ica", { exclude: cfg.ica.exclude.includes(k) ? cfg.ica.exclude.filter((x) => x !== k) : [...cfg.ica.exclude, k].sort((a, b) => a - b) })} expId={exp.id} onSources={setIca} />
          )}
          {runError && (
            <Alert tone="danger" title="Pipeline validation failed">
              {runError.message}
              {runError.result && <div className="mt-2"><Checklist items={runError.result.tensor_checks.map((c) => ({ id: c.id, label: c.label, status: c.ok, detail: c.detail }))} /></div>}
            </Alert>
          )}
          {exp.preprocessing_summary && saved && <RunResultPanel result={exp.preprocessing_summary} />}
        </div>
      </div>
    </>
  );
}

function PreviewPanel({ pv, loading, allChannels, channels, setChannels, trialId, setTrialId }: {
  pv: PreviewResponse | undefined;
  loading: boolean;
  allChannels: string[];
  channels: string[] | null;
  setChannels: (c: string[] | null) => void;
  trialId: string | null;
  setTrialId: (t: string | null) => void;
}) {
  const r = pv?.report;
  const rejected = r?.rejection.rejected ?? [];
  const trialOptions = useMemo(() => {
    const opts = new Map<string, string>();
    for (const t of pv?.trials ?? []) opts.set(t.trial_id, `${t.trial_id} · ${t.label}${t.group ? ` · participant ${t.group}` : ""}${t.run ? ` · run ${t.run}` : ""}`);
    if (pv?.trial_id && !opts.has(pv.trial_id)) opts.set(pv.trial_id, pv.trial_id);
    rejected.slice(0, 50).forEach((x) => opts.set(x.trial_id, `${x.trial_id} (rejected)`));
    return Array.from(opts, ([value, label]) => ({ value, label }));
  }, [pv?.trial_id, pv?.trials, rejected]);
  const win = r?.epoching.windows;
  return (
    <Card>
      <CardHeader
        title={<span className="flex items-center gap-2">Live preview {loading && <Badge tone="warning">updating…</Badge>}</span>}
        subtitle={pv?.note ?? "Raw versus processed signals, spectra and epoch counts for the current settings."}
      />
      <CardBody className="space-y-5">
        {!pv ? (
          <Loading label="Computing preview…" />
        ) : !pv.ok ? (
          <Alert tone="warning" title="Preview unavailable">Fix the configuration errors to preview.</Alert>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Stat label="Candidates" value={r!.input.n_candidates} hint={`${r!.input.n_channels} ch @ ${r!.input.sfreq} Hz`} />
              <Stat label="Rejected" value={r!.rejection.total} hint={`amp ${r!.rejection.amplitude} · flat ${r!.rejection.flat} · bad ${r!.rejection.bad_segment}`} tone={r!.rejection.total ? "danger" : undefined} />
              <Stat label={win ? "Windows" : "Kept epochs"} value={r!.output.n_epochs} hint={win ? `from ${win.n_parent_trials} segment(s)/events` : r!.epoching.excluded_class ? `${r!.epoching.excluded_class} excluded by class` : undefined} />
              <Stat label="Tensor" value={<Mono>{r!.output.shape.slice(1).join("×")}</Mono>} hint={`N = ${fmtInt(r!.output.shape[0])} · ${bytes(r!.output.shape.reduce((a, b) => a * b, 1) * 4)}`} />
            </div>
            {r!.scope?.subset && (
              <Alert tone="info" title={`Preview sample: ${r!.scope.recordings} of ${fmtInt(r!.scope.of)} recordings (${r!.scope.participants} participants)`}>
                This dataset is large, so the live preview processes recordings spread over participants. Counts here are for that sample; <b>Run &amp; validate pipeline</b> processes every recording (streamed to disk, so it does not need to fit in memory).
              </Alert>
            )}
            <div className="flex flex-wrap items-end gap-3">
              <div className="w-80">
                <SelectField label={win ? "Window" : "Trial"} value={trialId ?? pv.trial_id ?? ""} onChange={(v) => setTrialId(v)} options={trialOptions} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="mb-1.5 text-[13px] font-bold text-ink-muted">Channels shown</div>
                <div className="flex flex-wrap gap-1.5">
                  {allChannels.slice(0, 32).map((c) => {
                    const on = (channels ?? pv.channels ?? []).includes(c);
                    return (
                      <button key={c} type="button" aria-pressed={on} onClick={() => { const cur = channels ?? pv.channels ?? []; setChannels(on ? cur.filter((x) => x !== c) : [...cur, c]); }} className={classNames("rounded-lg border px-2 py-0.5 font-mono text-[12px]", on ? "border-primary bg-primary-soft text-link" : "border-line text-ink-subtle")}>
                        {c}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
            <div className="grid gap-4 2xl:grid-cols-2">
              <div>
                <div className="mb-1 text-[13px] font-bold text-ink-muted">Uploaded signal ({pv.raw?.sfreq} Hz)</div>
                {pv.raw && pv.channels && <StackedSignals times={pv.raw.times} data={pv.raw.data} channels={pv.channels} unit={pv.raw.unit ?? "µV"} ariaLabel="Raw signal" />}
              </div>
              <div>
                <div className="mb-1 text-[13px] font-bold text-ink-muted">Processed ({pv.processed?.sfreq ?? "—"} Hz){pv.processed?.label ? ` · ${pv.processed.label}` : ""}</div>
                {pv.processed && pv.channels ? (
                  <StackedSignals times={pv.processed.times} data={pv.processed.data} channels={pv.channels} unit="µV" ariaLabel="Processed signal" />
                ) : (
                  <Alert tone="danger" title="Trial rejected">{pv.rejected_reason}</Alert>
                )}
              </div>
            </div>
            {(r!.by_participant?.length ?? 0) > 0 && (
              <details className="rounded-xl border border-line bg-surface px-4 py-2.5 text-[13px]" open={(r!.by_participant?.length ?? 0) <= 10}>
                <summary className="cursor-pointer font-bold text-ink">Per participant ({r!.by_participant!.length})</summary>
                <div className="mt-2 max-h-64 overflow-auto scrollbar-thin">
                  <table className="w-full tabular">
                    <thead className="text-left text-ink-subtle"><tr><th className="py-1 pr-3">Participant</th><th className="pr-3">Candidate trials</th><th className="pr-3">Kept trials</th><th className="pr-3">{win ? "Windows" : "Epochs"}</th><th>Per class</th></tr></thead>
                    <tbody>
                      {r!.by_participant!.map((p) => (
                        <tr key={p.participant ?? "none"} className="border-t border-line">
                          <td className="py-1 pr-3 font-bold">{p.participant ?? "—"}</td>
                          <td className="pr-3">{p.candidates}</td>
                          <td className={classNames("pr-3", p.kept_trials < p.candidates && "text-warning")}>{p.kept_trials}</td>
                          <td className="pr-3">{p.kept_rows}</td>
                          <td className="text-ink-muted">{Object.entries(p.class_rows).map(([k, v]) => `${k} ${v}`).join(" · ")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
            {pv.psd_raw && pv.psd_processed && (
              <div>
                <div className="mb-1 text-[13px] font-bold text-ink-muted">Average power spectrum (Welch, up to 60 epochs)</div>
                <PsdChart raw={pv.psd_raw} proc={pv.psd_processed} />
              </div>
            )}
            <div className="grid gap-4 lg:grid-cols-2">
              <div>
                <div className="mb-1 text-[13px] font-bold text-ink-muted">Class distribution before → after</div>
                <Bars labels={r!.output.classes} series={[{ name: "before", data: r!.output.classes.map((c) => r!.input.class_distribution[c] ?? 0) }, { name: "after", data: r!.output.classes.map((c) => r!.output.class_distribution[c] ?? 0) }]} height={200} ariaLabel="Class distribution before and after" valueDigits={0} />
              </div>
              <div className="space-y-2 text-[13.5px]">
                <div className="text-[13px] font-bold text-ink-muted">Processing log</div>
                <KeyValue items={[
                  ["Filter", r!.filter.description],
                  ["Resample", r!.resample.applied ? `${r!.resample.from} → ${r!.resample.to} Hz (×${r!.resample.up}/${r!.resample.down})` : "none"],
                  ["Epoch window", `${r!.epoching.tmin.toFixed(3)} to ${r!.epoching.tmax.toFixed(3)} s (${r!.epoching.n_times} samples)`],
                  ["Out of bounds", r!.epoching.out_of_bounds],
                  ["Normalisation", r!.normalization.doc],
                ]} />
                {r!.filter.messages.map((m) => <p key={m} className="text-[12px] text-warning">{m}</p>)}
              </div>
            </div>
            {rejected.length > 0 && (
              <details className="rounded-xl border border-line bg-surface px-4 py-2.5 text-[13px]">
                <summary className="cursor-pointer font-bold text-ink">Rejected epochs ({r!.rejection.total})</summary>
                <ul className="mt-2 max-h-48 space-y-0.5 overflow-y-auto scrollbar-thin">
                  {rejected.map((x) => <li key={x.trial_id}><Mono>{x.trial_id}</Mono> — {x.reason}</li>)}
                </ul>
              </details>
            )}
          </>
        )}
      </CardBody>
    </Card>
  );
}

function PsdChart({ raw, proc }: { raw: { freqs: number[]; power_db: number[] }; proc: { freqs: number[]; power_db: number[] } }) {
  // align on the union of frequencies by plotting both on a numeric axis
  const x = raw.freqs.length >= proc.freqs.length ? raw.freqs : proc.freqs;
  const interp = (src: { freqs: number[]; power_db: number[] }) => x.map((f) => {
    if (f > src.freqs[src.freqs.length - 1]) return null;
    let i = src.freqs.findIndex((g) => g >= f);
    if (i <= 0) return src.power_db[0];
    const t = (f - src.freqs[i - 1]) / (src.freqs[i] - src.freqs[i - 1]);
    return src.power_db[i - 1] + t * (src.power_db[i] - src.power_db[i - 1]);
  });
  return <LinePlot x={x} series={[{ name: "uploaded", data: interp(raw), dashed: true }, { name: "processed", data: interp(proc) }]} xName="frequency (Hz)" yName="power (dB)" height={230} ariaLabel="Power spectrum before and after preprocessing" />;
}

function IcaPanel({ ica, exclude, onToggle, expId, onSources }: { ica: IcaView; exclude: number[]; onToggle: (k: number) => void; expId: string; onSources: (v: IcaView) => void }) {
  const [sel, setSel] = useState(0);
  const d = ica.diagnostics;
  const comp = d.components[sel];
  return (
    <Card>
      <CardHeader title="ICA components" subtitle={`${d.n_components} components (${d.method}) fitted on ${ica.reference.n_fit_trials}${(ica.reference.n_training_trials ?? 0) > ica.reference.n_fit_trials ? ` of ${ica.reference.n_training_trials} (random subset, memory limit)` : ""} training epochs · ${ica.reference.fit_scope}`} />
      <CardBody className="space-y-4">
        {d.notes.map((n) => <Alert key={n} tone="info">{n}</Alert>)}
        <div className="overflow-x-auto rounded-xl border border-line">
          <table className="w-full text-[13px] tabular">
            <thead className="bg-subtle text-left text-ink-subtle">
              <tr>
                <th className="px-3 py-2 font-semibold">#</th>
                <th className="px-3 py-2 font-semibold whitespace-nowrap text-right">Expl. var.</th>
                <th className="px-3 py-2 font-semibold whitespace-nowrap text-right">Kurtosis</th>
                <th className="px-3 py-2 font-semibold whitespace-nowrap text-right">Frontal |r|</th>
                <th className="px-3 py-2 font-semibold whitespace-nowrap text-right">1–4 Hz</th>
                <th className="px-3 py-2 font-semibold whitespace-nowrap text-right">&gt;20 Hz</th>
                <th className="px-3 py-2 pr-4 font-semibold whitespace-nowrap text-right">Remove</th>
              </tr>
            </thead>
            <tbody>
              {d.components.map((c) => (
                <tr key={c.index} onClick={() => setSel(c.index)} className={classNames("cursor-pointer border-t border-line transition-colors hover:bg-surface", sel === c.index && "bg-primary-soft/50")}>
                  <td className="px-3 py-1.5 font-bold">IC{c.index}</td>
                  <td className="px-3 py-1.5 whitespace-nowrap text-right tabular">{c.explained_variance !== null ? `${(c.explained_variance * 100).toFixed(1)} %` : "—"}</td>
                  <td className="px-3 py-1.5 whitespace-nowrap text-right tabular">{c.kurtosis.toFixed(1)}</td>
                  <td className={classNames("px-3 py-1.5 whitespace-nowrap text-right tabular", (c.frontal_corr ?? 0) > 0.6 && "font-bold text-warning")}>{c.frontal_corr?.toFixed(2) ?? "—"}</td>
                  <td className="px-3 py-1.5 whitespace-nowrap text-right tabular">{(c.low_freq_ratio * 100).toFixed(0)} %</td>
                  <td className="px-3 py-1.5 whitespace-nowrap text-right tabular">{(c.high_freq_ratio * 100).toFixed(0)} %</td>
                  <td className="px-3 py-1.5 pr-4 text-right" onClick={(e) => e.stopPropagation()}><input type="checkbox" aria-label={`remove IC${c.index}`} className="accent-[var(--c-danger)]" checked={exclude.includes(c.index)} onChange={() => onToggle(c.index)} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {comp && (
          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <div className="mb-1 text-[13px] font-bold text-ink-muted">IC{comp.index} spatial pattern (mixing weights)</div>
              <Bars labels={d.ch_names} series={[{ name: "weight", data: comp.pattern }]} height={210} ariaLabel={`IC${comp.index} pattern`} />
            </div>
            <div>
              <div className="mb-1 text-[13px] font-bold text-ink-muted">IC{comp.index} spectrum</div>
              <LinePlot x={d.freqs} series={[{ name: "power (dB)", data: comp.psd_db }]} xName="Hz" height={210} legend={false} ariaLabel={`IC${comp.index} spectrum`} />
            </div>
          </div>
        )}
        {ica.sources && (
          <div>
            <div className="mb-1 flex items-center gap-2 text-[13px] font-bold text-ink-muted">
              Component time courses · training trial <Mono>{ica.sources.trial_id}</Mono>
              <Button size="sm" variant="ghost" onClick={() => api.get<IcaView>(`/api/experiments/${expId}/preprocessing/ica`).then(onSources)}>refresh</Button>
            </div>
            <Heatmap matrix={ica.sources.data} yLabels={ica.sources.data.map((_, i) => `IC${i}`)} xLabels={ica.sources.times.map((t) => t.toFixed(2))} diverging height={Math.max(200, 18 * ica.sources.data.length + 70)} ariaLabel="ICA source time courses" highlightRow={sel} xName="time (s)" />
          </div>
        )}
        <p className="text-[12.5px] text-ink-subtle">Selected for removal: {exclude.length ? exclude.map((k) => `IC${k}`).join(", ") : "none"}. Save the configuration to record the selection; during training each fold refits ICA on its own training data and removes the matching components (spatial-pattern correlation ≥ 0.8).</p>
      </CardBody>
    </Card>
  );
}

function RunResultPanel({ result }: { result: PreprocessingResult }) {
  const lp = result.learned_preview;
  return (
    <Card>
      <CardHeader title={<span className="flex items-center gap-2">Validated pipeline output <Badge tone="success">ready</Badge></span>} subtitle={`fingerprint ${result.fingerprint} · ${new Date(result.computed_at).toLocaleString()}`} />
      <CardBody className="space-y-4">
        <Checklist items={result.tensor_checks.map((c) => ({ id: c.id, label: c.label, status: c.ok, detail: c.detail }))} />
        <Alert tone="info">{result.learned_note}</Alert>
        {lp && (
          <KeyValue items={[
            ["Fit scope", `${lp.fit_scope} (${lp.n_fit_trials} epochs)`],
            ["Normalisation", lp.normalization.doc],
            ...(lp.ica ? [["ICA", lp.ica.note ?? `${lp.ica.n_components} components, removed ${JSON.stringify(lp.ica.excluded)}`] as [string, string]] : []),
          ]} />
        )}
      </CardBody>
    </Card>
  );
}
