import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Database, FlaskConical, Layers, Shuffle, Upload, Users } from "lucide-react";
import { api, errorText, qs } from "../api/client";
import { useCandidates, useDataset, useDatasets, useRefreshExperiment } from "../api/hooks";
import type { DatasetRecord, Experiment, IntegrityReport, SplitConfig, SplitResponse } from "../api/types";
import { RequireExperiment } from "../components/layout/RequireExperiment";
import { LifecycleBanner } from "../components/layout/LifecycleBanner";
import { Button } from "../components/ui/Button";
import { Card, CardBody, CardHeader, Stat } from "../components/ui/Card";
import { Badge, SyntheticBadge } from "../components/ui/Chip";
import { Checkbox, NumberField, RadioCard, Segmented, SelectField, Toggle } from "../components/ui/Field";
import { Alert, Checklist, Loading, PageHeader, SectionTitle } from "../components/ui/Misc";
import { Bars, StackedSignals } from "../components/charts/Charts";
import { classNames, fmtInt, seconds, stableEqual } from "../lib/format";
import { useUI } from "../store/ui";

const DEFAULT_SPLIT: SplitConfig = {
  workflow: "single",
  strategy: "random",
  test_ratio: 0.2,
  stratify: true,
  seed: 42,
  test_groups: null,
  n_test_groups: null,
  kfold_mode: "stratified_trial",
  n_folds: 5,
};

export function DatasetSplitsPage() {
  return (
    <>
      <PageHeader
        eyebrow="Step 1 of the workflow"
        title="Dataset"
        accent="& evaluation splits."
        lead="Attach validated EEG data and decide what question the evaluation answers: new trials from known participants, or entirely unseen participants."
      />
      <RequireExperiment>{(exp) => <DatasetSplitsBody exp={exp} />}</RequireExperiment>
    </>
  );
}

function DatasetSplitsBody({ exp }: { exp: Experiment }) {
  return (
    <>
      <LifecycleBanner exp={exp} here="dataset" />
      <DatasetPicker exp={exp} />
      {exp.dataset_id && (
        <>
          <DatasetSummaryCard id={exp.dataset_id} title={exp.workflow === "separate" ? "Training dataset" : "Dataset summary"} />
          {exp.workflow === "separate" && exp.test_dataset_id && <DatasetSummaryCard id={exp.test_dataset_id} title="Test dataset" />}
          <SplitConfigurator exp={exp} />
        </>
      )}
    </>
  );
}

function DatasetPicker({ exp }: { exp: Experiment }) {
  const { data: datasets } = useDatasets();
  const setUploadOpen = useUI((s) => s.setUploadOpen);
  const refresh = useRefreshExperiment();
  const [workflow, setWorkflow] = useState<"single" | "separate">(exp.workflow);
  const [train, setTrain] = useState(exp.dataset_id ?? "");
  const [test, setTest] = useState(exp.test_dataset_id ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [compat, setCompat] = useState<{ ok: boolean; checks: { id: string; label: string; status: "pass" | "warn" | "fail"; detail: string }[] } | null>(null);

  useEffect(() => {
    setWorkflow(exp.workflow);
    setTrain(exp.dataset_id ?? "");
    setTest(exp.test_dataset_id ?? "");
  }, [exp.workflow, exp.dataset_id, exp.test_dataset_id]);
  useEffect(() => {
    if (!train && datasets?.length) setTrain(datasets[0].id);
  }, [datasets, train]);
  useEffect(() => {
    setCompat(null);
    if (workflow === "separate" && train && test && train !== test) {
      api.get<typeof compat>(`/api/datasets/${train}/compatibility/${test}`).then(setCompat).catch(() => setCompat(null));
    }
  }, [workflow, train, test]);

  const unchanged = exp.dataset_id === train && exp.workflow === workflow && (workflow === "single" || exp.test_dataset_id === test);
  const attach = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.put(`/api/experiments/${exp.id}/dataset`, { dataset_id: train, test_dataset_id: workflow === "separate" ? test : null, workflow });
      refresh(exp.id);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  const opts = (datasets ?? []).map((d) => ({
    value: d.id,
    label: `${d.synthetic ? "◆ " : ""}${d.name} · ${d.summary.n_participants || "?"} participant(s) · ${d.summary.n_recordings ?? d.summary.n_epochs} ${d.summary.structure === "continuous" ? "recordings" : "epochs"} · ${d.summary.n_trials} trials`,
  }));
  return (
    <Card className="mb-6">
      <CardHeader
        icon={<Database className="size-5" />}
        title="Data source"
        subtitle="Choose one dataset with a configurable split, or independent training and test datasets."
        actions={
          <>
            <Button variant="secondary" size="sm" icon={<Upload className="size-4" />} onClick={() => setUploadOpen(true, { attachTo: exp.id })}>Import dataset</Button>
            <Button variant="soft" size="sm" icon={<FlaskConical className="size-4" />} onClick={() => setUploadOpen(true, { attachTo: exp.id })}>Synthetic test data</Button>
          </>
        }
      />
      <CardBody className="space-y-5">
        <Segmented
          ariaLabel="Data workflow"
          value={workflow}
          onChange={setWorkflow}
          options={[
            { value: "single", label: "One dataset, configurable split" },
            { value: "separate", label: "Separate training & test datasets" },
          ]}
        />
        {!datasets?.length ? (
          <Alert tone="info" title="No datasets yet">Upload EEG files or generate a synthetic test dataset to begin.</Alert>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            <SelectField label={workflow === "separate" ? "Training dataset" : "Dataset"} value={train} onChange={setTrain} options={opts} />
            {workflow === "separate" && (
              <SelectField label="Test dataset" value={test} onChange={setTest} options={[{ value: "", label: "Select…" }, ...opts.filter((o) => o.value !== train)]} />
            )}
          </div>
        )}
        {compat && (
          <div className="rounded-2xl border border-line bg-surface px-4 py-2">
            <div className="pt-1 text-[13px] font-bold text-ink-muted">Train / test compatibility</div>
            <Checklist items={compat.checks.map((c) => ({ id: c.id, label: c.label, status: c.status, detail: c.detail }))} />
          </div>
        )}
        {error && <Alert tone="danger">{error}</Alert>}
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={attach} loading={busy} disabled={!train || unchanged || (workflow === "separate" && !test) || !exp.capabilities.can_attach_dataset}>
            {exp.dataset_id ? "Change dataset" : "Attach dataset"}
          </Button>
          {exp.dataset_id && !unchanged && RANK_HINT}
        </div>
      </CardBody>
    </Card>
  );
}

const RANK_HINT = (
  <span className="text-[13px] text-warning">Changing the data resets the split and preprocessing and marks any trained results stale.</span>
);

function sourceLabel(ds: DatasetRecord): string | null {
  const src = ds.source ?? {};
  const mode = src["mode"];
  if (mode === "local_path") return `Imported in place from ${String(src["local_path"])}`;
  if (mode === "folder") return `Imported from the uploaded folder “${String(src["root_label"] ?? "")}”`;
  if (mode === "files") return "Imported from selected files";
  if (mode === "revise") return "Revised import";
  if (mode === "api_upload") return "Uploaded through the API";
  return null;
}

function DatasetSummaryCard({ id, title }: { id: string; title: string }) {
  const { data: ds, isLoading } = useDataset(id);
  const setUploadOpen = useUI((st) => st.setUploadOpen);
  const [revising, setRevising] = useState(false);
  const [reviseError, setReviseError] = useState<string | null>(null);
  if (isLoading || !ds) return <Loading />;
  const s = ds.summary;
  const canRevise = ["folder", "files", "local_path", "revise"].includes(String(ds.source?.["mode"] ?? ""));
  const revise = async () => {
    setRevising(true);
    setReviseError(null);
    try {
      const sess = await api.post<{ id: string }>(`/api/datasets/${ds.id}/revise`);
      setUploadOpen(true, { sessionId: sess.id });
    } catch (e) {
      setReviseError(errorText(e));
    } finally {
      setRevising(false);
    }
  };
  return (
    <Card className="mb-6">
      <CardHeader
        title={
          <span className="flex flex-wrap items-center gap-2">
            {title}: {ds.name} {ds.synthetic && <SyntheticBadge />}
          </span>
        }
        subtitle={<>{sourceLabel(ds) && <span className="block">{sourceLabel(ds)}</span>}{s.processing_hint}</>}
        actions={
          <>
            <Badge tone="primary">{ds.format === "synthetic" ? "generated" : ds.format.toUpperCase()}</Badge>
            {canRevise && <Button size="sm" variant="secondary" loading={revising} onClick={revise}>Revise dataset</Button>}
          </>
        }
      />
      <CardBody className="space-y-5">
        {ds.synthetic && <Alert tone="warning">{ds.notes[0]}</Alert>}
        {reviseError && <Alert tone="danger">{reviseError}</Alert>}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <Stat label="Structure" value={s.structure} />
          <Stat label={s.structure === "epoched" ? "Epochs" : "Recordings"} value={fmtInt(s.structure === "epoched" ? s.n_epochs : s.n_recordings)} hint={s.structure === "continuous" ? `${s.n_trials} trials${s.n_segments ? ` (${s.n_segments} segments)` : ""}` : s.n_recordings ? `from ${s.n_recordings} file(s)` : undefined} />
          <Stat label="Channels" value={s.n_channels} hint={s.unit} />
          <Stat label="Sampling" value={`${s.sfreq} Hz`} hint={`${fmtInt(s.n_samples)} samples`} />
          <Stat label="Epoch length" value={s.epoch_duration_s ? seconds(s.epoch_duration_s) : "—"} hint={s.tmin !== null ? `from ${s.tmin} s` : "defined at epoching"} />
          <Stat label="Participants" value={s.n_participants || "—"} hint={[s.n_sessions ? `${s.n_sessions} session(s)` : "", s.n_runs ? `${s.n_runs} run(s)` : ""].filter(Boolean).join(" · ") || "no session IDs"} />
        </div>
        <div className="grid gap-6 lg:grid-cols-2">
          <div>
            <div className="mb-2 text-[13px] font-bold text-ink-muted">Class distribution ({s.n_classes} classes)</div>
            <Bars labels={Object.keys(s.class_distribution)} series={[{ name: "trials", data: Object.values(s.class_distribution) }]} height={200} ariaLabel="Class distribution" valueDigits={0} />
          </div>
          <div>
            <div className="mb-2 text-[13px] font-bold text-ink-muted">Channels</div>
            <div className="flex flex-wrap gap-1.5">
              {s.ch_names.map((c, i) => (
                <span key={c} className={classNames("rounded-lg border px-2 py-0.5 font-mono text-[12.5px]", s.ch_types[i] === "eeg" ? "border-line bg-surface text-ink" : "border-warning-line bg-warning-soft text-warning")} title={s.ch_types[i]}>
                  {c}{s.ch_types[i] !== "eeg" && <span className="ml-1 text-[10px] uppercase">{s.ch_types[i]}</span>}
                </span>
              ))}
            </div>
            {s.missing_metadata.length > 0 && (
              <div className="mt-4 space-y-2">
                {s.missing_metadata.map((m) => <Alert key={m} tone="warning">{m}</Alert>)}
              </div>
            )}
            {ds.issues.filter((i) => i.level === "warning").map((i) => <Alert key={i.code} tone="warning" className="mt-2">{i.message}</Alert>)}
          </div>
        </div>
        {(s.participants_table?.length ?? 0) > 0 && (
          <details className="rounded-2xl border border-line bg-surface px-4 py-3" open={(s.participants_table?.length ?? 0) <= 12}>
            <summary className="cursor-pointer text-[14px] font-bold text-ink">Participants ({s.participants_table!.length})</summary>
            <div className="mt-2 max-h-72 overflow-auto scrollbar-thin">
              <table className="w-full text-[13px] tabular">
                <thead className="text-left text-ink-subtle"><tr><th className="py-1 pr-3">Participant</th><th className="pr-3">Recordings</th><th className="pr-3">Sessions</th><th className="pr-3">Runs</th><th className="pr-3">Trials</th><th>Per class</th></tr></thead>
                <tbody>
                  {s.participants_table!.map((p) => (
                    <tr key={p.participant ?? "none"} className="border-t border-line">
                      <td className="py-1 pr-3 font-bold">{p.participant ?? <span className="text-warning">no ID</span>}</td>
                      <td className="pr-3">{p.n_recordings}</td>
                      <td className="pr-3">{p.sessions.join(", ") || "—"}</td>
                      <td className="pr-3">{p.runs.join(", ") || "—"}</td>
                      <td className="pr-3">{p.n_trials}</td>
                      <td className="text-[12px] text-ink-muted">{Object.entries(p.class_counts).map(([k, v]) => `${k} ${v}`).join(" · ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
        {s.recordings.length > 0 && (
          <details className="rounded-2xl border border-line bg-surface px-4 py-3">
            <summary className="cursor-pointer text-[14px] font-bold text-ink">Recordings ({s.recordings.length})</summary>
            <div className="mt-2 max-h-80 overflow-auto scrollbar-thin">
              <table className="w-full text-[13px] tabular">
                <thead className="sticky top-0 bg-surface text-left text-ink-subtle"><tr><th className="py-1 pr-3">Recording ID</th><th className="pr-3">Participant</th><th className="pr-3">Session</th><th className="pr-3">Run</th><th className="pr-3">Source file</th><th className="pr-3">Events from</th><th className="pr-3">Duration</th><th className="pr-3">Trials</th><th>BAD</th></tr></thead>
                <tbody>
                  {s.recordings.map((r) => (
                    <tr key={r.recording_id ?? r.name} className="border-t border-line">
                      <td className="py-1 pr-3 font-mono text-[12px]">{r.recording_id ?? r.name}</td>
                      <td className="pr-3">{r.group ?? "—"}</td>
                      <td className="pr-3">{r.session ?? "—"}</td>
                      <td className="pr-3">{r.run ?? "—"}</td>
                      <td className="max-w-[260px] truncate pr-3 font-mono text-[11.5px] text-ink-muted" title={r.source_file ?? ""}>{r.source_file ?? "—"}</td>
                      <td className="max-w-[200px] truncate pr-3 text-[12px]" title={r.event_source ?? ""}>{r.event_source ?? "—"}</td>
                      <td className="pr-3">{r.duration_s ? seconds(r.duration_s) : "—"}</td>
                      <td className="pr-3">{r.n_trials ?? r.n_events}</td>
                      <td>{r.bad_segments}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
        <SignalPreview ds={ds} />
      </CardBody>
    </Card>
  );
}

function SignalPreview({ ds }: { ds: DatasetRecord }) {
  const { data: trials } = useQuery({ queryKey: ["ds-trials", ds.id], queryFn: () => api.get<{ trial_id: string; label: string; group: string | null; session: string | null }[]>(`/api/datasets/${ds.id}/trials`) });
  const [tid, setTid] = useState<string>("");
  useEffect(() => {
    if (!tid && trials?.length) setTid(trials[0].trial_id);
  }, [trials, tid]);
  const { data: sig } = useQuery({
    queryKey: ["ds-signal", ds.id, tid],
    queryFn: () => api.get<{ times: number[]; data: number[][]; ch_names: string[]; unit: string; label: string }>(`/api/datasets/${ds.id}/signal${qs({ trial_id: tid })}`),
    enabled: !!tid,
  });
  if (!trials?.length) return null;
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-end gap-3">
        <div className="text-[13px] font-bold text-ink-muted">Raw uploaded signal (before preprocessing)</div>
        <div className="ml-auto w-72">
          <SelectField label="Trial" value={tid} onChange={setTid} options={trials.slice(0, 1500).map((t) => ({ value: t.trial_id, label: `${t.trial_id} · ${t.label}${t.group ? ` · participant ${t.group}` : ""}` }))} />
        </div>
      </div>
      {sig && <StackedSignals times={sig.times} data={sig.data} channels={sig.ch_names} unit={sig.unit} ariaLabel={`Raw signal of trial ${tid}`} highlight={ds.structure === "continuous" ? [0, 0.02] : null} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Split configuration
// ---------------------------------------------------------------------------

function SplitConfigurator({ exp }: { exp: Experiment }) {
  const navigate = useNavigate();
  const refresh = useRefreshExperiment();
  const { data: cand } = useCandidates(exp.id, !!exp.dataset_id);
  const separate = exp.workflow === "separate";
  const [cfg, setCfg] = useState<SplitConfig>(exp.split_config ?? DEFAULT_SPLIT);
  const [preview, setPreview] = useState<SplitResponse | null>(null);
  const [busy, setBusy] = useState<"preview" | "save" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { data: saved } = useQuery({
    queryKey: ["split", exp.id, exp.split_fingerprint],
    queryFn: () => api.get<SplitResponse>(`/api/experiments/${exp.id}/split`),
    enabled: !!exp.split_fingerprint,
  });
  useEffect(() => {
    if (exp.split_config) setCfg(exp.split_config);
  }, [exp.split_config]);
  const effective = separate ? { ...cfg, workflow: "separate" as const, strategy: "separate" as const } : cfg;
  const groupsAvailable = !!cand?.has_groups;
  // Revalidate split integrity automatically whenever the dataset or the split settings change.
  const effectiveKey = JSON.stringify(effective);
  useEffect(() => {
    setPreview(null);
    if (!exp.capabilities.can_configure_split) return;
    const t = setTimeout(() => {
      api.post<SplitResponse>(`/api/experiments/${exp.id}/split/preview`, JSON.parse(effectiveKey))
        .then((r) => { setPreview(r); setError(null); })
        .catch((e) => setError(errorText(e)));
    }, 450);
    return () => clearTimeout(t);
  }, [effectiveKey, exp.id, exp.dataset_id, exp.test_dataset_id, exp.capabilities.can_configure_split]);

  const doPreview = async () => {
    setBusy("preview");
    setError(null);
    try {
      setPreview(await api.post<SplitResponse>(`/api/experiments/${exp.id}/split/preview`, effective));
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  };
  const confirm = async () => {
    setBusy("save");
    setError(null);
    try {
      await api.put(`/api/experiments/${exp.id}/split`, effective);
      refresh(exp.id);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  };
  const isSaved = !!exp.split_fingerprint && saved?.assignment.fingerprint === exp.split_fingerprint;
  const configMatchesSaved = isSaved && stableEqual(saved?.assignment.config, effective);
  const shown = preview ?? (configMatchesSaved ? saved : null);
  const set = (patch: Partial<SplitConfig>) => setCfg((c) => ({ ...c, ...patch }));
  const disabled = !exp.capabilities.can_configure_split;

  return (
    <section className="mt-10">
      <SectionTitle accent="protocol." sub="Trial-wise and participant-independent splits answer different scientific questions; neither is labelled as the other.">
        Evaluation
      </SectionTitle>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <Card>
          <CardBody className="space-y-5 pt-5">
            {separate ? (
              <Alert tone="info" title="Separate datasets">The test dataset is evaluated as provided. A validation subset for early stopping is carved from the training dataset only.</Alert>
            ) : (
              <div role="radiogroup" aria-label="Split strategy" className="space-y-2.5">
                {cand && (
                  <p className="text-[13.5px] text-ink-muted">
                    Splits operate on <b className="text-ink">{fmtInt(cand.n)}</b> candidate trials ({cand.trial_unit}) from{" "}
                    <b className="text-ink">{cand.n_recordings}</b> recording(s) of <b className="text-ink">{cand.groups.length || "no identified"}</b> participant(s).
                    {cand.n_segments > 0 && " Windows cut from a segment always stay in that segment's partition."}
                  </p>
                )}
                <RadioCard checked={cfg.strategy === "random"} onSelect={() => set({ strategy: "random" })} icon={<Shuffle className="size-5" />} title="Random trial-level hold-out" description="Trials from the same participant may appear in both sets: estimates within-participant performance." disabled={disabled} />
                <RadioCard checked={cfg.strategy === "group"} onSelect={() => set({ strategy: "group" })} icon={<Users className="size-5" />} title="Participant-level hold-out" description={groupsAvailable ? "Selected participants are held out entirely: estimates generalisation to unseen people." : "Needs participant IDs for every trial."} disabled={disabled || !groupsAvailable} />
                <RadioCard checked={cfg.strategy === "kfold"} onSelect={() => set({ strategy: "kfold" })} icon={<Layers className="size-5" />} title="K-fold cross-validation" description="Trial-wise, stratified, group-wise or stratified group-wise folds (GroupKFold / StratifiedGroupKFold)." disabled={disabled} />
              </div>
            )}
            {!separate && cfg.strategy === "random" && (
              <div className="grid gap-4 sm:grid-cols-2">
                <NumberField label="Test fraction" value={cfg.test_ratio} step={0.05} min={0.05} max={0.9} onChange={(v) => set({ test_ratio: v ?? 0.2 })} />
                <NumberField label="Random seed" value={cfg.seed} onChange={(v) => set({ seed: v ?? 0 })} />
                <div className="sm:col-span-2"><Toggle label="Stratify by class" checked={cfg.stratify} onChange={(v) => set({ stratify: v })} description="Falls back (with a warning) when a class is too small." /></div>
              </div>
            )}
            {!separate && cfg.strategy === "group" && cand && (
              <div className="space-y-4">
                <Segmented ariaLabel="Group selection" value={cfg.test_groups ? "manual" : "random"} onChange={(v) => set(v === "manual" ? { test_groups: cand.groups.slice(-1), n_test_groups: null } : { test_groups: null, n_test_groups: Math.max(1, Math.round(cand.groups.length * 0.2)) })} options={[{ value: "random", label: "Random allocation" }, { value: "manual", label: "Choose participants" }]} />
                {cfg.test_groups ? (
                  <div>
                    <div className="mb-2 text-[13px] font-bold text-ink-muted">Held-out test participants</div>
                    <div className="flex flex-wrap gap-2">
                      {cand.groups.map((g) => (
                        <Checkbox key={g} checked={cfg.test_groups!.includes(g)} label={<span title={(() => { const p = cand.participants.find((x) => x.participant === g); return p ? `${p.n_recordings} recording(s), runs ${p.runs.join(", ") || "—"}, sessions ${p.sessions.join(", ") || "—"}` : ""; })()}>{g} <span className="text-ink-subtle">({cand.group_counts[g]} trials · {cand.participants.find((x) => x.participant === g)?.n_recordings ?? 0} rec.)</span></span>} onChange={(v) => set({ test_groups: v ? [...cfg.test_groups!, g] : cfg.test_groups!.filter((x) => x !== g) })} />
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="grid gap-4 sm:grid-cols-2">
                    <NumberField label={`Test participants (of ${cand.groups.length})`} value={cfg.n_test_groups ?? Math.max(1, Math.round(cand.groups.length * 0.2))} min={1} max={cand.groups.length - 1} onChange={(v) => set({ n_test_groups: v })} />
                    <NumberField label="Random seed" value={cfg.seed} onChange={(v) => set({ seed: v ?? 0 })} />
                  </div>
                )}
              </div>
            )}
            {!separate && cfg.strategy === "kfold" && (
              <div className="grid gap-4 sm:grid-cols-2">
                <SelectField label="Fold type" value={cfg.kfold_mode} onChange={(v) => set({ kfold_mode: v })} options={[
                  { value: "trial", label: "Trial-wise K-fold" },
                  { value: "stratified_trial", label: "Stratified trial-wise" },
                  { value: "group", label: "Group-wise (GroupKFold)", disabled: !groupsAvailable },
                  { value: "stratified_group", label: "Stratified group-wise", disabled: !groupsAvailable },
                ]} />
                <NumberField label="Folds (K)" value={cfg.n_folds} min={2} max={20} onChange={(v) => set({ n_folds: v ?? 5 })} />
                <NumberField label="Random seed" value={cfg.seed} onChange={(v) => set({ seed: v ?? 0 })} />
              </div>
            )}
            {error && <Alert tone="danger">{error}</Alert>}
            <div className="flex flex-wrap gap-3">
              <Button variant="secondary" onClick={doPreview} loading={busy === "preview"} disabled={disabled}>Preview split &amp; integrity</Button>
              <Button onClick={confirm} loading={busy === "save"} disabled={disabled || !shown?.integrity.ok || configMatchesSaved}>
                {configMatchesSaved ? "Split confirmed" : "Confirm split"}
              </Button>
            </div>
            {configMatchesSaved && exp.capabilities.can_configure_preprocessing && (
              <Button variant="ghost" icon={<ArrowRight className="size-4" />} onClick={() => navigate("/preprocessing")}>Continue to Preprocessing Studio</Button>
            )}
          </CardBody>
        </Card>
        <IntegrityPanel response={shown} saved={configMatchesSaved} />
      </div>
    </section>
  );
}

function IntegrityPanel({ response, saved }: { response: SplitResponse | null | undefined; saved: boolean }) {
  if (!response)
    return (
      <Card>
        <CardBody className="py-14 text-center text-[15px] text-ink-muted">Preview the split to see the integrity summary before confirming.</CardBody>
      </Card>
    );
  const { assignment: a, integrity } = response;
  return (
    <Card>
      <CardHeader
        title={<span className="flex items-center gap-2">Split integrity {saved ? <Badge tone="success">confirmed</Badge> : <Badge>preview</Badge>}</span>}
        subtitle={a.protocol}
        actions={<Badge tone={integrity.ok ? "success" : "danger"}>{integrity.ok ? "passes" : "fails"}</Badge>}
      />
      <CardBody className="space-y-4">
        <Alert tone="info" title="What this evaluation measures">{a.question}</Alert>
        {a.warnings.map((w) => <Alert key={w} tone="warning">{w}</Alert>)}
        <Checklist items={integrity.checks.map((c) => ({ id: c.id, label: c.label, status: c.status, detail: c.detail }))} />
        <PartitionTable integrity={integrity} />
        <p className="text-[12.5px] text-ink-subtle">Fingerprint {a.fingerprint} · seed {a.seed} · {a.n_candidates} candidate trials. Assignments are stored and reproducible.</p>
      </CardBody>
    </Card>
  );
}

function PartitionTable({ integrity }: { integrity: IntegrityReport }) {
  const classes = integrity.classes;
  const folds = useMemo(() => integrity.folds.slice(0, 10), [integrity]);
  return (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full text-[13px] tabular">
        <thead className="bg-subtle text-left text-ink-subtle">
          <tr>
            <th className="px-3 py-2">{folds.length > 1 ? "Fold" : "Partition"}</th>
            <th className="px-3 py-2">Train</th>
            <th className="px-3 py-2">Test</th>
            {classes.map((c) => <th key={c} className="px-3 py-2">{c} (tr/te)</th>)}
            <th className="px-3 py-2">Test participants</th>
          </tr>
        </thead>
        <tbody>
          {folds.map((f) => (
            <tr key={f.index} className="border-t border-line">
              <td className="px-3 py-1.5 font-bold">{folds.length > 1 ? f.index + 1 : "hold-out"}</td>
              <td className="px-3 py-1.5">{f.n_train}</td>
              <td className="px-3 py-1.5">{f.n_test}</td>
              {classes.map((c) => (
                <td key={c} className={classNames("px-3 py-1.5", (f.class_counts.test[c] ?? 0) === 0 && "text-warning")}>
                  {f.class_counts.train[c] ?? 0} / {f.class_counts.test[c] ?? 0}
                </td>
              ))}
              <td className="max-w-[220px] truncate px-3 py-1.5" title={f.test_groups.join(", ")}>
                {f.test_groups.join(", ") || "—"}
                {f.group_overlap.length > 0 && <span className="ml-1 text-warning">(overlap: {f.group_overlap.length})</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
