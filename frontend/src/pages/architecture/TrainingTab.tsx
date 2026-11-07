import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Play, RefreshCw, Square } from "lucide-react";
import { api, errorText } from "../../api/client";
import { useHealth, useRefreshExperiment } from "../../api/hooks";
import type { Experiment, HistoryRow, PretrainingChecks, TrainingConfig } from "../../api/types";
import { LinePlot } from "../../components/charts/Charts";
import { Button } from "../../components/ui/Button";
import { Card, CardBody, CardHeader, Stat } from "../../components/ui/Card";
import { Badge, Dot } from "../../components/ui/Chip";
import { NumberField, SelectField, Toggle } from "../../components/ui/Field";
import { Alert, Checklist, Mono } from "../../components/ui/Misc";
import { dateTime, isBusyState, pct, seconds, stableEqual } from "../../lib/format";
import { useTrainingStream } from "../../lib/useTrainingStream";

function TrainingForm({ cfg, set, disabled, cuda }: { cfg: TrainingConfig; set: (p: Partial<TrainingConfig>) => void; disabled: boolean; cuda: boolean }) {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3">
        <NumberField label="Epochs" value={cfg.epochs} min={1} disabled={disabled} onChange={(v) => set({ epochs: v ?? 1 })} />
        <NumberField label="Batch size" value={cfg.batch_size} min={2} disabled={disabled} onChange={(v) => set({ batch_size: v ?? 32 })} />
        <NumberField label="Learning rate" value={cfg.learning_rate} step={0.0001} disabled={disabled} onChange={(v) => set({ learning_rate: v ?? 0.001 })} />
        <SelectField label="Optimizer" value={cfg.optimizer} disabled={disabled} onChange={(v) => set({ optimizer: v })} options={[{ value: "adam", label: "Adam" }, { value: "adamw", label: "AdamW" }, { value: "sgd", label: "SGD + momentum" }]} />
        <NumberField label="Weight decay" value={cfg.weight_decay} step={0.0001} disabled={disabled} onChange={(v) => set({ weight_decay: v ?? 0 })} />
        {cfg.optimizer === "sgd" && <NumberField label="Momentum" value={cfg.momentum} step={0.05} disabled={disabled} onChange={(v) => set({ momentum: v ?? 0.9 })} />}
        <SelectField label="Loss" value={cfg.loss} disabled={disabled} onChange={(v) => set({ loss: v })} hint="Cross-entropy over mutually exclusive classes (softmax outputs)." options={[{ value: "cross_entropy", label: "Cross-entropy" }]} />
        <NumberField label="Label smoothing" value={cfg.label_smoothing} step={0.05} disabled={disabled} onChange={(v) => set({ label_smoothing: v ?? 0 })} />
        <NumberField label="Random seed" value={cfg.seed} disabled={disabled} onChange={(v) => set({ seed: v ?? 0 })} />
        <SelectField label="Device" value={cfg.device} disabled={disabled} onChange={(v) => set({ device: v })} options={[{ value: "auto", label: `Auto (${cuda ? "CUDA" : "CPU"})` }, { value: "cpu", label: "CPU" }, { value: "cuda", label: "CUDA", disabled: !cuda }]} />
        <SelectField label="Class weighting" value={cfg.class_weighting} disabled={disabled} onChange={(v) => set({ class_weighting: v })} options={[{ value: "none", label: "None" }, { value: "balanced", label: "Balanced (from training labels)" }]} />
        <NumberField label="Saved activation trials" value={cfg.representative_trials} min={1} max={32} disabled={disabled} hint="Representative validation trials whose activations are persisted with the checkpoint; others are re-extracted on demand." onChange={(v) => set({ representative_trials: v ?? 4 })} />
      </div>
      <fieldset className="space-y-3 rounded-2xl border border-line bg-surface p-4">
        <legend className="px-1 text-[12px] font-bold uppercase tracking-[0.07em] text-ink-subtle">Validation &amp; model selection</legend>
        <div className="grid grid-cols-2 gap-3">
          <SelectField label="Validation strategy" value={cfg.validation_mode} disabled={disabled} hint="Carved from the training partition only. 'auto' is group-wise for participant-level protocols." onChange={(v) => set({ validation_mode: v, ...(v === "none" ? { early_stopping: false, restore_best: false, scheduler: cfg.scheduler === "plateau" ? "none" : cfg.scheduler } : {}) })} options={[{ value: "auto", label: "Auto" }, { value: "stratified", label: "Stratified random" }, { value: "random", label: "Random" }, { value: "group", label: "Group-wise" }, { value: "none", label: "None" }]} />
          <NumberField label="Validation fraction" value={cfg.validation_fraction} step={0.05} disabled={disabled || cfg.validation_mode === "none"} onChange={(v) => set({ validation_fraction: v ?? 0.2 })} />
        </div>
        <Toggle label="Early stopping" checked={cfg.early_stopping} disabled={disabled || cfg.validation_mode === "none"} onChange={(v) => set({ early_stopping: v })} description="Monitors the validation subset; test data never influence stopping." />
        <div className="grid grid-cols-2 gap-3">
          <NumberField label="Patience (epochs)" value={cfg.patience} disabled={disabled || !cfg.early_stopping} onChange={(v) => set({ patience: v ?? 10 })} />
          <SelectField label="Monitor" value={cfg.monitor} disabled={disabled || cfg.validation_mode === "none"} onChange={(v) => set({ monitor: v })} options={[{ value: "val_loss", label: "Validation loss" }, { value: "val_balanced_accuracy", label: "Validation balanced accuracy" }]} />
        </div>
        <Toggle label="Restore best validation epoch" checked={cfg.restore_best} disabled={disabled || cfg.validation_mode === "none"} onChange={(v) => set({ restore_best: v })} />
      </fieldset>
      <div className="grid grid-cols-2 gap-3">
        <SelectField label="LR schedule" value={cfg.scheduler} disabled={disabled} onChange={(v) => set({ scheduler: v })} options={[{ value: "none", label: "Constant" }, { value: "step", label: "Step decay" }, { value: "cosine", label: "Cosine annealing" }, { value: "plateau", label: "Reduce on plateau", disabled: cfg.validation_mode === "none" }]} />
        {cfg.scheduler === "step" && <NumberField label="Step size" value={cfg.step_size} disabled={disabled} onChange={(v) => set({ step_size: v ?? 20 })} />}
        {(cfg.scheduler === "step" || cfg.scheduler === "plateau") && <NumberField label="Decay factor γ" value={cfg.gamma} step={0.05} disabled={disabled} onChange={(v) => set({ gamma: v ?? 0.5 })} />}
      </div>
    </div>
  );
}

function Curves({ history, epochs }: { history: HistoryRow[]; epochs: number | null }) {
  const x = history.map((h) => h.epoch);
  const hasVal = history.some((h) => h.val_loss !== undefined);
  const lossSeries = useMemo(() => [
    { name: "train loss", data: history.map((h) => h.train_loss) },
    ...(hasVal ? [{ name: "validation loss", data: history.map((h) => h.val_loss ?? null) }] : []),
  ], [history, hasVal]);
  const accSeries = useMemo(() => [
    { name: "train acc (dropout on)", data: history.map((h) => h.train_acc) },
    ...(hasVal ? [{ name: "validation acc", data: history.map((h) => h.val_acc ?? null) }, { name: "validation bal. acc", data: history.map((h) => h.val_balanced_accuracy ?? null), dashed: true }] : []),
  ], [history, hasVal]);
  const lrSeries = useMemo(() => [{ name: "learning rate", data: history.map((h) => h.lr), step: true }], [history]);
  if (!history.length) return <p className="py-10 text-center text-[14px] text-ink-subtle">Waiting for the first epoch…</p>;
  const last = history[history.length - 1];
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Epoch" value={`${last.epoch}${epochs ? ` / ${epochs}` : ""}`} hint={seconds(last.elapsed_s)} />
        <Stat label="Train loss" value={last.train_loss.toFixed(4)} />
        <Stat label="Val loss" value={last.val_loss !== undefined ? last.val_loss.toFixed(4) : "—"} />
        <Stat label="Val bal. acc" value={last.val_balanced_accuracy !== undefined ? pct(last.val_balanced_accuracy) : "—"} />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <LinePlot x={x} series={lossSeries} xName="epoch" yName="loss" ariaLabel="Loss curves" height={240} />
        <LinePlot x={x} series={accSeries} xName="epoch" yName="accuracy" yMin={0} yMax={1} ariaLabel="Accuracy curves" height={240} />
      </div>
      <LinePlot x={x} series={lrSeries} xName="epoch" yName="lr" ariaLabel="Learning rate" height={140} legend={false} />
    </div>
  );
}

export function TrainingTab({ exp }: { exp: Experiment }) {
  const navigate = useNavigate();
  const refresh = useRefreshExperiment();
  const { data: health } = useHealth();
  const [cfg, setCfg] = useState<TrainingConfig>(exp.training_config);
  const [busy, setBusy] = useState<"train" | "cancel" | "verify" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const run = exp.active_run;
  const live = isBusyState(exp.state);
  const stream = useTrainingStream(exp.id, run?.id ?? null, live, () => refresh(exp.id));
  const [foldView, setFoldView] = useState(0);
  useEffect(() => setCfg(exp.training_config), [exp.training_config]);
  useEffect(() => {
    if (live) setFoldView(stream.currentFold);
  }, [live, stream.currentFold]);

  const { data: pre } = useQuery({
    queryKey: ["pretraining", exp.id, exp.state, exp.updated_at],
    queryFn: () => api.get<PretrainingChecks>(`/api/experiments/${exp.id}/pretraining-checks`),
    enabled: exp.capabilities.can_train,
  });
  const histories: Record<number, HistoryRow[]> = live ? stream.folds : Object.fromEntries((run?.folds ?? []).map((f) => [f.fold, f.history]));
  const nFolds = live ? stream.nFolds : (run?.folds.length ?? 1);
  const changed = !stableEqual(cfg, exp.training_config);
  const trained = !!run && run.status === "completed" && !run.stale;
  const set = (p: Partial<TrainingConfig>) => setCfg((c) => ({ ...c, ...p }));

  const start = async () => {
    setBusy("train");
    setError(null);
    try {
      await api.post(`/api/experiments/${exp.id}/train`, { config: cfg });
      refresh(exp.id);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  };
  const cancel = async () => {
    setBusy("cancel");
    try {
      await api.post(`/api/experiments/${exp.id}/cancel`);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  };
  const reverify = async () => {
    setBusy("verify");
    try {
      await api.post(`/api/experiments/${exp.id}/artifacts/verify`);
      refresh(exp.id);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
      <div className="space-y-6">
        <Card>
          <CardHeader title="Training settings" subtitle="Training runs in a background worker; progress below is streamed from the real training loop." />
          <CardBody>
            <TrainingForm cfg={cfg} set={set} disabled={exp.capabilities.busy} cuda={!!health?.device.cuda_available} />
          </CardBody>
        </Card>
        {pre && (
          <Card>
            <CardHeader title={<span className="flex items-center gap-2">Pre-training checks <Badge tone={pre.ok ? "success" : "danger"}>{pre.ok ? "ready" : "blocked"}</Badge></span>} subtitle="Dataset, channels, sampling rate, class mapping, event files and split are re-validated before every run." />
            <CardBody><Checklist items={pre.checks.map((c) => ({ id: c.id, label: c.label, status: c.ok, detail: c.detail }))} /></CardBody>
          </Card>
        )}
        {trained && changed && <Alert tone="warning" title="Changing settings after training">Starting a run with different settings creates a new run; the current results will be marked stale.</Alert>}
        {error && <Alert tone="danger">{error}</Alert>}
        <div className="flex flex-wrap gap-3">
          {exp.capabilities.can_cancel ? (
            <Button variant="danger" icon={<Square className="size-4" />} onClick={cancel} loading={busy === "cancel"}>Cancel training</Button>
          ) : (
            <Button icon={<Play className="size-4" />} onClick={start} loading={busy === "train"} disabled={!exp.capabilities.can_train || pre?.ok === false}>
              {run ? "Start a new training run" : "Start training"}
            </Button>
          )}
        </div>
      </div>
      <div className="min-w-0 space-y-6">
        <Card>
          <CardHeader
            title={<span className="flex flex-wrap items-center gap-2">Live training {live && <Badge tone="warning"><Dot tone="warning" pulse /> {stream.status ?? exp.state}</Badge>} {run && !live && <Badge tone={run.status === "completed" ? "success" : run.status === "failed" ? "danger" : "neutral"}>{run.status}</Badge>}</span>}
            subtitle={run ? <>Run <Mono>{run.id}</Mono> · {run.device ?? stream.device ?? "device pending"} · started {dateTime(run.started_at ?? run.created_at)}</> : "No run yet."}
            actions={nFolds > 1 && (
              <div className="w-36"><SelectField label="Fold" value={String(foldView)} onChange={(v) => setFoldView(Number(v))} options={Array.from({ length: nFolds }, (_, i) => ({ value: String(i), label: `Fold ${i + 1}` }))} /></div>
            )}
          />
          <CardBody className="space-y-4">
            {run?.status === "failed" && !live && <Alert tone="danger" title="Training failed">{run.error} — no post-training tools were unlocked. Adjust the settings and start a new run.</Alert>}
            {run?.status === "cancelled" && !live && <Alert tone="warning" title="Training cancelled">The run was cancelled; learned-model tools stay locked.</Alert>}
            {run ? <Curves history={histories[foldView] ?? []} epochs={live ? stream.epochs : cfg.epochs} /> : <p className="py-10 text-center text-[14px] text-ink-subtle">Configure the settings and start training.</p>}
            {run && (run.folds[foldView]?.val_description || run.folds[foldView]?.info.selected_epoch) && (
              <p className="text-[13px] text-ink-muted">
                {run.folds[foldView]?.val_description} {run.folds[foldView]?.info.selected_epoch ? `Selected epoch ${run.folds[foldView].info.selected_epoch} (${run.folds[foldView].info.selection_basis}).` : ""}
              </p>
            )}
            {(live ? stream.log : (run?.log ?? [])).length > 0 && (
              <details className="rounded-xl border border-line bg-surface px-4 py-2.5" open={live}>
                <summary className="cursor-pointer text-[13px] font-bold text-ink">Training log</summary>
                <pre className="mt-2 max-h-48 overflow-y-auto font-mono text-[12px] whitespace-pre-wrap text-ink-muted scrollbar-thin">{(live ? stream.log : run!.log).join("\n")}</pre>
              </details>
            )}
          </CardBody>
        </Card>
        {run?.artifact_status && (
          <Card>
            <CardHeader
              title={<span className="flex items-center gap-2">Artifact verification <Badge tone={run.artifact_status.ready ? "success" : "danger"}>{run.artifact_status.ready ? "ready" : "incomplete"}</Badge></span>}
              subtitle={`Post-training tools unlock only when every check passes. Verified ${dateTime(run.artifact_status.verified_at)}.`}
              actions={exp.capabilities.can_reverify && <Button size="sm" variant="secondary" icon={<RefreshCw className="size-4" />} loading={busy === "verify"} onClick={reverify}>Re-verify</Button>}
            />
            <CardBody>
              <Checklist items={run.artifact_status.checks.map((c) => ({ id: c.id, label: c.label, status: c.ok, detail: c.detail, extra: c.recovery ? <p className="mt-1 text-[12.5px] font-bold text-danger">Recovery: {c.recovery}</p> : undefined }))} />
              {exp.capabilities.can_inspect_learned && (
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button variant="ghost" icon={<ArrowRight className="size-4" />} onClick={() => navigate("/architecture/results")}>Evaluate & view results</Button>
                  <Button variant="ghost" icon={<ArrowRight className="size-4" />} onClick={() => navigate("/architecture/kernels")}>Kernel Playground</Button>
                </div>
              )}
            </CardBody>
          </Card>
        )}
        {exp.runs.length > 1 && (
          <Card>
            <CardHeader title="Run history" subtitle="Earlier runs are preserved; stale runs belong to previous configurations." />
            <CardBody>
              <ul className="divide-y divide-line text-[14px]">
                {exp.runs.map((r) => (
                  <li key={r.id} className="flex flex-wrap items-center gap-3 py-2">
                    <Mono>{r.id}</Mono>
                    <Badge tone={r.status === "completed" ? "success" : r.status === "failed" ? "danger" : "neutral"}>{r.status}</Badge>
                    {r.stale && <Badge tone="warning" title={r.stale_reason ?? undefined}>stale</Badge>}
                    {r.id === exp.active_run_id && <Badge tone="primary">active</Badge>}
                    <span className="ml-auto text-[12.5px] text-ink-subtle">{dateTime(r.created_at)}</span>
                  </li>
                ))}
              </ul>
            </CardBody>
          </Card>
        )}
      </div>
    </div>
  );
}
