import { useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { Activity, Grid3x3, Play } from "lucide-react";
import { api, errorText } from "../../api/client";
import { useRefreshExperiment } from "../../api/hooks";
import type { Experiment, Metrics, Prediction, ResultsResponse } from "../../api/types";
import { Bars, Heatmap, LinePlot } from "../../components/charts/Charts";
import { Button } from "../../components/ui/Button";
import { Card, CardBody, CardHeader, Stat } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Chip";
import { Segmented } from "../../components/ui/Field";
import { Alert, EmptyState, ErrorState, KeyValue, Loading, Mono } from "../../components/ui/Misc";
import { classNames, dateTime, pct } from "../../lib/format";

export function ResultsTab({ exp }: { exp: Experiment }) {
  const refresh = useRefreshExperiment();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const has = exp.capabilities.has_results;
  const { data, error: loadError } = useQuery({
    queryKey: ["results", exp.id, exp.active_run_id, exp.state],
    queryFn: () => api.get<ResultsResponse>(`/api/experiments/${exp.id}/results`),
    enabled: has,
  });
  const evaluate = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.post(`/api/experiments/${exp.id}/evaluate`);
      refresh(exp.id);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  if (!has)
    return (
      <EmptyState
        icon={<Play className="size-6" />}
        title="The model has not been evaluated yet"
        action={<Button onClick={evaluate} loading={busy || exp.state === "evaluation_running"} disabled={!exp.capabilities.can_evaluate}>Evaluate model on the test partition</Button>}
      >
        Training is complete and the artifacts are verified. Evaluation runs the saved checkpoint(s) on the held-out test data and persists predictions and metrics. Validation metrics are never shown in place of test metrics.
        {error && <div className="mt-4 text-left"><Alert tone="danger">{error}</Alert></div>}
      </EmptyState>
    );
  if (loadError) return <ErrorState error={loadError} />;
  if (!data) return <Loading label="Loading results…" />;
  return <ResultsView exp={exp} data={data} onReevaluate={evaluate} busy={busy} />;
}

function MetricCards({ m, foldSummary }: { m: Metrics; foldSummary?: Record<string, { mean: number | null; std: number }> | null }) {
  const fs = (k: string) => (foldSummary?.[k] ? `folds: ${pct(foldSummary[k].mean)} ± ${pct(foldSummary[k].std)}` : undefined);
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      <Stat label="Accuracy" value={pct(m.accuracy)} hint={fs("accuracy") ?? `${m.n} test trials`} />
      <Stat label="Balanced acc." value={pct(m.balanced_accuracy)} hint={fs("balanced_accuracy") ?? `chance ${pct(m.chance_level)}`} />
      <Stat label="Macro F1" value={pct(m.averages.macro.f1)} hint={fs("macro_f1")} />
      <Stat label="ROC-AUC (OvR)" value={m.roc_auc_ovr_macro !== null ? m.roc_auc_ovr_macro.toFixed(3) : "undefined"} hint="macro one-vs-rest" />
      <Stat label="Avg. precision" value={m.average_precision_macro !== null ? m.average_precision_macro.toFixed(3) : "undefined"} hint="macro PR-AUC" />
      <Stat label="Cohen's κ" value={m.cohen_kappa !== null ? m.cohen_kappa.toFixed(3) : "—"} />
    </div>
  );
}

function ConfusionMatrix({ m }: { m: Metrics }) {
  const [norm, setNorm] = useState<"counts" | "row">("counts");
  const matrix = useMemo(() => (norm === "counts" ? m.confusion_matrix : m.confusion_matrix.map((r) => { const s = r.reduce((a, b) => a + b, 0); return r.map((v) => (s ? v / s : 0)); })), [m, norm]);
  return (
    <div className="space-y-2">
      <Segmented ariaLabel="Confusion matrix normalisation" size="sm" value={norm} onChange={setNorm} options={[{ value: "counts", label: "Counts" }, { value: "row", label: "Row-normalised (recall)" }]} />
      <Heatmap matrix={matrix} xLabels={m.class_names} yLabels={m.class_names} xName="predicted" yName="true" showValues valueDigits={norm === "counts" ? 0 : 2} height={Math.max(240, 60 * m.class_names.length + 80)} ariaLabel="Confusion matrix" />
    </div>
  );
}

function PerClassTable({ m }: { m: Metrics }) {
  const f = (v: number | null) => (v === null ? <span className="text-ink-subtle">undefined</span> : v.toFixed(3));
  return (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full text-[13.5px] tabular">
        <thead className="bg-subtle text-left text-ink-subtle"><tr>{["Class", "Precision", "Recall", "F1", "Support", "Notes"].map((h) => <th key={h} className="px-3 py-2">{h}</th>)}</tr></thead>
        <tbody>
          {m.per_class.map((c) => (
            <tr key={c.class} className="border-t border-line"><td className="px-3 py-1.5 font-bold">{c.class}</td><td className="px-3">{f(c.precision)}</td><td className="px-3">{f(c.recall)}</td><td className="px-3">{f(c.f1)}</td><td className="px-3">{c.support}</td><td className="px-3 text-[12.5px] text-warning">{c.notes.join("; ")}</td></tr>
          ))}
          {(["macro", "weighted", "micro"] as const).map((a) => (
            <tr key={a} className="border-t border-line bg-surface"><td className="px-3 py-1.5 font-bold">{a} avg.</td><td className="px-3">{f(m.averages[a].precision)}</td><td className="px-3">{f(m.averages[a].recall)}</td><td className="px-3">{f(m.averages[a].f1)}</td><td className="px-3">{m.n}</td><td className="px-3 text-[12px] text-ink-subtle">{m.averaging_doc[a]}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultsView({ exp, data, onReevaluate, busy }: { exp: Experiment; data: ResultsResponse; onReevaluate: () => void; busy: boolean }) {
  const navigate = useNavigate();
  const ev = data.evaluation;
  const m = ev.test.pooled;
  const [filter, setFilter] = useState<"all" | "correct" | "wrong">("all");
  const [foldView, setFoldView] = useState(0);
  const { data: preds } = useQuery({ queryKey: ["predictions", exp.id, ev.created_at], queryFn: () => api.get<Prediction[]>(`/api/experiments/${exp.id}/predictions`) });
  const shown = (preds ?? []).filter((p) => filter === "all" || (filter === "correct" ? p.correct : !p.correct));
  const hist = data.history[foldView]?.history ?? [];
  const splitRows = ev.validation.map((v, i) => ({ fold: v.fold, train: ev.training[i]?.metrics, val: v.metrics, test: ev.test.per_fold[i]?.metrics }));
  return (
    <div className="space-y-6">
      {exp.synthetic && <Alert tone="warning" title="Synthetic data">These metrics come from synthetic test data and are not evidence of real-world EEG decoding performance.</Alert>}
      <Card>
        <CardHeader
          title={<span className="flex items-center gap-2">Test results <Badge tone="primary">held-out test data</Badge></span>}
          subtitle={`${ev.test.note} Evaluated ${dateTime(ev.created_at)} · ${ev.n_test} trials · protocol: ${ev.protocol ?? "—"}.`}
          actions={<Button size="sm" variant="secondary" onClick={onReevaluate} loading={busy}>Re-evaluate</Button>}
        />
        <CardBody className="space-y-5">
          <MetricCards m={m} foldSummary={ev.test.fold_summary} />
          {m.undefined.map((u) => <Alert key={u} tone="warning">{u}</Alert>)}
          <div className="grid gap-6 xl:grid-cols-2">
            <div>
              <div className="mb-2 text-[13px] font-bold text-ink-muted">Confusion matrix (test)</div>
              <ConfusionMatrix m={m} />
            </div>
            <div className="space-y-4">
              <div className="text-[13px] font-bold text-ink-muted">Class-wise performance (test)</div>
              <Bars labels={m.class_names} series={[{ name: "precision", data: m.per_class.map((c) => c.precision) }, { name: "recall", data: m.per_class.map((c) => c.recall) }, { name: "F1", data: m.per_class.map((c) => c.f1) }]} yMax={1} height={220} ariaLabel="Per-class metrics" />
              <PerClassTable m={m} />
            </div>
          </div>
        </CardBody>
      </Card>
      {((ev.test.by_participant?.length ?? 0) > 0 || ev.test.by_trial) && (
        <Card>
          <CardHeader title="Held-out results by participant" subtitle={`Test participants: ${ev.test.participants?.join(", ") || "—"}. Each prediction is linked to its participant, run and recording.`} />
          <CardBody className="space-y-4">
            {ev.test.by_trial && (
              <Alert tone="info" title={`Per segment: ${pct(ev.test.by_trial.metrics.accuracy)} accuracy · ${pct(ev.test.by_trial.metrics.balanced_accuracy)} balanced`}>
                {ev.test.by_trial.note} {ev.test.by_trial.n_trials} segments from {ev.test.by_trial.n_windows} windows. The metrics above are per window.
              </Alert>
            )}
            {(ev.test.by_participant?.length ?? 0) > 0 && (
              <div className="overflow-x-auto rounded-xl border border-line">
                <table className="w-full text-[13.5px] tabular">
                  <thead className="bg-subtle text-left text-ink-subtle"><tr><th className="px-3 py-2">Participant</th><th className="px-3">Test samples</th><th className="px-3">Accuracy</th><th className="px-3">Balanced acc.</th><th className="px-3">Per class</th></tr></thead>
                  <tbody>
                    {ev.test.by_participant!.map((p) => (
                      <tr key={p.group ?? "none"} className="border-t border-line">
                        <td className="px-3 py-1.5 font-bold">{p.group ?? "—"}</td>
                        <td className="px-3">{p.n}</td>
                        <td className="px-3">{pct(p.accuracy)}</td>
                        <td className="px-3">{pct(p.balanced_accuracy)}</td>
                        <td className="px-3 text-[12.5px] text-ink-muted">{Object.entries(p.class_counts).map(([k, v]) => `${k} ${v}`).join(" · ")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardBody>
        </Card>
      )}
      <Card>
        <CardHeader title="Training, validation and test, kept apart" subtitle="Training metrics are optimistic (fitted data); validation selected the checkpoint; test was never used for training, early stopping or selection." />
        <CardBody className="space-y-5">
          <div className="overflow-x-auto rounded-xl border border-line">
            <table className="w-full text-[13.5px] tabular">
              <thead className="bg-subtle text-left text-ink-subtle"><tr><th className="px-3 py-2">Fold</th><th>Training acc.</th><th>Validation bal. acc.</th><th>Test bal. acc.</th><th>Test n</th><th className="pr-3">Checkpoint · epoch</th></tr></thead>
              <tbody>
                {splitRows.map((r) => (
                  <tr key={r.fold} className="border-t border-line">
                    <td className="px-3 py-1.5 font-bold">{r.fold + 1}</td>
                    <td>{pct(r.train?.accuracy)}</td>
                    <td>{pct(r.val?.balanced_accuracy)}</td>
                    <td className="font-bold">{pct(r.test?.balanced_accuracy)}</td>
                    <td>{r.test?.n ?? "—"}</td>
                    <td className="pr-3"><Mono>{ev.checkpoints[r.fold]?.checkpoint_id}</Mono> · {ev.checkpoints[r.fold]?.selected_epoch}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-[13px] font-bold text-ink-muted">Learning curves</div>
            {data.history.length > 1 && <Segmented size="sm" ariaLabel="Fold" value={String(foldView)} onChange={(v) => setFoldView(Number(v))} options={data.history.map((h) => ({ value: String(h.fold), label: `Fold ${h.fold + 1}` }))} />}
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            <LinePlot x={hist.map((h) => h.epoch)} series={[{ name: "train loss", data: hist.map((h) => h.train_loss) }, { name: "validation loss", data: hist.map((h) => h.val_loss ?? null) }]} xName="epoch" yName="loss" height={230} ariaLabel="Loss curves" />
            <LinePlot x={hist.map((h) => h.epoch)} series={[{ name: "train acc", data: hist.map((h) => h.train_acc) }, { name: "validation bal. acc", data: hist.map((h) => h.val_balanced_accuracy ?? null) }]} xName="epoch" yName="accuracy" yMin={0} yMax={1} height={230} ariaLabel="Accuracy curves" />
          </div>
          <p className="text-[13px] text-ink-muted">{data.history[foldView]?.val_description} Selected epoch {data.history[foldView]?.info.selected_epoch} ({data.history[foldView]?.info.selection_basis}).</p>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Predictions" subtitle="Softmax probabilities for mutually exclusive classes, per test trial." actions={<Segmented size="sm" ariaLabel="Filter predictions" value={filter} onChange={setFilter} options={[{ value: "all", label: `All (${preds?.length ?? 0})` }, { value: "correct", label: "Correct" }, { value: "wrong", label: "Incorrect" }]} />} />
        <CardBody>
          <div className="max-h-[480px] overflow-auto rounded-xl border border-line scrollbar-thin">
            <table className="w-full text-[13px] tabular">
              <thead className="sticky top-0 bg-subtle text-left text-ink-subtle">
                <tr><th className="px-3 py-2">Trial</th><th>Fold</th><th>Participant</th><th>Run</th><th>True</th><th>Predicted</th>{m.class_names.map((c) => <th key={c} className="px-1">p({c})</th>)}<th className="pr-3">Inspect</th></tr>
              </thead>
              <tbody>
                {shown.slice(0, 1000).map((p) => (
                  <tr key={p.trial_id} className={classNames("border-t border-line", !p.correct && "bg-danger-soft/40")}>
                    <td className="px-3 py-1 font-mono text-[12px]">{p.trial_id}</td>
                    <td>{p.fold + 1}</td>
                    <td>{p.group ?? "—"}</td>
                    <td title={p.recording_id ?? ""}>{p.run ?? "—"}</td>
                    <td>{p.true}</td>
                    <td className={classNames("font-bold", p.correct ? "text-success" : "text-danger")}>{p.predicted}</td>
                    {p.probabilities.map((v, i) => (
                      <td key={i} className="px-1">
                        <div className="flex items-center gap-1.5"><div className="h-1.5 w-12 overflow-hidden rounded-full bg-subtle"><div className="h-full bg-primary" style={{ width: `${v * 100}%` }} /></div><span className="text-[11.5px]">{v.toFixed(2)}</span></div>
                      </td>
                    ))}
                    <td className="pr-3 whitespace-nowrap">
                      <button type="button" title="Signal Journey" aria-label={`Signal journey for ${p.trial_id}`} className="mr-1 rounded-md p-1 text-link hover:bg-primary-soft" onClick={() => navigate(`/architecture/journey?trial=${encodeURIComponent(p.trial_id)}`)}><Activity className="size-4" /></button>
                      <button type="button" title="Kernel Playground" aria-label={`Kernel playground for ${p.trial_id}`} className="rounded-md p-1 text-link hover:bg-primary-soft" onClick={() => navigate(`/architecture/kernels?trial=${encodeURIComponent(p.trial_id)}&fold=${p.fold}`)}><Grid3x3 className="size-4" /></button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Evaluation metadata" />
        <CardBody>
          <KeyValue items={[
            ["Run", <Mono key="r">{data.metadata.run_id}</Mono>],
            ["Device", data.metadata.device ?? "—"],
            ["Seed", data.metadata.seed],
            ["Split · preprocessing · architecture", <Mono key="f">{data.metadata.split_fingerprint} · {data.metadata.preprocessing_fingerprint} · {data.metadata.architecture_fingerprint}</Mono>],
            ["Software", Object.entries(data.metadata.software).map(([k, v]) => `${k} ${v}`).join(", ")],
            ["Averaging", m.averaging_doc.balanced_accuracy],
            ["ROC-AUC", m.averaging_doc.roc_auc],
          ]} />
        </CardBody>
      </Card>
    </div>
  );
}
