import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Play, Plus, Swords } from "lucide-react";
import { api, errorText } from "../api/client";
import { useExperiment } from "../api/hooks";
import type { ArchitectureConfig, CompareResponse, ExperimentSummary, TrainingConfig } from "../api/types";
import { Bars, Heatmap, LinePlot } from "../components/charts/Charts";
import { Button } from "../components/ui/Button";
import { Card, CardBody, CardHeader } from "../components/ui/Card";
import { Badge, StateChip, SyntheticBadge } from "../components/ui/Chip";
import { NumberField, SelectField, TextField } from "../components/ui/Field";
import { Alert, EmptyState, Loading, Mono, PageHeader } from "../components/ui/Misc";
import { classNames, fmtInt, isBusyState, pct, seconds, shape } from "../lib/format";

export function BattleArenaPage() {
  const qc = useQueryClient();
  const { data: list } = useQuery({
    queryKey: ["experiments", "arena"],
    queryFn: () => api.get<ExperimentSummary[]>("/api/experiments"),
    refetchInterval: (q) => (q.state.data?.some((e) => isBusyState(e.state)) ? 2000 : false),
  });
  const [selected, setSelected] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const toggle = (id: string) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  const { data: cmp, isFetching, error: cmpError } = useQuery({
    queryKey: ["compare", selected.join(","), list?.map((e) => e.state).join(",")],
    queryFn: () => api.post<CompareResponse>("/api/experiments/compare", { ids: selected }),
    enabled: selected.length >= 2,
    placeholderData: (p) => p,
  });
  const train = async (id: string) => {
    setError(null);
    try {
      await api.post(`/api/experiments/${id}/train`);
      qc.invalidateQueries({ queryKey: ["experiments"] });
    } catch (e) {
      setError(errorText(e));
    }
  };
  const evaluate = async (id: string) => {
    setError(null);
    try {
      await api.post(`/api/experiments/${id}/evaluate`);
      qc.invalidateQueries({ queryKey: ["experiments"] });
      qc.invalidateQueries({ queryKey: ["compare"] });
    } catch (e) {
      setError(errorText(e));
    }
  };

  return (
    <>
      <PageHeader
        eyebrow="Separate area"
        title="Hyperparameter"
        accent="battle arena."
        lead="Create matched variants of an experiment and compare them side by side. Configurations are ranked by validation metrics; test metrics are reported for the final protocol only and must not be used to pick hyperparameters."
      />
      <div className="grid gap-6 xl:grid-cols-[400px_minmax(0,1fr)]">
        <div className="space-y-6">
          <VariantCreator list={list ?? []} onCreated={(id) => { setSelected((s) => [...s, id]); qc.invalidateQueries({ queryKey: ["experiments"] }); }} />
          <Card>
            <CardHeader title="Experiments" subtitle="Select two or more to compare." />
            <CardBody className="space-y-1.5">
              {!list?.length && <p className="text-[14px] text-ink-subtle">No experiments yet.</p>}
              {list?.map((e) => (
                <div key={e.id} className={classNames("flex items-center gap-3 rounded-xl border px-3 py-2", selected.includes(e.id) ? "border-primary bg-primary-soft/40" : "border-line")}>
                  <input type="checkbox" aria-label={`compare ${e.name}`} className="size-4 accent-[var(--c-primary)]" checked={selected.includes(e.id)} onChange={() => toggle(e.id)} />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[14px] font-bold text-ink">{e.name}</div>
                    <div className="mt-0.5 flex flex-wrap items-center gap-1.5"><StateChip state={e.state} />{e.synthetic && <SyntheticBadge />}{e.stale && <Badge tone="warning">stale</Badge>}</div>
                  </div>
                  {(e.state === "architecture_configured" || e.state === "training_failed" || e.state === "training_cancelled") && <Button size="sm" variant="soft" icon={<Play className="size-3.5" />} onClick={() => train(e.id)}>Train</Button>}
                  {e.state === "artifacts_ready" && <Button size="sm" variant="soft" onClick={() => evaluate(e.id)}>Evaluate</Button>}
                </div>
              ))}
              {error && <Alert tone="danger">{error}</Alert>}
            </CardBody>
          </Card>
        </div>
        <div className="min-w-0">
          {selected.length < 2 ? (
            <EmptyState icon={<Swords className="size-6" />} title="Pick at least two contenders">Variants derived from the same experiment share the dataset, split assignment and preprocessing, so the comparison is controlled.</EmptyState>
          ) : cmpError ? (
            <Alert tone="danger">{errorText(cmpError)}</Alert>
          ) : !cmp ? (
            <Loading label="Comparing…" />
          ) : (
            <Comparison cmp={cmp} updating={isFetching} />
          )}
        </div>
      </div>
    </>
  );
}

function VariantCreator({ list, onCreated }: { list: ExperimentSummary[]; onCreated: (id: string) => void }) {
  const eligible = list.filter((e) => ["architecture_configured", "artifacts_ready", "evaluation_completed", "training_failed", "training_cancelled", "artifacts_incomplete"].includes(e.state));
  const [baseId, setBaseId] = useState("");
  useEffect(() => {
    if (!baseId && eligible.length) setBaseId(eligible[0].id);
  }, [eligible, baseId]);
  const { data: base } = useExperiment(baseId || null);
  const [arch, setArch] = useState<ArchitectureConfig | null>(null);
  const [tr, setTr] = useState<TrainingConfig | null>(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (base) {
      setArch(base.active_run?.snapshot.architecture_config ?? base.architecture);
      setTr(base.training_config);
    }
  }, [base]);
  const create = async () => {
    if (!base || !arch || !tr) return;
    setBusy(true);
    setError(null);
    try {
      const v = await api.post<{ id: string }>(`/api/experiments/${base.id}/variants`, { name: name || `${base.name} · F1=${arch.F1} D=${arch.D} F2=${arch.F2} lr=${tr.learning_rate}`, architecture: { ...arch, preset: "custom" }, training: tr });
      setName("");
      onCreated(v.id);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  if (!eligible.length)
    return (
      <Card>
        <CardHeader title="Create a variant" subtitle="Needs an experiment with a configured architecture (after preprocessing)." />
      </Card>
    );
  const A = (k: keyof ArchitectureConfig, label: string, step = 1) => arch && <NumberField label={label} step={step} value={arch[k] as number} onChange={(v) => setArch({ ...arch, [k]: v ?? 0 })} />;
  const T = (k: keyof TrainingConfig, label: string, step = 1) => tr && <NumberField label={label} step={step} value={tr[k] as number} onChange={(v) => setTr({ ...tr, [k]: v ?? 0 })} />;
  return (
    <Card>
      <CardHeader icon={<Plus className="size-5" />} title="Create a variant" subtitle="Same data protocol, different hyperparameters." />
      <CardBody className="space-y-4">
        <SelectField label="Base experiment" value={baseId} onChange={setBaseId} options={eligible.map((e) => ({ value: e.id, label: e.name }))} />
        {arch && tr && (
          <>
            <div className="grid grid-cols-3 gap-3">
              {A("F1", "F1")}
              {A("D", "D")}
              {A("F2", "F2")}
              {A("temporal_kernel", "Kernel")}
              {A("pool1_kernel", "Pool 1")}
              {A("pool2_kernel", "Pool 2")}
              {A("dropout", "Dropout", 0.05)}
              {T("learning_rate", "LR", 0.0001)}
              {T("batch_size", "Batch")}
              {T("epochs", "Epochs")}
            </div>
            <SelectField label="Optimizer" value={tr.optimizer} onChange={(v) => setTr({ ...tr, optimizer: v })} options={[{ value: "adam", label: "Adam" }, { value: "adamw", label: "AdamW" }, { value: "sgd", label: "SGD" }]} />
            <TextField label="Name" value={name} onChange={setName} placeholder="auto" />
          </>
        )}
        {error && <Alert tone="danger">{error}</Alert>}
        <Button onClick={create} loading={busy} disabled={!arch}>Create variant</Button>
      </CardBody>
    </Card>
  );
}

function Comparison({ cmp, updating }: { cmp: CompareResponse; updating: boolean }) {
  const ex = cmp.experiments;
  const bestVal = Math.max(...ex.map((e) => e.best_validation ?? -1));
  const rows: [string, (e: (typeof ex)[number]) => ReactNode][] = [
    ["State", (e) => <StateChip state={e.state} />],
    ["Dataset", (e) => e.datasets?.map((d) => d.name).join(" + ") ?? "—"],
    ["Classes", (e) => e.class_names?.join(", ") ?? "—"],
    ["Epoching", (e) => e.trial_unit ?? "—"],
    ["Parameters", (e) => fmtInt(e.architecture?.params)],
    ["MACs / trial", (e) => fmtInt(e.architecture?.macs)],
    ["Receptive field", (e) => e.architecture?.receptive_field_samples ?? "—"],
    ["F1 · D · F2 · kernel", (e) => e.architecture ? <Mono>{e.architecture.config.F1} · {e.architecture.config.D} · {e.architecture.config.F2} · {e.architecture.config.temporal_kernel}</Mono> : "—"],
    ["Pools · dropout", (e) => e.architecture ? <Mono>{e.architecture.config.pool1_kernel}/{e.architecture.config.pool2_kernel} · {e.architecture.config.dropout}</Mono> : "—"],
    ["Optimizer · LR · batch", (e) => <Mono>{e.training_config.optimizer} · {e.training_config.learning_rate} · {e.training_config.batch_size}</Mono>],
    ["Split (strategy · seed)", (e) => <span>{e.split.strategy ?? "—"} · {e.split.seed ?? "—"} <Mono className="text-ink-subtle">{e.split.fingerprint?.slice(0, 8)}</Mono></span>],
    ["Preprocessing", (e) => <Mono>{e.preprocessing_fingerprint?.slice(0, 10) ?? "—"}</Mono>],
    ["Training time", (e) => (e.train_time_s ? `${seconds(e.train_time_s)} · ${e.device ?? ""}` : "—")],
    ["Best validation bal. acc.", (e) => (e.best_validation !== null ? <b className={classNames(e.best_validation === bestVal && "text-success")}>{pct(e.best_validation)}{e.best_validation === bestVal ? " ★" : ""}</b> : <span className="text-ink-subtle">not trained</span>)],
    ["Test accuracy (final)", (e) => (e.evaluation ? pct(e.evaluation.pooled.accuracy) : <span className="text-ink-subtle">{e.trained ? "not evaluated" : "—"}</span>)],
    ["Test balanced acc. (final)", (e) => (e.evaluation ? pct(e.evaluation.pooled.balanced_accuracy) : "—")],
    ["Artifacts", (e) => (e.artifact_status ? <Badge tone={e.artifact_status.ready ? "success" : "danger"}>{e.artifact_status.ready ? "verified" : "incomplete"}</Badge> : "—")],
  ];
  const trained = ex.filter((e) => e.trained && e.histories.length);
  const maxEp = Math.max(0, ...trained.map((e) => e.histories[0].history.length));
  const x = Array.from({ length: maxEp }, (_, i) => i + 1);
  const valSeries = useMemo(() => trained.map((e) => ({ name: e.name, data: x.map((ep) => e.histories[0].history.find((h) => h.epoch === ep)?.val_balanced_accuracy ?? null) })), [trained, x]);
  const evaluated = ex.filter((e) => e.evaluation);
  return (
    <div className="space-y-6">
      {cmp.warnings.map((w) => <Alert key={w} tone="warning">{w}</Alert>)}
      <Alert tone={cmp.matched_protocol ? "success" : "warning"} title={cmp.matched_protocol ? "Matched protocol" : "Protocols differ"}>{cmp.selection_rule}</Alert>
      <Card>
        <CardHeader title={<span className="flex items-center gap-2">Side by side {updating && <Badge tone="warning">updating…</Badge>}</span>} />
        <CardBody>
          <div className="overflow-x-auto rounded-xl border border-line">
            <table className="w-full text-[13.5px] tabular">
              <thead className="bg-subtle text-left"><tr><th className="px-3 py-2 text-ink-subtle" />{ex.map((e) => <th key={e.id} className="min-w-44 px-3 py-2 font-bold text-ink">{e.name}</th>)}</tr></thead>
              <tbody>
                {rows.map(([label, fn]) => (
                  <tr key={label} className="border-t border-line"><td className="px-3 py-1.5 whitespace-nowrap text-ink-subtle">{label}</td>{ex.map((e) => <td key={e.id} className="px-3 py-1.5">{fn(e)}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardBody>
      </Card>
      {trained.length > 0 && (
        <Card>
          <CardHeader title="Validation curves (fold 1)" subtitle="The selection signal: balanced accuracy on each experiment's validation subset." />
          <CardBody><LinePlot x={x} series={valSeries} xName="epoch" yName="val. balanced accuracy" yMin={0} yMax={1} height={260} ariaLabel="Validation curves" /></CardBody>
        </Card>
      )}
      <Card>
        <CardHeader title="Architectures" subtitle="Output shape of every layer (theoretical for untrained configurations)." />
        <CardBody>
          <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${ex.length}, minmax(180px, 1fr))` }}>
            {ex.map((e) => (
              <div key={e.id} className="rounded-xl border border-line bg-surface p-3">
                <div className="mb-2 truncate text-[13px] font-bold text-ink">{e.name}</div>
                <ul className="space-y-0.5 text-[12px]">
                  {e.architecture?.layers.map((l) => (
                    <li key={l.name} className="flex justify-between gap-2"><span className="truncate text-ink-muted">{l.label}</span><Mono className="text-link">{shape(l.out_shape)}</Mono></li>
                  ))}
                </ul>
                <div className="mt-2 text-[11.5px] text-ink-subtle">{e.architecture?.source}</div>
              </div>
            ))}
          </div>
        </CardBody>
      </Card>
      {evaluated.length > 0 && (
        <Card>
          <CardHeader title="Final test evaluation" subtitle="Reported for the final protocol; not a basis for choosing hyperparameters." />
          <CardBody className="space-y-5">
            <Bars labels={evaluated[0].evaluation!.pooled.class_names} series={evaluated.map((e) => ({ name: e.name, data: e.evaluation!.pooled.per_class.map((c) => c.f1) }))} yMax={1} yName="F1 per class" height={240} ariaLabel="Per-class F1" />
            <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${Math.min(evaluated.length, 3)}, minmax(0, 1fr))` }}>
              {evaluated.map((e) => (
                <div key={e.id}>
                  <div className="mb-1 truncate text-[13px] font-bold text-ink-muted">{e.name}</div>
                  <Heatmap matrix={e.evaluation!.pooled.confusion_matrix} xLabels={e.evaluation!.pooled.class_names} yLabels={e.evaluation!.pooled.class_names} showValues valueDigits={0} height={240} ariaLabel={`Confusion matrix ${e.name}`} />
                </div>
              ))}
            </div>
            {cmp.disagreement && (
              <div>
                <div className="mb-2 text-[13px] font-bold text-ink-muted">Prediction disagreement on identical test trials</div>
                <ul className="space-y-1 text-[14px]">
                  {cmp.disagreement.pairs.map((p) => (
                    <li key={p.a + p.b}>{ex.find((e) => e.id === p.a)?.name} vs {ex.find((e) => e.id === p.b)?.name}: <b>{p.disagreements}</b> of {p.n} trials ({pct(p.rate)})</li>
                  ))}
                </ul>
                <p className="mt-1 text-[12.5px] text-ink-subtle">{cmp.disagreement.note}</p>
              </div>
            )}
          </CardBody>
        </Card>
      )}
    </div>
  );
}
