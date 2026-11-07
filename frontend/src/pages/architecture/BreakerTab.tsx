import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { FlaskConical, Lock, RotateCcw } from "lucide-react";
import { api, errorText } from "../../api/client";
import { useRefreshExperiment, useTrials } from "../../api/hooks";
import type { ArchitectureConfig, BreakerResult, BreakerTotals, Experiment } from "../../api/types";
import { ArchForm } from "../../components/arch/ArchForm";
import { Bars } from "../../components/charts/Charts";
import { Button } from "../../components/ui/Button";
import { Card, CardBody, CardHeader } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Chip";
import { TextField } from "../../components/ui/Field";
import { Alert, CheckIcon, Loading, Mono } from "../../components/ui/Misc";
import { useUI } from "../../store/ui";
import { classNames, fmtInt, shape } from "../../lib/format";

const KIND: Record<string, { tone: "danger" | "warning" | "success" | "neutral"; label: string }> = {
  structural: { tone: "danger", label: "structural · weights incompatible" },
  computation: { tone: "warning", label: "changes learned computation" },
  training_only: { tone: "success", label: "training-time only" },
  display_only: { tone: "neutral", label: "display only" },
  invalid: { tone: "danger", label: "invalid" },
};

function diff(base: ArchitectureConfig, v: ArchitectureConfig): Partial<ArchitectureConfig> {
  const out: Record<string, unknown> = {};
  (Object.keys(v) as (keyof ArchitectureConfig)[]).forEach((k) => {
    if (k !== "preset" && v[k] !== base[k]) out[k] = v[k];
  });
  return out as Partial<ArchitectureConfig>;
}

function TotalsRow({ label, a, b, fmt = (x: number | null) => fmtInt(x) }: { label: string; a: number | null; b: number | null; fmt?: (x: number | null) => string }) {
  const changed = a !== b;
  const pctChange = a && b !== null ? ((b - a) / a) * 100 : null;
  return (
    <tr className="border-t border-line">
      <td className="px-3 py-1.5 text-ink-muted">{label}</td>
      <td className="px-3 py-1.5 tabular">{fmt(a)}</td>
      <td className={classNames("px-3 py-1.5 font-bold tabular", changed && "text-link")}>{fmt(b)}</td>
      <td className="px-3 py-1.5 text-[12.5px] text-ink-subtle tabular">{changed && pctChange !== null ? `${pctChange > 0 ? "+" : ""}${pctChange.toFixed(0)} %` : ""}</td>
    </tr>
  );
}

function Totals({ base, variant, sfreq }: { base: BreakerTotals; variant: BreakerTotals; sfreq: number | null }) {
  return (
    <table className="w-full text-[13.5px]">
      <thead className="text-left text-ink-subtle"><tr><th className="px-3 py-1.5" /><th className="px-3">Trained</th><th className="px-3">Variant</th><th /></tr></thead>
      <tbody>
        <TotalsRow label="Parameters" a={base.params} b={variant.params} />
        <TotalsRow label="MACs per trial" a={base.macs} b={variant.macs} />
        <TotalsRow label="Activation memory (KB/trial)" a={base.activation_kb_per_sample} b={variant.activation_kb_per_sample} fmt={(x) => (x === null ? "—" : x.toFixed(1))} />
        <TotalsRow label="Weights (KB)" a={base.param_kb} b={variant.param_kb} fmt={(x) => (x === null ? "—" : x.toFixed(2))} />
        <TotalsRow label="Receptive field (samples)" a={base.receptive_field_samples} b={variant.receptive_field_samples} />
        {sfreq && <TotalsRow label="Receptive field (ms)" a={base.receptive_field_samples} b={variant.receptive_field_samples} fmt={(x) => (x === null ? "—" : ((x / sfreq) * 1000).toFixed(0))} />}
        <TotalsRow label="Classifier inputs" a={base.classifier_in} b={variant.classifier_in} />
      </tbody>
    </table>
  );
}

export function BreakerTab({ exp }: { exp: Experiment }) {
  const navigate = useNavigate();
  const refresh = useRefreshExperiment();
  const setCurrent = useUI((s) => s.setCurrentExperiment);
  const run = exp.active_run!;
  const base = run.snapshot.architecture_config;
  const [cfg, setCfg] = useState<ArchitectureConfig>(base);
  const [debounced, setDebounced] = useState(cfg);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { data: tl } = useTrials(exp.id, 0, true);
  const trial = tl?.trials.find((t) => t.partition === "test")?.trial_id ?? tl?.trials[0]?.trial_id;
  useEffect(() => {
    const t = setTimeout(() => setDebounced(cfg), 400);
    return () => clearTimeout(t);
  }, [cfg]);
  const changes = useMemo(() => diff(base, debounced), [base, debounced]);
  const { data: res, isFetching } = useQuery({
    queryKey: ["breaker", exp.id, JSON.stringify(changes), trial],
    queryFn: () => api.post<BreakerResult>(`/api/experiments/${exp.id}/breaker`, { changes, trial_id: trial }),
    placeholderData: (p) => p,
  });
  const nChanged = Object.keys(changes).length;

  const spawn = async () => {
    setBusy(true);
    setError(null);
    try {
      const v = await api.post<Experiment>(`/api/experiments/${exp.id}/breaker/spawn`, { changes, name: name || undefined });
      refresh(v.id);
      setCurrent(v.id);
      navigate("/architecture/training");
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <Alert tone="info" title="What happens if…?">
        Modify any parameter of the trained architecture. The trained experiment and checkpoint <Mono>{run.folds[0]?.checkpoint_id}</Mono> stay immutable; variants are analysed separately and can only produce learned results by training a new experiment.
      </Alert>
      <div className="grid gap-6 xl:grid-cols-[minmax(0,440px)_minmax(0,1fr)]">
        <div className="space-y-6">
          <Card>
            <CardHeader title="Variant parameters" subtitle="Changed fields are highlighted. Input dimensions can be changed for shape exploration." actions={<Button size="sm" variant="secondary" icon={<RotateCcw className="size-4" />} onClick={() => setCfg(base)} disabled={!nChanged}>Reset</Button>} />
            <CardBody><ArchForm value={cfg} onChange={setCfg} base={base} editableDims /></CardBody>
          </Card>
        </div>
        <div className="min-w-0 space-y-6">
          {!res ? (
            <Loading label="Analysing variant…" />
          ) : !res.base || !res.variant ? (
            <Alert tone="danger" title="Invalid parameter values">
              <ul className="list-disc pl-5">{res.field_errors?.map((e) => <li key={e.field}><b>{e.field}</b>: {e.message}</li>)}</ul>
            </Alert>
          ) : (
            <>
              <Card>
                <CardHeader
                  title={<span className="flex flex-wrap items-center gap-2">Verdict {isFetching && <Badge tone="warning">updating…</Badge>}</span>}
                  actions={nChanged > 0 && <Badge tone={res.checkpoint_valid ? "success" : "danger"} icon={res.checkpoint_valid ? undefined : <Lock className="size-3" />}>{res.checkpoint_valid ? "checkpoint still valid" : "learned weights unavailable"}</Badge>}
                />
                <CardBody className="space-y-4">
                  <Alert tone={!res.valid ? "danger" : res.checkpoint_valid || !nChanged ? "success" : "warning"}>{res.verdict}</Alert>
                  {res.changed && Object.keys(res.changed).length > 0 && (
                    <div className="overflow-x-auto rounded-xl border border-line">
                      <table className="w-full text-[13px]">
                        <thead className="bg-subtle text-left text-ink-subtle"><tr><th className="px-3 py-1.5">Parameter</th><th>Trained</th><th>Variant</th><th>Effect</th><th className="pr-3">Why</th></tr></thead>
                        <tbody>
                          {Object.entries(res.changed).map(([k, v]) => (
                            <tr key={k} className="border-t border-line">
                              <td className="px-3 py-1.5 font-bold">{k}</td>
                              <td><Mono>{String(v.before)}</Mono></td>
                              <td><Mono className="text-link">{String(v.after)}</Mono></td>
                              <td><Badge tone={KIND[v.kind]?.tone ?? "neutral"}>{KIND[v.kind]?.label ?? v.kind}</Badge></td>
                              <td className="pr-3 text-ink-muted">{v.note}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  <Totals base={res.base.totals} variant={res.variant.totals} sfreq={base.sfreq} />
                  {res.variant.issues.filter((i) => i.level !== "info").map((i) => <Alert key={i.message} tone={i.level === "error" ? "danger" : "warning"} title={i.message}>{i.fix}</Alert>)}
                  {res.learned_outputs && (
                    <div className="rounded-2xl border border-success-line bg-success-soft/50 p-4">
                      <div className="mb-2 text-[13px] font-bold text-ink">Learned outputs with the saved checkpoint · trial <Mono>{res.learned_outputs.trial_id}</Mono></div>
                      <Bars labels={res.learned_outputs.class_names} series={[{ name: "trained", data: res.learned_outputs.probabilities_base }, { name: "variant", data: res.learned_outputs.probabilities_variant }]} yMax={1} height={180} ariaLabel="Probabilities with checkpoint" />
                      <p className="mt-1 text-[12.5px] text-ink-muted">{res.learned_outputs.note} Max difference {res.learned_outputs.max_abs_difference.toExponential(1)}.</p>
                    </div>
                  )}
                  {nChanged > 0 && !res.checkpoint_valid && <p className="text-[13px] text-ink-muted">No learned activations or accuracy are shown for this variant: it has not been trained.</p>}
                  <p className="text-[12.5px] text-ink-subtle">{res.immutability}</p>
                </CardBody>
              </Card>
              <Card>
                <CardHeader title="Before / after tensor dimensions" />
                <CardBody>
                  <div className="overflow-x-auto rounded-xl border border-line">
                    <table className="w-full text-[13px] tabular">
                      <thead className="bg-subtle text-left text-ink-subtle"><tr><th className="px-3 py-1.5">Layer</th><th>Trained output</th><th>Variant output</th><th>Params</th><th className="pr-3">Valid</th></tr></thead>
                      <tbody>
                        {res.layers?.map((l) => (
                          <tr key={l.layer} className={classNames("border-t border-line", l.changed && "bg-warning-soft/40")}>
                            <td className="px-3 py-1.5 font-bold">{l.variant?.label ?? l.base?.label ?? l.layer}</td>
                            <td><Mono>{l.base ? shape(l.base.out_shape) : "— (absent)"}</Mono></td>
                            <td><Mono className={l.changed ? "font-bold text-link" : undefined}>{l.variant ? shape(l.variant.out_shape) : "— (removed)"}</Mono></td>
                            <td>{fmtInt(l.base?.params)} → {fmtInt(l.variant?.params)}</td>
                            <td className="pr-3">{l.variant ? <CheckIcon status={l.variant.valid} /> : "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardBody>
              </Card>
              <Card>
                <CardHeader title="Learned weight compatibility" subtitle="A checkpoint can only be reused when every tensor keeps its shape and the computation is unchanged." />
                <CardBody>
                  <div className="overflow-x-auto rounded-xl border border-line">
                    <table className="w-full text-[13px]">
                      <thead className="bg-subtle text-left text-ink-subtle"><tr><th className="px-3 py-1.5">Tensor</th><th>Trained</th><th>Variant</th><th className="pr-3">Shape-compatible</th></tr></thead>
                      <tbody>
                        {res.tensors?.map((t) => (
                          <tr key={t.tensor} className="border-t border-line">
                            <td className="px-3 py-1 font-mono text-[12.5px]">{t.tensor}</td>
                            <td><Mono>{t.base ? (t.base.length ? shape(t.base) : "scalar") : "—"}</Mono></td>
                            <td><Mono>{t.variant ? (t.variant.length ? shape(t.variant) : "scalar") : "—"}</Mono></td>
                            <td className="pr-3"><CheckIcon status={t.compatible} /></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardBody>
              </Card>
              {nChanged > 0 && res.valid && (
                <Card>
                  <CardHeader title="Train this variant" subtitle="Creates a new experiment with the same dataset, split assignment and preprocessing (a matched protocol), so results are comparable in the Battle Arena." />
                  <CardBody className="space-y-3">
                    <TextField label="Variant experiment name" value={name} onChange={setName} placeholder={`${exp.name} – variant`} />
                    {error && <Alert tone="danger">{error}</Alert>}
                    <Button icon={<FlaskConical className="size-4" />} onClick={spawn} loading={busy}>Create variant experiment</Button>
                  </CardBody>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
