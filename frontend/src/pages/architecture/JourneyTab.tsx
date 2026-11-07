import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, ChevronRight } from "lucide-react";
import { api, qs } from "../../api/client";
import { useTrials } from "../../api/hooks";
import type { Experiment, Journey, JourneyStage } from "../../api/types";
import { TrialPicker } from "../../components/TrialPicker";
import { Bars, Heatmap, StackedSignals } from "../../components/charts/Charts";
import { Button } from "../../components/ui/Button";
import { Card, CardBody, CardHeader } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Chip";
import { NumberField, Segmented, SelectField } from "../../components/ui/Field";
import { Alert, ErrorState, KeyValue, Loading, Mono } from "../../components/ui/Misc";
import { classNames, shape } from "../../lib/format";

const INSPECTABLE = new Set(["temporal_conv", "bn1", "depthwise_conv", "bn2", "act1", "pool1", "drop1", "separable_depthwise", "separable_pointwise", "bn3", "act2", "pool2", "drop2", "flatten", "classifier"]);

export function JourneyTab({ exp }: { exp: Experiment }) {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const learned = exp.capabilities.can_inspect_learned;
  const [mode, setMode] = useState<"trained" | "untrained">(learned ? "trained" : "untrained");
  const nFolds = learned ? (exp.active_run?.folds.length ?? 1) : 1;
  const [fold, setFold] = useState(0);
  const { data: tl } = useTrials(exp.id, fold, learned);
  const [trial, setTrial] = useState<string | null>(params.get("trial"));
  const [stageId, setStageId] = useState(params.get("layer") ?? "model_input");

  useEffect(() => {
    if (!learned) setMode("untrained");
  }, [learned]);
  useEffect(() => {
    if (tl && (!trial || !tl.trials.some((t) => t.trial_id === trial))) {
      const first = tl.trials.find((t) => t.partition === "test") ?? tl.trials[0];
      setTrial(first?.trial_id ?? null);
    }
  }, [tl, trial]);

  const { data: j, error, isFetching, refetch } = useQuery({
    queryKey: ["journey", exp.id, mode, fold, trial, exp.active_run_id],
    queryFn: () => api.get<Journey>(`/api/experiments/${exp.id}/journey${qs({ trial_id: trial, fold, mode })}`),
    enabled: !!trial,
    placeholderData: (p) => p,
  });
  const stage = j?.stages.find((s) => s.id === stageId) ?? j?.stages.find((s) => s.id === "model_input");

  return (
    <div className="space-y-6">
      <Card>
        <CardBody className="grid gap-5 pt-5 lg:grid-cols-[auto_auto_minmax(0,1fr)]">
          <div>
            <div className="mb-1.5 text-[13px] font-bold text-ink-muted">Model</div>
            <Segmented
              ariaLabel="Model state"
              value={mode}
              onChange={setMode}
              options={[
                { value: "trained", label: "Trained checkpoint", disabled: !learned, title: learned ? undefined : "Available after verified training" },
                { value: "untrained", label: "Untrained (random init)" },
              ]}
            />
          </div>
          {nFolds > 1 && mode === "trained" && (
            <div className="w-36">
              <SelectField label="Fold / checkpoint" value={String(fold)} onChange={(v) => setFold(Number(v))} options={Array.from({ length: nFolds }, (_, i) => ({ value: String(i), label: `Fold ${i + 1}` }))} />
            </div>
          )}
          <div className="min-w-0">
            {tl && <TrialPicker trials={tl.trials} value={trial} onChange={setTrial} showPartitions={learned && mode === "trained"} />}
          </div>
        </CardBody>
      </Card>
      {error && <ErrorState error={error} onRetry={() => refetch()} />}
      {!j ? (
        !error && <Loading label="Running the forward pass…" />
      ) : (
        <>
          <Alert tone={j.mode === "trained" ? "info" : "warning"} title={j.mode === "trained" ? "Learned activations" : "Untrained model"}>
            {j.banner} {j.note}
          </Alert>
          <nav aria-label="Stage navigator" className="flex gap-1 overflow-x-auto rounded-2xl border border-line bg-card p-1.5 shadow-card scrollbar-thin">
            {j.stages.map((s, i) => (
              <div key={s.id} className="flex items-center">
                {i > 0 && <ChevronRight className="size-3.5 shrink-0 text-ink-subtle" />}
                <button
                  type="button"
                  aria-current={stage?.id === s.id}
                  onClick={() => setStageId(s.id)}
                  className={classNames("shrink-0 rounded-xl px-3 py-1.5 text-left transition-colors", stage?.id === s.id ? "bg-primary text-on-primary" : "hover:bg-subtle")}
                >
                  <span className="block text-[12.5px] font-bold whitespace-nowrap">{s.label}</span>
                  <span className={classNames("block font-mono text-[11px] whitespace-nowrap", stage?.id === s.id ? "opacity-80" : "text-ink-subtle")}>{shape(s.shape)}</span>
                </button>
              </div>
            ))}
          </nav>
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
            {stage && (
              <Card>
                <CardHeader
                  title={<span className="flex flex-wrap items-center gap-2">{stage.label} {stage.type && <Badge>{stage.type}</Badge>} {isFetching && <Badge tone="warning">updating…</Badge>}</span>}
                  subtitle={stage.purpose ?? stage.description}
                  actions={
                    j.mode === "trained" && INSPECTABLE.has(stage.id) ? (
                      <Button size="sm" variant="soft" icon={<ArrowRight className="size-4" />} onClick={() => navigate(`/architecture/kernels?layer=${stage.id}&trial=${encodeURIComponent(j.trial.trial_id)}&fold=${fold}`)}>
                        Inspect in Kernel Playground
                      </Button>
                    ) : undefined
                  }
                />
                <CardBody className="space-y-4">
                  <KeyValue items={[
                    ["Shape", <Mono key="s">{stage.in_shape ? `${shape(stage.in_shape)} → ` : ""}{shape(stage.shape)}</Mono>],
                    ["Values", stage.unit === "activation" ? "activations (arbitrary units), not EEG voltages" : stage.unit],
                    ...(stage.stats ? [["Range", `${stage.stats.min.toPrecision(3)} … ${stage.stats.max.toPrecision(3)} (mean ${stage.stats.mean.toPrecision(3)}, sd ${stage.stats.std.toPrecision(3)})`] as [string, string]] : []),
                    ...(stage.time_jump && stage.time_jump > 1 ? [["Time step", `one column = ${stage.time_jump} input samples; receptive field ${stage.receptive_field} samples`] as [string, string]] : []),
                    ...(stage.decimation && stage.decimation > 1 ? [["Display", `decimated ×${stage.decimation} for display only`] as [string, string]] : []),
                  ]} />
                  <StageView stage={stage} classNames={j.class_names} />
                </CardBody>
              </Card>
            )}
            <Card>
              <CardHeader title="Prediction" subtitle={`Trial ${j.trial.trial_id} · ${j.provenance.partition}`} />
              <CardBody className="space-y-4">
                <div className="flex items-baseline gap-3">
                  <span className="font-display text-[30px] text-ink">{j.predicted}</span>
                  <Badge tone={j.predicted === j.trial.label ? "success" : "danger"}>{j.predicted === j.trial.label ? "correct" : `true: ${j.trial.label}`}</Badge>
                </div>
                <Bars labels={j.class_names} series={[{ name: "probability", data: j.probabilities }]} yMax={1} height={200} ariaLabel="Class probabilities" highlightIndex={j.class_names.indexOf(j.predicted)} />
                <KeyValue items={[
                  ["Participant", j.trial.group ?? "—"],
                  ["Session · run", `${j.trial.session ?? "—"} · ${j.trial.run ?? "—"}`],
                  ["Recording", <Mono key="r">{j.trial.recording_id ?? "—"}</Mono>],
                  ...(j.trial.parent && j.trial.parent !== j.trial.trial_id ? [["Window", `${j.trial.offset_s?.toFixed(2)} s into segment ${j.trial.parent}`] as [string, string]] : []),
                  ["Logits", <Mono key="l">{j.logits.map((v) => v.toFixed(3)).join(", ")}</Mono>],
                  ["Checkpoint", j.provenance.checkpoint_id ? <Mono key="c">{j.provenance.checkpoint_id}</Mono> : "none (untrained)"],
                ]} />
                {j.mode === "untrained" && <p className="text-[12.5px] text-warning">Untrained probabilities are meaningless as predictions.</p>}
              </CardBody>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

export function StageView({ stage, classNames: cls }: { stage: JourneyStage; classNames: string[] }) {
  const [map, setMap] = useState(0);
  const content = useMemo(() => {
    if (stage.kind === "signal") {
      return <StackedSignals times={stage.times ?? []} data={stage.data as number[][]} channels={stage.ch_names ?? []} unit={stage.unit} ariaLabel={`${stage.label} signal`} />;
    }
    if (stage.kind === "vector") {
      const v = stage.data as number[];
      const labels = stage.id === "classifier" ? cls : v.map((_, i) => i);
      return <Bars labels={labels} series={[{ name: stage.label, data: v }]} height={stage.id === "classifier" ? 200 : 240} ariaLabel={`${stage.label} values`} />;
    }
    const d = stage.data as number[][][];
    const [F, H] = [d.length, d[0]?.length ?? 0];
    if (H === 1) {
      return <Heatmap matrix={d.map((f) => f[0])} yLabels={d.map((_, i) => `map ${i}`)} diverging height={Math.max(220, 16 * F + 80)} xName="time step" ariaLabel={`${stage.label} feature maps`} />;
    }
    const m = Math.min(map, F - 1);
    return (
      <div className="space-y-3">
        <div className="w-48"><NumberField label={`Feature map (0–${F - 1})`} value={m} min={0} max={F - 1} onChange={(v) => setMap(Math.max(0, Math.min(F - 1, v ?? 0)))} /></div>
        <Heatmap matrix={d[m]} yLabels={stage.ch_names && stage.ch_names.length === H ? stage.ch_names : undefined} diverging height={Math.max(220, 14 * H + 80)} xName="time sample" ariaLabel={`${stage.label} map ${m}`} />
      </div>
    );
  }, [stage, cls, map]);
  return content;
}
