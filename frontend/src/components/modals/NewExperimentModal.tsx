import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { Database, Upload } from "lucide-react";
import { api, errorText } from "../../api/client";
import { useCurrentExperiment, useDatasets, useRefreshExperiment } from "../../api/hooks";
import type { Experiment } from "../../api/types";
import { useUI } from "../../store/ui";
import { Button } from "../ui/Button";
import { SyntheticBadge } from "../ui/Chip";
import { Checkbox, RadioCard, SelectField, TextField } from "../ui/Field";
import { Alert, Modal } from "../ui/Misc";

type Source = "current" | "existing" | "upload";

export function NewExperimentModal() {
  const open = useUI((s) => s.newExperimentOpen);
  const setOpen = useUI((s) => s.setNewExperimentOpen);
  const setUploadOpen = useUI((s) => s.setUploadOpen);
  const setCurrent = useUI((s) => s.setCurrentExperiment);
  const { data: current } = useCurrentExperiment();
  const { data: datasets } = useDatasets();
  const refresh = useRefreshExperiment();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [source, setSource] = useState<Source>("current");
  const [datasetId, setDatasetId] = useState("");
  const [inherit, setInherit] = useState<Record<string, boolean>>({ split: false, preprocessing: false, architecture: false, training: false });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasCurrentDataset = !!current?.dataset_id;
  useEffect(() => {
    if (open) {
      setSource(hasCurrentDataset ? "current" : datasets?.length ? "existing" : "upload");
      setDatasetId(datasets?.[0]?.id ?? "");
      setError(null);
    }
  }, [open, hasCurrentDataset, datasets]);

  const create = async () => {
    setBusy(true);
    setError(null);
    try {
      const body: Record<string, unknown> = { name: name || undefined };
      if (source === "current" && current) {
        body.source_experiment_id = current.id;
        body.inherit = ["dataset", ...Object.entries(inherit).filter(([, v]) => v).map(([k]) => k)];
      } else if (source === "existing") {
        body.dataset_id = datasetId;
      }
      const exp = await api.post<Experiment>("/api/experiments", body);
      setCurrent(exp.id);
      refresh(exp.id);
      setOpen(false);
      setName("");
      navigate("/dataset");
      if (source === "upload") setUploadOpen(true, { attachTo: exp.id });
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const selected = datasets?.find((d) => d.id === datasetId);
  return (
    <Modal
      open={open}
      onClose={() => setOpen(false)}
      title="Start a new experiment"
      footer={
        <>
          <Button variant="secondary" onClick={() => setOpen(false)}>Cancel</Button>
          <Button onClick={create} loading={busy} disabled={source === "existing" && !datasetId}>Create experiment</Button>
        </>
      }
    >
      <div className="space-y-5 pb-2">
        <p className="text-[15px] leading-relaxed text-ink-muted">
          A new experiment gets its own ID and a clean configuration. Earlier experiments, models and results are never
          overwritten. You will start at <b className="text-ink">Dataset &amp; Splits</b> and confirm the evaluation protocol.
        </p>
        <TextField label="Experiment name" value={name} onChange={setName} placeholder="e.g. EEGNet-8,2 cross-subject" />
        <div role="radiogroup" aria-label="Data source" className="space-y-2.5">
          {hasCurrentDataset && (
            <RadioCard
              checked={source === "current"}
              onSelect={() => setSource("current")}
              icon={<Database className="size-5" />}
              title={`Use the current dataset`}
              description={`From “${current?.name}”: ${current?.datasets.map((d) => d.name).join(" + ")}`}
              badge={current?.synthetic ? <SyntheticBadge /> : undefined}
            />
          )}
          <RadioCard checked={source === "existing"} onSelect={() => setSource("existing")} icon={<Database className="size-5" />} title="Choose an uploaded dataset" disabled={!datasets?.length} description={datasets?.length ? undefined : "No datasets uploaded yet."} />
          <RadioCard checked={source === "upload"} onSelect={() => setSource("upload")} icon={<Upload className="size-5" />} title="Import another dataset" description="Creates the experiment, then opens the dataset import (folder, files or local path); the imported dataset is attached to it." />
        </div>
        {source === "existing" && datasets && (
          <SelectField label="Dataset" value={datasetId} onChange={setDatasetId} options={datasets.map((d) => ({ value: d.id, label: `${d.name} · ${d.summary.n_trials} trials` }))} />
        )}
        {source === "existing" && selected?.synthetic && <Alert tone="warning">This dataset is synthetic (software testing only).</Alert>}
        {source === "current" && current && (
          <div className="rounded-2xl border border-line bg-surface px-4 py-3">
            <div className="mb-2 text-[13px] font-bold text-ink-muted">Reuse as a starting point (shown as inherited, must be re-confirmed)</div>
            <div className="grid grid-cols-2 gap-2">
              {(["split", "preprocessing", "architecture", "training"] as const).map((k) => (
                <Checkbox key={k} checked={!!inherit[k]} onChange={(v) => setInherit({ ...inherit, [k]: v })} label={`${k[0].toUpperCase()}${k.slice(1)} settings`} />
              ))}
            </div>
          </div>
        )}
        {error && <Alert tone="danger">{error}</Alert>}
      </div>
    </Modal>
  );
}
