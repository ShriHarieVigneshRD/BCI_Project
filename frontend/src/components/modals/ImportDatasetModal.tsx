import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, FileStack, FlaskConical, FolderOpen, FolderSearch, HardDrive, Loader2 } from "lucide-react";
import { api, errorText, uploadWithProgress } from "../../api/client";
import { useCurrentExperiment, useExperiment, useFormats, useRefreshExperiment } from "../../api/hooks";
import type { DatasetManifest, DatasetRecord, Experiment, ImportPlan, ImportSession } from "../../api/types";
import { useUI } from "../../store/ui";
import { bytes, classNames, fmtInt } from "../../lib/format";
import { Button } from "../ui/Button";
import { Badge, SyntheticBadge } from "../ui/Chip";
import { NumberField, RadioCard, SelectField, TextField, Toggle } from "../ui/Field";
import { Alert, Modal } from "../ui/Misc";
import { DatasetReview, type ManifestPatch } from "../import/DatasetReview";

type Method = "folder" | "files" | "local_path" | "synthetic";
type Step = "source" | "plan" | "uploading" | "scanning" | "review" | "importing" | "done";

interface Picked {
  file: File;
  path: string;
}

const BATCH_BYTES = 48 * 1024 * 1024;
const BATCH_FILES = 64;

function ProgressBar({ value, label }: { value: number; label: string }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-[14px] text-ink-muted"><Loader2 className="size-4 animate-spin text-link" />{label}</div>
      <div className="h-2 overflow-hidden rounded-full bg-subtle" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full rounded-full bg-primary transition-[width]" style={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }} />
      </div>
    </div>
  );
}

export function ImportDatasetModal() {
  const open = useUI((s) => s.uploadOpen);
  const setOpen = useUI((s) => s.setUploadOpen);
  const attachTo = useUI((s) => s.importAttachTo);
  const resumeId = useUI((s) => s.importSessionId);
  const setCurrent = useUI((s) => s.setCurrentExperiment);
  const { data: current } = useCurrentExperiment();
  const { data: target } = useExperiment(attachTo);
  const { data: formats } = useFormats();
  const qc = useQueryClient();
  const refresh = useRefreshExperiment();
  const navigate = useNavigate();

  const [method, setMethod] = useState<Method>("folder");
  const [step, setStep] = useState<Step>("source");
  const [name, setName] = useState("");
  const [localPath, setLocalPath] = useState("");
  const [picked, setPicked] = useState<Picked[]>([]);
  const [plan, setPlan] = useState<ImportPlan | null>(null);
  const [session, setSession] = useState<ImportSession | null>(null);
  const [manifest, setManifest] = useState<DatasetManifest | null>(null);
  const [uploaded, setUploaded] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<DatasetRecord | null>(null);
  const [syn, setSyn] = useState({ kind: "epoched" as "epoched" | "continuous", n_participants: 6, n_classes: 3, trials_per_class: 40, n_channels: 16, include_frontal: true, sfreq: 128, n_sessions: 1, seed: 7 });
  const folderInput = useRef<HTMLInputElement>(null);
  const filesInput = useRef<HTMLInputElement>(null);

  const reset = useCallback(() => {
    setStep("source");
    setPicked([]);
    setPlan(null);
    setSession(null);
    setManifest(null);
    setUploaded(0);
    setError(null);
    setCreated(null);
    setBusy(false);
  }, []);

  const close = () => {
    if (session && !["done", "cancelled", "importing", "scanning"].includes(session.status) && session.mode !== "revise") {
      api.del(`/api/imports/${session.id}`).catch(() => undefined);
    }
    setOpen(false);
    reset();
  };

  // resume an existing session (e.g. "Revise dataset")
  useEffect(() => {
    if (open && resumeId) {
      api.get<ImportSession>(`/api/imports/${resumeId}`).then((s) => {
        setSession(s);
        setManifest(s.manifest);
        setName(s.manifest?.name ?? s.name ?? "");
        setStep(s.status === "review" ? "review" : s.status === "done" ? "done" : "scanning");
      }).catch((e) => setError(errorText(e)));
    }
  }, [open, resumeId]);

  // poll background scan / import
  useEffect(() => {
    if (!session || !(step === "scanning" || step === "importing")) return;
    const t = setInterval(async () => {
      try {
        const s = await api.get<ImportSession>(`/api/imports/${session.id}`);
        setSession(s);
        if (s.status === "review") {
          setManifest(s.manifest);
          setStep("review");
        } else if (s.status === "done" && s.dataset_id) {
          const rec = await api.get<DatasetRecord>(`/api/datasets/${s.dataset_id}`);
          setCreated(rec);
          qc.invalidateQueries({ queryKey: ["datasets"] });
          setStep("done");
        } else if (s.status === "failed") {
          setError(s.error ?? "The step failed.");
          setStep(s.manifest ? "review" : "source");
        }
      } catch (e) {
        setError(errorText(e));
      }
    }, 700);
    return () => clearInterval(t);
  }, [session, step, qc]);

  const pick = (list: FileList | null, folder: boolean) => {
    const files = Array.from(list ?? []);
    setPicked(files.map((f) => ({ file: f, path: folder ? (f.webkitRelativePath || f.name) : f.name })));
    setPlan(null);
    setError(null);
  };

  const makePlan = async () => {
    setBusy(true);
    setError(null);
    try {
      const s = await api.post<ImportSession>("/api/imports", { mode: method, name: name || null, attach_experiment_id: attachTo });
      setSession(s);
      const p = await api.post<ImportPlan>(`/api/imports/${s.id}/plan`, { files: picked.map((f) => ({ path: f.path, size: f.file.size })) });
      setPlan(p);
      setStep("plan");
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const uploadAndScan = async () => {
    if (!session || !plan) return;
    setStep("uploading");
    setError(null);
    const byPath = new Map(picked.map((p) => [p.path, p.file]));
    const wanted = plan.files.filter((f) => f.role !== "ignored").map((f) => ({ path: f.path, file: byPath.get(f.path)! })).filter((f) => f.file);
    let done = 0;
    try {
      let batch: typeof wanted = [];
      let size = 0;
      const flush = async () => {
        if (!batch.length) return;
        const form = new FormData();
        form.append("paths", JSON.stringify(batch.map((b) => b.path)));
        batch.forEach((b) => form.append("files", b.file, b.file.name));
        const base = done;
        await uploadWithProgress(`/api/imports/${session.id}/files`, form, (loaded) => setUploaded(Math.min(plan.upload_bytes, base + loaded)));
        done = base + batch.reduce((a, b) => a + b.file.size, 0);
        setUploaded(done);
        batch = [];
        size = 0;
      };
      for (const f of wanted) {
        if (batch.length && (size + f.file.size > BATCH_BYTES || batch.length >= BATCH_FILES)) await flush();
        batch.push(f);
        size += f.file.size;
      }
      await flush();
      const s = await api.post<ImportSession>(`/api/imports/${session.id}/scan`, {});
      setSession(s);
      setStep("scanning");
    } catch (e) {
      setError(errorText(e));
      setStep("plan");
    }
  };

  const scanLocal = async () => {
    setBusy(true);
    setError(null);
    try {
      const s = await api.post<ImportSession>("/api/imports", { mode: "local_path", local_path: localPath, name: name || null, attach_experiment_id: attachTo });
      setSession(s);
      setStep("scanning");
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const patch = async (p: ManifestPatch) => {
    if (!session) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.put<{ manifest: DatasetManifest }>(`/api/imports/${session.id}/manifest`, p);
      setManifest(r.manifest);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const runImport = async () => {
    if (!session) return;
    setBusy(true);
    setError(null);
    try {
      const s = await api.post<ImportSession>(`/api/imports/${session.id}/import`);
      setSession(s);
      setStep("importing");
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      setCreated(await api.post<DatasetRecord>("/api/datasets/synthetic", syn));
      qc.invalidateQueries({ queryKey: ["datasets"] });
      setStep("done");
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const attach = async (expId: string) => {
    if (!created) return;
    setBusy(true);
    try {
      await api.put(`/api/experiments/${expId}/dataset`, { dataset_id: created.id, workflow: "single" });
      setCurrent(expId);
      refresh(expId);
      close();
      navigate("/dataset");
    } catch (e) {
      setError(errorText(e));
      setBusy(false);
    }
  };

  const startNew = async () => {
    if (!created) return;
    setBusy(true);
    try {
      const exp = await api.post<Experiment>("/api/experiments", { dataset_id: created.id });
      setCurrent(exp.id);
      refresh(exp.id);
      close();
      navigate("/dataset");
    } catch (e) {
      setError(errorText(e));
      setBusy(false);
    }
  };

  const title = step === "review" ? "Review dataset" : step === "done" ? "Dataset imported" : session?.mode === "revise" ? "Revise dataset" : "Import dataset";
  const isRevise = session?.mode === "revise";
  const footer = (() => {
    if (step === "source")
      return (
        <>
          <Button variant="secondary" onClick={close}>Cancel</Button>
          {method === "synthetic" ? (
            <Button onClick={generate} loading={busy} icon={<FlaskConical className="size-4" />}>Generate synthetic dataset</Button>
          ) : method === "local_path" ? (
            <Button onClick={scanLocal} loading={busy} disabled={!localPath.trim()} icon={<FolderSearch className="size-4" />}>Scan folder</Button>
          ) : (
            <Button onClick={makePlan} loading={busy} disabled={!picked.length}>Continue</Button>
          )}
        </>
      );
    if (step === "plan")
      return (
        <>
          <Button variant="secondary" onClick={() => setStep("source")}>Back</Button>
          <Button onClick={uploadAndScan} disabled={!plan?.upload_files}>Upload {plan?.upload_files} files ({bytes(plan?.upload_bytes)}) &amp; scan</Button>
        </>
      );
    if (step === "review")
      return (
        <>
          <Button variant="secondary" onClick={close}>Cancel</Button>
          <Button onClick={runImport} loading={busy} disabled={!manifest?.importable || busy}>
            {isRevise ? "Import as a new dataset" : `Import ${manifest ? fmtInt(manifest.summary.n_trials) : ""} trials from ${manifest?.summary.n_included ?? 0} recordings`}
          </Button>
        </>
      );
    if (step === "done" && created)
      return (
        <>
          <Button variant="secondary" onClick={close}>Close</Button>
          {target && <Button onClick={() => attach(target.id)} loading={busy}>Attach to “{target.name}”</Button>}
          {!target && current && <Button variant="secondary" onClick={() => attach(current.id)} loading={busy}>Use in current experiment</Button>}
          {!target && <Button onClick={startNew} loading={busy}>Start new experiment with it</Button>}
        </>
      );
    return <Button variant="secondary" onClick={close}>Close</Button>;
  })();

  return (
    <Modal open={open} onClose={close} title={title} size={step === "review" ? "xl" : "lg"} footer={footer}>
      <div className="space-y-5 pb-2">
        {step === "source" && (
          <>
            <p className="text-[15px] leading-relaxed text-ink-muted">
              Import a whole study: every participant, session and run, with their event files and metadata. The folder is scanned first and you
              review what was found before anything is imported. Data stay on this machine.
            </p>
            <div role="radiogroup" aria-label="Import method" className="grid gap-2.5 md:grid-cols-2">
              <RadioCard checked={method === "folder"} onSelect={() => setMethod("folder")} icon={<FolderOpen className="size-5" />} title="Select dataset folder" description="A folder with participants, sessions, recordings, event files and metadata (nested folders are scanned)." />
              <RadioCard checked={method === "files"} onSelect={() => setMethod("files")} icon={<FileStack className="size-5" />} title="Select multiple files" description="Several recordings together with their event / companion / metadata files." />
              <RadioCard checked={method === "local_path"} onSelect={() => setMethod("local_path")} icon={<HardDrive className="size-5" />} title="Use local folder path" description="Read a folder in place on this computer — no upload, best for large datasets." />
              <RadioCard checked={method === "synthetic"} onSelect={() => setMethod("synthetic")} icon={<FlaskConical className="size-5" />} title="Synthetic test data" description="Generated, explicitly labelled SYNTHETIC data for testing the software." />
            </div>
            {method !== "synthetic" && <TextField label="Dataset name (optional)" value={name} onChange={setName} placeholder="defaults to the folder name or dataset_description.json" />}
            {(method === "folder" || method === "files") && (
              <div className="space-y-3">
                <input ref={folderInput} type="file" className="sr-only" aria-label="Dataset folder" onChange={(e) => pick(e.target.files, true)} {...({ webkitdirectory: "", directory: "" } as Record<string, string>)} />
                <input ref={filesInput} type="file" multiple className="sr-only" aria-label="EEG files" onChange={(e) => pick(e.target.files, false)} />
                <button type="button" onClick={() => (method === "folder" ? folderInput : filesInput).current?.click()} className="flex w-full flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-line-strong bg-surface px-6 py-8 text-center hover:border-primary">
                  {method === "folder" ? <FolderOpen className="size-7 text-link" /> : <FileStack className="size-7 text-link" />}
                  <span className="text-[15px] font-bold text-ink">
                    {picked.length ? `${picked.length} file(s) selected · ${bytes(picked.reduce((a, p) => a + p.file.size, 0))}` : method === "folder" ? "Choose a dataset folder" : "Choose files"}
                  </span>
                  <span className="text-[13px] text-ink-subtle">Only the file list is read now; contents are uploaded after you confirm which files are used.</span>
                </button>
              </div>
            )}
            {method === "local_path" && (
              <div className="space-y-2">
                <TextField label="Folder path on this computer" value={localPath} onChange={setLocalPath} placeholder="e.g. D:\\data\\motor_imagery_study" />
                <p className="text-[12.5px] text-ink-subtle">The folder is read in place and never modified. Network shares, drive roots and the app's own data folder are refused.</p>
              </div>
            )}
            {method === "synthetic" && (
              <>
                <Alert tone="warning" title="Explicitly synthetic">Generated motor-imagery-like signals for testing and teaching; not evidence of real-world EEG performance.</Alert>
                <div className="grid gap-4 sm:grid-cols-3">
                  <SelectField label="Structure" value={syn.kind} onChange={(v) => setSyn({ ...syn, kind: v })} options={[{ value: "epoched", label: "Epoched" }, { value: "continuous", label: "Continuous + events" }]} />
                  <NumberField label="Participants" value={syn.n_participants} min={1} max={40} onChange={(v) => setSyn({ ...syn, n_participants: v ?? 1 })} />
                  <NumberField label="Sessions" value={syn.n_sessions} min={1} max={4} onChange={(v) => setSyn({ ...syn, n_sessions: v ?? 1 })} />
                  <NumberField label="Classes" value={syn.n_classes} min={2} max={4} onChange={(v) => setSyn({ ...syn, n_classes: v ?? 2 })} />
                  <NumberField label="Trials per class" value={syn.trials_per_class} min={4} max={400} onChange={(v) => setSyn({ ...syn, trials_per_class: v ?? 4 })} />
                  <SelectField label="Channels" value={String(syn.n_channels) as "8" | "16" | "22"} onChange={(v) => setSyn({ ...syn, n_channels: Number(v) })} options={[{ value: "8", label: "8" }, { value: "16", label: "16" }, { value: "22", label: "22" }]} />
                  <NumberField label="Sampling rate (Hz)" value={syn.sfreq} min={64} max={1024} onChange={(v) => setSyn({ ...syn, sfreq: v ?? 128 })} />
                  <NumberField label="Seed" value={syn.seed} onChange={(v) => setSyn({ ...syn, seed: v ?? 0 })} />
                  <div className="pt-6"><Toggle label="Frontal Fp1/Fp2" checked={syn.include_frontal} onChange={(v) => setSyn({ ...syn, include_frontal: v })} /></div>
                </div>
                <div className="flex items-center gap-2 text-[13px] text-ink-subtle"><SyntheticBadge /> attached to the dataset record</div>
              </>
            )}
            <details className="rounded-2xl border border-line bg-surface px-4 py-3">
              <summary className="cursor-pointer text-[14px] font-bold text-ink">Supported formats, event files and naming</summary>
              <div className="mt-3 divide-y divide-line">
                {formats?.map((f) => (
                  <div key={f.format} className="py-2.5 text-[13px]">
                    <div className="flex flex-wrap items-center gap-2"><span className="font-bold text-ink">{f.format}</span>{f.extensions.map((e) => <Badge key={e}>{e}</Badge>)}</div>
                    <p className="mt-1 text-ink-muted">{f.requirements}</p>
                  </div>
                ))}
                <p className="pt-2.5 text-[13px] text-ink-muted">Participant / session / run IDs are recognised from BIDS names (sub-01/ses-01/…run-02), PhysioNet (S001R04), BCI-IV (A01T) and folder-per-participant layouts, and can be corrected in the review.</p>
              </div>
            </details>
          </>
        )}

        {step === "plan" && plan && (
          <div className="space-y-4">
            <p className="text-[15px] text-ink-muted">Found <b className="text-ink">{plan.files.length}</b> files{plan.root_label ? <> in <b className="text-ink">{plan.root_label}</b></> : null}. Only files that can contribute are uploaded:</p>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
              {(["eeg", "events", "companion", "metadata", "ignored"] as const).map((r) => (
                <div key={r} className={classNames("rounded-xl border px-3 py-2.5", r === "ignored" ? "border-line bg-subtle" : "border-line bg-surface")}>
                  <div className="text-[12px] font-bold uppercase tracking-[0.06em] text-ink-subtle">{r === "eeg" ? "recordings" : r}</div>
                  <div className="font-display text-[24px] text-ink">{plan.counts[r] ?? 0}</div>
                  <div className="text-[12px] text-ink-subtle">{bytes(plan.bytes[r] ?? 0)}</div>
                </div>
              ))}
            </div>
            {!!plan.counts.ignored && (
              <details className="rounded-xl border border-line bg-surface px-4 py-2 text-[13px]">
                <summary className="cursor-pointer font-bold text-ink">Not uploaded ({plan.counts.ignored})</summary>
                <ul className="mt-2 max-h-40 space-y-0.5 overflow-y-auto scrollbar-thin">
                  {plan.files.filter((f) => f.role === "ignored").map((f) => <li key={f.path}><span className="font-mono">{f.path}</span> — <span className="text-ink-subtle">{f.note}</span></li>)}
                </ul>
              </details>
            )}
            <p className="text-[12.5px] text-ink-subtle">CSV/TSV tables are classified by their content during the scan (event tables vs. signal tables).</p>
          </div>
        )}

        {step === "uploading" && plan && <ProgressBar value={plan.upload_bytes ? uploaded / plan.upload_bytes : 0} label={`Uploading ${bytes(uploaded)} of ${bytes(plan.upload_bytes)} (batches of up to ${BATCH_FILES} files)…`} />}
        {step === "scanning" && (
          <ProgressBar
            value={session?.progress?.total ? (session.progress.done ?? 0) / session.progress.total : 0.05}
            label={session?.progress?.message ?? "Scanning recordings (headers only)…"}
          />
        )}
        {step === "importing" && (
          <ProgressBar value={session?.progress?.total ? (session.progress.done ?? 0) / session.progress.total : 0.05} label={session?.progress?.message ?? "Importing recordings…"} />
        )}
        {step === "review" && manifest && (
          <>
            {isRevise && <Alert tone="info" title="Revision">Importing creates a new dataset. Existing experiments keep using the original until you attach the new one; doing so marks their results stale.</Alert>}
            <DatasetReview manifest={manifest} onPatch={patch} busy={busy} />
          </>
        )}
        {step === "done" && created && (
          <div className="space-y-4">
            <Alert tone="success" title={`Dataset ready: ${created.name}`}>
              {created.summary.n_participants || "No"} participant(s), {created.summary.n_recordings ?? created.summary.n_epochs} {created.summary.structure === "continuous" ? "recording(s)" : "epochs"},{" "}
              {fmtInt(created.summary.n_trials)} trials in {created.summary.n_classes} classes, {created.summary.n_channels} channels at {created.summary.sfreq} Hz.
            </Alert>
            {created.synthetic && <Alert tone="warning" title="Synthetic data">{created.notes[0]}</Alert>}
            {!created.trainable && <Alert tone="danger" title="Not usable for training">No labelled trials were found.</Alert>}
            {target && <p className="text-[14px] text-ink-muted">This dataset was imported for experiment <b className="text-ink">{target.name}</b>.</p>}
            <div className="flex items-center gap-2 text-[14px] text-success"><CheckCircle2 className="size-4" /> Fingerprint <span className="font-mono">{created.fingerprint.slice(0, 16)}</span></div>
          </div>
        )}
        {error && (
          <Alert tone="danger" title="Something needs attention">
            {error}
            {session?.status === "failed" && <div className="mt-1 text-[12.5px]">You can adjust the review and try again.</div>}
          </Alert>
        )}
      </div>
    </Modal>
  );
}

