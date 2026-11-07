import { useEffect, useState, type ReactNode } from "react";
import { useNavigate } from "react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Copy, Download, FolderOpen, Package, Pencil, Trash2 } from "lucide-react";
import { api, errorText, qs } from "../api/client";
import { useExperiment, useExperiments } from "../api/hooks";
import type { ArtifactsResponse, ExperimentState, ExportRecord, LifecycleEvent } from "../api/types";
import { useUI } from "../store/ui";
import { Button } from "../components/ui/Button";
import { Card, CardBody, CardHeader } from "../components/ui/Card";
import { Badge, StateChip, SyntheticBadge } from "../components/ui/Chip";
import { Checkbox, SelectField, TextField } from "../components/ui/Field";
import { Alert, ConfirmDialog, EmptyState, Loading, Modal, Mono, PageHeader } from "../components/ui/Misc";
import { bytes, classNames, dateTime, pct, STATE_LABEL } from "../lib/format";

export function ExperimentsPage() {
  const [search, setSearch] = useState("");
  const [state, setState] = useState("");
  const { data: list, isLoading } = useExperiments(search || undefined, state || undefined);
  const current = useUI((s) => s.currentExperimentId);
  const setCurrent = useUI((s) => s.setCurrentExperiment);
  const [selected, setSelected] = useState<string | null>(current);
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [rename, setRename] = useState<{ id: string; name: string } | null>(null);
  const [del, setDel] = useState<{ id: string; name: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshList = () => qc.invalidateQueries({ queryKey: ["experiments"] });
  const open = async (id: string) => {
    setCurrent(id);
    const e = await api.get<{ capabilities: { next_page: string } }>(`/api/experiments/${id}`);
    navigate(`/${e.capabilities.next_page}`);
  };
  const duplicate = async (id: string) => {
    setError(null);
    try {
      const e = await api.post<{ id: string }>(`/api/experiments/${id}/duplicate`, {});
      refreshList();
      setSelected(e.id);
    } catch (e) {
      setError(errorText(e));
    }
  };
  const doRename = async () => {
    if (!rename) return;
    setBusy(true);
    try {
      await api.patch(`/api/experiments/${rename.id}`, { name: rename.name });
      refreshList();
      qc.invalidateQueries({ queryKey: ["experiment", rename.id] });
      setRename(null);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  const doDelete = async () => {
    if (!del) return;
    setBusy(true);
    try {
      await api.del(`/api/experiments/${del.id}${qs({ confirm: true })}`);
      if (current === del.id) setCurrent(null);
      if (selected === del.id) setSelected(null);
      refreshList();
      setDel(null);
    } catch (e) {
      setError(errorText(e));
      setDel(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader eyebrow="Separate area" title="Experiments" accent="& exports." lead="Every experiment keeps its dataset fingerprint, split assignments, preprocessing, architecture, training settings, software versions, checkpoints and metrics. Nothing is overwritten." />
      <div className="mb-5 flex flex-wrap items-end gap-3">
        <div className="w-80">
          <TextField label="Search" value={search} onChange={setSearch} placeholder="name or ID" />
        </div>
        <div className="w-60">
          <SelectField label="State" value={state} onChange={setState} options={[{ value: "", label: "All states" }, ...(Object.keys(STATE_LABEL) as ExperimentState[]).map((s) => ({ value: s, label: STATE_LABEL[s] }))]} />
        </div>
      </div>
      {error && <Alert tone="danger" className="mb-4">{error}</Alert>}
      {isLoading ? (
        <Loading />
      ) : !list?.length ? (
        <EmptyState title="No experiments match">Start a new experiment from the header.</EmptyState>
      ) : (
        <div className="grid gap-6 2xl:grid-cols-[minmax(0,1fr)_520px]">
          <Card className="overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-[13.5px] tabular">
                <thead className="bg-subtle text-left text-ink-subtle">
                  <tr><th className="px-4 py-2.5">Experiment</th><th>State</th><th>Dataset</th><th>Protocol</th><th>Best val.</th><th>Test bal. acc.</th><th>Created</th><th className="pr-4 text-right">Actions</th></tr>
                </thead>
                <tbody>
                  {list.map((e) => (
                    <tr key={e.id} onClick={() => setSelected(e.id)} className={classNames("cursor-pointer border-t border-line hover:bg-surface", selected === e.id && "bg-primary-soft/40")}>
                      <td className="px-4 py-2">
                        <div className="flex items-center gap-2 font-bold text-ink">{e.name}{e.id === current && <Badge tone="primary">current</Badge>}</div>
                        <div className="flex items-center gap-1.5 text-[12px] text-ink-subtle"><Mono>{e.id}</Mono>{e.parent_id && <span>· derived</span>}{e.synthetic && <SyntheticBadge />}</div>
                      </td>
                      <td><StateChip state={e.state} />{e.stale && <Badge tone="warning" className="ml-1">stale</Badge>}</td>
                      <td className="max-w-48 truncate text-ink-muted">{e.dataset_name ?? "—"}</td>
                      <td className="text-ink-muted">{e.strategy ?? "—"}</td>
                      <td>{e.best_val_balanced_accuracy !== null ? pct(e.best_val_balanced_accuracy) : "—"}</td>
                      <td className="font-bold">{e.test_balanced_accuracy !== null ? pct(e.test_balanced_accuracy) : "—"}</td>
                      <td className="text-[12.5px] whitespace-nowrap text-ink-subtle">{dateTime(e.created_at)}</td>
                      <td className="pr-4" onClick={(ev) => ev.stopPropagation()}>
                        <div className="flex justify-end gap-0.5">
                          <IconBtn label="Open" onClick={() => open(e.id)}><FolderOpen className="size-4" /></IconBtn>
                          <IconBtn label="Rename" onClick={() => setRename({ id: e.id, name: e.name })}><Pencil className="size-4" /></IconBtn>
                          <IconBtn label="Duplicate" onClick={() => duplicate(e.id)}><Copy className="size-4" /></IconBtn>
                          <IconBtn label="Delete" danger onClick={() => setDel({ id: e.id, name: e.name })}><Trash2 className="size-4" /></IconBtn>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          {selected && <ExperimentDetail id={selected} />}
        </div>
      )}
      <Modal open={!!rename} onClose={() => setRename(null)} title="Rename experiment" footer={<><Button variant="secondary" onClick={() => setRename(null)}>Cancel</Button><Button onClick={doRename} loading={busy} disabled={!rename?.name.trim()}>Save</Button></>}>
        {rename && <TextField label="Name" value={rename.name} onChange={(v) => setRename({ ...rename, name: v })} />}
      </Modal>
      <ConfirmDialog
        open={!!del}
        danger
        loading={busy}
        title="Delete experiment?"
        confirmLabel="Delete permanently"
        onClose={() => setDel(null)}
        onConfirm={doDelete}
        body={<>This permanently deletes <b className="text-ink">{del?.name}</b>, including its runs, checkpoints, predictions and exports. Datasets are kept. This cannot be undone.</>}
      />
    </>
  );
}

function IconBtn({ label, onClick, children, danger }: { label: string; onClick: () => void; children: ReactNode; danger?: boolean }) {
  return (
    <button type="button" title={label} aria-label={label} onClick={onClick} className={classNames("rounded-lg p-1.5", danger ? "text-danger hover:bg-danger-soft" : "text-link hover:bg-primary-soft")}>
      {children}
    </button>
  );
}

const ITEM_ORDER = ["dataset_manifest", "trial_provenance", "model_config", "training_config", "preprocessing_config", "split_manifest", "metrics", "predictions", "history", "confusion", "plots", "checkpoints", "report"];

function ExperimentDetail({ id }: { id: string }) {
  const { data: exp } = useExperiment(id);
  const { data: items } = useQuery({ queryKey: ["export-items", id], queryFn: () => api.get<Record<string, string>>(`/api/experiments/${id}/export/items`) });
  const { data: events } = useQuery({ queryKey: ["events", id, exp?.state], queryFn: () => api.get<LifecycleEvent[]>(`/api/experiments/${id}/events`) });
  const { data: arts } = useQuery({ queryKey: ["artifacts", id, exp?.state], queryFn: () => api.get<ArtifactsResponse>(`/api/experiments/${id}/artifacts`) });
  const [chosen, setChosen] = useState<string[]>([]);
  const [rec, setRec] = useState<ExportRecord | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!exp) return;
    const caps = exp.capabilities;
    const trained = exp.active_run?.status === "completed";
    setChosen(ITEM_ORDER.filter((k) => {
      if (["metrics", "predictions", "confusion"].includes(k)) return caps.has_results;
      if (["history", "plots", "checkpoints", "report"].includes(k)) return trained;
      if (k === "model_config") return !!exp.architecture;
      if (k === "preprocessing_config") return !!exp.preprocessing_config;
      if (k === "split_manifest" || k === "trial_provenance") return !!exp.split_fingerprint;
      if (k === "dataset_manifest") return !!exp.dataset_id;
      return true;
    }));
    setRec(null);
  }, [exp?.id, exp?.state]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!exp) return <Loading />;
  const build = async () => {
    setBusy(true);
    setError(null);
    try {
      setRec(await api.post<ExportRecord>(`/api/experiments/${id}/export`, { items: chosen }));
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  const ckpts = arts?.manifest.filter((a) => a.type === "checkpoint") ?? [];
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader title={exp.name} subtitle={<Mono>{exp.id}</Mono>} actions={<StateChip state={exp.state} />} />
        <CardBody className="space-y-2 text-[13.5px]">
          {exp.datasets.map((d) => (
            <div key={d.id} className="flex flex-wrap items-center gap-2">
              <span className="text-ink-subtle">Dataset</span> <b className="text-ink">{d.name}</b> <Mono className="text-ink-subtle">{d.fingerprint.slice(0, 12)}</Mono>
              {d.synthetic && <SyntheticBadge />}
              <a className="link-dotted ml-auto text-[13px]" href={`/api/datasets/${d.id}/export`}>Export dataset (explicit)</a>
            </div>
          ))}
          <div><span className="text-ink-subtle">Split</span> {exp.split_config?.strategy ?? "—"} <Mono className="text-ink-subtle">{exp.split_fingerprint ?? ""}</Mono></div>
          <div><span className="text-ink-subtle">Preprocessing</span> <Mono>{exp.preprocessing_fingerprint ?? "—"}</Mono></div>
          <div><span className="text-ink-subtle">Runs</span> {exp.runs.length} {exp.runs.some((r) => r.stale) && <Badge tone="warning">some stale</Badge>}</div>
          {exp.active_run && <div><span className="text-ink-subtle">Software</span> {Object.entries(exp.active_run.software).slice(0, 6).map(([k, v]) => `${k} ${v}`).join(" · ")}</div>}
          {exp.warnings.map((w) => <Alert key={w} tone="warning">{w}</Alert>)}
        </CardBody>
      </Card>
      <Card>
        <CardHeader icon={<Package className="size-5" />} title="Export bundle" subtitle="A ZIP with a manifest of every file, its sha256 and provenance. Raw EEG data are never included automatically." />
        <CardBody className="space-y-4">
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {ITEM_ORDER.filter((k) => items?.[k]).map((k) => (
              <Checkbox key={k} checked={chosen.includes(k)} onChange={(v) => setChosen((c) => (v ? [...c, k] : c.filter((x) => x !== k)))} label={items![k]} />
            ))}
          </div>
          {error && <Alert tone="danger">{error}</Alert>}
          <Button icon={<Package className="size-4" />} onClick={build} loading={busy} disabled={!chosen.length}>Build export</Button>
          {rec && (
            <div className="space-y-2 rounded-2xl border border-success-line bg-success-soft/40 p-4">
              <div className="flex flex-wrap items-center gap-3">
                <b className="text-ink">{rec.filename}</b> <span className="text-[13px] text-ink-muted">{bytes(rec.size_bytes)}</span>
                <a href={`/api/exports/${rec.id}/download`} className="ml-auto inline-flex items-center gap-1.5 rounded-full bg-primary px-4 py-1.5 text-[13.5px] font-bold text-on-primary"><Download className="size-4" /> Download</a>
              </div>
              <ul className="max-h-56 space-y-0.5 overflow-y-auto text-[12px] scrollbar-thin">
                <li className="font-mono text-ink">manifest.json</li>
                {rec.manifest.files.map((f) => <li key={f.path} className="flex gap-2"><Mono className="text-ink">{f.path}</Mono><span className="text-ink-subtle">{bytes(f.bytes)} · {f.sha256.slice(0, 10)}… · {f.provenance}</span></li>)}
              </ul>
            </div>
          )}
          {ckpts.length > 0 && (
            <div className="text-[13px]">
              <div className="mb-1 font-bold text-ink-muted">Checkpoints</div>
              {ckpts.map((c) => <a key={c.id} className="link-dotted mr-3" href={`/api/experiments/${id}/checkpoints/${c.id}/download`}>{c.id} ({bytes(c.size_bytes)})</a>)}
            </div>
          )}
        </CardBody>
      </Card>
      {arts && arts.manifest.length > 0 && (
        <Card>
          <CardHeader title="Artifact manifest" subtitle="Storage references are relative keys inside the local store; no filesystem paths are exposed." />
          <CardBody>
            <div className="max-h-64 overflow-auto rounded-xl border border-line scrollbar-thin">
              <table className="w-full text-[12px]">
                <thead className="sticky top-0 bg-subtle text-left text-ink-subtle"><tr><th className="px-2 py-1.5">Type</th><th>ID</th><th>Checkpoint</th><th>Config fp</th><th className="pr-2">sha256</th></tr></thead>
                <tbody>{arts.manifest.map((a) => <tr key={a.id} className="border-t border-line"><td className="px-2 py-1">{a.type}</td><td><Mono>{a.id}</Mono></td><td><Mono>{a.checkpoint_id?.slice(0, 15) ?? "—"}</Mono></td><td><Mono>{a.config_fingerprint ?? "—"}</Mono></td><td className="pr-2"><Mono>{a.sha256?.slice(0, 10) ?? "—"}</Mono></td></tr>)}</tbody>
              </table>
            </div>
          </CardBody>
        </Card>
      )}
      <Card>
        <CardHeader title="Lifecycle log" />
        <CardBody>
          <ol className="max-h-72 space-y-1.5 overflow-y-auto text-[13px] scrollbar-thin">
            {events?.map((e, i) => (
              <li key={i} className="flex gap-3">
                <span className="w-36 shrink-0 text-ink-subtle">{dateTime(e.ts)}</span>
                <span className={classNames("font-bold", e.kind === "rollback" ? "text-warning" : "text-ink")}>{e.from_state ? `${e.from_state} → ` : ""}{e.to_state}</span>
                <span className="min-w-0 truncate text-ink-muted" title={e.reason ?? ""}>{e.reason}</span>
              </li>
            ))}
          </ol>
        </CardBody>
      </Card>
    </div>
  );
}
