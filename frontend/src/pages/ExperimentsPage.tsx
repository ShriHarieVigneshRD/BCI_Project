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
import { SelectField, TextField } from "../components/ui/Field";
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

  // Automatically select an experiment if none is selected
  useEffect(() => {
    if ((!selected || !list?.some((e) => e.id === selected)) && list && list.length > 0) {
      setSelected(current && list.some((e) => e.id === current) ? current : list[0].id);
    }
  }, [list, current, selected]);

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
      <PageHeader
        eyebrow="Separate area"
        title="Experiments"
        accent="& exports."
        lead="Every experiment keeps its dataset fingerprint, split assignments, preprocessing, architecture, training settings, software versions, checkpoints and metrics. Nothing is overwritten."
      />

      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="w-72">
            <TextField label="Search" value={search} onChange={setSearch} placeholder="name or ID" />
          </div>
          <div className="w-56">
            <SelectField
              label="State"
              value={state}
              onChange={setState}
              options={[
                { value: "", label: "All states" },
                ...(Object.keys(STATE_LABEL) as ExperimentState[]).map((s) => ({ value: s, label: STATE_LABEL[s] })),
              ]}
            />
          </div>
        </div>
        {list && list.length > 0 && (
          <div className="text-[13px] text-ink-subtle">
            Showing <b className="text-ink">{list.length}</b> {list.length === 1 ? "experiment" : "experiments"}
          </div>
        )}
      </div>

      {error && <Alert tone="danger" className="mb-4">{error}</Alert>}

      {isLoading ? (
        <Loading />
      ) : !list?.length ? (
        <EmptyState title="No experiments match">Start a new experiment from the header.</EmptyState>
      ) : (
        <div className="space-y-8">
          {/* Top master table: full width, clean headers, right-aligned metrics, no cramped columns */}
          <Card className="overflow-hidden">
            <div className="max-h-[380px] overflow-auto scrollbar-thin">
              <table className="w-full text-[13.5px] tabular">
                <thead className="sticky top-0 z-10 border-b border-line bg-subtle text-left text-ink-subtle shadow-sm">
                  <tr>
                    <th className="px-4 py-3 font-semibold whitespace-nowrap text-ink">Experiment</th>
                    <th className="px-3.5 py-3 font-semibold whitespace-nowrap">State</th>
                    <th className="px-3.5 py-3 font-semibold whitespace-nowrap">Dataset</th>
                    <th className="px-3.5 py-3 font-semibold whitespace-nowrap">Protocol</th>
                    <th className="px-3.5 py-3 font-semibold whitespace-nowrap text-right">Best val.</th>
                    <th className="px-3.5 py-3 font-semibold whitespace-nowrap text-right">Test bal. acc.</th>
                    <th className="px-3.5 py-3 font-semibold whitespace-nowrap">Created</th>
                    <th className="pr-4 pl-3.5 py-3 font-semibold whitespace-nowrap text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {list.map((e) => {
                    const isSelected = selected === e.id;
                    return (
                      <tr
                        key={e.id}
                        onClick={() => setSelected(e.id)}
                        className={classNames(
                          "cursor-pointer border-t border-line transition-colors hover:bg-surface",
                          isSelected ? "bg-primary-soft/45 font-medium" : ""
                        )}
                      >
                        <td className="px-4 py-2.5">
                          <div className="flex items-center gap-2 font-bold text-ink">
                            <span className="truncate">{e.name}</span>
                            {e.id === current && <Badge tone="primary">current</Badge>}
                            {isSelected && <Badge tone="neutral">selected</Badge>}
                          </div>
                          <div className="mt-0.5 flex items-center gap-1.5 text-[12px] text-ink-subtle">
                            <Mono>{e.id}</Mono>
                            {e.parent_id && <span>· derived</span>}
                            {e.synthetic && <SyntheticBadge />}
                          </div>
                        </td>
                        <td className="px-3.5 py-2.5 whitespace-nowrap">
                          <div className="flex items-center gap-1">
                            <StateChip state={e.state} />
                            {e.stale && <Badge tone="warning">stale</Badge>}
                          </div>
                        </td>
                        <td className="px-3.5 py-2.5 max-w-44 truncate text-ink-muted whitespace-nowrap" title={e.dataset_name ?? undefined}>
                          {e.dataset_name ?? "—"}
                        </td>
                        <td className="px-3.5 py-2.5 text-ink-muted whitespace-nowrap">
                          {e.strategy ?? "—"}
                        </td>
                        <td className="px-3.5 py-2.5 text-right font-medium whitespace-nowrap tabular text-ink">
                          {e.best_val_balanced_accuracy !== null ? pct(e.best_val_balanced_accuracy) : "—"}
                        </td>
                        <td className="px-3.5 py-2.5 text-right font-bold whitespace-nowrap tabular text-ink">
                          {e.test_balanced_accuracy !== null ? pct(e.test_balanced_accuracy) : "—"}
                        </td>
                        <td className="px-3.5 py-2.5 text-[12.5px] whitespace-nowrap text-ink-subtle tabular">
                          {dateTime(e.created_at)}
                        </td>
                        <td className="pr-4 pl-3.5 py-2.5 text-right whitespace-nowrap" onClick={(ev) => ev.stopPropagation()}>
                          <div className="flex justify-end gap-1">
                            <IconBtn label="Open" onClick={() => open(e.id)}><FolderOpen className="size-4" /></IconBtn>
                            <IconBtn label="Rename" onClick={() => setRename({ id: e.id, name: e.name })}><Pencil className="size-4" /></IconBtn>
                            <IconBtn label="Duplicate" onClick={() => duplicate(e.id)}><Copy className="size-4" /></IconBtn>
                            <IconBtn label="Delete" danger onClick={() => setDel({ id: e.id, name: e.name })}><Trash2 className="size-4" /></IconBtn>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Selected experiment details & export sections */}
          {selected && <ExperimentDetail id={selected} onOpen={open} />}
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

const ITEM_ORDER = [
  "dataset_manifest",
  "trial_provenance",
  "model_config",
  "training_config",
  "preprocessing_config",
  "split_manifest",
  "metrics",
  "predictions",
  "history",
  "confusion",
  "plots",
  "checkpoints",
  "report",
];

function ExperimentDetail({ id, onOpen }: { id: string; onOpen?: (id: string) => void }) {
  const { data: exp } = useExperiment(id);
  const current = useUI((s) => s.currentExperimentId);
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

  const selectAll = () => {
    if (!items) return;
    setChosen(ITEM_ORDER.filter((k) => items[k]));
  };
  const clearAll = () => setChosen([]);

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
    <div className="space-y-6 pt-1">
      {/* Detail header bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-line pb-3">
        <div>
          <div className="text-[12px] font-bold uppercase tracking-wider text-ink-subtle">Selected experiment details</div>
          <div className="mt-0.5 flex flex-wrap items-center gap-2.5">
            <h2 className="text-[20px] font-bold text-ink">{exp.name}</h2>
            <StateChip state={exp.state} />
            {exp.id === current && <Badge tone="primary">current</Badge>}
            {exp.runs.some((r) => r.stale) && <Badge tone="warning">stale</Badge>}
          </div>
        </div>
        {onOpen && (
          <Button variant="secondary" icon={<FolderOpen className="size-4" />} onClick={() => onOpen(exp.id)}>
            Open in workflow
          </Button>
        )}
      </div>

      {/* Balanced 2-column layout: Overview & Export Bundle on Left, Artifacts & Lifecycle on Right */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Left Column: Configuration & Export */}
        <div className="space-y-6">
          {/* Overview Card */}
          <Card>
            <CardHeader
              title="Configuration & provenance"
              subtitle={<Mono className="text-[13px]">{exp.id}</Mono>}
            />
            <CardBody className="space-y-3">
              <div className="divide-y divide-line/60 rounded-xl border border-line bg-surface/30 text-[13px]">
                {exp.datasets.map((d) => (
                  <div key={d.id} className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="w-28 shrink-0 font-medium text-ink-subtle">Dataset</span>
                      <b className="text-ink">{d.name}</b>
                      <Mono className="text-[12px] text-ink-subtle">{d.fingerprint.slice(0, 12)}</Mono>
                      {d.synthetic && <SyntheticBadge />}
                    </div>
                    <a className="link-dotted text-[12.5px]" href={`/api/datasets/${d.id}/export`}>
                      Export dataset (explicit)
                    </a>
                  </div>
                ))}
                <div className="flex items-center gap-2 px-4 py-2.5">
                  <span className="w-28 shrink-0 font-medium text-ink-subtle">Split protocol</span>
                  <span className="font-medium text-ink">{exp.split_config?.strategy ?? "—"}</span>
                  {exp.split_fingerprint && <Mono className="text-[12px] text-ink-subtle">{exp.split_fingerprint.slice(0, 14)}</Mono>}
                </div>
                <div className="flex items-center gap-2 px-4 py-2.5">
                  <span className="w-28 shrink-0 font-medium text-ink-subtle">Preprocessing</span>
                  <Mono className="text-[12px] text-ink">{exp.preprocessing_fingerprint ?? "—"}</Mono>
                </div>
                <div className="flex items-center gap-2 px-4 py-2.5">
                  <span className="w-28 shrink-0 font-medium text-ink-subtle">Runs</span>
                  <span className="font-medium text-ink">{exp.runs.length}</span>
                  {exp.runs.some((r) => r.stale) && <Badge tone="warning">some stale</Badge>}
                </div>
                {exp.active_run && (
                  <div className="flex flex-wrap items-start gap-2 px-4 py-2.5">
                    <span className="w-28 shrink-0 pt-0.5 font-medium text-ink-subtle">Software</span>
                    <div className="flex min-w-0 flex-1 flex-wrap gap-1.5">
                      {Object.entries(exp.active_run.software).slice(0, 6).map(([k, v]) => (
                        <span key={k} className="rounded-md border border-line/60 bg-subtle px-2 py-0.5 font-mono text-[11.5px] text-ink-muted">
                          {k} {v}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              {exp.warnings.map((w) => <Alert key={w} tone="warning">{w}</Alert>)}
            </CardBody>
          </Card>

          {/* Export Bundle Card */}
          <Card>
            <CardHeader
              icon={<Package className="size-5" />}
              title="Export bundle"
              subtitle="A ZIP with a manifest of every file, its sha256 and provenance. Raw EEG data are never included automatically."
              actions={
                items && Object.keys(items).length > 0 ? (
                  <div className="flex items-center gap-2 text-[12.5px]">
                    <button type="button" onClick={selectAll} className="font-medium text-link hover:underline">
                      Select all
                    </button>
                    <span className="text-ink-subtle">·</span>
                    <button type="button" onClick={clearAll} className="font-medium text-ink-subtle hover:text-ink hover:underline">
                      Clear
                    </button>
                  </div>
                ) : undefined
              }
            />
            <CardBody className="space-y-4">
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {ITEM_ORDER.filter((k) => items?.[k]).map((k) => {
                  const isChecked = chosen.includes(k);
                  return (
                    <label
                      key={k}
                      className={classNames(
                        "flex cursor-pointer items-start gap-2.5 rounded-xl border p-2.5 text-[13px] transition-colors",
                        isChecked
                          ? "border-primary/50 bg-primary-soft/30 text-ink"
                          : "border-line bg-surface/30 text-ink-muted hover:border-line-strong hover:bg-surface/70"
                      )}
                    >
                      <input
                        type="checkbox"
                        className="mt-0.5 size-4 shrink-0 rounded border-line-strong accent-[var(--c-primary)]"
                        checked={isChecked}
                        onChange={(e) => setChosen((c) => (e.target.checked ? [...c, k] : c.filter((x) => x !== k)))}
                      />
                      <span className="leading-snug">{items![k]}</span>
                    </label>
                  );
                })}
              </div>

              {error && <Alert tone="danger">{error}</Alert>}

              <div className="flex items-center gap-3 pt-1">
                <Button icon={<Package className="size-4" />} onClick={build} loading={busy} disabled={!chosen.length}>
                  Build export {chosen.length > 0 && `(${chosen.length})`}
                </Button>
              </div>

              {rec && (
                <div className="space-y-2.5 rounded-2xl border border-success-line bg-success-soft/40 p-4">
                  <div className="flex flex-wrap items-center gap-3">
                    <b className="text-ink">{rec.filename}</b>
                    <span className="text-[13px] text-ink-muted">{bytes(rec.size_bytes)}</span>
                    <a
                      href={`/api/exports/${rec.id}/download`}
                      className="ml-auto inline-flex items-center gap-1.5 rounded-full bg-primary px-4 py-1.5 text-[13.5px] font-bold text-on-primary shadow-sm hover:opacity-95"
                    >
                      <Download className="size-4" /> Download
                    </a>
                  </div>
                  <ul className="max-h-56 space-y-1 overflow-y-auto rounded-xl border border-success-line/60 bg-card/60 p-2.5 text-[12px] scrollbar-thin">
                    <li className="font-mono font-bold text-ink">manifest.json</li>
                    {rec.manifest.files.map((f) => (
                      <li key={f.path} className="flex flex-wrap items-center justify-between gap-2 border-t border-line/40 pt-1 font-mono">
                        <Mono className="text-ink">{f.path}</Mono>
                        <span className="text-ink-subtle">{bytes(f.bytes)} · {f.sha256.slice(0, 10)}… · {f.provenance}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {ckpts.length > 0 && (
                <div className="space-y-1.5 pt-1">
                  <div className="text-[12px] font-bold uppercase tracking-wider text-ink-subtle">Checkpoints</div>
                  <div className="flex flex-wrap gap-2">
                    {ckpts.map((c) => (
                      <a
                        key={c.id}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface/60 px-2.5 py-1 text-[12.5px] text-link hover:border-primary/50 hover:bg-primary-soft/30"
                        href={`/api/experiments/${id}/checkpoints/${c.id}/download`}
                      >
                        <Download className="size-3.5" />
                        <Mono>{c.id}</Mono>
                        <span className="text-ink-subtle">({bytes(c.size_bytes)})</span>
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </CardBody>
          </Card>
        </div>

        {/* Right Column: Artifacts & Lifecycle Log */}
        <div className="space-y-6">
          {/* Artifact Manifest Card */}
          <Card>
            <CardHeader
              title="Artifact manifest"
              subtitle="Storage references are relative keys inside the local store; no filesystem paths are exposed."
              actions={arts?.manifest.length ? <Badge tone="neutral">{arts.manifest.length} artifacts</Badge> : undefined}
            />
            <CardBody>
              {!arts?.manifest.length ? (
                <div className="rounded-xl border border-line/60 bg-surface/30 p-4 text-center text-[13px] text-ink-subtle">
                  No artifacts generated yet. Run training to generate checkpoints and activations.
                </div>
              ) : (
                <div className="max-h-72 overflow-auto rounded-xl border border-line scrollbar-thin">
                  <table className="w-full text-[12.5px] tabular">
                    <thead className="sticky top-0 z-10 border-b border-line bg-subtle text-left text-ink-subtle">
                      <tr>
                        <th className="px-3 py-2 font-semibold whitespace-nowrap">Type</th>
                        <th className="px-3 py-2 font-semibold whitespace-nowrap">ID</th>
                        <th className="px-3 py-2 font-semibold whitespace-nowrap">Checkpoint</th>
                        <th className="px-3 py-2 font-semibold whitespace-nowrap">Config fp</th>
                        <th className="pr-3 pl-2 py-2 font-semibold whitespace-nowrap text-right">sha256</th>
                      </tr>
                    </thead>
                    <tbody>
                      {arts.manifest.map((a) => (
                        <tr key={a.id} className="border-t border-line transition-colors hover:bg-surface/50">
                          <td className="px-3 py-2 font-medium text-ink whitespace-nowrap">{a.type}</td>
                          <td className="px-3 py-2 whitespace-nowrap"><Mono className="text-[12px]">{a.id}</Mono></td>
                          <td className="px-3 py-2 whitespace-nowrap"><Mono className="text-[12px]">{a.checkpoint_id?.slice(0, 15) ?? "—"}</Mono></td>
                          <td className="px-3 py-2 whitespace-nowrap"><Mono className="text-[12px]">{a.config_fingerprint ? a.config_fingerprint.slice(0, 12) : "—"}</Mono></td>
                          <td className="pr-3 pl-2 py-2 text-right whitespace-nowrap"><Mono className="text-[12px]">{a.sha256?.slice(0, 10) ?? "—"}</Mono></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardBody>
          </Card>

          {/* Lifecycle Log Card */}
          <Card>
            <CardHeader
              title="Lifecycle log"
              subtitle="Audit trail of experiment state transitions"
              actions={events?.length ? <Badge tone="neutral">{events.length} events</Badge> : undefined}
            />
            <CardBody>
              {!events?.length ? (
                <div className="rounded-xl border border-line/60 bg-surface/30 p-4 text-center text-[13px] text-ink-subtle">
                  No lifecycle events recorded yet.
                </div>
              ) : (
                <div className="max-h-72 overflow-y-auto rounded-xl border border-line scrollbar-thin">
                  <div className="divide-y divide-line text-[12.5px]">
                    {events.map((e, i) => (
                      <div key={i} className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3.5 py-2 transition-colors hover:bg-surface/50">
                        <span className="w-36 shrink-0 text-ink-subtle tabular-nums">{dateTime(e.ts)}</span>
                        <div className={classNames("flex shrink-0 items-center gap-1.5 font-mono text-[12px] font-bold", e.kind === "rollback" ? "text-warning" : "text-ink")}>
                          {e.from_state ? (
                            <>
                              <span className="rounded bg-subtle px-1.5 py-0.5 text-ink-muted">{e.from_state}</span>
                              <span className="text-ink-subtle">→</span>
                            </>
                          ) : null}
                          <span className="rounded bg-primary-soft/60 px-1.5 py-0.5 text-link">{e.to_state}</span>
                        </div>
                        <span className="min-w-0 flex-1 truncate text-ink-muted" title={e.reason ?? ""}>
                          {e.reason}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardBody>
          </Card>
        </div>
      </div>
    </div>
  );
}
