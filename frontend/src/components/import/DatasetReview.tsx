import { useEffect, useMemo, useState, type ReactNode } from "react";
import { AlertTriangle, CheckCircle2, CircleSlash, FileText, Search, Wand2 } from "lucide-react";
import type { DatasetManifest, FileRole, ManifestRecording, MappingAction, MappingRow } from "../../api/types";
import { classNames, fmtInt, seconds } from "../../lib/format";
import { Button } from "../ui/Button";
import { Stat } from "../ui/Card";
import { Badge } from "../ui/Chip";
import { NumberField, Segmented, SelectField, TextField } from "../ui/Field";
import { Alert, Mono, Tabs } from "../ui/Misc";

export interface ManifestPatch {
  name?: string;
  recordings?: { index: number; participant?: string | null; session?: string | null; run?: string | null; task?: string | null; included?: boolean }[];
  mapping?: { task: string | null; event: string; action: MappingAction; label: string | null }[];
  rules?: { pattern: string; action: MappingAction; label: string | null; task?: string | null }[];
  channel_policy?: "intersection" | "strict";
  sfreq_policy?: "resample" | "strict";
  target_sfreq?: number | null;
  array_options?: DatasetManifest["array_options"];
}

type Tab = "participants" | "recordings" | "mapping" | "channels" | "files";

const ACTIONS: { value: MappingAction; label: string }[] = [
  { value: "class", label: "Class (event onset)" },
  { value: "segment_start", label: "Segment start" },
  { value: "segment_end", label: "Segment end" },
  { value: "ignore", label: "Ignore" },
];

const ROLE_TONE: Record<FileRole, "primary" | "success" | "neutral" | "warning"> = {
  eeg: "primary",
  events: "success",
  companion: "neutral",
  metadata: "neutral",
  ignored: "warning",
};

/** Text cell that commits on blur / Enter (one request per edit, not per keystroke). */
function EditCell({ value, onCommit, label, width = "w-20" }: { value: string | null; onCommit: (v: string | null) => void; label: string; width?: string }) {
  const [v, setV] = useState(value ?? "");
  useEffect(() => setV(value ?? ""), [value]);
  const commit = () => {
    const nv = v.trim() || null;
    if (nv !== (value ?? null)) onCommit(nv);
  };
  return (
    <input
      aria-label={label}
      value={v}
      onChange={(e) => setV(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
      className={classNames("h-7 rounded-md border border-line bg-card px-1.5 font-mono text-[12px] focus:border-primary focus:outline-none", width)}
    />
  );
}

function NameField({ value, onCommit }: { value: string; onCommit: (v: string) => void }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return (
    <div className="max-w-md" onBlur={() => v.trim() && v.trim() !== value && onCommit(v.trim())}>
      <TextField label="Dataset name" value={v} onChange={setV} />
    </div>
  );
}

function IssueIcons({ r }: { r: ManifestRecording }) {
  const errors = r.issues.filter((i) => i.level === "error");
  const warns = r.issues.filter((i) => i.level === "warning");
  const infos = r.issues.filter((i) => i.level === "info");
  const excluded = !r.included && !errors.length;
  return (
    <span className="inline-flex items-center gap-1">
      {errors.length > 0 && <span title={errors.map((i) => i.message).join("\n")}><CircleSlash className="size-4 text-danger" aria-label="errors" /></span>}
      {warns.length > 0 && <span title={warns.map((i) => i.message).join("\n")}><AlertTriangle className="size-4 text-warning" aria-label="warnings" /></span>}
      {excluded && (
        <span title={`Excluded: ${r.exclude_reason ?? "deselected"}`} className="inline-flex items-center gap-1 text-[11.5px] text-ink-subtle">
          <CircleSlash className="size-4" aria-label="excluded" />
          {!warns.length && (r.exclude_reason ?? "excluded")}
        </span>
      )}
      {!errors.length && !warns.length && !excluded && <span title={infos.map((i) => i.message).join("\n") || "OK"}><CheckCircle2 className="size-4 text-success" aria-label="ok" /></span>}
    </span>
  );
}

function ClassChips({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts);
  if (!entries.length) return <span className="text-ink-subtle">—</span>;
  return (
    <span className="flex flex-wrap gap-1">
      {entries.map(([k, v]) => (
        <span key={k} className="rounded-md bg-subtle px-1.5 py-0.5 text-[11.5px] text-ink-muted">
          {k} <b className="text-ink">{v}</b>
        </span>
      ))}
    </span>
  );
}

export function DatasetReview({ manifest, onPatch, busy }: { manifest: DatasetManifest; onPatch: (p: ManifestPatch) => void; busy: boolean }) {
  const [tab, setTab] = useState<Tab>("recordings");
  const [filter, setFilter] = useState("");
  const [rule, setRule] = useState<{ pattern: string; action: MappingAction; label: string }>({ pattern: "", action: "class", label: "" });
  const [roleFilter, setRoleFilter] = useState<FileRole | "all">("all");
  const s = manifest.summary;
  const errors = manifest.issues.filter((i) => i.level === "error");
  const recErrors = manifest.recordings.filter((r) => r.included && r.issues.some((i) => i.level === "error"));
  const warns = manifest.issues.filter((i) => i.level === "warning");
  const infos = manifest.issues.filter((i) => i.level === "info");
  const hasArrays = manifest.recordings.some((r) => ["npz", "npy", "csv", "tsv"].includes(r.format));
  const hasTasks = manifest.mapping.some((m) => m.task);

  const recs = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return manifest.recordings.filter((r) => !q || r.file.toLowerCase().includes(q) || (r.participant ?? "").toLowerCase().includes(q));
  }, [manifest.recordings, filter]);

  const setRec = (r: ManifestRecording, patch: Partial<ManifestRecording>) => onPatch({ recordings: [{ index: r.index, ...patch }] });
  const setMap = (m: MappingRow, patch: Partial<MappingRow>) =>
    onPatch({ mapping: [{ task: m.task, event: m.event, action: patch.action ?? m.action, label: patch.label !== undefined ? patch.label : m.label }] });

  const files = manifest.files.filter((f) => roleFilter === "all" || f.role === roleFilter);
  const roleCounts = manifest.files.reduce<Record<string, number>>((acc, f) => ({ ...acc, [f.role]: (acc[f.role] ?? 0) + 1 }), {});

  const tabs: { id: Tab; label: ReactNode }[] = [
    { id: "participants", label: <>Participants <Badge>{s.n_participants}</Badge></> },
    { id: "recordings", label: <>Recordings <Badge>{s.n_included}/{s.n_recordings}</Badge></> },
    { id: "mapping", label: <>Event mapping <Badge>{manifest.mapping.length}</Badge></> },
    { id: "channels", label: "Channels & rates" },
    { id: "files", label: <>Files <Badge>{s.n_files}</Badge></> },
  ];

  return (
    <div className="space-y-5">
      <NameField value={manifest.name} onCommit={(name) => onPatch({ name })} />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-7">
        <Stat label="Participants" value={s.n_participants} hint={s.sessions.length ? `${s.sessions.length} session(s)` : "no session IDs"} />
        <Stat label="Recordings" value={`${s.n_included}/${s.n_recordings}`} hint={s.n_unreadable ? `${s.n_unreadable} unreadable` : "included / found"} tone={s.n_unreadable ? "danger" : undefined} />
        <Stat label="Trials" value={fmtInt(s.n_trials)} hint={s.class_seconds ? "segments + events" : s.structure === "epoched" ? "epochs" : "events"} />
        <Stat label="Classes" value={s.classes.length} hint={s.classes.slice(0, 4).join(", ")} tone={s.classes.length < 2 ? "danger" : undefined} />
        <Stat label="Sampling" value={s.target_sfreq ? `${s.target_sfreq} Hz` : "—"} hint={s.sfreqs.length > 1 ? `from ${s.sfreqs.join(" / ")} Hz` : "all recordings"} />
        <Stat label="Channels" value={s.common_channels.length} hint={s.dropped_channels.length ? `${s.dropped_channels.length} not shared` : "shared by all"} />
        <Stat label="Duration" value={seconds(s.total_duration_s)} hint="included recordings" />
      </div>
      {(errors.length > 0 || recErrors.length > 0) && (
        <Alert tone="danger" title="Fix before importing">
          <ul className="list-disc space-y-0.5 pl-5">
            {errors.map((i) => <li key={i.code}>{i.message}</li>)}
            {recErrors.slice(0, 6).map((r) => <li key={r.index}><Mono>{r.file}</Mono>: {r.issues.filter((i) => i.level === "error").map((i) => i.message).join(" ")}</li>)}
            {recErrors.length > 6 && <li>…and {recErrors.length - 6} more recording(s)</li>}
          </ul>
        </Alert>
      )}
      {warns.map((i) => <Alert key={i.code} tone="warning">{i.message}</Alert>)}
      {infos.length > 0 && (
        <details className="rounded-2xl border border-line bg-surface px-4 py-2 text-[13.5px] text-ink-muted">
          <summary className="cursor-pointer font-bold text-ink">Notes ({infos.length})</summary>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">{infos.map((i) => <li key={i.code}>{i.message}</li>)}</ul>
        </details>
      )}
      {s.classes.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-[13px]">
          <span className="font-bold text-ink-muted">Trials per class:</span>
          {s.classes.map((c) => (
            <Badge key={c} tone="primary">{c} · {s.class_counts[c]}{s.class_seconds?.[c] ? ` · ${seconds(s.class_seconds[c])}` : ""}</Badge>
          ))}
        </div>
      )}
      <Tabs<Tab> ariaLabel="Dataset review sections" value={tab} onChange={setTab} tabs={tabs} />

      {tab === "participants" && (
        <div className="overflow-x-auto rounded-xl border border-line">
          <table className="w-full text-[13px] tabular">
            <thead className="bg-subtle text-left text-ink-subtle">
              <tr><th className="px-3 py-2">Participant</th><th className="px-3">Recordings</th><th className="px-3">Sessions</th><th className="px-3">Runs</th><th className="px-3">Trials</th><th className="px-3">Per class</th><th className="px-3">Attributes</th></tr>
            </thead>
            <tbody>
              {s.participants.map((p) => (
                <tr key={p.participant ?? "none"} className="border-t border-line">
                  <td className="px-3 py-1.5 font-bold">{p.participant ?? <span className="text-warning">no ID</span>}</td>
                  <td className="px-3">{p.n_recordings}</td>
                  <td className="px-3">{p.sessions.join(", ") || "—"}</td>
                  <td className="px-3">{p.runs.join(", ") || "—"}</td>
                  <td className="px-3 font-bold">{p.n_trials}</td>
                  <td className="px-3 py-1"><ClassChips counts={p.class_counts} /></td>
                  <td className="px-3 text-[12px] text-ink-muted">{Object.entries(p.attributes || {}).map(([k, v]) => `${k}: ${v}`).join(" · ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="px-3 py-2 text-[12px] text-ink-subtle">Participant IDs drive participant-level splits and overlap checks. Edit them in the Recordings tab.</p>
        </div>
      )}

      {tab === "recordings" && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative w-72">
              <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-ink-subtle" />
              <input aria-label="Filter recordings" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="filter by file or participant" className="h-9 w-full rounded-full border border-line bg-card pr-3 pl-8 text-[13px]" />
            </div>
            <Button size="sm" variant="secondary" disabled={busy} onClick={() => onPatch({ recordings: manifest.recordings.filter((r) => r.readable && r.n_trials > 0).map((r) => ({ index: r.index, included: true })) })}>Include all with trials</Button>
            <Button size="sm" variant="ghost" disabled={busy} onClick={() => onPatch({ recordings: manifest.recordings.filter((r) => r.issues.some((i) => i.level === "error")).map((r) => ({ index: r.index, included: false })) })}>Exclude recordings with errors</Button>
          </div>
          <div className="max-h-[46vh] overflow-auto rounded-xl border border-line scrollbar-thin">
            <table className="w-full text-[12.5px] tabular">
              <thead className="sticky top-0 z-10 bg-subtle text-left text-ink-subtle">
                <tr>
                  <th className="px-2 py-2">Use</th><th className="px-2">File</th><th className="px-2">Participant</th><th className="px-2">Session</th><th className="px-2">Run</th>
                  {hasTasks && <th className="px-2">Task</th>}
                  <th className="px-2">Rate</th><th className="px-2">Ch.</th><th className="px-2">Length</th><th className="px-2">Events from</th><th className="px-2">Trials</th><th className="px-2" />
                </tr>
              </thead>
              <tbody>
                {recs.map((r) => (
                  <tr key={r.index} className={classNames("border-t border-line", !r.included && "bg-subtle/60 text-ink-subtle")}>
                    <td className="px-2 py-1"><input type="checkbox" aria-label={`include ${r.file}`} className="accent-[var(--c-primary)]" checked={r.included} disabled={!r.readable || busy} onChange={(e) => setRec(r, { included: e.target.checked })} /></td>
                    <td className="max-w-[280px] truncate px-2 font-mono text-[11.5px]" title={`${r.file}${r.exclude_reason ? ` — ${r.exclude_reason}` : ""}`}>{r.file}</td>
                    <td className="px-2">{r.inner_groups ? <span title={Object.keys(r.inner_groups).join(", ")}>{Object.keys(r.inner_groups).length} in file</span> : <EditCell label={`participant of ${r.file}`} value={r.participant} onCommit={(v) => setRec(r, { participant: v })} />}</td>
                    <td className="px-2"><EditCell label={`session of ${r.file}`} value={r.session} width="w-14" onCommit={(v) => setRec(r, { session: v })} /></td>
                    <td className="px-2"><EditCell label={`run of ${r.file}`} value={r.run} width="w-14" onCommit={(v) => setRec(r, { run: v })} /></td>
                    {hasTasks && <td className="px-2"><EditCell label={`task of ${r.file}`} value={r.task} width="w-16" onCommit={(v) => setRec(r, { task: v })} /></td>}
                    <td className="px-2 whitespace-nowrap">{r.sfreq ? `${r.sfreq} Hz` : <span className="text-danger">?</span>}</td>
                    <td className="px-2">{r.n_channels}</td>
                    <td className="px-2 whitespace-nowrap">{r.structure === "epoched" ? `${r.n_epochs} ep.` : seconds(r.duration_s)}</td>
                    <td className="max-w-[200px] truncate px-2" title={r.events.file ?? r.events.note ?? ""}>
                      <Badge tone={r.events.kind === "none" ? "danger" : r.events.kind === "events_file" ? "success" : "neutral"}>{r.events.kind === "events_file" ? "event file" : r.events.kind}</Badge>
                      <span className="ml-1 text-ink-subtle">{r.events.n_events}</span>
                    </td>
                    <td className="px-2 py-1"><span className="font-bold">{r.n_trials}</span></td>
                    <td className="px-2"><IssueIcons r={r} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[12px] text-ink-subtle">Recording IDs are built from participant / session / task / run (e.g. <Mono>{manifest.recordings.find((r) => r.included)?.recording_id ?? "sub-01_run-1"}</Mono>) and travel with every trial through preprocessing, training and results.</p>
        </div>
      )}

      {tab === "mapping" && (
        <div className="space-y-3">
          <p className="text-[13.5px] text-ink-muted">
            Map raw event descriptions to classes. Use <b>segment start / end</b> for block designs: a start marker opens a block that the matching end marker
            (same name without “start/end”) closes, and each block becomes one trial with its duration. Ignored events never become trials.
          </p>
          <div className="flex flex-wrap items-end gap-2 rounded-2xl border border-line bg-surface p-3">
            <div className="w-56"><TextField label="Apply to events matching" value={rule.pattern} onChange={(v) => setRule({ ...rule, pattern: v })} placeholder="e.g. FC-* start  or  T1" /></div>
            <div className="w-48"><SelectField label="as" value={rule.action} onChange={(v) => setRule({ ...rule, action: v })} options={ACTIONS} /></div>
            <div className="w-40"><TextField label="class label" value={rule.label} onChange={(v) => setRule({ ...rule, label: v })} placeholder="e.g. FC" /></div>
            <Button size="sm" icon={<Wand2 className="size-4" />} disabled={!rule.pattern.trim() || busy} onClick={() => onPatch({ rules: [{ pattern: rule.pattern, action: rule.action, label: rule.action === "ignore" ? null : rule.label || null }] })}>Apply</Button>
            <span className="text-[12px] text-ink-subtle">Wildcards: * any text, ? one character (case-insensitive).</span>
          </div>
          <div className="max-h-[44vh] overflow-auto rounded-xl border border-line scrollbar-thin">
            <table className="w-full text-[12.5px] tabular">
              <thead className="sticky top-0 z-10 bg-subtle text-left text-ink-subtle">
                <tr>{hasTasks && <th className="px-3 py-2">Task</th>}<th className="px-3 py-2">Event description</th><th className="px-3">Count</th><th className="px-3">Becomes</th><th className="px-3">Class label</th></tr>
              </thead>
              <tbody>
                {manifest.mapping.map((m) => (
                  <tr key={`${m.task}|${m.event}`} className={classNames("border-t border-line", m.action === "ignore" && "text-ink-subtle")}>
                    {hasTasks && <td className="px-3 py-1">{m.task ?? "—"}</td>}
                    <td className="px-3 py-1 font-mono text-[12px]">{m.event}</td>
                    <td className="px-3">{m.count}</td>
                    <td className="px-3 py-1">
                      <select aria-label={`action for ${m.event}`} value={m.action} disabled={busy} onChange={(e) => setMap(m, { action: e.target.value as MappingAction })} className="h-7 rounded-md border border-line bg-card px-1.5 text-[12px]">
                        {ACTIONS.map((a) => <option key={a.value} value={a.value}>{a.label}</option>)}
                      </select>
                    </td>
                    <td className="px-3 py-1">{m.action === "ignore" || m.action === "segment_end" ? <span className="text-ink-subtle">{m.action === "segment_end" ? "closes its block" : "—"}</span> : <EditCell label={`label for ${m.event}`} value={m.label} width="w-32" onCommit={(v) => setMap(m, { label: v })} />}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === "channels" && (
        <div className="grid gap-5 lg:grid-cols-2">
          <div className="space-y-3">
            <div className="text-[13px] font-bold text-ink-muted">Channels shared by every included recording ({s.common_channels.length})</div>
            <div className="flex flex-wrap gap-1.5">
              {s.common_channels.map((c, i) => (
                <span key={c} className={classNames("rounded-lg border px-2 py-0.5 font-mono text-[12px]", s.common_channel_types[i] === "eeg" ? "border-line bg-surface text-ink" : "border-warning-line bg-warning-soft text-warning")} title={s.common_channel_types[i]}>
                  {c}{s.common_channel_types[i] !== "eeg" && <span className="ml-1 text-[10px] uppercase">{s.common_channel_types[i]}</span>}
                </span>
              ))}
            </div>
            {s.dropped_channels.length > 0 && <Alert tone="warning" title="Not present in every recording">{s.dropped_channels.join(", ")}</Alert>}
            <Segmented ariaLabel="Channel policy" value={manifest.channel_policy} onChange={(v) => onPatch({ channel_policy: v })} options={[{ value: "intersection", label: "Use shared channels" }, { value: "strict", label: "Require identical sets" }]} />
            <p className="text-[12px] text-ink-subtle">Non-EEG channels (EMG, ECG, references) are imported but left out of the model input by default in the Preprocessing Studio.</p>
          </div>
          <div className="space-y-3">
            <div className="text-[13px] font-bold text-ink-muted">Sampling rates found: {s.sfreqs.join(", ") || "—"} Hz</div>
            <Segmented ariaLabel="Sampling-rate policy" value={manifest.sfreq_policy} onChange={(v) => onPatch({ sfreq_policy: v })} options={[{ value: "resample", label: "Resample to one rate" }, { value: "strict", label: "Require identical rates" }]} />
            <div className="w-56"><NumberField label="Target rate at import (Hz)" value={manifest.target_sfreq} nullable placeholder={s.target_sfreq ? `${s.target_sfreq} (most common)` : "auto"} onChange={(v) => onPatch({ target_sfreq: v })} hint="Leave empty to use the most common rate. Further downsampling (e.g. to 128 Hz) can be done in the Preprocessing Studio." /></div>
            {hasArrays && (
              <div className="space-y-3 rounded-2xl border border-line bg-surface p-3">
                <div className="text-[13px] font-bold text-ink">Array file options (NumPy / CSV)</div>
                <div className="grid grid-cols-2 gap-3">
                  <NumberField label="Sampling rate (Hz)" value={manifest.array_options.sfreq} nullable onChange={(v) => onPatch({ array_options: { ...manifest.array_options, sfreq: v } })} />
                  <SelectField label="Unit" value={manifest.array_options.unit} onChange={(v) => onPatch({ array_options: { ...manifest.array_options, unit: v } })} options={[{ value: "uV", label: "µV" }, { value: "mV", label: "mV" }, { value: "V", label: "V" }, { value: "nV", label: "nV" }, { value: "unknown", label: "unknown" }]} />
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {tab === "files" && (
        <div className="space-y-3">
          <Segmented<FileRole | "all">
            size="sm"
            ariaLabel="File role filter"
            value={roleFilter}
            onChange={setRoleFilter}
            options={[{ value: "all", label: `All (${s.n_files})` }, ...(["eeg", "events", "companion", "metadata", "ignored"] as FileRole[]).filter((r) => roleCounts[r]).map((r) => ({ value: r, label: `${r === "eeg" ? "recordings" : r} (${roleCounts[r]})` }))]}
          />
          <div className="max-h-[44vh] overflow-auto rounded-xl border border-line scrollbar-thin">
            <table className="w-full text-[12.5px]">
              <thead className="sticky top-0 bg-subtle text-left text-ink-subtle"><tr><th className="px-3 py-2">File</th><th className="px-3">Role</th><th className="px-3">Size</th><th className="px-3">Note</th></tr></thead>
              <tbody>
                {files.slice(0, 2000).map((f) => (
                  <tr key={f.path} className="border-t border-line">
                    <td className="max-w-[420px] truncate px-3 py-1 font-mono text-[11.5px]" title={f.path}><FileText className="mr-1 inline size-3.5 text-ink-subtle" />{f.path}</td>
                    <td className="px-3"><Badge tone={ROLE_TONE[f.role]}>{f.role === "eeg" ? "recording" : f.role}</Badge></td>
                    <td className="px-3 whitespace-nowrap text-ink-subtle">{f.size ? `${(f.size / 1024 / 1024).toFixed(2)} MB` : "—"}</td>
                    <td className="px-3 text-ink-muted">{f.note ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
