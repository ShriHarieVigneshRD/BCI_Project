import { useMemo, useState } from "react";
import type { TrialRow } from "../api/types";
import { SelectField } from "./ui/Field";
import { Segmented } from "./ui/Field";

type Part = "all" | "train" | "validation" | "test";

export function TrialPicker({
  trials,
  value,
  onChange,
  label = "Trial",
  showPartitions = true,
}: {
  trials: TrialRow[];
  value: string | null;
  onChange: (id: string) => void;
  label?: string;
  showPartitions?: boolean;
}) {
  const hasParts = showPartitions && trials.some((t) => t.partition !== "unknown");
  const [part, setPart] = useState<Part>("all");
  const [who, setWho] = useState("");
  const participants = useMemo(() => Array.from(new Set(trials.map((t) => t.group).filter(Boolean) as string[])).sort(), [trials]);
  const filtered = useMemo(
    () => trials.filter((t) => (part === "all" || t.partition === part) && (!who || t.group === who)),
    [trials, part, who],
  );
  const opts = filtered.slice(0, 2000).map((t) => ({
    value: t.trial_id,
    label: `${t.trial_id} · ${t.label}${t.group ? ` · P ${t.group}` : ""}${t.run ? ` · run ${t.run}` : ""}${t.partition !== "unknown" ? ` · ${t.partition}` : ""}`,
  }));
  if (value && !opts.some((o) => o.value === value)) {
    const t = trials.find((x) => x.trial_id === value);
    if (t) opts.unshift({ value: t.trial_id, label: `${t.trial_id} · ${t.label} · ${t.partition}` });
  }
  return (
    <div className="space-y-2">
      {participants.length > 1 && (
        <select aria-label={`${label} participant filter`} value={who} onChange={(e) => setWho(e.target.value)} className="h-7 rounded-full border border-line bg-card px-3 text-[12.5px] text-ink-muted">
          <option value="">All participants ({participants.length})</option>
          {participants.map((p) => <option key={p} value={p}>Participant {p}</option>)}
        </select>
      )}
      {hasParts && (
        <Segmented<Part>
          size="sm"
          ariaLabel={`${label} partition filter`}
          value={part}
          onChange={setPart}
          options={[{ value: "all", label: "All" }, { value: "train", label: "Train" }, { value: "validation", label: "Validation" }, { value: "test", label: "Test" }]}
        />
      )}
      <SelectField label={label} value={value ?? ""} onChange={onChange} options={opts.length ? opts : [{ value: "", label: "No trials" }]} />
    </div>
  );
}
