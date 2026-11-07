import type { ExperimentState } from "../api/types";

export const STATE_LABEL: Record<ExperimentState, string> = {
  created: "Created",
  dataset_required: "Dataset required",
  dataset_ready: "Dataset ready",
  split_configured: "Split configured",
  preprocessing_configured: "Preprocessing configured",
  preprocessing_completed: "Preprocessing completed",
  architecture_configured: "Architecture configured",
  training_queued: "Training queued",
  training: "Training",
  training_failed: "Training failed",
  training_cancelled: "Training cancelled",
  training_completed: "Training completed",
  evaluation_running: "Evaluating",
  evaluation_completed: "Evaluated",
  artifacts_incomplete: "Artifacts incomplete",
  artifacts_ready: "Artifacts ready",
};

export type Tone = "neutral" | "primary" | "success" | "warning" | "danger";

export function stateTone(s: ExperimentState): Tone {
  if (s === "training_failed" || s === "artifacts_incomplete") return "danger";
  if (s === "training" || s === "training_queued" || s === "evaluation_running" || s === "training_completed") return "warning";
  if (s === "artifacts_ready" || s === "evaluation_completed") return "success";
  if (s === "training_cancelled") return "neutral";
  return "primary";
}

export const isBusyState = (s: ExperimentState) =>
  s === "training" || s === "training_queued" || s === "evaluation_running" || s === "training_completed";

export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(2)} M`;
  if (Math.abs(v) >= 1e4) return Math.round(v).toLocaleString("en-US");
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  return v.toFixed(digits);
}

export function fmtInt(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return Math.round(v).toLocaleString("en-US");
}

export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "undefined";
  return `${(v * 100).toFixed(digits)} %`;
}

export function shape(s: number[] | null | undefined): string {
  if (!s || !s.length) return "—";
  return s.join(" × ");
}

export function bytes(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(2)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function seconds(s: number | null | undefined): string {
  if (s === null || s === undefined) return "—";
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function shortId(id: string | null | undefined): string {
  return id ? id.replace(/^[a-z]+_/, "").slice(0, 8) : "—";
}

export function classNames(...xs: (string | false | null | undefined)[]): string {
  return xs.filter(Boolean).join(" ");
}

export const LAYER_SHORT: Record<string, string> = {
  temporal_conv: "Temporal conv",
  bn1: "BatchNorm",
  depthwise_conv: "Depthwise spatial",
  bn2: "BatchNorm",
  act1: "Activation",
  pool1: "Pool 1",
  drop1: "Dropout",
  separable_depthwise: "Separable · depthwise",
  separable_pointwise: "Separable · pointwise",
  bn3: "BatchNorm",
  act2: "Activation",
  pool2: "Pool 2",
  drop2: "Dropout",
  flatten: "Flatten",
  classifier: "Classifier",
};

/** Structural equality independent of object key order. */
export function stableEqual(a: unknown, b: unknown): boolean {
  const norm = (v: unknown): unknown =>
    Array.isArray(v)
      ? v.map(norm)
      : v && typeof v === "object"
        ? Object.fromEntries(Object.keys(v as object).sort().map((k) => [k, norm((v as Record<string, unknown>)[k])]))
        : v;
  return JSON.stringify(norm(a)) === JSON.stringify(norm(b));
}
