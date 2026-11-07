import type { ReactNode } from "react";
import type { ExperimentState } from "../../api/types";
import { classNames, isBusyState, STATE_LABEL, stateTone, type Tone } from "../../lib/format";

const tones: Record<Tone, string> = {
  neutral: "bg-subtle text-ink-muted border-line",
  primary: "bg-primary-soft text-link border-transparent",
  success: "bg-success-soft text-success border-success-line",
  warning: "bg-warning-soft text-warning border-warning-line",
  danger: "bg-danger-soft text-danger border-danger-line",
};

const dots: Record<Tone, string> = {
  neutral: "bg-ink-subtle",
  primary: "bg-primary",
  success: "bg-success",
  warning: "bg-warning",
  danger: "bg-danger",
};

export function Badge({ tone = "neutral", children, icon, className, title }: { tone?: Tone; children: ReactNode; icon?: ReactNode; className?: string; title?: string }) {
  return (
    <span
      title={title}
      className={classNames(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2.5 py-0.5 text-[12.5px] font-bold leading-5",
        tones[tone],
        className,
      )}
    >
      {icon}
      {children}
    </span>
  );
}

export function Dot({ tone = "neutral", pulse }: { tone?: Tone; pulse?: boolean }) {
  return <span aria-hidden className={classNames("inline-block size-2 rounded-full", dots[tone], pulse && "animate-pulse-dot")} />;
}

export function StateChip({ state, className }: { state: ExperimentState; className?: string }) {
  const tone = stateTone(state);
  return (
    <Badge tone={tone} className={className}>
      <Dot tone={tone} pulse={isBusyState(state)} />
      {STATE_LABEL[state]}
    </Badge>
  );
}

export function SyntheticBadge() {
  return (
    <Badge tone="warning" title="Synthetic data for software testing - not evidence of real-world performance">
      Synthetic
    </Badge>
  );
}
