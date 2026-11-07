import { useEffect, useEffectEvent, useRef, type ReactNode } from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, CircleSlash, Info, Loader2, Lock, X, XCircle } from "lucide-react";
import { classNames } from "../../lib/format";
import { Button } from "./Button";

export function PageHeader({
  eyebrow,
  title,
  accent,
  lead,
  actions,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  accent?: ReactNode;
  lead?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-8 flex flex-wrap items-end gap-6">
      <div className="min-w-0 flex-1">
        {eyebrow && <div className="mb-3 text-[13px] font-bold uppercase tracking-[0.08em] text-ink-subtle">{eyebrow}</div>}
        <h1 className="font-display text-[44px] leading-[1.04] text-ink sm:text-[52px]">
          {title} {accent && <em className="italic">{accent}</em>}
        </h1>
        {lead && <p className="mt-4 max-w-3xl text-[17px] leading-relaxed text-ink-muted">{lead}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-3">{actions}</div>}
    </header>
  );
}

export function SectionTitle({ children, accent, sub }: { children: ReactNode; accent?: ReactNode; sub?: ReactNode }) {
  return (
    <div className="mb-4">
      <h2 className="font-display text-[30px] leading-tight text-ink">
        {children} {accent && <em className="italic">{accent}</em>}
      </h2>
      {sub && <p className="mt-1.5 max-w-3xl text-[15px] text-ink-muted">{sub}</p>}
    </div>
  );
}

type AlertTone = "info" | "warning" | "danger" | "success";
const alertTone: Record<AlertTone, string> = {
  info: "border-line bg-primary-soft/60 text-ink",
  warning: "border-warning-line bg-warning-soft text-ink",
  danger: "border-danger-line bg-danger-soft text-ink",
  success: "border-success-line bg-success-soft text-ink",
};
const alertIcon: Record<AlertTone, ReactNode> = {
  info: <Info className="size-4.5 text-link" />,
  warning: <AlertTriangle className="size-4.5 text-warning" />,
  danger: <AlertCircle className="size-4.5 text-danger" />,
  success: <CheckCircle2 className="size-4.5 text-success" />,
};

export function Alert({ tone = "info", title, children, className, action }: { tone?: AlertTone; title?: ReactNode; children?: ReactNode; className?: string; action?: ReactNode }) {
  return (
    <div role={tone === "danger" ? "alert" : "status"} className={classNames("flex gap-3 rounded-2xl border px-4 py-3", alertTone[tone], className)}>
      <div className="mt-0.5 shrink-0">{alertIcon[tone]}</div>
      <div className="min-w-0 flex-1 text-[14px] leading-snug">
        {title && <div className={classNames("font-bold", tone === "danger" ? "text-danger" : "text-ink")}>{title}</div>}
        {children && <div className={classNames("text-ink-muted", title ? "mt-0.5" : undefined)}>{children}</div>}
      </div>
      {action && <div className="shrink-0 self-center">{action}</div>}
    </div>
  );
}

export function Loading({ label = "Loading…", className }: { label?: string; className?: string }) {
  return (
    <div role="status" className={classNames("flex items-center gap-2 py-8 text-[14px] text-ink-subtle", className)}>
      <Loader2 className="size-4 animate-spin" /> {label}
    </div>
  );
}

export function EmptyState({ icon, title, children, action }: { icon?: ReactNode; title: ReactNode; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center rounded-2xl border border-dashed border-line-strong bg-surface px-6 py-14 text-center">
      {icon && <div className="mb-4 grid size-14 place-items-center rounded-full bg-primary-soft text-link">{icon}</div>}
      <h3 className="font-display text-[26px] text-ink">{title}</h3>
      {children && <div className="mt-2 max-w-xl text-[15px] leading-relaxed text-ink-muted">{children}</div>}
      {action && <div className="mt-6 flex flex-wrap justify-center gap-3">{action}</div>}
    </div>
  );
}

export function LockedState({ title, reason, action }: { title: ReactNode; reason: ReactNode; action?: ReactNode }) {
  return <EmptyState icon={<Lock className="size-6" />} title={title} action={action}>{reason}</EmptyState>;
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <Alert tone="danger" title="Something went wrong" action={onRetry && <Button size="sm" variant="secondary" onClick={onRetry}>Retry</Button>}>
      {msg}
    </Alert>
  );
}

export function CheckIcon({ status }: { status: "pass" | "warn" | "fail" | boolean | "skip" }) {
  if (status === "pass" || status === true) return <CheckCircle2 className="size-4.5 shrink-0 text-success" aria-label="pass" />;
  if (status === "warn") return <AlertTriangle className="size-4.5 shrink-0 text-warning" aria-label="warning" />;
  if (status === "skip") return <CircleSlash className="size-4.5 shrink-0 text-ink-subtle" aria-label="skipped" />;
  return <XCircle className="size-4.5 shrink-0 text-danger" aria-label="fail" />;
}

export function Checklist({ items }: { items: { id: string; label: ReactNode; status: "pass" | "warn" | "fail" | boolean; detail?: ReactNode; extra?: ReactNode }[] }) {
  return (
    <ul className="divide-y divide-line">
      {items.map((c) => (
        <li key={c.id} className="flex gap-3 py-2.5">
          <div className="mt-0.5">
            <CheckIcon status={c.status} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-[14.5px] font-bold text-ink">{c.label}</div>
            {c.detail && <div className="text-[13px] leading-snug text-ink-muted">{c.detail}</div>}
            {c.extra}
          </div>
        </li>
      ))}
    </ul>
  );
}

export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
  wide,
  size,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
  size?: "md" | "lg" | "xl";
}) {
  const ref = useRef<HTMLDivElement>(null);
  // callers pass inline onClose: depending on it re-ran this effect on every keystroke and stole focus from inputs
  const close = useEffectEvent(onClose);
  useEffect(() => {
    if (!open) return;
    const prev = document.activeElement as HTMLElement | null;
    ref.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      prev?.focus?.();
    };
  }, [open]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/40 p-4 pt-[6vh] backdrop-blur-[2px]" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-label={typeof title === "string" ? title : undefined}
        tabIndex={-1}
        className={classNames(
          "w-full rounded-3xl border border-line bg-card shadow-float outline-none",
          size === "xl" ? "max-w-[1240px]" : wide || size === "lg" ? "max-w-4xl" : "max-w-xl",
        )}
      >
        <div className="flex items-start gap-4 px-6 pt-6 pb-2">
          <h2 className="font-display flex-1 text-[28px] leading-tight text-ink">{title}</h2>
          <button type="button" onClick={onClose} aria-label="Close" className="rounded-full p-1.5 text-ink-subtle hover:bg-subtle hover:text-ink">
            <X className="size-5" />
          </button>
        </div>
        <div className={classNames("overflow-y-auto px-6 py-3 scrollbar-thin", size === "xl" ? "max-h-[76vh]" : "max-h-[70vh]")}>{children}</div>
        {footer && <div className="flex flex-wrap justify-end gap-3 border-t border-line px-6 py-4">{footer}</div>}
      </div>
    </div>
  );
}

export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = "Confirm",
  danger,
  onConfirm,
  onClose,
  loading,
}: {
  open: boolean;
  title: string;
  body: ReactNode;
  confirmLabel?: string;
  danger?: boolean;
  onConfirm: () => void;
  onClose: () => void;
  loading?: boolean;
}) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      title={title}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>Cancel</Button>
          <Button variant={danger ? "danger" : "primary"} onClick={onConfirm} loading={loading}>{confirmLabel}</Button>
        </>
      }
    >
      <div className="pb-2 text-[15px] leading-relaxed text-ink-muted">{body}</div>
    </Modal>
  );
}

export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
  ariaLabel,
}: {
  tabs: { id: T; label: ReactNode; icon?: ReactNode; locked?: boolean; badge?: ReactNode }[];
  value: T;
  onChange: (v: T) => void;
  ariaLabel: string;
}) {
  return (
    <div role="tablist" aria-label={ariaLabel} className="flex gap-1 overflow-x-auto border-b border-line scrollbar-thin">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          type="button"
          aria-selected={value === t.id}
          onClick={() => onChange(t.id)}
          className={classNames(
            "-mb-px inline-flex shrink-0 items-center gap-2 border-b-2 px-3.5 py-3 text-[14.5px] font-bold whitespace-nowrap transition-colors",
            value === t.id ? "border-primary text-ink" : "border-transparent text-ink-subtle hover:text-ink",
          )}
        >
          {t.icon}
          {t.label}
          {t.locked && <Lock className="size-3.5 text-ink-subtle" aria-label="locked" />}
          {t.badge}
        </button>
      ))}
    </div>
  );
}

export function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={classNames("font-mono text-[0.92em] tabular", className)}>{children}</span>;
}

export function KeyValue({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1.5 text-[14px]">
      {items.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-ink-subtle">{k}</dt>
          <dd className="min-w-0 break-words text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
