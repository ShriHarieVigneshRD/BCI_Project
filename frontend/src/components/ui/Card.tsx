import type { HTMLAttributes, ReactNode } from "react";
import { classNames } from "../../lib/format";

export function Card({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={classNames("rounded-2xl border border-line bg-card shadow-card", className)} {...rest}>
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  actions,
  icon,
  className,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  icon?: ReactNode;
  className?: string;
}) {
  return (
    <div className={classNames("flex flex-wrap items-start gap-3 px-5 pt-5 pb-3", className)}>
      {icon && (
        <div className="grid size-10 shrink-0 place-items-center rounded-full bg-primary-soft text-link">{icon}</div>
      )}
      <div className="min-w-0 flex-1">
        <h3 className="text-[17px] font-bold leading-tight text-ink">{title}</h3>
        {subtitle && <p className="mt-1 text-[14px] leading-snug text-ink-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function CardBody({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={classNames("px-5 pb-5", className)}>{children}</div>;
}

/** Table-like definition rows, as in the reference "doctor view" card. */
export function DefRow({ label, children, className }: { label: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={classNames("grid grid-cols-[130px_1fr] gap-4 border-t border-line px-5 py-3.5 first:border-t-0", className)}>
      <div className="pt-0.5 text-[13px] font-bold text-ink-subtle">{label}</div>
      <div className="min-w-0 text-[15px] text-ink">{children}</div>
    </div>
  );
}

export function Stat({ label, value, hint, tone }: { label: ReactNode; value: ReactNode; hint?: ReactNode; tone?: "danger" | "success" }) {
  return (
    <div className="min-w-0 rounded-xl border border-line bg-surface px-4 py-3">
      <div className="text-[12px] font-bold uppercase tracking-[0.06em] text-ink-subtle">{label}</div>
      <div
        className={classNames(
          "font-display mt-1 truncate text-[26px] leading-tight tabular",
          tone === "danger" ? "text-danger" : tone === "success" ? "text-success" : "text-ink",
        )}
      >
        {value}
      </div>
      {hint && <div className="mt-0.5 text-[12.5px] text-ink-subtle">{hint}</div>}
    </div>
  );
}
