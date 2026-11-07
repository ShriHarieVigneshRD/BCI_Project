import { useId, useState, type ReactNode } from "react";
import { Info } from "lucide-react";
import { classNames } from "../../lib/format";

export function InfoTip({ text }: { text: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span className="relative inline-flex align-middle">
      <button
        type="button"
        aria-describedby={open ? id : undefined}
        aria-label="More information"
        className="text-ink-subtle hover:text-link"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
      >
        <Info className="size-3.5" />
      </button>
      {open && (
        <span
          role="tooltip"
          id={id}
          className="absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-xl border border-line bg-card px-3 py-2 text-[12.5px] font-normal leading-snug text-ink-muted shadow-float"
        >
          {text}
        </span>
      )}
    </span>
  );
}

export function Label({ children, hint, htmlFor }: { children: ReactNode; hint?: ReactNode; htmlFor?: string }) {
  return (
    <label htmlFor={htmlFor} className="mb-1.5 flex items-center gap-1.5 text-[13px] font-bold text-ink-muted">
      {children}
      {hint && <InfoTip text={hint} />}
    </label>
  );
}

const inputCls =
  "h-10 w-full rounded-xl border border-line-strong bg-card px-3 text-[15px] text-ink tabular placeholder:text-ink-subtle " +
  "focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20 disabled:opacity-50";

export function NumberField({
  label,
  value,
  onChange,
  hint,
  step,
  min,
  max,
  disabled,
  nullable,
  placeholder,
  suffix,
  changed,
}: {
  label: ReactNode;
  value: number | null;
  onChange: (v: number | null) => void;
  hint?: ReactNode;
  step?: number;
  min?: number;
  max?: number;
  disabled?: boolean;
  nullable?: boolean;
  placeholder?: string;
  suffix?: string;
  changed?: boolean;
}) {
  const id = useId();
  // text being typed; an empty or partial entry ("-", "0.") must not be replaced by 0 mid-typing
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <div>
      <Label htmlFor={id} hint={hint}>
        {label}
        {changed && <span className="size-1.5 rounded-full bg-warning" aria-label="changed" />}
      </Label>
      <div className="relative">
        <input
          id={id}
          type="number"
          className={classNames(inputCls, suffix && "pr-12", changed && "border-warning-line bg-warning-soft/40")}
          value={draft ?? value ?? ""}
          step={step}
          min={min}
          max={max}
          placeholder={placeholder}
          disabled={disabled}
          onChange={(e) => {
            const raw = e.target.value;
            setDraft(raw);
            if (raw === "") {
              if (nullable) onChange(null);
            } else if (Number.isFinite(Number(raw))) onChange(Number(raw));
          }}
          onBlur={() => setDraft(null)}
        />
        {suffix && <span className="pointer-events-none absolute top-1/2 right-3 -translate-y-1/2 text-[13px] text-ink-subtle">{suffix}</span>}
      </div>
    </div>
  );
}

export function TextField({
  label,
  value,
  onChange,
  hint,
  placeholder,
  disabled,
}: {
  label: ReactNode;
  value: string;
  onChange: (v: string) => void;
  hint?: ReactNode;
  placeholder?: string;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div>
      <Label htmlFor={id} hint={hint}>
        {label}
      </Label>
      <input id={id} className={inputCls} value={value} placeholder={placeholder} disabled={disabled} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

export function SelectField<T extends string>({
  label,
  value,
  onChange,
  options,
  hint,
  disabled,
  changed,
}: {
  label: ReactNode;
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string; disabled?: boolean }[];
  hint?: ReactNode;
  disabled?: boolean;
  changed?: boolean;
}) {
  const id = useId();
  return (
    <div>
      <Label htmlFor={id} hint={hint}>
        {label}
        {changed && <span className="size-1.5 rounded-full bg-warning" aria-label="changed" />}
      </Label>
      <select
        id={id}
        className={classNames(inputCls, "appearance-none bg-[length:12px] pr-8", changed && "border-warning-line bg-warning-soft/40")}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value as T)}
        style={{
          backgroundImage:
            "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12'%3E%3Cpath d='M3 4.5l3 3 3-3' fill='none' stroke='%23888' stroke-width='1.5'/%3E%3C/svg%3E\")",
          backgroundRepeat: "no-repeat",
          backgroundPosition: "right 12px center",
        }}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export function Toggle({
  label,
  checked,
  onChange,
  hint,
  disabled,
  description,
}: {
  label: ReactNode;
  checked: boolean;
  onChange: (v: boolean) => void;
  hint?: ReactNode;
  disabled?: boolean;
  description?: ReactNode;
}) {
  const id = useId();
  return (
    <div className="flex items-start gap-3">
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={classNames(
          "relative mt-0.5 h-6 w-10 shrink-0 rounded-full transition-colors disabled:opacity-45",
          checked ? "bg-primary" : "bg-line-strong",
        )}
      >
        <span
          className={classNames(
            "absolute top-0.5 left-0.5 size-5 rounded-full bg-card shadow transition-transform",
            checked && "translate-x-4",
          )}
        />
      </button>
      <label htmlFor={id} className="min-w-0 cursor-pointer">
        <span className="flex items-center gap-1.5 text-[14.5px] font-bold text-ink">
          {label}
          {hint && <InfoTip text={hint} />}
        </span>
        {description && <span className="block text-[13px] leading-snug text-ink-muted">{description}</span>}
      </label>
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  ariaLabel,
  size = "md",
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: ReactNode; title?: string; disabled?: boolean }[];
  ariaLabel: string;
  size?: "sm" | "md";
}) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="inline-flex rounded-full border border-line bg-subtle p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          title={o.title}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
          className={classNames(
            "inline-flex items-center gap-1.5 rounded-full font-bold transition-colors disabled:opacity-40",
            size === "sm" ? "h-7 px-2.5 text-[12.5px]" : "h-8 px-3.5 text-[13.5px]",
            value === o.value ? "bg-card text-ink shadow-sm" : "text-ink-subtle hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function RadioCard({
  checked,
  onSelect,
  title,
  description,
  icon,
  disabled,
  badge,
}: {
  checked: boolean;
  onSelect: () => void;
  title: ReactNode;
  description?: ReactNode;
  icon?: ReactNode;
  disabled?: boolean;
  badge?: ReactNode;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={checked}
      disabled={disabled}
      onClick={onSelect}
      className={classNames(
        "flex w-full items-start gap-3 rounded-2xl border p-4 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-45",
        checked ? "border-primary bg-primary-soft/50 ring-1 ring-primary" : "border-line bg-card hover:border-line-strong",
      )}
    >
      {icon && <div className="grid size-10 shrink-0 place-items-center rounded-full bg-primary-soft text-link">{icon}</div>}
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2 text-[15px] font-bold text-ink">
          {title}
          {badge}
        </div>
        {description && <div className="mt-1 text-[13.5px] leading-snug text-ink-muted">{description}</div>}
      </div>
    </button>
  );
}

export function Checkbox({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode; disabled?: boolean }) {
  return (
    <label className={classNames("inline-flex cursor-pointer items-center gap-2 text-[14px] text-ink", disabled && "opacity-50")}>
      <input
        type="checkbox"
        className="size-4 rounded border-line-strong accent-[var(--c-primary)]"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}
