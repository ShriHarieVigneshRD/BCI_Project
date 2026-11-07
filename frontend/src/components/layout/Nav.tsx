import { NavLink } from "react-router";
import { Check, Lock } from "lucide-react";
import type { Stage } from "../../api/types";
import { classNames } from "../../lib/format";

export const PRIMARY_NAV = [
  { to: "/dataset", label: "Dataset & Splits", step: 1, stage: "dataset" },
  { to: "/preprocessing", label: "Preprocessing Studio", step: 2, stage: "preprocessing" },
  { to: "/architecture", label: "Architecture Explorer", step: 3, stage: "architecture" },
  { to: "/arena", label: "Hyperparameter Battle Arena" },
  { to: "/tensors", label: "Tensor Detective" },
  { to: "/experiments", label: "Experiments & Exports" },
] as const;

function StepMark({ step, status }: { step: number; status?: Stage["status"] }) {
  if (status === "complete")
    return (
      <span className="grid size-5 place-items-center rounded-full bg-success text-[11px] text-card" aria-label="complete">
        <Check className="size-3" strokeWidth={3} />
      </span>
    );
  if (status === "locked")
    return (
      <span className="grid size-5 place-items-center rounded-full bg-subtle text-ink-subtle" aria-label="locked">
        <Lock className="size-3" />
      </span>
    );
  return <span className="grid size-5 place-items-center rounded-full bg-primary-soft text-[11px] font-bold text-link">{step}</span>;
}

export function PrimaryNav({ stages }: { stages?: Stage[] }) {
  const status = (id?: string) => stages?.find((s) => s.id === id)?.status;
  return (
    <nav aria-label="Primary" className="border-b border-line bg-canvas">
      <div className="mx-auto flex max-w-[1400px] items-center gap-1 overflow-x-auto px-5 lg:px-8 scrollbar-thin">
        {PRIMARY_NAV.map((item, i) => (
          <div key={item.to} className="flex items-center">
            {i === 3 && <span aria-hidden className="mx-3 h-5 w-px bg-line-strong" />}
            <NavLink
              to={item.to}
              className={({ isActive }) =>
                classNames(
                  "-mb-px flex items-center gap-2 border-b-2 px-3 py-3 text-[14.5px] font-bold whitespace-nowrap transition-colors",
                  isActive ? "border-primary text-ink" : "border-transparent text-ink-subtle hover:text-ink",
                )
              }
            >
              {"step" in item && <StepMark step={item.step} status={status(item.stage)} />}
              {item.label}
            </NavLink>
          </div>
        ))}
      </div>
    </nav>
  );
}
