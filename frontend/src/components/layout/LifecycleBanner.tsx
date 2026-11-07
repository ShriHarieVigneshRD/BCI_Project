import { Link } from "react-router";
import { ArrowRight, History } from "lucide-react";
import type { Experiment } from "../../api/types";
import { StateChip, SyntheticBadge } from "../ui/Chip";
import { Alert } from "../ui/Misc";

/** Shows where the experiment is in its lifecycle and what the backend says is required next. */
export function LifecycleBanner({ exp, here }: { exp: Experiment; here: string }) {
  const caps = exp.capabilities;
  const nextHere = caps.next_page.startsWith(here);
  return (
    <div className="mb-8 space-y-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-2xl border border-line bg-card px-5 py-3.5 shadow-card">
        <StateChip state={exp.state} />
        {exp.synthetic && <SyntheticBadge />}
        <div className="min-w-0 flex-1 text-[14.5px] text-ink-muted">
          <span className="font-bold text-ink">Next: </span>
          {caps.next_step}
        </div>
        {!nextHere && (
          <Link to={`/${caps.next_page}`} className="inline-flex items-center gap-1 text-[14px] font-bold text-link hover:underline">
            Go there <ArrowRight className="size-4" />
          </Link>
        )}
      </div>
      {caps.run_stale && (
        <Alert tone="warning" title="Results are stale">
          The configuration changed after the last training run, so its checkpoint and metrics describe a different
          configuration. They are kept in the registry but are not shown as results for the current settings.
        </Alert>
      )}
      {exp.inherited.length > 0 && (
        <details className="rounded-2xl border border-line bg-surface px-5 py-3 text-[14px] text-ink-muted">
          <summary className="flex cursor-pointer items-center gap-2 font-bold text-ink">
            <History className="size-4 text-link" /> Inherited settings ({exp.inherited.length})
          </summary>
          <ul className="mt-2 list-disc space-y-1 pl-6">
            {exp.inherited.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
