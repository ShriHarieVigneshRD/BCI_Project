import type { ReactNode } from "react";
import { useNavigate } from "react-router";
import { FlaskConical, Plus } from "lucide-react";
import { useCurrentExperiment } from "../../api/hooks";
import type { Experiment } from "../../api/types";
import { useUI } from "../../store/ui";
import { Button } from "../ui/Button";
import { EmptyState, ErrorState, Loading } from "../ui/Misc";

export function RequireExperiment({ children }: { children: (exp: Experiment) => ReactNode }) {
  const { id, data, isLoading, error, refetch } = useCurrentExperiment();
  const setNewOpen = useUI((s) => s.setNewExperimentOpen);
  const navigate = useNavigate();
  if (!id)
    return (
      <EmptyState
        icon={<FlaskConical className="size-6" />}
        title="No experiment selected"
        action={
          <>
            <Button icon={<Plus className="size-4" />} onClick={() => setNewOpen(true)}>Start new experiment</Button>
            <Button variant="secondary" onClick={() => navigate("/experiments")}>Open an existing one</Button>
          </>
        }
      >
        Every experiment starts at Dataset &amp; Splits, continues through the Preprocessing Studio and is trained in the
        Architecture Explorer. Earlier experiments are always preserved.
      </EmptyState>
    );
  if (isLoading || !data) return error ? <ErrorState error={error} onRetry={() => refetch()} /> : <Loading label="Loading experiment…" />;
  return <>{children(data)}</>;
}
