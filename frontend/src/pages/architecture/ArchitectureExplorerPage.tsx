import type { ReactNode } from "react";
import { useNavigate, useParams } from "react-router";
import { Activity, BarChart3, Boxes, Cpu, GitBranch, Grid3x3 } from "lucide-react";
import type { Experiment } from "../../api/types";
import { RequireExperiment } from "../../components/layout/RequireExperiment";
import { LifecycleBanner } from "../../components/layout/LifecycleBanner";
import { LockedState, PageHeader, Tabs } from "../../components/ui/Misc";
import { Button } from "../../components/ui/Button";
import { ConfigTab } from "./ConfigTab";
import { JourneyTab } from "./JourneyTab";
import { TrainingTab } from "./TrainingTab";
import { KernelTab } from "./KernelTab";
import { BreakerTab } from "./BreakerTab";
import { ResultsTab } from "./ResultsTab";

export type ArchTab = "config" | "journey" | "training" | "kernels" | "breaker" | "results";
const TABS: ArchTab[] = ["config", "journey", "training", "kernels", "breaker", "results"];

export function ArchitectureExplorerPage() {
  return (
    <>
      <PageHeader
        eyebrow="Step 3 · central workspace"
        title="Architecture"
        accent="explorer."
        lead="Configure EEGNet, follow a trial through every layer, and train. Kernel inspection, the Architecture Breaker and results unlock only after a training run's checkpoint and metadata are saved and verified."
      />
      <RequireExperiment>{(exp) => <ExplorerBody exp={exp} />}</RequireExperiment>
    </>
  );
}

function ExplorerBody({ exp }: { exp: Experiment }) {
  const params = useParams();
  const navigate = useNavigate();
  const tab = (TABS.includes(params.tab as ArchTab) ? params.tab : "config") as ArchTab;
  const caps = exp.capabilities;
  const learned = caps.can_inspect_learned;
  return (
    <>
      <LifecycleBanner exp={exp} here="architecture" />
      <div className="mb-6">
        <Tabs<ArchTab>
          ariaLabel="Architecture Explorer sections"
          value={tab}
          onChange={(t) => navigate(`/architecture/${t}`)}
          tabs={[
            { id: "config", label: "Architecture Configuration", icon: <Boxes className="size-4" />, locked: !caps.can_configure_architecture },
            { id: "journey", label: "EEG Signal Journey", icon: <Activity className="size-4" />, locked: !caps.can_inspect_untrained && !learned },
            { id: "training", label: "Training Simulator", icon: <Cpu className="size-4" />, locked: !caps.can_train && !caps.busy && !exp.active_run },
            { id: "kernels", label: "Kernel & Convolution Playground", icon: <Grid3x3 className="size-4" />, locked: !learned },
            { id: "breaker", label: "Architecture Breaker", icon: <GitBranch className="size-4" />, locked: !caps.can_use_breaker },
            { id: "results", label: "Results & Predictions", icon: <BarChart3 className="size-4" />, locked: !learned },
          ]}
        />
      </div>
      {tab === "config" && (caps.can_configure_architecture || exp.architecture ? <ConfigTab exp={exp} /> : <Gate exp={exp} what="Architecture configuration" need="a validated preprocessing run (the input shape comes from the preprocessed data)" />)}
      {tab === "journey" && (caps.can_inspect_untrained || learned ? <JourneyTab exp={exp} /> : <Gate exp={exp} what="The Signal Journey" need="a validated architecture" />)}
      {tab === "training" && (caps.can_train || caps.busy || exp.active_run ? <TrainingTab exp={exp} /> : <Gate exp={exp} what="Training" need="valid data, preprocessing and a saved architecture" />)}
      {tab === "kernels" && (learned ? <KernelTab exp={exp} /> : <PostTrainingGate exp={exp} what="The Kernel & Convolution Playground" />)}
      {tab === "breaker" && (caps.can_use_breaker ? <BreakerTab exp={exp} /> : <PostTrainingGate exp={exp} what="The Architecture Breaker" />)}
      {tab === "results" && (learned ? <ResultsTab exp={exp} /> : <PostTrainingGate exp={exp} what="Results & Predictions" />)}
    </>
  );
}

function Gate({ exp, what, need }: { exp: Experiment; what: string; need: string }) {
  const navigate = useNavigate();
  return (
    <LockedState
      title={`${what} is not available yet`}
      reason={<>It needs {need}. {exp.capabilities.next_step}</>}
      action={<Button onClick={() => navigate(`/${exp.capabilities.next_page}`)}>Go to the next step</Button>}
    />
  );
}

function PostTrainingGate({ exp, what }: { exp: Experiment; what: string }) {
  const navigate = useNavigate();
  const run = exp.active_run;
  const failed = run?.artifact_status?.checks.filter((c) => !c.ok) ?? [];
  let reason: ReactNode = "It uses the saved, verified checkpoint of a completed training run and its recorded intermediates.";
  if (exp.capabilities.run_stale) reason = "The last run is stale because the configuration changed after training. Retrain to inspect the current configuration.";
  else if (run?.status === "failed") reason = `The last training run failed (${run.error}). Failed runs never unlock learned-model tools.`;
  else if (run?.status === "cancelled") reason = "The last training run was cancelled. Cancelled runs never unlock learned-model tools.";
  else if (failed.length) reason = <>Artifact verification failed: {failed.map((c) => c.label).join("; ")}.</>;
  else if (exp.capabilities.busy) reason = "Training or verification is in progress.";
  return (
    <LockedState
      title={`${what} unlocks after verified training`}
      reason={reason}
      action={<Button onClick={() => navigate("/architecture/training")}>Open the Training Simulator</Button>}
    />
  );
}
