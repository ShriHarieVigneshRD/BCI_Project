import { useEffect } from "react";
import { Outlet } from "react-router";
import { ShieldCheck } from "lucide-react";
import { useCurrentExperiment } from "../../api/hooks";
import { useUI } from "../../store/ui";
import { useThemeSync } from "../../lib/useTheme";
import { Header } from "./Header";
import { PrimaryNav } from "./Nav";
import { ImportDatasetModal } from "../modals/ImportDatasetModal";
import { NewExperimentModal } from "../modals/NewExperimentModal";

export function AppShell() {
  useThemeSync();
  const { data: exp, error } = useCurrentExperiment();
  const setCurrent = useUI((s) => s.setCurrentExperiment);
  useEffect(() => {
    if (error && (error as { status?: number }).status === 404) setCurrent(null);
  }, [error, setCurrent]);
  return (
    <div className="flex min-h-screen flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:m-2 focus:rounded-full focus:bg-card focus:px-4 focus:py-2">
        Skip to content
      </a>
      <Header />
      <PrimaryNav stages={exp?.capabilities.stages} />
      <main id="main" className="mx-auto w-full max-w-[1400px] flex-1 px-5 py-10 lg:px-8">
        <Outlet />
      </main>
      <footer className="border-t border-line bg-surface">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-3 px-5 py-5 text-[13px] text-ink-subtle lg:px-8">
          <ShieldCheck className="size-4 text-success" />
          Runs locally. EEG data are processed on this machine and never sent to external services.
          <span className="ml-auto">EEGNet Architecture Playground · educational &amp; experimental use</span>
        </div>
      </footer>
      <ImportDatasetModal />
      <NewExperimentModal />
    </div>
  );
}
