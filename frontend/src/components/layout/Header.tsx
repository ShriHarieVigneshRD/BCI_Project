import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useNavigate } from "react-router";
import { Activity, ChevronDown, Cpu, Monitor, Moon, Plus, Sun, Upload, Zap } from "lucide-react";
import { useCurrentExperiment, useExperiments, useHealth } from "../../api/hooks";
import { useUI, type ThemePref } from "../../store/ui";
import { classNames, STATE_LABEL } from "../../lib/format";
import { Button } from "../ui/Button";
import { Dot, StateChip } from "../ui/Chip";
import { stateTone } from "../../lib/format";

export function Logo() {
  return (
    <Link to="/" className="flex items-center gap-2.5" aria-label="EEGNet Playground home">
      <span className="grid size-8 place-items-center rounded-[9px] bg-primary text-on-primary shadow-sm">
        <Activity className="size-[18px]" strokeWidth={2.4} />
      </span>
      <span className="text-[19px] font-bold tracking-[-0.01em] text-ink">
        EEGNet <span className="font-normal text-ink-muted">Playground</span>
      </span>
    </Link>
  );
}

export function ThemeSelector() {
  const theme = useUI((s) => s.theme);
  const setTheme = useUI((s) => s.setTheme);
  const opts: { value: ThemePref; icon: ReactNode; label: string }[] = [
    { value: "system", icon: <Monitor className="size-4" />, label: "System theme" },
    { value: "light", icon: <Sun className="size-4" />, label: "Light theme" },
    { value: "dark", icon: <Moon className="size-4" />, label: "Dark theme" },
  ];
  return (
    <div role="radiogroup" aria-label="Theme" className="inline-flex rounded-full border border-line bg-subtle p-0.5">
      {opts.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={theme === o.value}
          aria-label={o.label}
          title={o.label}
          onClick={() => setTheme(o.value)}
          className={classNames(
            "grid size-8 place-items-center rounded-full transition-colors",
            theme === o.value ? "bg-card text-ink shadow-sm" : "text-ink-subtle hover:text-ink",
          )}
        >
          {o.icon}
        </button>
      ))}
    </div>
  );
}

function DeviceBadge() {
  const { data } = useHealth();
  if (!data) return null;
  const cuda = data.device.cuda_available;
  return (
    <span
      className="hidden items-center gap-1.5 rounded-full border border-line bg-card px-3 py-1 text-[12.5px] font-bold text-ink-muted xl:inline-flex"
      title={cuda ? `${data.device.cuda_device} · CUDA ${data.device.cuda_version}` : "CPU only (no CUDA device)"}
    >
      {cuda ? <Zap className="size-3.5 text-success" /> : <Cpu className="size-3.5" />}
      {cuda ? "CUDA" : "CPU"}
      {data.training_worker.running_run_id && <Dot tone="warning" pulse />}
    </span>
  );
}

function ExperimentSwitcher() {
  const { id, data: exp } = useCurrentExperiment();
  const { data: list } = useExperiments();
  const setCurrent = useUI((s) => s.setCurrentExperiment);
  const setNewOpen = useUI((s) => s.setNewExperimentOpen);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  useEffect(() => {
    const onDoc = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);
  return (
    <div ref={ref} className="relative min-w-0">
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="flex max-w-[420px] min-w-0 items-center gap-2.5 rounded-full border border-line bg-card py-1.5 pr-3 pl-4 shadow-sm hover:border-line-strong"
      >
        <span className="min-w-0 text-left">
          <span className="block text-[11px] font-bold uppercase tracking-[0.07em] text-ink-subtle">Experiment</span>
          <span className="block truncate text-[14px] font-bold text-ink">{exp?.name ?? (id ? "Loading…" : "None selected")}</span>
        </span>
        {exp && <StateChip state={exp.state} className="hidden md:inline-flex" />}
        <ChevronDown className="size-4 shrink-0 text-ink-subtle" />
      </button>
      {open && (
        <div role="listbox" className="absolute top-full left-0 z-40 mt-2 w-[380px] rounded-2xl border border-line bg-card p-2 shadow-float">
          <div className="max-h-80 overflow-y-auto scrollbar-thin">
            {(list ?? []).slice(0, 12).map((e) => (
              <button
                key={e.id}
                role="option"
                aria-selected={e.id === id}
                type="button"
                onClick={() => {
                  setCurrent(e.id);
                  setOpen(false);
                }}
                className={classNames(
                  "flex w-full items-center gap-3 rounded-xl px-3 py-2 text-left hover:bg-subtle",
                  e.id === id && "bg-primary-soft/60",
                )}
              >
                <Dot tone={stateTone(e.state)} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[14px] font-bold text-ink">{e.name}</span>
                  <span className="block truncate text-[12px] text-ink-subtle">
                    {STATE_LABEL[e.state]} · {e.dataset_name ?? "no dataset"}
                  </span>
                </span>
              </button>
            ))}
            {!list?.length && <p className="px-3 py-4 text-[14px] text-ink-subtle">No experiments yet.</p>}
          </div>
          <div className="mt-1 flex gap-2 border-t border-line px-1 pt-2">
            <Button size="sm" variant="ghost" onClick={() => { setOpen(false); navigate("/experiments"); }}>
              All experiments
            </Button>
            <Button size="sm" variant="ghost" icon={<Plus className="size-4" />} onClick={() => { setOpen(false); setNewOpen(true); }}>
              New
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

export function Header() {
  const setUploadOpen = useUI((s) => s.setUploadOpen);
  const setNewOpen = useUI((s) => s.setNewExperimentOpen);
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-canvas/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1400px] items-center gap-4 px-5 py-3 lg:px-8">
        <Logo />
        <div className="ml-2 hidden min-w-0 md:block">
          <ExperimentSwitcher />
        </div>
        <div className="ml-auto flex items-center gap-2.5">
          <DeviceBadge />
          <Button variant="ghost" size="md" icon={<Upload className="size-4" />} onClick={() => setUploadOpen(true)}>
            <span className="hidden sm:inline">Upload dataset</span>
          </Button>
          <Button size="md" icon={<Plus className="size-4" />} onClick={() => setNewOpen(true)}>
            <span className="hidden sm:inline">Start new experiment</span>
          </Button>
          <ThemeSelector />
        </div>
      </div>
      <div className="px-5 pb-3 md:hidden">
        <ExperimentSwitcher />
      </div>
    </header>
  );
}
