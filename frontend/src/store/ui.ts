import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ThemePref = "system" | "light" | "dark";

interface UIState {
  theme: ThemePref;
  resolvedTheme: "light" | "dark";
  currentExperimentId: string | null;
  uploadOpen: boolean;
  newExperimentOpen: boolean;
  /** Experiment that should receive the dataset when the import finishes (e.g. a just-created experiment). */
  importAttachTo: string | null;
  /** Re-open an existing import session (e.g. "Revise dataset") instead of starting a new one. */
  importSessionId: string | null;
  setTheme: (t: ThemePref) => void;
  setResolvedTheme: (t: "light" | "dark") => void;
  setCurrentExperiment: (id: string | null) => void;
  setUploadOpen: (v: boolean, opts?: { attachTo?: string | null; sessionId?: string | null }) => void;
  setNewExperimentOpen: (v: boolean) => void;
}

export const useUI = create<UIState>()(
  persist(
    (set) => ({
      theme: "system",
      resolvedTheme: "light",
      currentExperimentId: null,
      uploadOpen: false,
      newExperimentOpen: false,
      importAttachTo: null,
      importSessionId: null,
      setTheme: (theme) => set({ theme }),
      setResolvedTheme: (resolvedTheme) => set({ resolvedTheme }),
      setCurrentExperiment: (currentExperimentId) => set({ currentExperimentId }),
      setUploadOpen: (uploadOpen, opts) =>
        set(uploadOpen ? { uploadOpen, importAttachTo: opts?.attachTo ?? null, importSessionId: opts?.sessionId ?? null }
          : { uploadOpen, importAttachTo: null, importSessionId: null }),
      setNewExperimentOpen: (newExperimentOpen) => set({ newExperimentOpen }),
    }),
    {
      name: "eegnet-ui",
      partialize: (s) => ({ theme: s.theme, currentExperimentId: s.currentExperimentId }),
    },
  ),
);

/** Resolve a preference against the OS setting and apply it to <html>. */
export function resolveTheme(pref: ThemePref, systemDark: boolean): "light" | "dark" {
  return pref === "system" ? (systemDark ? "dark" : "light") : pref;
}

export function applyTheme(resolved: "light" | "dark", root: HTMLElement = document.documentElement): void {
  root.classList.toggle("dark", resolved === "dark");
  root.style.colorScheme = resolved;
}
