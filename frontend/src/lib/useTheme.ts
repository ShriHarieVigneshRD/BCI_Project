import { useEffect } from "react";
import { applyTheme, resolveTheme, useUI } from "../store/ui";

/** Keep <html class="dark"> in sync with the persisted preference and the OS setting. */
export function useThemeSync(): void {
  const pref = useUI((s) => s.theme);
  const setResolved = useUI((s) => s.setResolvedTheme);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => {
      const resolved = resolveTheme(pref, mq.matches);
      applyTheme(resolved);
      setResolved(resolved);
    };
    update();
    mq.addEventListener?.("change", update);
    return () => mq.removeEventListener?.("change", update);
  }, [pref, setResolved]);
}
