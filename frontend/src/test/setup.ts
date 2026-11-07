import "@testing-library/jest-dom/vitest";

// jsdom lacks matchMedia; tests control the OS preference through this flag.
let systemDark = false;
const listeners = new Set<() => void>();
export function setSystemDark(v: boolean) {
  systemDark = v;
  listeners.forEach((l) => l());
}
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    get matches() {
      return query.includes("dark") ? systemDark : false;
    },
    media: query,
    onchange: null,
    addEventListener: (_: string, l: () => void) => listeners.add(l),
    removeEventListener: (_: string, l: () => void) => listeners.delete(l),
    addListener: () => undefined,
    removeListener: () => undefined,
    dispatchEvent: () => false,
  }),
});
