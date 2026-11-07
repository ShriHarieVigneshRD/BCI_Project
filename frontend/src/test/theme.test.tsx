import { act, render, renderHook, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { applyTheme, resolveTheme, useUI } from "../store/ui";
import { useThemeSync } from "../lib/useTheme";
import { ThemeSelector } from "../components/layout/Header";
import { setSystemDark } from "./setup";

const isDark = () => document.documentElement.classList.contains("dark");

describe("theme modes", () => {
  beforeEach(() => {
    localStorage.clear();
    setSystemDark(false);
    act(() => useUI.getState().setTheme("system"));
    document.documentElement.classList.remove("dark");
  });

  it("resolves system, light and dark preferences", () => {
    expect(resolveTheme("system", true)).toBe("dark");
    expect(resolveTheme("system", false)).toBe("light");
    expect(resolveTheme("light", true)).toBe("light");
    expect(resolveTheme("dark", false)).toBe("dark");
    applyTheme("dark");
    expect(isDark()).toBe(true);
    applyTheme("light");
    expect(isDark()).toBe(false);
  });

  it("applies light, dark and system modes to <html> and follows OS changes", () => {
    renderHook(() => useThemeSync());
    act(() => useUI.getState().setTheme("dark"));
    expect(isDark()).toBe(true);
    expect(useUI.getState().resolvedTheme).toBe("dark");
    act(() => useUI.getState().setTheme("light"));
    expect(isDark()).toBe(false);
    act(() => useUI.getState().setTheme("system"));
    expect(isDark()).toBe(false);
    act(() => setSystemDark(true));
    expect(isDark()).toBe(true);
    act(() => setSystemDark(false));
    expect(isDark()).toBe(false);
  });

  it("persists the preference locally", () => {
    act(() => useUI.getState().setTheme("dark"));
    const stored = JSON.parse(localStorage.getItem("eegnet-ui") ?? "{}");
    expect(stored.state.theme).toBe("dark");
  });

  it("theme selector switches modes", async () => {
    renderHook(() => useThemeSync());
    render(<ThemeSelector />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("radio", { name: "Dark theme" }));
    expect(useUI.getState().theme).toBe("dark");
    expect(isDark()).toBe(true);
    expect(screen.getByRole("radio", { name: "Dark theme" })).toHaveAttribute("aria-checked", "true");
    await user.click(screen.getByRole("radio", { name: "Light theme" }));
    expect(isDark()).toBe(false);
    await user.click(screen.getByRole("radio", { name: "System theme" }));
    expect(useUI.getState().theme).toBe("system");
  });
});
