import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeQueryClient, routes } from "../App";
import { useUI } from "../store/ui";
import { experiment, HEALTH, mockFetch } from "./fixtures";

function renderAt(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(
    <QueryClientProvider client={makeQueryClient()}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return router;
}

describe("primary navigation and workflow gating", () => {
  beforeEach(() => {
    act(() => useUI.getState().setCurrentExperiment("exp_test"));
  });
  afterEach(() => vi.unstubAllGlobals());

  it("shows the six primary areas in workflow order and navigates between them", async () => {
    mockFetch({ "/api/health": HEALTH, "/api/experiments/exp_test": experiment(), "/api/experiments": [], "/api/datasets": [], "/api/meta/formats": [] });
    const router = renderAt("/dataset");
    const nav = screen.getByRole("navigation", { name: "Primary" });
    const links = within(nav).getAllByRole("link");
    expect(links.map((l) => l.textContent?.replace(/^\d/, "").trim())).toEqual([
      "Dataset & Splits",
      "Preprocessing Studio",
      "Architecture Explorer",
      "Hyperparameter Battle Arena",
      "Tensor Detective",
      "Experiments & Exports",
    ]);
    expect(within(nav).getByRole("link", { name: /Dataset & Splits/ })).toHaveAttribute("aria-current", "page");
    // global header actions
    expect(screen.getByRole("button", { name: /Upload dataset/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Start new experiment/ })).toBeInTheDocument();
    expect(screen.getByRole("radiogroup", { name: "Theme" })).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(within(nav).getByRole("link", { name: /Tensor Detective/ }));
    expect(router.state.location.pathname).toBe("/tensors");
    expect(within(nav).getByRole("link", { name: /Tensor Detective/ })).toHaveAttribute("aria-current", "page");
    await user.click(within(nav).getByRole("link", { name: /Architecture Explorer/ }));
    expect(router.state.location.pathname).toBe("/architecture");
    await waitFor(() => expect(screen.getByRole("tablist", { name: "Architecture Explorer sections" })).toBeInTheDocument());
  });

  it("marks completed workflow steps from backend-reported stages", async () => {
    mockFetch({ "/api/health": HEALTH, "/api/experiments/exp_test": experiment(), "/api/experiments": [], "/api/datasets": [], "/api/meta/formats": [] });
    renderAt("/experiments");
    const nav = screen.getByRole("navigation", { name: "Primary" });
    await waitFor(() => expect(within(nav).getAllByLabelText("complete")).toHaveLength(3));
  });

  it("locks post-training tools until the backend reports verified artifacts", async () => {
    mockFetch({ "/api/health": HEALTH, "/api/experiments/exp_test": experiment(), "/api/experiments": [], "/api/datasets": [] });
    renderAt("/architecture/kernels");
    await waitFor(() => expect(screen.getByText(/unlocks after verified training/)).toBeInTheDocument());
    const tabs = screen.getByRole("tablist", { name: "Architecture Explorer sections" });
    for (const name of [/Kernel & Convolution Playground/, /Architecture Breaker/, /Results & Predictions/]) {
      expect(within(within(tabs).getByRole("tab", { name })).getByLabelText("locked")).toBeInTheDocument();
    }
    expect(within(within(tabs).getByRole("tab", { name: /Architecture Configuration/ })).queryByLabelText("locked")).toBeNull();
  });

  it("does not unlock learned-model tools for a failed run", async () => {
    const failed = experiment(
      { active_run_id: "run_x", active_run: { id: "run_x", status: "failed", error: "boom", folds: [], artifact_status: null } as never },
      { state: "training_failed", can_inspect_learned: false },
    );
    mockFetch({ "/api/health": HEALTH, "/api/experiments/exp_test": failed, "/api/experiments": [], "/api/datasets": [] });
    renderAt("/architecture/breaker");
    await waitFor(() => expect(screen.getByText(/Failed runs never unlock learned-model tools/)).toBeInTheDocument());
  });

  it("asks for an experiment when none is selected", async () => {
    act(() => useUI.getState().setCurrentExperiment(null));
    mockFetch({ "/api/health": HEALTH, "/api/experiments": [], "/api/datasets": [] });
    renderAt("/preprocessing");
    expect(await screen.findByText("No experiment selected")).toBeInTheDocument();
  });
});
