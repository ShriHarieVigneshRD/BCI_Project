import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeQueryClient, routes } from "../App";
import type { DatasetManifest, ImportSession } from "../api/types";
import { DatasetReview } from "../components/import/DatasetReview";
import { useUI } from "../store/ui";
import { HEALTH } from "./fixtures";

function manifest(over: Partial<DatasetManifest> = {}): DatasetManifest {
  const rec = (i: number, p: string, run: string, included = true) => ({
    index: i, file: `sub-${p}/eeg/sub-${p}_run-${run}_eeg.edf`, format: "edf", companions: [], structure: "continuous" as const,
    participant: p, session: null, run, task: "mi", id_source: "file name pattern", sfreq: 128, n_channels: 8,
    ch_names: ["C3", "Cz", "C4"], ch_types: ["eeg", "eeg", "eeg"], n_samples: 128 * 60, duration_s: 60, n_epochs: null,
    inner_groups: null, unit: "uV", events: { kind: "events_file", file: `sub-${p}_run-${run}_events.tsv`, n_events: 16, note: null },
    event_counts: { left: 8, right: 8 }, included, exclude_reason: included ? null : "no events", readable: true, issues: [],
    recording_id: `sub-${p}_task-mi_run-${run}`, n_trials: included ? 16 : 0,
    class_counts: (included ? { left: 8, right: 8 } : {}) as Record<string, number>,
  });
  return {
    name: "Nested study",
    source: { mode: "local_path" },
    recordings: [rec(0, "01", "1"), rec(1, "01", "2"), rec(2, "02", "1"), rec(3, "02", "2", false)],
    mapping: [
      { task: "mi", event: "left", action: "class", label: "left", count: 24 },
      { task: "mi", event: "right", action: "class", label: "right", count: 24 },
    ],
    channel_policy: "intersection",
    sfreq_policy: "resample",
    target_sfreq: null,
    array_options: { sfreq: null, unit: "uV", ch_names: null, tmin: 0 },
    files: [
      { path: "sub-01/eeg/sub-01_run-1_eeg.edf", size: 1000, role: "eeg", recording: 0, note: null },
      { path: "sub-01/eeg/sub-01_run-1_events.tsv", size: 100, role: "events", recording: 0, note: "events for …" },
      { path: "participants.tsv", size: 50, role: "metadata", recording: null, note: null },
      { path: "stimuli/arrow.png", size: 10, role: "ignored", recording: null, note: "not an EEG, event or metadata file" },
    ],
    metadata: {},
    issues: [{ level: "warning", code: "orphan_event_files", message: "1 event file(s) could not be matched to a recording." }],
    importable: true,
    summary: {
      n_files: 4, n_recordings: 4, n_included: 3, n_unreadable: 0, structure: "continuous", n_participants: 2,
      participants: [
        { participant: "01", n_recordings: 2, sessions: [], runs: ["1", "2"], n_trials: 32, class_counts: { left: 16, right: 16 }, attributes: { age: 24 } },
        { participant: "02", n_recordings: 1, sessions: [], runs: ["1"], n_trials: 16, class_counts: { left: 8, right: 8 }, attributes: {} },
      ],
      n_trials: 48, classes: ["left", "right"], class_counts: { left: 24, right: 24 }, class_seconds: null, sfreqs: [128],
      target_sfreq: 128, common_channels: ["C3", "Cz", "C4"], common_channel_types: ["eeg", "eeg", "eeg"], dropped_channels: [],
      total_duration_s: 180, sessions: [], runs: ["1", "2"], tasks: ["mi"],
    },
    ...over,
  };
}

describe("dataset review", () => {
  it("shows dynamic counts and sends edits", async () => {
    const onPatch = vi.fn();
    render(<DatasetReview manifest={manifest()} onPatch={onPatch} busy={false} />);
    expect(screen.getAllByText("3/4").length).toBeGreaterThan(0); // recordings included / found
    expect(screen.getByText("48")).toBeInTheDocument(); // trials from the manifest summary, not a placeholder
    expect(screen.getByText(/could not be matched/)).toBeInTheDocument();
    // an excluded recording shows why it is excluded rather than an "ok" mark
    expect(screen.getByTitle("Excluded: no events")).toBeInTheDocument();
    // recordings tab: edit a participant ID (commit on blur) and exclude a recording
    const input = screen.getByLabelText("participant of sub-02/eeg/sub-02_run-1_eeg.edf");
    fireEvent.change(input, { target: { value: "02b" } });
    fireEvent.blur(input);
    expect(onPatch).toHaveBeenCalledWith({ recordings: [{ index: 2, participant: "02b" }] });
    fireEvent.click(screen.getByLabelText("include sub-01/eeg/sub-01_run-2_eeg.edf"));
    expect(onPatch).toHaveBeenCalledWith({ recordings: [{ index: 1, included: false }] });
    // participants tab
    const user = userEvent.setup();
    await user.click(screen.getByRole("tab", { name: /Participants/ }));
    const table = screen.getByRole("table");
    expect(within(table).getByText("01")).toBeInTheDocument();
    expect(within(table).getByText(/age: 24/)).toBeInTheDocument();
    // mapping tab: per-event action and a wildcard rule
    await user.click(screen.getByRole("tab", { name: /Event mapping/ }));
    fireEvent.change(screen.getByLabelText("action for right"), { target: { value: "ignore" } });
    expect(onPatch).toHaveBeenCalledWith({ mapping: [{ task: "mi", event: "right", action: "ignore", label: "right" }] });
    await user.type(screen.getByLabelText("Apply to events matching"), "le*");
    await user.type(screen.getByLabelText("class label"), "L");
    await user.click(screen.getByRole("button", { name: /Apply/ }));
    expect(onPatch).toHaveBeenCalledWith({ rules: [{ pattern: "le*", action: "class", label: "L" }] });
  });
});

describe("import dialog", () => {
  beforeEach(() => act(() => useUI.getState().setCurrentExperiment(null)));
  afterEach(() => vi.unstubAllGlobals());

  it("offers the three import methods and walks a local folder through scan and review", async () => {
    const reviewing: ImportSession = {
      id: "import_x", created_at: "", updated_at: "", mode: "local_path", status: "review", name: null, root_label: "study",
      local_path: "D:\\study", staged_files: 0, staged_bytes: 0, progress: {}, error: null, dataset_id: null, revises: null,
      attach_experiment_id: null, manifest: manifest({ importable: false, issues: [{ level: "error", code: "too_few_classes", message: "The mapping yields 1 class(es)" }] }),
    };
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input).split("?")[0];
      const method = init?.method ?? "GET";
      const json = (b: unknown) => new Response(JSON.stringify(b), { status: 200, headers: { "Content-Type": "application/json" } });
      if (url === "/api/health") return json(HEALTH);
      if (url === "/api/meta/formats") return json([]);
      if (url === "/api/experiments" || url === "/api/datasets") return json([]);
      if (url === "/api/imports" && method === "POST") return json({ ...reviewing, status: "scanning", manifest: null });
      if (url === "/api/imports/import_x") return json(reviewing);
      return new Response(JSON.stringify({ detail: "unmocked" }), { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    const router = createMemoryRouter(routes, { initialEntries: ["/dataset"] });
    render(<QueryClientProvider client={makeQueryClient()}><RouterProvider router={router} /></QueryClientProvider>);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Upload dataset/ }));
    const dialog = await screen.findByRole("dialog");
    for (const name of [/Select dataset folder/, /Select multiple files/, /Use local folder path/]) {
      expect(within(dialog).getByRole("radio", { name })).toBeInTheDocument();
    }
    await user.click(within(dialog).getByRole("radio", { name: /Use local folder path/ }));
    fireEvent.change(within(dialog).getByLabelText("Folder path on this computer"), { target: { value: "D:\\study" } });
    await user.click(within(dialog).getByRole("button", { name: "Scan folder" }));
    const body = JSON.parse(String(fetchMock.mock.calls.find((c) => String(c[0]) === "/api/imports")![1]!.body));
    expect(body).toMatchObject({ mode: "local_path", local_path: "D:\\study" });
    await waitFor(() => expect(screen.getByRole("dialog", { name: "Review dataset" })).toBeInTheDocument(), { timeout: 4000 });
    expect(screen.getByText(/Fix before importing/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Import 48 trials from 3 recordings/ })).toBeDisabled();
  });
});
