import { vi } from "vitest";
import type { Capabilities, Experiment } from "../api/types";

export function capabilities(over: Partial<Capabilities> = {}): Capabilities {
  return {
    state: "architecture_configured",
    busy: false,
    can_attach_dataset: true,
    can_configure_split: true,
    can_configure_preprocessing: true,
    can_run_preprocessing: true,
    can_configure_architecture: true,
    can_train: true,
    can_cancel: false,
    can_inspect_untrained: true,
    can_inspect_learned: false,
    can_use_breaker: false,
    can_evaluate: false,
    has_results: false,
    can_reverify: false,
    run_stale: false,
    next_page: "architecture/training",
    next_step: "Review training settings and start training.",
    stages: [
      { id: "dataset", label: "Dataset & Splits", status: "complete" },
      { id: "preprocessing", label: "Preprocessing Studio", status: "complete" },
      { id: "architecture", label: "Architecture Explorer", status: "complete" },
      { id: "training", label: "Training", status: "available" },
      { id: "artifacts", label: "Verified artifacts", status: "locked" },
      { id: "evaluation", label: "Evaluation", status: "locked" },
    ],
    ...over,
  };
}

export function experiment(over: Partial<Experiment> = {}, caps: Partial<Capabilities> = {}): Experiment {
  const c = capabilities(caps);
  return {
    id: "exp_test",
    name: "Test experiment",
    created_at: "2026-10-31T10:00:00+00:00",
    updated_at: "2026-10-31T10:00:00+00:00",
    state: c.state,
    parent_id: null,
    inherited: [],
    workflow: "single",
    dataset_id: null,
    test_dataset_id: null,
    split_config: null,
    split_fingerprint: null,
    preprocessing_config: null,
    preprocessing_fingerprint: null,
    preprocessing_summary: null,
    ica_reference: null,
    architecture_id: null,
    architecture: null,
    training_config: {
      epochs: 60, batch_size: 32, learning_rate: 0.001, optimizer: "adam", momentum: 0.9, weight_decay: 0, loss: "cross_entropy",
      label_smoothing: 0, seed: 42, device: "auto", validation_mode: "auto", validation_fraction: 0.2, early_stopping: true,
      patience: 15, monitor: "val_loss", restore_best: true, scheduler: "none", step_size: 20, gamma: 0.5, class_weighting: "none",
      representative_trials: 4,
    },
    active_run_id: null,
    warnings: [],
    stale_notes: [],
    capabilities: c,
    active_run: null,
    runs: [],
    datasets: [],
    synthetic: false,
    ...over,
  };
}

/** Route-aware fetch mock: exact path match; anything unmocked returns 404. */
export function mockFetch(routes: Record<string, unknown>) {
  const fn = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    const path = url.split("?")[0];
    const key = Object.keys(routes).find((k) => path === k);
    if (!key) return new Response(JSON.stringify({ detail: `unmocked ${path}` }), { status: 404 });
    return new Response(JSON.stringify(routes[key]), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

export const HEALTH = {
  status: "ok",
  app_version: "1.0.0",
  software: { python: "3.11" },
  device: { cuda_available: false, default_device: "cpu", cuda_device: null, cuda_version: null, torch_threads: 4 },
  training_worker: { running_run_id: null },
  privacy: "local",
};
