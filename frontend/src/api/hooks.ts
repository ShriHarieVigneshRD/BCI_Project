import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback } from "react";
import { api, qs } from "./client";
import type {
  ArchitectureState,
  Candidates,
  DatasetRecord,
  Experiment,
  ExperimentSummary,
  FormatInfo,
  Health,
  TrialRow,
} from "./types";
import { useUI } from "../store/ui";

export const keys = {
  health: ["health"] as const,
  datasets: ["datasets"] as const,
  dataset: (id: string) => ["dataset", id] as const,
  experiments: (search?: string, state?: string) => ["experiments", search ?? "", state ?? ""] as const,
  experiment: (id: string) => ["experiment", id] as const,
  architecture: (id: string) => ["architecture", id] as const,
  trials: (id: string, fold: number, trained: boolean) => ["trials", id, fold, trained] as const,
};

export function useHealth() {
  return useQuery({ queryKey: keys.health, queryFn: () => api.get<Health>("/api/health"), staleTime: 60_000 });
}

export function useFormats() {
  return useQuery({ queryKey: ["formats"], queryFn: () => api.get<FormatInfo[]>("/api/meta/formats"), staleTime: Infinity });
}

export function useDatasets() {
  return useQuery({ queryKey: keys.datasets, queryFn: () => api.get<DatasetRecord[]>("/api/datasets") });
}

export function useDataset(id: string | null | undefined) {
  return useQuery({
    queryKey: keys.dataset(id ?? ""),
    queryFn: () => api.get<DatasetRecord>(`/api/datasets/${id}`),
    enabled: !!id,
  });
}

export function useExperiments(search?: string, state?: string) {
  return useQuery({
    queryKey: keys.experiments(search, state),
    queryFn: () => api.get<ExperimentSummary[]>(`/api/experiments${qs({ search, state })}`),
  });
}

/** The selected experiment. Polls while the backend reports a busy lifecycle state. */
export function useExperiment(id: string | null | undefined) {
  return useQuery({
    queryKey: keys.experiment(id ?? ""),
    queryFn: () => api.get<Experiment>(`/api/experiments/${id}`),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data?.capabilities.busy ? 1500 : false),
    retry: (count, err) => (err as { status?: number }).status !== 404 && count < 2,
  });
}

export function useCurrentExperiment() {
  const id = useUI((s) => s.currentExperimentId);
  return { id, ...useExperiment(id) };
}

export function useArchitecture(id: string | null | undefined, enabled = true) {
  return useQuery({
    queryKey: keys.architecture(id ?? ""),
    queryFn: () => api.get<ArchitectureState>(`/api/experiments/${id}/architecture`),
    enabled: !!id && enabled,
  });
}

export function useCandidates(id: string | null | undefined, enabled = true) {
  return useQuery({
    queryKey: ["candidates", id],
    queryFn: () => api.get<Candidates>(`/api/experiments/${id}/candidates`),
    enabled: !!id && enabled,
  });
}

export function useTrials(id: string | null | undefined, fold = 0, trained = false, enabled = true) {
  return useQuery({
    queryKey: keys.trials(id ?? "", fold, trained),
    queryFn: () =>
      api.get<{ trials: TrialRow[]; class_names: string[]; fold: number; trained: boolean }>(
        `/api/experiments/${id}/trials${qs({ fold })}`,
      ),
    enabled: !!id && enabled,
  });
}

/** Invalidate everything that depends on an experiment after a mutation. */
export function useRefreshExperiment() {
  const qc = useQueryClient();
  return useCallback(
    (id?: string | null) => {
      if (id) {
        qc.invalidateQueries({ queryKey: keys.experiment(id) });
        qc.invalidateQueries({ queryKey: keys.architecture(id) });
        qc.invalidateQueries({ queryKey: ["trials", id] });
        qc.invalidateQueries({ queryKey: ["candidates", id] });
        qc.invalidateQueries({ queryKey: ["split", id] });
        qc.invalidateQueries({ queryKey: ["results", id] });
        qc.invalidateQueries({ queryKey: ["kernels", id] });
        qc.invalidateQueries({ queryKey: ["artifacts", id] });
      }
      qc.invalidateQueries({ queryKey: ["experiments"] });
    },
    [qc],
  );
}
