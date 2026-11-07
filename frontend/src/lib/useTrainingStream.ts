import { useEffect, useRef, useState } from "react";
import type { HistoryRow, ProgressEvent } from "../api/types";

export interface StreamState {
  connected: boolean;
  status: string | null;
  device: string | null;
  folds: Record<number, HistoryRow[]>;
  nFolds: number;
  epochs: number | null;
  currentFold: number;
  log: string[];
  done: boolean;
  artifactsReady: boolean | null;
  message: string | null;
}

const EMPTY: StreamState = {
  connected: false, status: null, device: null, folds: {}, nFolds: 1, epochs: null, currentFold: 0, log: [], done: false,
  artifactsReady: null, message: null,
};

const TYPES: ProgressEvent["type"][] = ["snapshot", "status", "log", "epoch", "fold_done", "done", "failed", "cancelled"];

/** Subscribe to the backend's Server-Sent Events for a training run. Every value comes from the real training loop. */
export function useTrainingStream(expId: string, runId: string | null, enabled: boolean, onTerminal?: () => void): StreamState {
  const [state, setState] = useState<StreamState>(EMPTY);
  const terminalRef = useRef(onTerminal);
  terminalRef.current = onTerminal;

  useEffect(() => {
    if (!runId || !enabled || typeof EventSource === "undefined") return;
    setState(EMPTY);
    const es = new EventSource(`/api/experiments/${expId}/progress?run_id=${encodeURIComponent(runId)}`);
    const handle = (raw: MessageEvent) => {
      const ev = JSON.parse(raw.data) as ProgressEvent;
      setState((s) => {
        const next: StreamState = { ...s, connected: true };
        switch (ev.type) {
          case "snapshot": {
            const folds: Record<number, HistoryRow[]> = {};
            ev.folds?.forEach((f) => (folds[f.fold] = f.history));
            return { ...next, status: ev.status ?? null, device: ev.device ?? null, folds, nFolds: Math.max(1, ev.folds?.length ?? 1), log: ev.log ?? [], message: ev.error ?? null };
          }
          case "status":
            return { ...next, status: ev.status ?? s.status, device: ev.device ?? s.device, message: ev.message ?? s.message };
          case "log":
            return { ...next, log: [...s.log, ev.message ?? ""].slice(-200) };
          case "epoch": {
            const f = ev.fold ?? 0;
            const row: HistoryRow = {
              epoch: ev.epoch!, train_loss: ev.train_loss!, train_acc: ev.train_acc!, val_loss: ev.val_loss, val_acc: ev.val_acc,
              val_balanced_accuracy: ev.val_balanced_accuracy, lr: ev.lr!, elapsed_s: ev.elapsed_s!,
            };
            const prev = (s.folds[f] ?? []).filter((r) => r.epoch !== row.epoch);
            return { ...next, status: "training", folds: { ...s.folds, [f]: [...prev, row] }, nFolds: ev.n_folds ?? s.nFolds, epochs: ev.epochs ?? s.epochs, currentFold: f };
          }
          case "fold_done":
            return { ...next, log: [...s.log, `Fold ${(ev.fold ?? 0) + 1} finished; checkpoint saved (epoch ${(ev as { selected_epoch?: number }).selected_epoch ?? "?"}).`] };
          case "done":
          case "failed":
          case "cancelled":
            return { ...next, done: true, status: ev.type === "done" ? (ev.status ?? "completed") : ev.type, artifactsReady: ev.artifacts_ready ?? null, message: ev.message ?? null };
          default:
            return next;
        }
      });
      if (ev.type === "done" || ev.type === "failed" || ev.type === "cancelled") {
        es.close();
        terminalRef.current?.();
      }
    };
    TYPES.forEach((t) => es.addEventListener(t, handle as EventListener));
    es.onerror = () => setState((s) => ({ ...s, connected: false }));
    return () => es.close();
  }, [expId, runId, enabled]);

  return state;
}
