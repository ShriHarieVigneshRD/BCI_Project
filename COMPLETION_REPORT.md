# Completion report — EEGNet Architecture Playground

Date: 31 October 2026 · Version 1.0.0 · updated 1 November 2026 (dataset import changes, section 6)

This report lists what is implemented and working, which tests were actually run (with their results), how the
application was verified in a browser, and what remains limited or out of scope.

## 1. Delivered

| Path | Contents |
|---|---|
| `backend/` | FastAPI + PyTorch + MNE application (`app/`), 86 pytest tests (`tests/`), `pyproject.toml` with `uv.lock` (Python 3.11; PyTorch as `cpu` / `cuda` extras) |
| `frontend/` | React 19 + TypeScript (strict) + Vite + Tailwind v4 app, Vitest tests, `package-lock.json` with exact versions |
| `scripts/` | `start.ps1` / `start.sh` (one-command start), `dev.ps1` (hot-reload development) |
| `README.md` | Installation, usage, data formats, protocols, exports, tests, troubleshooting, limitations |
| `.env.example` | Optional settings (data directory, host/port, upload and import limits, local import roots, CORS) |

Resolved versions: torch 2.14.1+cu130 (CUDA) / 2.15.0 (CPU), MNE 1.13.2, scikit-learn 1.9.1, FastAPI 0.162,
React 19.3, React Router 8.5, Vite 8.3, TypeScript 7.0, ECharts 6.1.

## 2. Features by delivery phase

**Phase 1 — model and tracing.** A configurable EEGNet: F1, D, F2, temporal kernel and stride, spatial kernel height,
pool sizes and strides, pool type, same/valid padding, dropout and its type, activation, BatchNorm and its
momentum/epsilon, and max-norm constraints. Presets EEGNet-8,2 / 4,2 / 16,2. Analytic shape propagation with
per-layer validity messages and fixes, cross-checked against a real PyTorch forward pass on every validation.
Parameters, MACs, activation and weight memory, receptive fields, coordinate-to-input mapping and flatten index
mapping. Forward-hook tracing.

**Phase 2 — data.** Loaders for EDF/EDF+/BDF, FIF (raw and epochs), EEGLAB (raw and epochs, companion `.fdt`), NPY,
NPZ and CSV/TSV (long-format epoched and continuous), with validation (NaN, units, channels, labels, events, class
counts) and per-file participant/session metadata. Separate train/test dataset compatibility checks. Split strategies:

- random trial-level hold-out with stratification and fallback;
- participant-level hold-out (manual or random allocation);
- K-fold: trial-wise, stratified trial-wise, group-wise and stratified group-wise.

Backend integrity checks cover zero group overlap, disjoint trials and folds, class coverage, and participant/session
overlap reports. Preprocessing covers channels and mapping, FIR/IIR filtering with phase options and notch, BAD
segments, amplitude and flat rejection, ICA (reference fit on training data with full diagnostics), polyphase
resampling, epoching from events or crop, baseline and normalisation. Learned stages are fitted per fold on training
data only. Deterministic outputs are cached by content fingerprint.

**Phase 3 — lifecycle and training.** The 16-state lifecycle with explicit transitions and rollbacks, plus an audit
log. SQLite registry. Background training worker with SSE progress and cancellation; early stopping and model
selection on validation data only; schedulers; class weighting. Each fold persists its checkpoint, learned transform,
partition and representative activations, with sha256-hashed artifact records. The training-completion gate (eight
checks, including a strict reload and a forward pass that reproduces the persisted activations) controls
`artifacts_ready` / `artifacts_incomplete`. Interrupted runs are recovered as failed on start-up.

**Phase 4 — shell.** Global header (Upload Dataset, Start New Experiment, experiment switcher with state, CUDA/CPU
indicator, System/Light/Dark theme persisted in localStorage), primary navigation with backend-derived step status,
the lifecycle banner, Dataset & Splits and the Preprocessing Studio (live preview, spectra, rejection list, ICA panel).

**Phase 5 — Architecture Explorer.** Configuration with a zoomable and pannable SVG diagram (node selection, shape
tracing), layer table, versions (locked when trained; edits create v2) and a configuration library. The EEG Signal
Journey works with the trained model or a labelled untrained initialisation. The Training Simulator streams live
charts and logs and shows the verification checklist.

**Phase 6 — post-training tools.** The Kernel & Convolution Playground: per-element arithmetic for all 15 layer types,
verified against the hooks, with trial comparison and the kernel gallery (including temporal frequency responses).
The Convolution Arena: depthwise group mapping, separable depthwise then pointwise, and a labelled stride/padding
demonstration. The Architecture Breaker: per-parameter classification into structural / computation / training-only
/ display, tensor compatibility, before/after totals, learned outputs only for valid checkpoints, and spawning a
variant experiment. Results & Predictions: an explicit Evaluate action, pooled and per-fold test metrics with
documented averaging and undefined-metric handling, a confusion matrix (counts or row-normalised), per-class table,
training/validation/test kept apart, prediction table linked to the Journey and the Playground, and metadata.

**Phase 7 — separate areas.** Battle Arena: matched variants, train/evaluate from the list, side-by-side table,
validation curves, architectures, confusion matrices, per-class F1, disagreement on identical trials, protocol
warnings. Tensor Detective: current / trained / custom configurations, step-by-step trace with formulas, coordinate
detective and flatten map. Experiments & Exports: search and filter, open, rename, duplicate, delete with
confirmation, a ZIP export with manifest and HTML report, checkpoint downloads, explicit dataset export, artifact
manifest and lifecycle log.

**Phase 8 — integration.** README, startup scripts, `.env.example`, pinned lockfiles, and the tests and browser
verification below.

## 3. Tests actually run (31 Oct 2026)

| Command | Result |
|---|---|
| `cd backend && uv run --extra cuda pytest` | **70 passed** (≈14 s) |
| `cd frontend && npm test` (Vitest + jsdom) | **9 passed** in 2 files |
| `cd frontend && npm run typecheck` (strict `tsc`) | **0 errors** |
| `cd frontend && npm run build` | **succeeds** (JS 1.27 MB / 402 KB gzip, ECharts included) |

Mapping to the 20 required test areas:

| # | Requirement | Test(s) |
|---|---|---|
| 1 | New experiment begins at Dataset & Splits | `test_lifecycle.py::test_new_experiment_begins_at_dataset_and_splits` |
| 2 | Preprocessing blocked for invalid dataset metadata | `test_preprocessing.py::test_preprocessing_blocked_for_dataset_without_events`, `test_invalid_preprocessing_config_rejected` |
| 3 | No training before valid data + preprocessing | `test_lifecycle.py::test_training_blocked_before_data_and_preprocessing` |
| 4 | Invalid architectures rejected | `test_model.py::test_invalid_architectures_rejected`, `test_field_constraints_rejected_by_schema`, `test_validate_endpoint_rejects_invalid`, `test_lifecycle.py::test_architecture_must_match_preprocessed_data` |
| 5 | Progress comes from real training | `test_training.py::test_progress_comes_from_real_training` (streamed values = persisted history; validation loss recomputed from the checkpoint matches) |
| 6 | Failed / cancelled runs do not unlock | `test_failed_training_does_not_unlock`, `test_cancelled_training_does_not_unlock` (real cancellation mid-run) |
| 7 | Verified success unlocks tools | `test_successful_training_unlocks_post_training_tools` |
| 8 | Missing / corrupt checkpoints block readiness | `test_corrupt_or_missing_checkpoint_blocks_readiness` (including recovery) |
| 9 | Activations use the correct checkpoint + input | `test_activation_extraction_uses_checkpoint_and_input` |
| 10 | Kernels are real weights | `test_kernel_visualisation_uses_real_weights` |
| 11 | Breaker leaves the original untouched | `test_breaker_variants_leave_original_untouched` |
| 12 | Structural changes invalidate weights | same test (structural vs computation vs training-only) |
| 13 | Results only after evaluation | `test_results_only_after_evaluation`, `test_export_needs_evaluation_for_metrics` |
| 14 | Overlapping groups rejected | `test_splits.py::test_overlapping_group_assignment_is_rejected`, `test_group_split_api_rejects_unknown_or_all_groups` |
| 15 | Training-only, fold-specific fitting | `test_learned_transforms_fitted_on_training_subset_per_fold`, `test_ica_fitted_on_training_partition_only` |
| 16 | New experiments preserve old records | `test_new_experiment_preserves_previous_records` |
| 17 | Data / preprocessing changes mark results stale | `test_changes_after_training_mark_results_stale`, `test_architecture_change_after_training_creates_new_version` |
| 18 | Export metadata matches the experiment | `test_export_metadata_matches_experiment` (sha256 of every file verified) |
| 19 | Theme modes and primary navigation | `frontend/src/test/theme.test.tsx` (4 tests), `navigation.test.tsx` (5 tests, including lock gating) |
| 20 | End-to-end on synthetic data | `test_e2e.py::test_complete_workflow` (continuous synthetic data → group split → filter/notch/resample/epoch/baseline/rejection → train with SSE → inspect → evaluate → matched variant → compare → export) |

Further scientific tests: receptive fields checked against input-gradient support, flatten index round-trip, max-norm,
every split mode, the validation subset never touching test data, Nyquist/phase validation, epoching and rejection
accounting, and readers for every format (EDF and EEGLAB files are generated with MNE's exporters).

## 4. Browser verification (performed, not part of the automated suites)

- The production build served by FastAPI was driven in Microsoft Edge (headless, through `playwright-core`, run
  ad hoc from a scratch folder outside the repository):
  - created an experiment from the modal and confirmed a participant-level split;
  - ran preprocessing and saved the architecture;
  - trained with live SSE updates and saw the verification checklist;
  - evaluated, then opened the Kernel Playground from a prediction row.

  The run completed with **no browser console errors**.
- Screenshots of all six areas and all six Architecture Explorer tabs were reviewed in light and dark mode. This
  review found and fixed: a diagram toolbar overlapping a node, unreadable confusion-matrix labels, unformatted axis
  labels, cramped table cells, and Tensor Detective choosing its source before the experiment loaded.
- CUDA training was verified on the local RTX 3050 Ti; the automated tests train on the CPU.
- Synthetic sanity check: EEGNet-8,2 reached about 94–98 % test accuracy on the synthetic data (trial-wise,
  participant-level and group-wise 3-fold). This only shows that the pipeline learns; the synthetic signal is
  deliberately easy.

## 5. Remaining limitations

- **Resuming training is not supported.** Runs interrupted by a server restart are marked failed; retry creates a new
  run. One run trains at a time (a FIFO queue).
- **Reports are HTML only** (no PDF).
- **Nested cross-validation is not implemented.** Selection uses one validation subset per fold.
- **ICA component removal across folds** depends on spatial-pattern matching (|r| ≥ 0.8). Unmatched components are
  kept and recorded. Spatial patterns are shown as bar charts (no scalp topographies, because electrode positions are
  not required).
- **Bad channels are excluded, not interpolated.** Rule-based rejection applies to every partition, as documented.
- **Stale runs are archived, not browsable.** Stale runs stay in the registry (run history, artifact manifest, exports), but the
  results pages always show the current configuration only.
- **Accessibility:** semantic roles, labels, keyboard operation, visible focus and reduced motion are implemented, but
  no automated accessibility audit (e.g. axe) was run.
- **Bundle size:** the frontend bundle is about 1.3 MB (dominated by ECharts). This is acceptable for local use and was
  not code-split.
- **Browser e2e:** the Playwright workflow script is not part of the repository's automated tests. Frontend automated
  tests run in jsdom.
- The EEGLAB "missing companion `.fdt`" error path is implemented but not covered by a test, because no fixture with
  external data was available.

## 6. Dataset import changes (`prompts/changes1.md`, 1 Nov 2026)

### Delivered

- **Import dialog with three methods:** *Select dataset folder* (browser folder picker; uploads only files that can
  contribute), *Select multiple files*, and *Use local folder path* (read in place). The local-path method only accepts
  requests from this machine. It rejects relative, non-existent, network (UNC), drive-root and data-store paths, and
  can be limited with `EEGNET_IMPORT_ROOTS`. Upload paths are sanitised and the server never returns absolute paths.
- **Scanner** (`backend/app/data/scanner.py`, `entities.py`): recursive scan; header-only reads; recognises
  participant / session / run / task from BIDS and common name patterns or folder depth. It associates event files
  (`_events.tsv/.csv`, `-eve.fif`, `.eve`) with their recordings and never counts them as recordings. It reads
  `participants.tsv`, `_channels.tsv` (bad channels, types), `_eeg.json` and `dataset_description.json`, and reports
  orphan event files, unreadable files and ignored files. EDF header identity fields are never read.
- **Dataset manifest** (`manifest.py`): the single, reviewable description of a dataset. It is stored with every
  imported dataset and used by every screen and export.
- **Dataset Review**: participants, recordings, rates, channels, event status and warnings; editable participant /
  session / run / task; include and exclude recordings; event mapping (class, segment start / end, ignore, wildcard
  rules, suggested defaults); channel and rate policies. All counts are recomputed by the backend after every edit,
  and blocking issues disable the import.
- **Canonical format v2:** per-recording memory-mapped arrays; events with durations; recording IDs
  (`sub-01_ses-01_task-mi_run-2`) carried into trial IDs. Legacy v1 datasets still load unchanged.
- **Block designs:** start/end markers are paired into segments (trials with durations). The new **sliding-window**
  epoching mode cuts segments into windows. Splits assign whole segments, so windows of one segment never cross
  partitions, and the held-out participant guard is applied at segment level.
- **Downstream screens:**
  - **Header:** opens the import dialog.
  - **Dataset & Splits:** participants table, recordings, *Revise import*, automatic integrity preview.
  - **Preprocessing Studio:** suggestions from the manifest, windows controls, per-participant and per-recording
    counts.
  - **Architecture:** dimensions from the dataset.
  - **Training:** pre-training checks (datasets, recording compatibility, events, classes, split integrity, freshness,
    model input).
  - **Journey and Kernel Playground:** participant / partition trial pickers with run provenance.
  - **Results:** held-out results by participant and per-segment aggregation.
  - **Battle Arena:** dataset / trial-unit warnings.
  - **Tensor Detective:** dataset dimensions.
  - **Exports:** dataset manifest and trial provenance CSV.
- **Compatibility rules:**
  - Revising an import creates a **new** dataset; the original and its experiments are untouched until the new dataset
    is attached, which marks the earlier run stale.
  - Learned preprocessing stays fitted on training data only.
  - Data stay local.

### Tests actually run (1 Nov 2026)

| Command | Result |
|---|---|
| `cd backend && uv run --extra cuda pytest` | **86 passed** (≈43 s; 70 earlier + 16 new in `tests/test_import.py`) |
| `cd frontend && npm test` | **11 passed** in 3 files (adds `import.test.tsx`: Dataset Review edits/counts/exclusion reasons, three import methods + local-path scan → review) |
| `cd frontend && npx tsc -b --noEmit` | **0 errors** |
| `cd frontend && npm run build` | **succeeds** (JS 1.31 MB / 412 KB gzip) |

New backend tests (`tests/test_import.py`, fixtures in `tests/nested_fixture.py`):

- `test_file_classification_never_counts_event_files_as_recordings`
- `test_entity_inference` (parametrised name patterns)
- `test_event_table_parsing`
- `test_mapping_suggestions_pair_start_end_blocks`
- `test_local_path_validation` (relative, missing, file instead of folder, UNC, drive root, data store, outside `EEGNET_IMPORT_ROOTS`, request from another machine → 403)
- `test_upload_paths_are_sanitised`
- `test_review_counts_are_dynamic_and_edits_apply`
- `test_folder_upload_matches_local_import`
- `test_acceptance_nested_dataset_all_split_methods`: a nested BIDS-like study with 4 participants × 2 runs, separate
  event files, a rest recording without events, an orphan event file and non-EEG files. Under random, participant-level
  and stratified group K-fold splits, it runs suggestions → preprocessing → pre-training checks → training → journey →
  convolution → evaluation → breaker → export. It checks trial counts against the manifest, that no held-out
  participant appears in training, prediction provenance, and that earlier experiments are left untouched.
- `test_block_design_windows_keep_blocks_together`
- `test_revised_dataset_marks_dependent_results_stale`
- `test_mixed_rates_and_channels_are_harmonised` (128/256 Hz, different channel sets and orders)

### Browser verification

The production build was driven in headless Microsoft Edge (`playwright-core`, scratch folder outside the repo):

- **Folder upload** of the nested BIDS-like study: plan (9 recordings, 9 event files, metadata, ignored files), upload,
  review, import, attach.
- **Local-path import** of a block-design study: segment mapping with ignored timestamps, ECG/EMG channels detected.
- **Full workflow on the block-design study:** participant-level split, preprocessing with window suggestions,
  architecture, training (CUDA), evaluation with results by participant, Signal Journey.

All steps completed with **no browser console errors**. The screenshot review found that excluded recordings showed a
green "OK" mark; they now show their exclusion reason.

### Additional limitations

- Event mapping is per task, not per recording.
- Sliding windows may overlap; per-window metrics are not independent, so a per-segment summary is reported alongside.
- *Revise import* needs the original scan and source files (kept staged copies, or the original local folder).
- Local-path import is available only when the browser and backend run on the same machine.
- Scanning reads headers only, but importing converts all included recordings to float32 in the app's data store,
  which can need about twice the size of 16-bit source files (EDF/BDF/GDF) in free disk space.
