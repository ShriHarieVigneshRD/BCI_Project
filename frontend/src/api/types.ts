// Types mirroring the FastAPI / Pydantic response schemas.

export type ExperimentState =
  | "created"
  | "dataset_required"
  | "dataset_ready"
  | "split_configured"
  | "preprocessing_configured"
  | "preprocessing_completed"
  | "architecture_configured"
  | "training_queued"
  | "training"
  | "training_failed"
  | "training_cancelled"
  | "training_completed"
  | "evaluation_running"
  | "evaluation_completed"
  | "artifacts_incomplete"
  | "artifacts_ready";

export type Level = "error" | "warning" | "info";
export type CheckStatus = "pass" | "warn" | "fail";

export interface Issue {
  level: Level;
  code: string;
  message: string;
  layer?: string | null;
  fix?: string | null;
}

export interface Health {
  status: string;
  app_version: string;
  software: Record<string, string>;
  device: {
    cuda_available: boolean;
    default_device: string;
    cuda_device: string | null;
    cuda_version: string | null;
    cuda_memory_gb?: number;
    torch_threads: number;
  };
  training_worker: { running_run_id: string | null };
  privacy: string;
}

export interface FormatInfo {
  format: string;
  extensions: string[];
  structure: string;
  requirements: string;
  metadata: string;
}

export interface DatasetSummary {
  structure: "epoched" | "continuous";
  n_channels: number;
  ch_names: string[];
  ch_types: string[];
  sfreq: number;
  unit: string;
  n_trials: number;
  n_classes: number;
  classes: string[];
  class_distribution: Record<string, number>;
  n_participants: number;
  participants: string[];
  n_sessions: number;
  sessions: string[];
  missing_metadata: string[];
  non_eeg_channels?: string[];
  n_segments?: number;
  segment_seconds?: Record<string, number> | null;
  n_runs?: number;
  participants_table?: { participant: string | null; n_trials: number; n_recordings: number; sessions: string[]; runs: string[]; class_counts: Record<string, number> }[];
  n_recordings: number | null;
  n_epochs: number | null;
  n_samples: number;
  epoch_samples: number | null;
  epoch_duration_s: number | null;
  tmin: number | null;
  recordings: {
    name: string;
    recording_id?: string | null;
    duration_s: number | null;
    n_events: number;
    n_trials?: number;
    group: string | null;
    session: string | null;
    run?: string | null;
    task?: string | null;
    source_file?: string | null;
    event_source?: string | null;
    event_types: string[];
    bad_segments: number;
  }[];
  processing_hint: string;
  highpass: number | null;
  lowpass: number | null;
}

export interface DatasetRecord {
  id: string;
  name: string;
  created_at: string;
  format: string;
  structure: string;
  fingerprint: string;
  synthetic: boolean;
  files: string[];
  summary: DatasetSummary;
  issues: { level: Level; code: string; message: string }[];
  trainable: boolean;
  notes: string[];
  manifest?: DatasetManifest | null;
  source?: Record<string, unknown>;
}

export interface Stage {
  id: string;
  label: string;
  status: "complete" | "available" | "locked";
}

export interface Capabilities {
  state: ExperimentState;
  busy: boolean;
  can_attach_dataset: boolean;
  can_configure_split: boolean;
  can_configure_preprocessing: boolean;
  can_run_preprocessing: boolean;
  can_configure_architecture: boolean;
  can_train: boolean;
  can_cancel: boolean;
  can_inspect_untrained: boolean;
  can_inspect_learned: boolean;
  can_use_breaker: boolean;
  can_evaluate: boolean;
  has_results: boolean;
  can_reverify: boolean;
  run_stale: boolean;
  next_page: string;
  next_step: string;
  stages: Stage[];
}

export interface SplitConfig {
  workflow: "single" | "separate";
  strategy: "random" | "group" | "kfold" | "separate";
  test_ratio: number;
  stratify: boolean;
  seed: number;
  test_groups: string[] | null;
  n_test_groups: number | null;
  kfold_mode: "trial" | "stratified_trial" | "group" | "stratified_group";
  n_folds: number;
}

export interface IntegrityCheck {
  id: string;
  label: string;
  status: CheckStatus;
  detail: string;
}

export interface IntegrityFold {
  index: number;
  n_train: number;
  n_test: number;
  train_groups: string[];
  test_groups: string[];
  group_overlap: string[];
  session_overlap: string[];
  class_counts: { train: Record<string, number>; test: Record<string, number> };
}

export interface IntegrityReport {
  ok: boolean;
  checks: IntegrityCheck[];
  folds: IntegrityFold[];
  classes: string[];
  group_level: boolean;
  protocol: string;
  question: string;
  seed: number;
  fingerprint: string;
}

export interface AssignmentBrief {
  strategy: string;
  protocol: string;
  question: string;
  seed: number;
  fingerprint: string;
  n_candidates: number;
  n_folds: number;
  warnings: string[];
  config: SplitConfig;
  folds: { index: number; n_train: number; n_test: number }[];
}

export interface SplitResponse {
  assignment: AssignmentBrief;
  integrity: IntegrityReport;
  changed?: boolean;
  membership?: { index: number; train: string[]; test: string[] }[];
}

export interface Candidates {
  n: number;
  groups: string[];
  group_counts: Record<string, number>;
  participants: { participant: string; n_trials: number; n_recordings: number; sessions: string[]; runs: string[]; class_counts: Record<string, number> }[];
  classes: string[];
  class_counts: Record<string, number>;
  n_recordings: number;
  n_segments: number;
  trial_unit: string;
  has_groups: boolean;
}

export interface PreprocessingConfig {
  channels: { selected: string[] | null; bad_channels: string[]; rename: Record<string, string> };
  filter: {
    enabled: boolean;
    kind: "bandpass" | "highpass" | "lowpass";
    method: "fir" | "iir";
    l_freq: number | null;
    h_freq: number | null;
    iir_order: number;
    phase: "zero" | "minimum" | "causal";
    notch_enabled: boolean;
    notch_freqs: number[];
  };
  artifacts: {
    reject_bad_segments: boolean;
    amplitude_enabled: boolean;
    amplitude_threshold_uv: number;
    flat_enabled: boolean;
    flat_threshold_uv: number;
  };
  ica: {
    enabled: boolean;
    n_components: number | null;
    method: "fastica" | "infomax";
    random_state: number;
    exclude: number[];
    match_threshold: number;
  };
  resample: { enabled: boolean; target_sfreq: number };
  epoching: { mode: "keep" | "crop" | "events" | "windows"; classes: string[] | null; tmin: number; tmax: number; window_s: number; step_s: number; segment_offset_s: number; max_windows: number | null };
  baseline: { enabled: boolean; tmin: number; tmax: number };
  normalization: { method: "none" | "channel_zscore" | "global_zscore" | "robust_channel" | "trial_zscore" };
}

export interface ValidationResult {
  errors: string[];
  warnings: string[];
  info: string[];
}

export interface PreprocessingReport {
  pipeline_order: string[];
  input: { structure: string; n_channels: number; sfreq: number; n_candidates: number; class_distribution: Record<string, number> };
  channels: { selected: string[]; output_names: string[]; excluded_bad: string[]; n: number };
  filter: { description: string; messages: string[] };
  resample: { applied: boolean; from?: number; to?: number; up?: number; down?: number; method?: string };
  epoching: { mode: string; tmin: number; tmax: number; n_times: number; excluded_class: number; out_of_bounds: number; windows: { window_s: number; step_s: number; segment_offset_s: number; max_windows: number | null; n_segments: number; n_parent_trials: number; n_windows: number } | null };
  by_participant?: { participant: string | null; candidates: number; kept_trials: number; kept_rows: number; class_rows: Record<string, number> }[];
  by_recording?: { recording_id: string; participant: string | null; run: string | null; rows: number; trials: number }[];
  baseline: { enabled: boolean; tmin: number; tmax: number };
  /** subset = the live preview processed only some recordings of a large dataset */
  scope?: { subset: boolean; recordings: number; of: number; participants: number };
  rejection: { bad_segment: number; amplitude: number; flat: number; total: number; rejected: { trial_id: string; reason: string }[] };
  output: { n_epochs: number; class_distribution: Record<string, number>; shape: number[]; sfreq: number; tmin: number; duration_s: number; classes: string[] };
  normalization: { method: string; doc: string };
  validation: ValidationResult;
}

export interface SignalBlock {
  times: number[];
  data: number[][];
  sfreq: number;
  unit?: string;
  label?: string;
}

export interface PreviewResponse {
  validation: ValidationResult;
  ok: boolean;
  det_fingerprint?: string;
  trial_id?: string;
  report?: PreprocessingReport;
  channels?: string[];
  all_channels?: string[];
  trials?: { trial_id: string; label: string; group: string | null; run: string | null; recording_id: string | null }[];
  raw?: SignalBlock;
  processed?: SignalBlock | null;
  rejected_reason?: string | null;
  psd_raw?: { freqs: number[]; power_db: number[] };
  psd_processed?: { freqs: number[]; power_db: number[] };
  note?: string;
}

export interface TensorCheck {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
}

export interface PreprocessingResult {
  report: PreprocessingReport;
  integrity: IntegrityReport;
  tensor_checks: TensorCheck[];
  valid: boolean;
  learned_preview: {
    n_fit_trials: number;
    fit_scope: string;
    ica?: { n_components?: number; excluded?: number[]; matches?: unknown[]; note?: string };
    normalization: { method: string; doc: string; stats: Record<string, number[]> };
  } | null;
  det_fingerprint: string;
  fingerprint: string;
  computed_at: string;
  learned_note: string;
}

export interface IcaComponent {
  index: number;
  explained_variance: number | null;
  kurtosis: number;
  high_freq_ratio: number;
  low_freq_ratio: number;
  frontal_corr: number | null;
  pattern: number[];
  psd_db: number[];
}

export interface IcaView {
  reference: { fingerprint: string; n_components: number; n_fit_trials: number; n_training_trials?: number; fit_scope: string; created_at: string; method: string };
  diagnostics: { n_components: number; ch_names: string[]; freqs: number[]; components: IcaComponent[]; method: string; notes: string[] };
  sources?: { trial_id: string; times: number[]; data: number[][] };
}

export interface ArchitectureConfig {
  preset: string;
  n_channels: number;
  n_times: number;
  n_classes: number;
  sfreq: number | null;
  F1: number;
  D: number;
  F2: number;
  temporal_kernel: number;
  temporal_stride: number;
  spatial_kernel: number | null;
  pool1_kernel: number;
  pool1_stride: number | null;
  separable_kernel: number;
  pool2_kernel: number;
  pool2_stride: number | null;
  pool_type: "avg" | "max";
  padding: "same" | "valid";
  dropout: number;
  dropout_type: "dropout" | "spatial";
  activation: "elu" | "relu" | "leaky_relu" | "gelu" | "silu";
  batch_norm: boolean;
  bn_momentum: number;
  bn_eps: number;
  depthwise_max_norm: number | null;
  classifier_max_norm: number | null;
}

export interface LayerSpec {
  name: string;
  type: string;
  label: string;
  block: string;
  purpose: string;
  in_shape: number[];
  out_shape: number[];
  kernel: number[] | null;
  stride: number[] | null;
  padding: number[] | null;
  groups: number | null;
  dilation: number[] | null;
  params: number;
  trainable: number;
  buffers: number;
  param_shapes: Record<string, number[]>;
  macs: number;
  rf_time: number;
  rf_channels: number;
  jump_time: number;
  start_time: number;
  valid: boolean;
  message: string | null;
  learnable: boolean;
}

export interface ArchitectureReport {
  config: ArchitectureConfig;
  fingerprint: string;
  input_shape: number[];
  input_convention: string;
  layers: LayerSpec[];
  issues: Issue[];
  valid: boolean;
  total_params: number;
  trainable_params: number;
  buffers: number;
  total_macs: number;
  activation_bytes_per_sample: number;
  param_bytes: number;
  receptive_field_samples: number | null;
  receptive_field_seconds: number | null;
  classifier_in_features: number | null;
  torch_verified: boolean;
  torch_message: string | null;
}

export interface ArchitectureVersion {
  id: string;
  name: string;
  created_at: string;
  experiment_id: string | null;
  version: number;
  parent_id: string | null;
  locked: boolean;
  config: ArchitectureConfig;
  fingerprint: string;
}

export interface ArchitectureState {
  config: ArchitectureConfig | null;
  default: ArchitectureConfig | null;
  data_dims: { n_channels: number; n_times: number; n_classes: number; sfreq: number; classes: string[]; ch_names?: string[]; epoch_s?: number; n_epochs?: number; epoching_mode?: string; datasets?: string[] } | null;
  report: ArchitectureReport | null;
  confirmed: boolean;
  versions: ArchitectureVersion[];
  current_version_id: string | null;
}

export interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: "adam" | "adamw" | "sgd";
  momentum: number;
  weight_decay: number;
  loss: "cross_entropy";
  label_smoothing: number;
  seed: number;
  device: "auto" | "cpu" | "cuda";
  validation_mode: "auto" | "stratified" | "random" | "group" | "none";
  validation_fraction: number;
  early_stopping: boolean;
  patience: number;
  monitor: "val_loss" | "val_balanced_accuracy";
  restore_best: boolean;
  scheduler: "none" | "step" | "cosine" | "plateau";
  step_size: number;
  gamma: number;
  class_weighting: "none" | "balanced";
  representative_trials: number;
}

export interface HistoryRow {
  epoch: number;
  train_loss: number;
  train_acc: number;
  val_loss?: number;
  val_acc?: number;
  val_balanced_accuracy?: number;
  lr: number;
  elapsed_s: number;
}

export interface FoldRun {
  fold: number;
  status: "pending" | "training" | "completed" | "failed" | "cancelled";
  checkpoint_id: string | null;
  fit_ids_count: number;
  val_ids_count: number;
  test_ids_count: number;
  val_description: string | null;
  history: HistoryRow[];
  info: { selected_epoch?: number; stopped_early?: boolean; epochs_run?: number; train_time_s?: number; monitor?: string | null; selection_basis?: string };
  transform_report: Record<string, unknown>;
  activations: { trial_ids: string[]; layers: string[]; checkpoint_id: string } | null;
}

export interface ArtifactCheck {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
  recovery: string | null;
}

export interface ArtifactStatus {
  ready: boolean;
  checks: ArtifactCheck[];
  verified_at: string;
  run_id: string;
}

export interface RunRecord {
  id: string;
  experiment_id: string;
  created_at: string;
  status: "queued" | "training" | "completed" | "failed" | "cancelled";
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  device: string | null;
  fingerprint: string;
  snapshot: {
    architecture_config: ArchitectureConfig;
    training_config: TrainingConfig;
    preprocessing_config: PreprocessingConfig;
    split_fingerprint: string;
    preprocessing_fingerprint: string;
    architecture_fingerprint: string;
    architecture_version: number;
    class_names: string[];
    seed: number;
    sfreq: number;
    dataset_fingerprints: string[];
    [k: string]: unknown;
  };
  folds: FoldRun[];
  artifact_status: ArtifactStatus | null;
  evaluation: EvaluationResult | null;
  stale: boolean;
  stale_reason: string | null;
  software: Record<string, string>;
  log: string[];
}

export interface Experiment {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  state: ExperimentState;
  parent_id: string | null;
  inherited: string[];
  workflow: "single" | "separate";
  dataset_id: string | null;
  test_dataset_id: string | null;
  split_config: SplitConfig | null;
  split_fingerprint: string | null;
  preprocessing_config: PreprocessingConfig | null;
  preprocessing_fingerprint: string | null;
  preprocessing_summary: PreprocessingResult | null;
  ica_reference: IcaView["reference"] | null;
  architecture_id: string | null;
  architecture: ArchitectureConfig | null;
  training_config: TrainingConfig;
  active_run_id: string | null;
  warnings: string[];
  stale_notes: string[];
  capabilities: Capabilities;
  active_run: RunRecord | null;
  runs: { id: string; status: string; created_at: string; stale: boolean; stale_reason: string | null; fingerprint: string }[];
  datasets: { id: string; name: string; fingerprint: string; synthetic: boolean; structure: string }[];
  synthetic: boolean;
}

export interface ExperimentSummary {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  state: ExperimentState;
  parent_id: string | null;
  dataset_name: string | null;
  strategy: string | null;
  architecture_preset: string | null;
  run_status: string | null;
  stale: boolean;
  test_accuracy: number | null;
  test_balanced_accuracy: number | null;
  best_val_balanced_accuracy: number | null;
  synthetic: boolean;
}

export interface TrialRow {
  trial_id: string;
  parent?: string;
  label: string;
  group: string | null;
  session: string | null;
  run?: string | null;
  recording_id?: string | null;
  offset_s?: number;
  partition: "train" | "validation" | "test" | "unused" | "unknown";
}

export interface JourneyStage {
  id: string;
  label: string;
  kind: "signal" | "activation" | "vector";
  unit: string;
  shape: number[];
  data: number[][] | number[][][] | number[];
  times?: number[];
  ch_names?: string[];
  sfreq?: number;
  description?: string;
  type?: string;
  block?: string;
  purpose?: string;
  in_shape?: number[];
  decimation?: number;
  stats?: { min: number; max: number; mean: number; std: number };
  time_jump?: number;
  receptive_field?: number;
}

export interface Journey {
  mode: "trained" | "untrained";
  banner: string;
  provenance: { checkpoint_id: string | null; sha256?: string; run_id?: string; fold?: number; partition: string };
  trial: { trial_id: string; label: string; group: string | null; session: string | null; run?: string | null; recording_id?: string | null; parent?: string; offset_s?: number };
  class_names: string[];
  stages: JourneyStage[];
  logits: number[];
  probabilities: number[];
  predicted: string;
  note: string;
}

export interface Kernels {
  checkpoint_id: string;
  sha256: string;
  fold: number;
  run_id: string;
  source: string;
  temporal: { weights: number[][]; shape: number[]; freqs: number[]; response_db: number[][]; peak_hz: number[]; sfreq: number; kernel_ms: number };
  depthwise: { weights: number[][]; shape: number[]; groups: number; D: number; rows: string[]; group_of_output: number[]; max_norm: number | null; norms: number[] };
  separable_depthwise: { weights: number[][]; shape: number[]; groups: number };
  pointwise: { weights: number[][]; shape: number[] };
  classifier: { weights: number[][]; bias: number[]; shape: number[]; feature_shape: number[]; class_names: string[] };
  batchnorm: Record<string, { weight: number[]; bias: number[]; running_mean: number[]; running_var: number[]; eps: number }>;
}

export interface ConvolutionResult {
  layer: string;
  type: string;
  label: string;
  purpose: string;
  operation: string;
  in_shape: number[];
  out_shape: number[];
  checkpoint_id: string;
  trial_id: string;
  partition: string;
  kernel: number[] | null;
  stride: number[] | null;
  padding: number[] | null;
  groups: number | null;
  selected: Record<string, number | string | null>;
  computed: number;
  actual: number;
  abs_diff: number;
  verified: boolean;
  input_row?: number[];
  input_map?: number[][];
  input_column?: number[];
  window?: { indices?: number[]; values: number[]; padded?: boolean[]; channels?: string[] };
  weights?: number[];
  products?: number[];
  output_row?: number[];
  output_map?: number[][] | null;
  params?: Record<string, number>;
  input_value?: number;
  curve?: { x: number[]; y: number[] };
  features?: number[];
  bias?: number;
  products_map?: number[][][];
  feature_shape?: number[];
  top_contributions?: { index: number; fhw: number[]; feature: number; weight: number; product: number }[];
  prediction: { probabilities: number[]; predicted: string; true: string; class_names: string[] };
  comparison?: {
    trial_a: string;
    trial_b: string;
    label_b: string;
    output_a: number[] | number[][];
    output_b: number[] | number[][];
    difference: number[] | number[][];
    input_difference_rms: number;
    activation_difference_rms: number;
    probabilities_a: number[];
    probabilities_b: number[];
    predicted_b: string;
  };
}

export interface ArenaResult {
  checkpoint_id: string;
  trial_id: string;
  map_index: number;
  depthwise: {
    layer: string;
    groups: number;
    D: number;
    kernel: number[];
    in_shape: number[];
    out_shape: number[];
    params: number;
    weights: number[][];
    ch_names: string[];
    mapping: { out: number; in: number }[];
    selected_input_map: number[][];
    selected_output: number[][];
    explanation: string;
  };
  separable: {
    depthwise: { layer: string; groups: number; kernel: number[]; padding: number[]; params: number; in_shape: number[]; out_shape: number[]; weights: number[][]; selected_input: number[]; selected_output: number[] };
    pointwise: { layer: string; params: number; in_shape: number[]; out_shape: number[]; weights: number[][]; outputs: number[][] };
    total_params: number;
    explanation: string;
  };
  demo: { stride: number; padding: string; pad: number[]; input_length: number; output_length: number; output: number[]; formula: string; label: string } | null;
}

export interface BreakerLayer {
  type: string;
  label: string;
  in_shape: number[];
  out_shape: number[];
  params: number;
  macs: number;
  rf_time: number;
  valid: boolean;
  message: string | null;
}

export interface BreakerTotals {
  params: number;
  trainable: number;
  macs: number;
  activation_kb_per_sample: number;
  param_kb: number;
  receptive_field_samples: number | null;
  receptive_field_s: number | null;
  classifier_in: number | null;
  valid: boolean;
}

export interface BreakerResult {
  valid: boolean;
  field_errors?: { field: string; message: string }[];
  base_checkpoint_id?: string;
  changed?: Record<string, { before: unknown; after: unknown; kind: string; note: string }>;
  base: { config: ArchitectureConfig; totals: BreakerTotals; issues: Issue[] } | null;
  variant: { config: ArchitectureConfig; totals: BreakerTotals; issues: Issue[]; torch_verified: boolean } | null;
  layers?: { layer: string; base: BreakerLayer | null; variant: BreakerLayer | null; changed: boolean }[];
  tensors?: { tensor: string; base: number[] | null; variant: number[] | null; compatible: boolean }[];
  checkpoint_valid?: boolean;
  verdict?: string;
  learned_outputs?: { trial_id: string; probabilities_base: number[]; probabilities_variant: number[]; max_abs_difference: number; class_names: string[]; note: string } | null;
  immutability?: string;
}

export interface PerClass {
  class: string;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  support: number;
  notes: string[];
}

export interface Metrics {
  n: number;
  accuracy: number | null;
  balanced_accuracy: number | null;
  cohen_kappa: number | null;
  chance_level: number;
  averages: Record<"macro" | "weighted" | "micro", { precision: number | null; recall: number | null; f1: number | null }>;
  per_class: PerClass[];
  roc_auc_ovr_macro: number | null;
  average_precision_macro: number | null;
  confusion_matrix: number[][];
  class_names: string[];
  undefined: string[];
  averaging_doc: Record<string, string>;
}

export interface FoldSummaryStat {
  mean: number | null;
  std: number;
  values: number[];
}

export interface EvaluationResult {
  created_at: string;
  run_id: string;
  n_test: number;
  class_names: string[];
  protocol: string | null;
  test: {
    pooled: Metrics;
    per_fold: { fold: number; metrics: Metrics | null }[];
    fold_summary: Record<string, FoldSummaryStat> | null;
    note: string;
    by_participant?: { group: string | null; n: number; accuracy: number | null; balanced_accuracy: number | null; class_counts: Record<string, number> }[];
    by_trial?: { metrics: Metrics; n_trials: number; n_windows: number; note: string } | null;
    participants?: string[];
  };
  validation: { fold: number; metrics: Metrics | null }[];
  training: { fold: number; metrics: Metrics | null }[];
  checkpoints: { fold: number; checkpoint_id: string; selected_epoch: number }[];
  labels: Record<string, string>;
  artifact_ids?: string[];
}

export interface ResultsResponse {
  evaluation: EvaluationResult;
  history: { fold: number; history: HistoryRow[]; info: FoldRun["info"]; val_description: string | null }[];
  metadata: {
    run_id: string;
    device: string | null;
    started_at: string | null;
    finished_at: string | null;
    seed: number;
    split_fingerprint: string;
    preprocessing_fingerprint: string;
    architecture_fingerprint: string;
    software: Record<string, string>;
    training_config: TrainingConfig;
  };
}

export interface Prediction {
  trial_id: string;
  parent_trial?: string;
  fold: number;
  group: string | null;
  session: string | null;
  run?: string | null;
  recording_id?: string | null;
  true: string;
  predicted: string;
  correct: boolean;
  probabilities: number[];
  checkpoint_id: string;
}

export interface CompareEntry {
  id: string;
  name: string;
  state: ExperimentState;
  trained: boolean;
  evaluated: boolean;
  stale: boolean;
  architecture: {
    config: ArchitectureConfig;
    valid: boolean;
    params: number;
    macs: number;
    receptive_field_samples: number | null;
    layers: { name: string; label: string; type: string; out_shape: number[]; params: number }[];
    source: string;
  } | null;
  training_config: TrainingConfig;
  preprocessing_config: PreprocessingConfig | null;
  preprocessing_fingerprint: string | null;
  split: { strategy: string | null; config: SplitConfig | null; fingerprint: string | null; seed: number | null };
  dataset_fingerprints: string[] | null;
  class_names: string[] | null;
  datasets?: { id: string; name: string }[];
  trial_unit?: string | null;
  histories: { fold: number; history: HistoryRow[] }[];
  train_time_s: number | null;
  device: string | null;
  evaluation: { pooled: Metrics; fold_summary: Record<string, FoldSummaryStat> | null; validation: { fold: number; metrics: Metrics | null }[] } | null;
  best_validation: number | null;
  artifact_status: ArtifactStatus | null;
}

export interface CompareResponse {
  experiments: CompareEntry[];
  warnings: string[];
  matched_protocol: boolean;
  disagreement: { pairs: { a: string; b: string; n: number; disagreements: number; rate: number; trials: string[] }[]; note: string } | null;
  selection_rule: string;
}

export interface ExportRecord {
  id: string;
  experiment_id: string;
  items: string[];
  size_bytes: number;
  filename: string;
  manifest: {
    export_id: string;
    created_at: string;
    experiment_id: string;
    run_id: string | null;
    run_fingerprint: string | null;
    synthetic_data: boolean;
    dataset_included: boolean;
    note: string;
    files: { path: string; sha256: string; bytes: number; provenance: string }[];
  };
}

export interface CoordinateMapping {
  layer: string;
  out_shape: number[];
  coordinate?: number[];
  flat_index?: number | null;
  unflattened?: number[];
  time_range: number[];
  time_range_unclipped?: number[];
  padded_samples?: number;
  channel_range: number[];
  receptive_field_samples?: number;
  jump?: number;
  depends_on?: string;
}

export interface ArtifactsResponse {
  state: ExperimentState;
  run_id: string | null;
  status: ArtifactStatus | null;
  stale: boolean;
  stale_reason: string | null;
  manifest: {
    id: string;
    type: string;
    run_id: string | null;
    checkpoint_id: string | null;
    created_at: string;
    source_ref: string | null;
    config_fingerprint: string | null;
    storage_ref: string;
    sha256: string | null;
    size_bytes: number | null;
  }[];
}

export interface LifecycleEvent {
  ts: string;
  from_state: string | null;
  to_state: string;
  kind: string;
  reason: string | null;
}

export interface ProgressEvent {
  type: "snapshot" | "status" | "log" | "epoch" | "fold_done" | "done" | "failed" | "cancelled";
  seq?: number;
  run_id?: string;
  status?: string;
  message?: string;
  device?: string | null;
  fold?: number;
  n_folds?: number;
  epochs?: number;
  epoch?: number;
  train_loss?: number;
  train_acc?: number;
  val_loss?: number;
  val_acc?: number;
  val_balanced_accuracy?: number;
  lr?: number;
  elapsed_s?: number;
  artifacts_ready?: boolean;
  error?: string | null;
  log?: string[];
  folds?: { fold: number; status: string; history: HistoryRow[] }[];
}

// ---------------------------------------------------------------------------
// Dataset import (folder / files / local path) and manifest
// ---------------------------------------------------------------------------

export type FileRole = "eeg" | "events" | "companion" | "metadata" | "ignored";
export type MappingAction = "class" | "segment_start" | "segment_end" | "ignore";

export interface ManifestIssue {
  level: Level;
  code: string;
  message: string;
  recording?: number | null;
}

export interface ManifestFile {
  path: string;
  size: number;
  role: FileRole;
  recording: number | null;
  note: string | null;
}

export interface ManifestRecording {
  index: number;
  file: string;
  format: string;
  companions: string[];
  structure: "continuous" | "epoched";
  participant: string | null;
  session: string | null;
  run: string | null;
  task: string | null;
  id_source: string | null;
  sfreq: number | null;
  n_channels: number;
  ch_names: string[];
  ch_types: string[];
  n_samples: number | null;
  duration_s: number | null;
  n_epochs: number | null;
  inner_groups: Record<string, number> | null;
  unit: string;
  events: { kind: string; file: string | null; n_events: number; note: string | null };
  event_counts: Record<string, number>;
  included: boolean;
  exclude_reason: string | null;
  readable: boolean;
  issues: ManifestIssue[];
  recording_id: string | null;
  n_trials: number;
  class_counts: Record<string, number>;
}

export interface MappingRow {
  task: string | null;
  event: string;
  action: MappingAction;
  label: string | null;
  count: number;
}

export interface ManifestParticipant {
  participant: string | null;
  n_recordings: number;
  sessions: string[];
  runs: string[];
  n_trials: number;
  class_counts: Record<string, number>;
  attributes: Record<string, unknown>;
}

export interface DatasetManifest {
  name: string;
  source: Record<string, unknown>;
  recordings: ManifestRecording[];
  mapping: MappingRow[];
  channel_policy: "intersection" | "strict";
  sfreq_policy: "resample" | "strict";
  target_sfreq: number | null;
  array_options: { sfreq: number | null; unit: "uV" | "mV" | "V" | "nV" | "unknown"; ch_names: string[] | null; tmin: number };
  files: ManifestFile[];
  metadata: {
    participants?: Record<string, Record<string, unknown>>;
    bad_channels?: Record<string, string[]>;
    power_line_hz?: number | null;
    dataset_description?: Record<string, unknown>;
  };
  issues: ManifestIssue[];
  importable: boolean;
  summary: {
    n_files: number;
    n_recordings: number;
    n_included: number;
    n_unreadable: number;
    structure: "continuous" | "epoched" | null;
    n_participants: number;
    participants: ManifestParticipant[];
    n_trials: number;
    classes: string[];
    class_counts: Record<string, number>;
    class_seconds: Record<string, number> | null;
    sfreqs: number[];
    target_sfreq: number | null;
    common_channels: string[];
    common_channel_types: string[];
    dropped_channels: string[];
    total_duration_s: number;
    sessions: string[];
    runs: string[];
    tasks: string[];
  };
}

export interface ImportSession {
  id: string;
  created_at: string;
  updated_at: string;
  mode: "folder" | "files" | "local_path" | "revise";
  status: "created" | "uploading" | "scanning" | "review" | "importing" | "done" | "failed" | "cancelled";
  name: string | null;
  root_label: string | null;
  local_path: string | null;
  staged_files: number;
  staged_bytes: number;
  progress: { done?: number; total?: number; message?: string };
  error: string | null;
  dataset_id: string | null;
  revises: string | null;
  attach_experiment_id: string | null;
  manifest: DatasetManifest | null;
}

export interface ImportPlan {
  files: ManifestFile[];
  counts: Partial<Record<FileRole, number>>;
  bytes: Partial<Record<FileRole, number>>;
  upload_bytes: number;
  upload_files: number;
  root_label: string | null;
}

export interface PretrainingChecks {
  ok: boolean;
  checks: { id: string; label: string; ok: boolean; detail: string }[];
}

export interface PreprocessingDefaults {
  config: PreprocessingConfig;
  notes: string[];
}
