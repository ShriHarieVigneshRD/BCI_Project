# Updated Build Prompt: EEGNet Architecture Playground

## A Stateful, Interactive EEG Deep-Learning Experiment and Visualization Platform

Act as a senior full-stack engineer, machine-learning engineer, and scientific visualization specialist. Build a complete, locally runnable application named **EEGNet Architecture Playground**.

The application is an educational and experimental environment for understanding EEGNet, exploring EEG preprocessing, configuring neural-network architectures, training models, inspecting learned operations, and comparing experimental results.

This is a functional scientific application, not a static UI prototype. Every model visualization, tensor dimension, activation, kernel, training curve, prediction, and metric must be based on actual computations and recorded experiment data.

## 1. Required technology stack

**Frontend**

- React + TypeScript with strict type checking.
- Vite.
- Tailwind CSS.
- React Router.
- Plotly.js or Apache ECharts for scientific plots.
- React Flow or custom SVG for interactive architecture diagrams.
- Zustand or an equivalent lightweight state-management solution.

**Backend**

- Python 3.11 or another explicitly supported version.
- FastAPI and Pydantic.
- PyTorch for EEGNet and training.
- MNE-Python for supported EEG formats and preprocessing.
- NumPy, SciPy, pandas, and scikit-learn as appropriate.
- Server-Sent Events (SSE) or WebSockets for live training progress.
- SQLite or a lightweight local metadata store.
- pytest and frontend testing tools.

**Execution environment:** Local machine, with CPU support and optional CUDA acceleration when available.

**Visual design:** Adaptive light/dark theme with system, light, and dark modes. Persist the preference locally. Use an accessible, modern scientific-workstation interface with responsive layouts, clear charts, consistent labels, tooltips, and loading/error states.

Do not send EEG datasets to external services.

---

# 2. Mandatory application workflow

The application must follow this workflow. Do not rearrange the primary user journey.

\<escape>
MAIN WORKFLOW

1. Dataset & Splits
   ↓
2. Preprocessing Studio
   ↓
3. Architecture Explorer
   ├── Configure architecture
   ├── EEG Signal Journey
   └── Training Simulator
   ↓
   Training completed
   ↓
   Persisted experiment artifacts
   ↓
   ├── Kernel & Convolution Playground
   ├── Architecture Breaker
   └── Results & Predictions

SEPARATE TOP-LEVEL AREAS

- Hyperparameter Battle Arena
- Tensor Detective
- Experiments & Exports

GLOBAL HEADER

- Upload Dataset
- Start New Experiment
- Theme selector

Implement the workflow as a stateful experiment lifecycle. The user must be able to understand which stage they are in, what has been completed, what is required next, and which results are available.

## Global navigation

Use the following primary navigation:

1. Dataset & Splits
2. Preprocessing Studio
3. Architecture Explorer
4. Hyperparameter Battle Arena
5. Tensor Detective
6. Experiments & Exports

The Architecture Explorer is the central workspace and contains these internal sections or tabs:

- Architecture Configuration
- EEG Signal Journey
- Training Simulator
- Kernel & Convolution Playground
- Architecture Breaker
- Results & Predictions

The last three sections—Kernel & Convolution Playground, Architecture Breaker, and Results & Predictions—must depend on the selected experiment's lifecycle state. Kernel inspection and Architecture Breaker must not become available until a successful training run has completed and the required artifacts have been persisted. Results & Predictions must also use actual evaluation results; show an appropriate state if evaluation has not yet run.

Do not create separate primary navigation entries for every internal Architecture Explorer section.

---

# 3. Global header and fresh-experiment workflow

Place a prominent header above the main application navigation.

Include:

- **Upload Dataset**: opens the upload workflow and allows the user to add a dataset.
- **Start New Experiment**: initiates a fresh experiment using the currently selected or newly uploaded dataset.
- Current experiment name and status.
- Theme selector.
- Optional device/status indicator.

## Start New Experiment behavior

When the user selects Start New Experiment:

1. Ask whether to create a new experiment using the current dataset or upload another dataset.
2. Create a unique experiment ID and a clean experiment configuration.
3. Return the user to Dataset & Splits.
4. Require the user to confirm or configure the data and evaluation protocol.
5. Continue to Preprocessing Studio.
6. Continue to Architecture Explorer for model configuration and training.

Preserve prior experiments. Never overwrite the previous model, configuration, metrics, or artifacts when starting a new experiment.

A new experiment may reuse a dataset and preprocessing configuration as a starting point, but it must create a separate experiment record and explicitly show any inherited settings.

If a user changes the dataset, split assignment, preprocessing configuration, or architecture after training, mark dependent results as stale. Do not present results from the old configuration as results for the modified configuration.

---

# 4. Experiment lifecycle and state management

Implement a formal backend-owned lifecycle.

Suggested states:

- `created`
- `dataset_required`
- `dataset_ready`
- `split_configured`
- `preprocessing_configured`
- `preprocessing_completed`
- `architecture_configured`
- `training_queued`
- `training`
- `training_failed`
- `training_cancelled`
- `training_completed`
- `evaluation_running`
- `evaluation_completed`
- `artifacts_incomplete`
- `artifacts_ready`

Refine these states if needed, but keep the transitions explicit and testable.

The frontend must derive available actions from backend-reported experiment state and artifact availability, not from a button click or a client-side boolean alone.

## Training completion gate

Training completion alone is not sufficient to enable learned-weight visualizations.

Before enabling Kernel & Convolution Playground and Architecture Breaker, the backend must verify that:

1. Training reached a valid terminal state.
2. A valid model checkpoint has been saved.
3. The exact model architecture and hyperparameters are recorded.
4. The selected dataset, preprocessing pipeline, and split assignments are recorded.
5. The training history and random seed are recorded.
6. Required intermediate-result metadata is available.
7. The checkpoint can be loaded successfully.
8. A forward pass and required visualization extraction can be performed for the selected model and input.

Only then mark the required artifacts as ready.

If training fails, is cancelled, or produces an invalid checkpoint, do not enable the post-training inspection tools as if training succeeded. Show the reason and provide a retry or recovery action.

A user may still inspect the untrained architecture, validate its tensor shapes, and explore theoretical layer parameters before training. Clearly distinguish this from inspecting learned weights or learned activations.

---

# 5. Dataset & Splits — first step

This must be the first stage of every new experiment.

## Upload support

Support implemented and validated EEG formats, including appropriate use of:

- EDF/EDF+.
- FIF.
- EEGLAB `.set` files with required companion files.
- NumPy `.npy` and `.npz`.
- CSV and TSV for documented structured signal data.

Display supported formats and their metadata requirements. Validate files before accepting them into an experiment.

Provide two data workflows:

### A. Separate training and test datasets

Allow independent uploads and validation of training and test datasets.

Check channel identities and order, channel counts, sampling rates, units where known, epoch lengths, tensor dimensions, event and class mappings, and metadata requirements.

Warn if participants or recording sessions overlap between the partitions when that information is available. Let users determine whether the dataset is appropriate for their intended evaluation objective.

### B. One dataset with a configurable split

Offer:

**Random trial/sample-level split**

- Configurable train/test ratio.
- Stratification when valid.
- Random seed.
- Participant/session overlap report.
- An explanation that trials from the same participant may occur in both sets.

**Participant/group-level split**

- Select participants or groups for training and testing.
- Allow a configurable allocation, such as eight participants for training and two held out for testing if the uploaded dataset supports it.
- Guarantee that groups do not overlap.
- Display partition membership and class coverage.

**K-fold cross-validation**

- Trial-wise K-fold.
- Stratified trial-wise K-fold when supported.
- Group-wise K-fold.
- Stratified group-wise K-fold when supported by the dataset and installed libraries.

Use appropriate splitters such as `GroupKFold` or `StratifiedGroupKFold` where applicable.

Explain that trial-wise splitting and participant-independent splitting answer different scientific questions. Neither should be mislabeled as the other.

## Dataset summary

Show:

- Number of recordings, trials, epochs, and samples.
- Number of channels.
- Sampling frequency.
- Epoch duration when available.
- Number of classes and class distribution.
- Participant and session counts where available.
- Channel names and metadata.
- Missing or invalid metadata.
- Whether the dataset appears raw, continuous, or already epoched.

Do not assume a fixed dataset size or participant count.

## Leakage safeguards

Implement backend validation that ensures:

- Group-aware partitions have zero group overlap.
- Learned transformations are fitted on training data only.
- Cross-validation transformations are fitted independently within each fold.
- Test data is not used for hyperparameter selection, early stopping, or model selection.
- Split assignments and random seeds are stored and reproducible where supported.
- Every experiment retains a traceable evaluation protocol.

Show a split-integrity summary before allowing the user to proceed.

---

# 6. Preprocessing Studio — second step

After Dataset & Splits is validated, guide the user to Preprocessing Studio.

Support both raw EEG and previously processed/epoched data.

## Pipeline controls

Implement applicable stages:

1. Channel inspection, selection, and mapping.
2. Filtering.
3. Artifact handling.
4. Resampling/downsampling.
5. Event detection and epoching.
6. Baseline correction.
7. Normalization.
8. Final tensor validation.

Provide controls for every implemented stage. Users should be able to preview changes and save a preprocessing configuration.

## Filtering

Allow the user to select a supported filter type, such as FIR band-pass, high-pass, low-pass, IIR where implemented, or notch filtering.

Expose appropriate cutoff frequencies, line-noise frequency, filter order or transition parameters, and phase behavior where applicable.

Validate cutoff frequencies against the sampling rate and Nyquist limit. Reject invalid configurations with clear explanations.

## Artifact handling

Provide supported, documented options such as bad-channel annotation, noisy-segment rejection, amplitude-based rejection, and ICA-based inspection and component rejection.

For ICA:

- Display actual component diagnostics and time courses.
- Allow users to select components to reject or retain.
- Explain the meaning of retained-component count or percentage if offered.
- Do not imply that explained variance alone identifies artifacts.
- Record the fitting scope and selected/rejected components.

Prevent learned preprocessing from compromising held-out evaluation. Where the correct scope depends on the experiment design, implement and document the appropriate training-fold or recording-level policy.

## Resampling and epoching

Allow configurable resampling with appropriate anti-aliasing and updated time axes.

Perform epoching using available event markers. Allow the user to select event/label mapping, epoch start and end, baseline interval, and rejection settings.

If the dataset is already epoched, detect that structure and allow the user to retain it or rebuild epochs when the necessary continuous data and event markers are available.

If event markers are absent or invalid, explain why epoching cannot safely proceed. Never silently invent trial boundaries.

## Normalization

Support documented normalization methods. Fit parameters on training data only, apply them to validation and test data, and fit them separately inside each cross-validation fold.

## Preprocessing comparison

Display:

- Raw and processed signals.
- Frequency-domain changes where appropriate.
- Channel and epoch counts.
- Rejected epochs and artifact statistics.
- Class distribution before and after processing.
- Final tensor dimensions.

Store the full preprocessing configuration and processing metadata with the experiment.

Only enable Continue to Architecture Explorer after the preprocessing pipeline has passed validation and produced a usable model input, unless a clearly defined no-op pipeline is appropriate for already processed data.

---

# 7. Architecture Explorer — central workspace

The Architecture Explorer is where the user configures the EEGNet architecture, follows signal transformations, and trains the model.

It must be a single main workspace with three initial tabs:

1. Architecture Configuration.
2. EEG Signal Journey.
3. Training Simulator.

After successful training and artifact verification, reveal the post-training tools:

4. Kernel & Convolution Playground.
5. Architecture Breaker.
6. Results & Predictions.

The UI should communicate that the post-training tools use the saved trained model and recorded intermediates.

## 7.1 Architecture Configuration

Before training, the user must configure the architecture.

Provide editable controls for supported architecture parameters:

- Number of temporal filters, F1.
- Depth multiplier, D.
- Number of pointwise filters, F2.
- Temporal kernel length.
- Spatial kernel dimensions.
- Pooling sizes and strides.
- Padding.
- Dropout probability.
- Activation functions.
- Normalization layers and settings where supported.
- Number of input channels.



- Number of classes.

Allow selection of a documented original-style EEGNet-8,2 baseline and custom configurations.

Explain which parameters are structural and which are training-related. Do not imply that changing an architecture parameter has improved accuracy without a valid experiment.

### Architecture visualization

Render the configured architecture from the actual model definition or shared architecture specification.

For each layer, show:

- Layer type and name.
- Input and output shapes.
- Kernel size, stride, padding, and groups where relevant.
- Parameter count.
- Trainable parameters.
- Layer purpose.
- Validity status.

Support zoom, pan, reset, node selection, and shape tracing.

Validate the architecture before training. Display dimension or configuration errors and explain how to resolve them.

The standard model input convention may be `[batch, 1, channels, time_samples]`, but record and validate the actual convention used.

### Saving configurations

Allow users to save, name, duplicate, and restore architecture configurations.

Changing the architecture after training must create a new configuration version. Do not silently mutate the configuration of a previously trained experiment.

## 7.2 EEG Signal Journey

The Signal Journey is part of the Architecture Explorer, not a separate top-level page.

Allow selection of a trial, epoch, participant/group where applicable, model layer, and trained experiment.

Show actual representations from:

- Uploaded or loaded signal.
- Processed signal.
- Epoch.
- Model input.
- Temporal convolution.
- Spatial/depthwise convolution.
- Separable convolution stages.
- Activation and pooling layers.
- Final feature representation.
- Classifier logits and probabilities.

Use actual forward-pass tracing, such as PyTorch forward hooks or an equivalent explicit mechanism.

Display the input/output shape at each stage. Label intermediate feature maps as activations rather than EEG voltages.

Provide linked plots, a layer/stage selector, tooltips, and a stage navigator. Make it easy to continue from a selected layer into the Kernel Playground after training.

## 7.3 Training Simulator

Training occurs within Architecture Explorer after the dataset, splits, preprocessing, and architecture are valid.

Provide controls for:

- Epoch count.
- Batch size.
- Learning rate.
- Optimizer.
- Weight decay.
- Loss function appropriate to the task.
- Random seed.
- Device.
- Validation strategy.
- Early stopping and patience.
- Learning-rate scheduling where implemented.
- Class weighting when appropriate.

Show actual backend-reported training progress, training and validation loss, metrics, learning rate, elapsed time, device, and cancellation state.

Training must run without blocking the API. Use SSE or WebSockets for progress updates.

Support safe cancellation, error recovery, checkpoint saving, and resuming only when correctly supported.

After training completes:

1. Persist the model checkpoint.
2. Persist the architecture and training configuration.
3. Persist preprocessing and split metadata.
4. Persist training history.
5. Persist model state and experiment provenance.
6. Record intermediate-result metadata and any saved activation artifacts.
7. Validate that the checkpoint reloads.
8. Verify that the model can perform a forward pass and supply required layer outputs.
9. Update the experiment state.
10. Reveal post-training tools only after artifact verification succeeds.

Never show fake training progress or placeholder metrics as actual training.

---

# 8. Post-training tools inside Architecture Explorer

These tools are accessible after training has completed successfully and the required artifacts have been persisted and verified.

If an artifact is missing, show exactly what is missing and provide a recovery action where possible.

## 8.1 Kernel & Convolution Playground

This feature must inspect the actual trained model.

Allow the user to select:

- Trained experiment/checkpoint.
- Trial or epoch.
- Layer.
- Input channel or feature map.
- Kernel/filter index when applicable.
- Output feature map.
- Convolution position/stride.
- A second trial for comparison.

Show real learned weights, relevant input slices, the selected convolution window, and the actual output activations.

Where practical, explain the multiplication and accumulation contributing to an output element. Show the relationship between the selected kernel and output feature map.

For other layer types, display the actual operation rather than inventing a kernel. For example, show pooling windows, activation transformations, normalization parameters, dropout behavior, or classifier weights as appropriate.

Allow comparison of two selected epochs at the same layer, including input differences, activation differences, and changes in model predictions.

Use a saved checkpoint and the exact associated architecture configuration. Default to evaluation mode for stable inference visualizations.

### Intermediate artifact strategy

Do not assume it is necessary or practical to save every possible activation for every trial and layer during training.

Implement a hybrid strategy:

- Always persist the checkpoint, model configuration, dataset and preprocessing references, split assignments, training history, and experiment metadata.
- Save intermediate activations for configured representative trials/layers when appropriate.
- Support deterministic activation re-extraction from the saved checkpoint and the exact stored input pipeline.
- Persist extracted results when requested or when useful for reproducibility.
- Record the input trial ID, layer ID, checkpoint ID, extraction configuration, and tensor metadata.
- Ensure that the extraction result corresponds to the selected model version.

The user must never see fabricated activations to compensate for missing saved tensors. If re-extraction is required, compute it from the real checkpoint and data.

## 8.2 Convolution Arena

Include a focused educational comparison of depthwise and separable convolution.

Do not turn this into a generic comparison of ordinary convolution versus depthwise-separable convolution.

For depthwise convolution, show actual weight tensors, channel/group relationships, selected kernels, input/output feature maps, and the effect of stride and padding.

For separable convolution, show both:

1. Depthwise filtering.
2. Pointwise channel mixing.

Use actual model weights and activations. If the current architecture does not contain the selected operation, explain that and let the user choose a compatible configuration or a clearly labeled isolated educational demonstration.

Provide explanatory diagrams, synchronized outputs, parameter counts, and tensor-shape changes.

## 8.3 Architecture Breaker

Let users ask “What happens if…?” by modifying supported architectural parameters.

Allow users to adjust F1, D, F2, kernel lengths, pooling, stride, dropout, input dimensions, and other implemented parameters.

Show:

- Before/after architecture.
- Tensor dimensions.
- Parameter counts.
- Compute estimates where supported.
- Memory estimates when available.
- Receptive-field changes where computable.
- Configuration validity.

### Strict trained-model behavior

The trained experiment must remain immutable.

When a user changes a structural parameter:

1. Create a new architecture variant derived from the saved configuration.
2. Validate the variant.
3. Show the consequences of the change.
4. Mark learned weights as unavailable or incompatible if their shapes no longer match.
5. Do not reuse incompatible weights.
6. Allow the user to start a new training run for the variant.

If only a non-structural display setting changes, preserve the checkpoint. If a parameter affects learned computation, clearly identify whether the existing checkpoint remains valid.

Show actual learned outputs only for compatible trained checkpoints. A modified, untrained architecture may show shape and parameter changes but must not display invented learned activations or claim measured performance.

If the user wants to compare accuracy, launch a new experiment with an appropriate, comparable evaluation protocol.

## 8.4 Results & Predictions

This tab is part of Architecture Explorer and depends on actual evaluation results.

If training is complete but final evaluation has not been run, show an Evaluate Model action. Never substitute validation metrics for test metrics without labeling them.

Show supported metrics, including:

- Accuracy.
- Balanced accuracy.
- Precision.
- Recall.
- F1.
- Per-class support and metrics.
- Confusion matrix.
- Appropriate ROC-AUC or precision-recall metrics when valid.

Document macro, micro, or weighted averaging. Handle undefined metrics and absent classes explicitly.

Provide:

- Interactive confusion matrix with counts and optional row normalization.
- Prediction table.
- Actual class probabilities for mutually exclusive multiclass classification.
- True and predicted labels.
- Trial identifiers.
- Class-wise performance.
- Training and validation curves.
- Model evaluation metadata.
- Links back to the Signal Journey and Kernel Playground.

Clearly distinguish training, validation, and test results.

Only calculate prediction disagreements between models when the evaluated samples and evaluation protocols are compatible.

---

# 9. Separate top-level tab: Hyperparameter Battle Arena

The Battle Arena is independent navigation but uses the same experiment registry and saved checkpoints.

Allow users to create and compare multiple EEGNet experiments, including changes to F1, D, F2, kernel size, dropout, pooling, optimizer, learning rate, batch size, and other implemented parameters.

Show:

- Side-by-side architecture diagrams.
- Parameter counts.
- Tensor dimensions.
- Training and validation curves.
- Evaluation metrics.
- Confusion matrices.
- Per-class metrics.
- Training time and compute cost when measured.
- Preprocessing differences.
- Split strategy and random seed.
- Checkpoint and artifact status.

Allow selection of two or more experiments for detailed comparison.

Use matched split assignments and equivalent evaluation conditions when comparing architectures for a controlled experiment. Warn when datasets, class mappings, preprocessing, or evaluation protocols differ.

Do not use final test metrics to choose hyperparameters. Use validation data or an appropriate nested evaluation procedure, and reserve final test evaluation for the final model-selection protocol.

The Battle Arena must not pretend that an untrained configuration has evaluation results. It may show theoretical shapes and parameter counts before training, clearly separated from measured outcomes.

---

# 10. Separate top-level tab: Tensor Detective

Provide a standalone Tensor Detective page for learning how tensors change throughout the architecture.

Allow users to select an experiment, architecture configuration, layer, and tensor coordinate.

Show:

- Layer input and output shapes.
- Shape transformations.
- Element counts.
- Kernel, stride, padding, dilation, and groups.
- Pooling behavior.
- Flattening and index mapping.
- Classifier input dimensions.
- Parameter counts.
- A step-by-step shape trace.

Use the actual model or verified shape-propagation logic. Do not hardcode dimensions for a specific example architecture.

The page must support both:

- Pre-training architecture validation.
- Post-training inspection of a saved model.

When inspecting a trained experiment, link to the relevant checkpoint and recorded configuration.

---

# 11. Separate top-level tab: Experiments & Exports

Provide a persistent experiment registry.

Each experiment must store:

- Unique ID and name.
- Creation time and lifecycle state.
- Dataset reference and fingerprint.
- Split configuration and assignments.
- Preprocessing configuration.
- Model architecture and hyperparameters.
- Training settings and seed.
- Software versions.
- Training history.
- Checkpoint references.
- Intermediate artifact metadata.
- Evaluation metrics and prediction artifacts.
- Known warnings and limitations.

Allow users to search, filter, open, duplicate, rename, and delete experiments. Confirm destructive actions.

Export:

- Model and preprocessing configurations as JSON.
- Split manifests.
- Metrics as CSV and JSON.
- Predictions as CSV.
- Training history as CSV or JSON.
- Confusion matrices as CSV.
- Plots as PNG or SVG where supported.
- Trained PyTorch checkpoints.
- An experiment report in HTML or PDF where implemented.
- A manifest of exported files and their provenance.

Do not automatically include raw EEG data in experiment bundles. Make dataset export an explicit action.

---

# 12. Backend API and data contracts

Use typed request and response schemas.

Provide routes for:

- Environment and health.
- Dataset upload, listing, summary, and validation.
- Split creation and integrity checks.
- Preprocessing preview and execution.
- Architecture validation and tensor tracing.
- Training start, progress, cancellation, and checkpoint retrieval.
- Intermediate activation and kernel extraction.
- Evaluation and predictions.
- Experiment comparison.
- Artifact status and export.

Suggested endpoint groups:

- `/api/datasets`
- `/api/splits`
- `/api/preprocessing`
- `/api/models`
- `/api/experiments`
- `/api/experiments/{id}/train`
- `/api/experiments/{id}/progress`
- `/api/experiments/{id}/artifacts`
- `/api/experiments/{id}/evaluate`
- `/api/experiments/{id}/results`
- `/api/experiments/{id}/predictions`
- `/api/experiments/compare`
- `/api/experiments/{id}/export`

Use the exact routes that best fit the implementation, but implement all required functionality.

Create a consistent artifact manifest containing artifact type, ID, experiment ID, checkpoint ID where relevant, creation time, source data reference, configuration fingerprint, and storage reference.

Do not expose arbitrary server filesystem paths to the frontend.

---

# 13. Scientific integrity and security

Implement explicit safeguards:

- Training-dependent preprocessing is fitted on training data only.
- Cross-validation preprocessing is fitted independently within each fold.
- Group-level split integrity is checked in the backend.
- Test data is not used for hyperparameter selection or early stopping.
- Model comparisons disclose incompatible protocols.
- Predictions and activations are linked to the exact checkpoint.
- Modifying a configuration invalidates dependent results as appropriate.
- Cached visualizations are keyed by checkpoint, input, layer, and configuration.
- Failed training does not unlock post-training tools.
- Incomplete artifacts are reported rather than fabricated.
- Uploaded files are validated and stored safely.
- The app binds to localhost by default and does not transmit EEG data externally.

---

# 14. Testing requirements

Test the workflow as a complete state machine.

Include tests that verify:

1. A new experiment begins at Dataset & Splits.
2. Preprocessing cannot proceed with invalid dataset metadata.
3. Architecture configuration cannot start training before valid data and preprocessing are ready.
4. Invalid architecture settings are rejected.
5. Training progress is generated by actual backend training.
6. Failed or cancelled training does not unlock learned-model inspection.
7. Successful training with valid persisted artifacts unlocks the post-training tools.
8. Missing or corrupt checkpoints prevent artifact readiness.
9. Activation extraction uses the correct checkpoint and input.
10. Kernel visualization uses real model weights.
11. Architecture Breaker creates variants without modifying the original experiment.
12. Structural changes invalidate incompatible learned weights.
13. Results are only shown when the corresponding evaluation artifacts exist.
14. Group-level split checks reject overlapping groups.
15. Training-only preprocessing and fold-specific fitting are enforced.
16. Starting a new experiment preserves earlier experiment records.
17. Changing data or preprocessing marks dependent artifacts stale.
18. Exported metadata matches the selected experiment.
19. Theme modes and primary navigation work.
20. The complete workflow passes an end-to-end test using a small, explicitly synthetic EEG dataset.

Use synthetic data for software testing only. Clearly label it as synthetic and do not present its metrics as evidence of real-world EEG performance.

---

# 15. Required delivery process

Implement in this order:

**Phase 1:** EEGNet model, architecture validation, tensor tracing, and scientific unit tests.

**Phase 2:** Dataset upload, validation, split strategies, leakage safeguards, and preprocessing.

**Phase 3:** Experiment lifecycle, persistence, actual training, checkpoints, and live progress.

**Phase 4:** Global header, workflow navigation, theme system, Dataset & Splits, and Preprocessing Studio.

**Phase 5:** Architecture Explorer, including architecture configuration, EEG Signal Journey, and Training Simulator.

**Phase 6:** Post-training artifact verification, Kernel & Convolution Playground, Architecture Breaker, and Results & Predictions.

**Phase 7:** Hyperparameter Battle Arena, Tensor Detective, and Experiments & Exports.

**Phase 8:** Integration testing, accessibility, performance, documentation, and local startup verification.

Do not create every page as an unconnected mockup and claim the application is complete. Each phase must integrate with the backend and the experiment lifecycle.

Deliver a complete local project with source code, pinned dependencies, automated tests, documentation, and startup commands. Include a completion report that states which features work, which tests were actually run, and any remaining limitations.

**The core acceptance rule is this: no post-training kernel or convolution inspection, and no learned-model Architecture Breaker visualization, may be enabled until the exact trained checkpoint and its required experiment metadata are saved, verified, and available for genuine computation.**


Additional Points to be noted: 
- Needed a Light, dark, and system theme modes work.
23. Documentation

Provide a complete README.md that includes:

Project overview.

Features.

Architecture diagram or description.

Prerequisites.

Installation steps.

Local startup instructions.

Dependency and environment setup.

How to upload datasets.

Required data schema and metadata.

Supported file formats.

How to configure preprocessing.

How split strategies differ.

How to run training and cross-validation.

How to compare experiments.

How to export results.

How to run tests.

Troubleshooting.

Scientific limitations and known constraints.

Include an .env.example if environment variables are needed, plus suitable dependency lock or pinned requirement files.
