## Change Request: Multi-Subject Dataset Upload & End-to-End Compatibility

Update the existing **Dataset Upload dialog and dataset ingestion pipeline** to support real-world, multi-subject EEG/BCI datasets. This is an enhancement to the existing application, not a rebuild.

### 1. Update the Dataset Upload dialog

Replace the single-file-centric upload flow with three options:

- **Select Dataset Folder** — import a directory containing multiple participants, sessions, EEG recordings, event files, and metadata.
- **Select Multiple Files** — select multiple EEG recordings and their associated event/metadata files together.
- **Use Local Folder Path** — allow a local dataset directory path when running the app on the user's machine, with safe path validation.

Automatically scan nested folders, identify participants, sessions, runs, recordings, and associated event files where possible. Support multiple recordings per participant. Do not treat an entire EDF recording as one trial or count event files as EEG recordings.

After scanning, show a **Dataset Review** with discovered participants, recordings, sampling rates, channels, event/label status, and validation warnings. Allow users to correct participant/session/run IDs, map event codes to classes, and exclude selected recordings. Calculate all counts dynamically; never use placeholder counts as real metadata.

### 2. Preserve and integrate the existing split functionality

**All three split methods are already implemented. Do not rebuild or replace them.** Adapt their existing inputs and data references to the newly discovered dataset manifest:

- Trial/sample-level split.
- Participant/group-level split.
- K-fold cross-validation, including the existing supported trial-wise and group-wise strategies.

Ensure the existing split UI receives correctly identified participants, recordings, trials, groups, labels, and metadata. Preserve current split controls and behavior wherever possible. Revalidate split integrity after dataset selection or changes.

### 3. Update every downstream screen to use the same dataset manifest

The uploaded dataset must remain the single source of truth throughout the application. Audit and update the following existing screens and features for compatibility:

- **Home screen / global header:** `Upload Dataset` and `Start New Experiment` must open the new import workflow and create or select the correct dataset/experiment.
- **Dataset & Splits:** use the discovered recordings, trials, participants, and labels with the existing split methods.
- **Preprocessing Studio:** process all selected recordings consistently, associate event files correctly, and preserve participant/session/run/trial provenance.
- **Architecture Explorer:** configure input channels, sampling rate, epoch length, and class count from the validated processed dataset.
- **EEG Signal Journey:** display actual trials/epochs from the selected recordings.
- **Training Simulator:** train on the correct split and labels, without accidentally mixing held-out participants into training.
- **Kernel & Convolution Playground:** retrieve inputs and activations for the correct trial and trained checkpoint.
- **Architecture Breaker:** retain compatibility with the experiment's dataset and model configuration.
- **Results & Predictions:** associate predictions and metrics with the correct held-out samples, labels, and participant IDs.
- **Hyperparameter Battle Arena:** compare experiments using compatible datasets, class mappings, and evaluation protocols.
- **Tensor Detective:** derive dimensions from the actual configured model and dataset.
- **Experiments & Exports:** persist and export the dataset manifest, file references, split assignments, preprocessing configuration, participant/trial provenance, model configuration, and results.

### 4. Compatibility and validation requirements

- Changing the dataset must invalidate or mark dependent splits, preprocessing outputs, models, and results as stale where appropriate.
- Starting a new experiment must not overwrite existing experiments.
- Validate channel compatibility, sampling rates, class mappings, missing event files, and incompatible recordings before training.
- Preserve group separation in group-based splits and fit learned preprocessing transformations on training data only.
- For large local datasets, scan metadata incrementally and load signal data lazily instead of loading the entire dataset into browser memory.
- Keep data local and do not transmit EEG recordings externally.

### Acceptance criterion

Test the complete workflow using a nested dataset with multiple participants, multiple runs, and associated event files. Verify that the existing split methods work with the imported dataset and that every downstream screen uses the correct dataset, trial identities, labels, and experiment artifacts.

**Before finishing, audit all existing pages and shared state/API contracts. Do not fix only the upload dialog while leaving downstream screens dependent on the previous single-file data structure.**