# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- **UncertaintyWeightedLoss** module (`src/training/uncertainty_weights.py`) — analytical uncertainty weighting (arXiv 2408.07985) for automatic multi-loss balancing
- **Surface Laplacian** spatial filter (`src/preprocessing/laplacian.py`) — FDN and spherical spline methods for improved spatial resolution over CAR
- **Synthetic data generator** (`scripts/generate_synthetic_data.py`) — headless labeled MI data for pipeline testing without hardware or display
- **GFlowNet save/load** persistence across sessions
- **GFlowNet integration** into SEAL adaptation loop in `run_eeg_cursor.py`
- **FilterBankCSP** wired into advanced pipeline Phase 2 (9 sub-bands + subject mu)
- **Probability calibration** for CSP+LDA via CalibratedClassifierCV (sigmoid/Platt)
- **Router z-loss** (ST-MoE) on adaptive router gating network
- **Per-module learning rates** for EEGNet and Neural SDE
- **CosineAnnealingWarmRestarts** scheduler for EEGNet and Neural SDE
- **Multi-OS support** in install.sh: Ubuntu, Fedora, Arch, openSUSE, macOS (MPS), WSL2
- **DataRecorder.drain()** for safe long-session recording (prevents ring buffer overflow)
- **Emergency save** on Ctrl+C during calibration (saves partial data)
- **BaseClassifier.load()** dispatches .pt to PyTorch, .pkl to joblib
- 18 new tests: Surface Laplacian (12) + synthetic data generator (6)

### Fixed
- **CRITICAL: Ring buffer overflow** — sessions >6 minutes silently lost data. Now drains periodically.
- **CRITICAL: Data leakage** — augmented data was used for cross-validation, inflating accuracy
- **CRITICAL: Double softmax** in adaptive router GatingMLP (Softmax + cross_entropy)
- **CRITICAL: Import crashes** — eegnet.py and neural_sde.py crashed when torch not installed
- **CRITICAL: train_advanced.py** — undefined `saved` variable crashed every run
- **CRITICAL: BaseClassifier.load()** — couldn't load .pt models (dispatched all to joblib)
- **HIGH: Numerical overflow** in causal_channels.py DAGMA gradient descent (expm)
- **HIGH: Hardcoded n_classes=5** in _SoftVotingClassifier fallback
- **HIGH: np.bincount** on non-contiguous labels silently skipped temporal profiling
- **HIGH: run_koopman.py** confused (center, bandwidth) with (low, high) for mu_band
- **HIGH: torch.load(weights_only=False)** security risk in model loading
- Neural SDE: train/inference jump dynamics mismatch (Bernoulli vs expected value)
- Neural SDE: unseeded RNG for train/val split
- EEGNet: unseeded RNG, weights_only=False
- Adaptive router: default sampling_rate 250 → 125 (Cyton+Daisy)
- SEAL: negative buffer class imbalance (now uses 2nd-best prediction)
- MultiModelTrainer: shallow dict copy mutated shared config
- EnsembleBuilder: fallback proba shape used wrong n_classes
- boot.sh: find_latest_file alpha-sort picked wrong file
- boot.sh: do_full_pipeline missing Step 3/3
- GFlowNet: trajectory balance source state fallback
- GUI: dead request_run signals, unused threading import
- Various: dead code removal, unused imports, misleading docstrings
- 35+ fixes total across 20+ files

### Changed
- Neural SDE training uses AdamW (was Adam) with per-module LR and cosine scheduler
- EEGNet training uses AdamW with per-module LR and cosine scheduler
- Advanced pipeline Phase 2 now includes FilterBankCSP before augmentation
- Cross-validation uses clean (non-augmented) data to prevent leakage
- Model files use correct extensions (.pt for PyTorch, .pkl for sklearn)
- n_classes default changed from 3 to 5 throughout
- settings.yaml expanded with neural_sde, adaptive_router config sections

## [1.0.0] - 2026-03-31

### Added
- Initial release: 5-class motor imagery cursor control
- 5 classifiers: CSP+LDA, EEGNet, Riemannian MDM, Neural SDE, Adaptive Router
- 10 SOTA modules: adaptive_router, causal_channels, state_monitor, koopman, pretrain, neural_sde, jacobian_features, ftle, variable_selector, gflownet_strategy
- SEAL self-adaptation engine with ErrP/P300 detection
- 5-tab PyQt5 GUI
- Graz motor imagery paradigm (pygame)
- Advanced 5-phase training pipeline
- 12 entry-point scripts
- 93 test functions
