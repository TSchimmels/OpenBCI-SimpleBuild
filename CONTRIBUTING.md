# Contributing to OpenBCI SimpleBuild

Thank you for your interest in contributing to this EEG Brain-Computer Interface project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/OpenBCI-SimpleBuild.git`
3. Install dependencies: `bash install.sh`
4. Run tests: `python -m pytest tests/ -v`
5. Create a feature branch: `git checkout -b feature/your-feature`

## Development Setup

```bash
bash install.sh          # Creates .venv, installs all deps
source .venv/bin/activate
python -m pytest tests/ -v --tb=short
```

## Project Structure

```
src/
  acquisition/     # BrainFlow board management
  preprocessing/   # Bandpass, CAR, Laplacian, artifact rejection
  features/        # CSP, chaos, bandpower, Jacobian-SVD, variable selection
  classification/  # 5 classifiers + factory + adaptive router
  control/         # Cursor mapping, mouse control, click detection
  training/        # Paradigm, recorder, trainer, advanced pipeline, UW loss
  analysis/        # ERP, ERDS, topography, Koopman, FTLE, causal, state monitor
  adaptation/      # ErrP/P300 detector, SEAL engine, GFlowNet optimizer

scripts/           # Entry-point scripts (CLI)
tests/             # Unit and integration tests
config/            # settings.yaml (all tunable parameters)
```

## Code Style

- Python 3.10+, type hints on all public functions
- Logging via `logging.getLogger(__name__)` (no `print()` in `src/`)
- Optional dependencies guarded with `try/except ImportError`
- PyTorch classes use `_TorchModule = nn.Module if _TORCH_AVAILABLE else object` pattern
- All config values read from `settings.yaml` via `load_config()`

## Testing

```bash
python -m pytest tests/ -v                    # All tests
python -m pytest tests/test_sota.py -v        # SOTA modules only
python -m pytest tests/ -k "laplacian" -v     # Specific tests
```

- Tests requiring PyTorch, MNE, or BrainFlow are auto-skipped when not installed
- Use `np.random.RandomState(42)` for reproducible tests
- Test both happy path and edge cases (empty arrays, NaN, single-trial)

## Pull Request Process

1. Ensure all tests pass: `python -m pytest tests/ --tb=short`
2. Run syntax check: `python -c "import ast, os; [ast.parse(open(os.path.join(r,f)).read()) for r,_,fs in os.walk('src') for f in fs if f.endswith('.py')]"`
3. Update `CHANGELOG.md` if applicable
4. Reference any related issues in the PR description

## Classifier Development

To add a new classifier:

1. Create `src/classification/your_classifier.py` extending `BaseClassifier`
2. Implement `fit()`, `predict()`, `predict_proba()`, `decision_function()`
3. Register in `ClassifierFactory` (`src/classification/pipeline.py`)
4. Add lazy import in `src/classification/__init__.py`
5. Add config section in `config/settings.yaml`
6. Add tests in `tests/`

## License

See LICENSE file.
