# EEG Cursor — Pure EEG Brain-Computer Interface

## What This Is
A pure EEG-based cursor control system using the OpenBCI Cyton+Daisy (16-channel).
Motor imagery from 5 classes drives cursor movement: left_hand→LEFT, right_hand→RIGHT, feet→DOWN, tongue→UP, rest→STOP.
Click is triggered by sustained high-confidence classification.

## Architecture
- **Acquisition**: BrainFlow (Cyton+Daisy 16ch @ 125Hz, or synthetic board)
- **Preprocessing**: Bandpass (8-30Hz), Notch (60Hz), CAR, artifact rejection
- **Features**: CSP spatial filters, chaos/nonlinear features, band power
- **Classification**: CSP+LDA (default), EEGNet (deep learning), Riemannian MDM
- **Control**: 5-class MI → 4-directional cursor + click via sustained imagery
- **Training**: Graz motor imagery paradigm with pygame visual cues

## Tech Stack
- Python 3.10+, BrainFlow, MNE-Python, scikit-learn, PyTorch, pyRiemann
- pyautogui (cursor), pygame (training paradigm), PyQt5+pyqtgraph (GUI)

## Key Files
- `config/settings.yaml` — All tunable parameters
- `src/` — Core library modules
- `scripts/` — Entry-point scripts
- `tests/` — Unit tests
- `install.sh` — Automated installer
- `boot.sh` — Interactive launcher

## Critical Design Decisions
- Training and inference MUST use identical preprocessing (mi_bandpass 8-30Hz + CAR)
- Per-window non-causal filtering at inference (not stateful causal) to match training
- Label mapping saved alongside model to prevent encoding mismatches
- All channel indices validated against actual board at runtime
