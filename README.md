# EEG Cursor — Pure EEG Brain-Computer Interface

> **A 16-channel motor imagery BCI that controls your computer's cursor using only brainwaves.**

No eye tracking. No EMG. No cameras. Just EEG signals from an OpenBCI Cyton+Daisy board translated into 4-directional cursor movement and click actions through real-time classification of motor imagery.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Hardware Requirements](#hardware-requirements)
- [Signal Processing Pipeline](#signal-processing-pipeline)
- [Classification System](#classification-system)
- [Cursor Control Design](#cursor-control-design)
- [Training Paradigm](#training-paradigm)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Design Attributes & Innovations](#design-attributes--innovations)
- [Open Source Foundations](#open-source-foundations)
- [Known Issues & Audit](#known-issues--audit)
- [License](#license)

---

## Overview

EEG Cursor implements a **5-class motor imagery (MI) brain-computer interface** that translates imagined movements into real computer cursor control:

| Mental Task | Cursor Action |
|-------------|--------------|
| **Left hand** imagery | Cursor moves LEFT |
| **Right hand** imagery | Cursor moves RIGHT |
| **Feet** imagery | Cursor moves DOWN |
| **Tongue** imagery | Cursor moves UP |
| **Rest** (relaxed) | No movement |
| **Sustained imagery** (>0.8s at high confidence) | Mouse CLICK |

The system achieves real-time control at **16 Hz update rate** with sub-100ms classification latency, using a modular pipeline that separates acquisition, preprocessing, feature extraction, classification, and control into independent, testable components.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EEG Cursor Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐          │
│  │ OpenBCI  │───>│ Preprocessing│───>│   Feature     │          │
│  │ Cyton+   │    │  - Bandpass  │    │  Extraction   │          │
│  │ Daisy    │    │  - Notch     │    │  - CSP        │          │
│  │ 16ch EEG │    │  - CAR       │    │  - Bandpower  │          │
│  │ @125 Hz  │    │  - Artifact  │    │  - Chaos      │          │
│  └──────────┘    └──────────────┘    └───────┬───────┘          │
│                                               │                  │
│                                    ┌──────────▼──────────┐      │
│                                    │   Classification    │      │
│                                    │  ┌───────────────┐  │      │
│                                    │  │ CSP+LDA       │  │      │
│                                    │  │ EEGNet (CNN)  │  │      │
│                                    │  │ Riemannian MDM│  │      │
│                                    │  └───────────────┘  │      │
│                                    └──────────┬──────────┘      │
│                                               │                  │
│                                    ┌──────────▼──────────┐      │
│                                    │  Cursor Controller  │      │
│                                    │  - Direction map    │      │
│                                    │  - Velocity scale   │      │
│                                    │  - EMA smoothing    │      │
│                                    │  - Click detection  │      │
│                                    └──────────┬──────────┘      │
│                                               │                  │
│                                    ┌──────────▼──────────┐      │
│                                    │   pyautogui         │      │
│                                    │   Cursor Movement   │      │
│                                    └─────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Thread Model

| Thread | Rate | Purpose |
|--------|------|---------|
| EEG Acquisition | 125 Hz | Reads from BrainFlow ring buffer |
| Classification Loop | 16 Hz | Preprocess → classify → move cursor |
| UI Rendering | 60 fps | PyQt5 signal display (optional) |

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| **EEG Board** | OpenBCI Cyton + Daisy (16 channels) |
| **Sampling Rate** | 125 Hz (Cyton+Daisy combined) |
| **Resolution** | 24-bit ADC |
| **Electrode Layout** | Standard 10-20 montage |
| **Connection** | USB dongle (serial) |

### 16-Channel Electrode Placement (10-20 System)

```
        Fp1  Fp2
     F3   Fz   F4
  T7   C3   Cz   C4   T8
     P3   Pz   P4
        O1   O2
```

**Critical channels for motor imagery:**
- **C3, C4** — Primary motor cortex (left/right hand discrimination via mu ERD/ERS)
- **Cz** — Central midline (feet imagery, supplementary motor area)
- **F3, F4** — Frontal motor areas (tongue imagery, planning)
- **P3, P4** — Parietal sensorimotor integration

---

## Signal Processing Pipeline

### Preprocessing (Training & Inference — Identical)

1. **Bandpass Filter** — 8-30 Hz Butterworth (order 4), capturing mu (8-12 Hz) and beta (13-30 Hz) rhythms
2. **Common Average Reference (CAR)** — Subtracts mean across all channels to remove global noise
3. **Artifact Rejection** — Peak-to-peak threshold (100 μV default) rejects contaminated epochs

> **Critical design decision:** Training uses zero-phase (`sosfiltfilt`) filtering on complete epochs. Inference uses the **same per-window non-causal approach** — NOT a stateful causal filter — to prevent training/inference mismatch (the #1 bug identified in the audit of the original project).

### Feature Extraction

| Feature Type | Method | Output |
|-------------|--------|--------|
| **CSP** | Common Spatial Patterns via MNE | 12 log-variance components |
| **Band Power** | Welch PSD + Simpson integration | mu/beta power per channel + ratio |
| **Chaos/Nonlinear** | Hjorth, permutation entropy, fractal dimensions via antropy | 6 features × 7 channels |

---

## Classification System

Three pluggable classifiers implement a common `BaseClassifier` interface:

### 1. CSP + Shrinkage LDA (Default, Recommended)

- **Spatial filtering:** CSP learns 12 spatial filters maximizing class-conditional variance ratios
- **Classification:** Linear Discriminant Analysis with Ledoit-Wolf automatic shrinkage
- **Strengths:** Fast (< 1ms inference), works well with small datasets (40+ trials/class), interpretable
- **Approach:** Well-established spatial filtering and linear classification methods

### 2. EEGNet (Deep Learning)

- **Architecture:** Compact CNN with temporal → depthwise spatial → separable convolutions
- **Training:** Adam optimizer with early stopping on 10% validation hold-out
- **Strengths:** Learns spatial and temporal features end-to-end, no manual feature engineering
- **Approach:** Standard compact CNN architecture designed for EEG classification

### 3. Riemannian MDM (Geometry-Aware)

- **Approach:** Maps EEG epochs to SPD covariance matrices, classifies via geodesic distances using standard Riemannian geometry methods
- **Strengths:** Naturally robust to non-stationarity, no explicit feature extraction

### Factory Pattern

```python
from src.classification.pipeline import ClassifierFactory

# Create any classifier from config
clf = ClassifierFactory.create(config)  # reads config.classification.model_type
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

---

## Cursor Control Design

### Probability-to-Velocity Mapping

Unlike the original hybrid project (which used eye tracking for positioning), this pure EEG system maps classification **confidence** to cursor **velocity**:

```
velocity = ((confidence - threshold) / (1 - threshold)) * max_velocity
```

- **Below threshold (< 0.5):** No movement (uncertain/rest)
- **At threshold (0.5):** Zero velocity
- **Maximum confidence (1.0):** Full max_velocity (25 px/frame)

### EMA Smoothing

Exponential moving average prevents jitter:
```
v_smoothed = (1 - alpha) * v_previous + alpha * v_new
```

### Click Detection (Sustained Imagery)

Instead of EMG jaw clench (removed), clicks are triggered by **sustained high-confidence classification**:

1. Any directional class held at confidence ≥ 0.7 for ≥ 0.8 seconds → **single click**
2. Two clicks within 1.5 seconds → **double click**
3. 0.5 second cooldown between clicks prevents accidental repeats

---

## Training Paradigm

Implements a standard **motor imagery calibration protocol**:

```
┌─────────┐  ┌──────────┐  ┌────────────────┐  ┌──────────┐
│ Fixation│  │Beep + Cue│  │ Motor Imagery  │  │   Rest   │
│   2.0s  │→ │  1.25s   │→ │    4.0s        │→ │ 1.5-3.0s │
│   "+"   │  │  Arrow   │  │  (keep going)  │  │  (blank) │
└─────────┘  └──────────┘  └────────────────┘  └──────────┘
```

**Visual cues (pygame fullscreen):**
- ← Left arrow = Imagine left hand movement
- → Right arrow = Imagine right hand movement
- ↓ Down arrow = Imagine feet movement
- ↑ Up arrow = Imagine tongue movement
- "+" only = Rest (no imagery)

**Default session:** 5 classes × 40 trials/class = 200 total trials split across 2 runs (~32 minutes)

---

## Project Structure

```
OpenBCI_SimpleBuild/
├── CLAUDE.md                       # Project rules & architecture
├── README.md                       # This file
├── config/
│   └── settings.yaml               # All 100+ tunable parameters
├── requirements.txt                # Python dependencies
├── install.sh                      # Automated installer (GPU-aware)
├── boot.sh                         # Interactive launcher menu
│
├── src/                            # Core library
│   ├── config.py                   # YAML config loader
│   ├── acquisition/
│   │   └── board.py                # BrainFlow BoardManager wrapper
│   ├── preprocessing/
│   │   ├── filters.py              # Bandpass, notch, CAR, Laplacian
│   │   └── artifacts.py            # Epoch rejection, bad channel detection
│   ├── features/
│   │   ├── csp.py                  # Common Spatial Patterns (MNE)
│   │   ├── chaos.py                # Nonlinear features (antropy)
│   │   └── bandpower.py            # Welch PSD band power
│   ├── classification/
│   │   ├── base.py                 # Abstract BaseClassifier interface
│   │   ├── csp_lda.py              # CSP + shrinkage LDA
│   │   ├── eegnet.py               # EEGNet CNN (PyTorch)
│   │   ├── pipeline.py             # Riemannian MDM + ClassifierFactory
│   │   ├── adaptive_router.py      # Dynamics-aware multi-classifier routing
│   │   └── neural_sde.py           # Stochastic differential equation classifier
│   ├── control/
│   │   ├── mouse.py                # pyautogui cursor driver
│   │   ├── mapping.py              # Signal normalization + velocity mapping
│   │   └── cursor_control.py       # EEG-driven cursor state machine (NEW)
│   ├── training/
│   │   ├── paradigm.py             # Graz MI protocol (pygame, 5-class)
│   │   ├── recorder.py             # Data recording with event markers
│   │   ├── trainer.py              # Offline training + cross-validation
│   │   └── pretrain.py             # Self-supervised pre-training
│   ├── adaptation/
│   │   ├── errp_detector.py        # Involuntary error/confirmation detection
│   │   ├── seal_engine.py          # Self-evolving adaptive learning
│   │   └── gflownet_strategy.py    # Adaptation strategy optimization
│   └── features/
│       ├── (core feature extractors)
│       ├── jacobian_features.py    # Dynamical system features
│       └── variable_selector.py    # Learned feature importance
│
├── scripts/                        # Entry points
│   ├── test_synthetic.py           # Pipeline test (no hardware needed)
│   ├── collect_training_data.py    # Run calibration paradigm
│   ├── train_model.py              # Train classifier on recorded data
│   ├── run_eeg_cursor.py           # Real-time cursor control (MAIN APP)
│   └── gui.py                      # PyQt5 graphical interface
│
├── tests/                          # Unit tests (pytest)
│   ├── test_preprocessing.py       # Filter & artifact tests
│   ├── test_features.py            # CSP, chaos, bandpower tests
│   ├── test_classification.py      # Classifier & factory tests
│   └── test_control.py             # Mapper, cursor, ERP, ERDS, topo tests
│
├── data/                           # Recorded EEG sessions
│   ├── raw/                        # .npz files from calibration
│   └── processed/                  # Processed epochs
├── models/                         # Trained classifier files
└── notebooks/                      # Jupyter analysis notebooks
```

---

## Installation

### Prerequisites

- **Python 3.10+** (check with `python3 --version`)
- **OpenBCI Cyton + Daisy** board (optional — synthetic mode works without hardware)
- **Linux / WSL2** recommended (also works on macOS/Windows with minor adjustments)

### One-Command Install

```bash
# Clone the repository
git clone https://github.com/UA-Consciousness-Studies-Club/OpenBCI-SimpleBuild.git
cd OpenBCI-SimpleBuild

# Run the automated installer
bash install.sh
```

The installer will:
1. Create a Python virtual environment (`.venv/`)
2. Detect your GPU (NVIDIA CUDA auto-detected for EEGNet acceleration)
3. Install all Python dependencies (~15 packages)
4. Install system libraries (SDL2 for audio, xdotool for cursor control)
5. Create project directories (`data/`, `models/`, `notebooks/`)
6. Validate every import
7. Run a BrainFlow smoke test with the synthetic board

### Quick Test (No Hardware Needed)

```bash
source .venv/bin/activate
python scripts/test_synthetic.py --verbose    # Test full pipeline
python -m pytest tests/ -v                     # Run unit tests
```

### Launch

```bash
bash launch.sh              # Launch the GUI (recommended)
bash launch.sh --cli        # Interactive terminal menu
bash launch.sh --test       # Run all tests
bash launch.sh --install    # Install dependencies + launch
```

---

## How to Use

### The GUI (Recommended)

`bash launch.sh` opens a 5-tab control center:

| Tab | What You Can Do |
|-----|-----------------|
| **Dashboard** | Live EEG signals, classification bars, cursor position, system status |
| **Data Collection** | Run calibration, ERP signal trainer, or JEPA pre-training |
| **Training** | Select model type, pick data file, train, see cross-validation results |
| **Live Control** | Start/stop cursor, toggle adaptation/monitoring/auto-undo/routing |
| **Settings** | Edit all configuration, toggle advanced modules, save/load config |

### Terminal Menu

`bash launch.sh --cli` gives you a numbered menu for every operation.

### Manual Commands

There are **three paths** depending on what you want to do:

---

### Path A: Just Explore Your Brain Signals (No Model Needed)

Use the **ERP Signal Trainer** to collect EEG data and see your motor imagery signals in real time. No classifier training required.

```bash
source .venv/bin/activate

# Start the ERP trainer (live collection + visual feedback)
python scripts/erp_trainer.py --verbose
```

**What happens:**
1. Connects to your OpenBCI board (or synthetic board if no hardware)
2. Presents arrow cues (left, right, up, down, rest)
3. After each trial, updates a 6-panel dashboard showing:
   - Your **ERP waveforms** (averaged brain response per class)
   - **ERDS% spectrogram** (time-frequency power map — look for blue = mu ERD)
   - **Mu band power** timecourse (the signal that drives the cursor)
   - **Scalp topographic map** (which brain areas are active)
   - **r² discriminability** (which channels best separate your classes)
   - **SNR feedback** (signal quality per channel)
4. Saves all data to `data/raw/erp_session_YYYYMMDD_HHMMSS.npz`

**To review a previous session:**
```bash
python scripts/erp_trainer.py --review data/raw/erp_session_20260326_143000.npz
```

**What to look for:**
- **Blue regions** in the spectrogram at 8-12 Hz (mu band) during imagery = good ERD
- **Contralateral activation**: Left hand imagery should show stronger ERD at C4 (right hemisphere), and vice versa
- **High r² values** on motor cortex channels (C3, C4, Cz) = your signal is classifiable

---

### Path B: Collect Data + Train a Classifier

This is the standard BCI calibration workflow.

#### Step 1: Collect Training Data
```bash
python scripts/collect_training_data.py --verbose
```

**What happens:**
- Full-screen Graz paradigm with arrow cues
- 5 classes x 40 trials = 200 trials (~25 minutes)
- Saves to `data/raw/session_YYYYMMDD_HHMMSS.npz`

**Tips for good data:**
- Stay relaxed, minimize eye blinks and jaw tension
- For **left hand**: imagine squeezing your left fist rhythmically
- For **right hand**: imagine squeezing your right fist
- For **feet**: imagine curling your toes or pressing a pedal
- For **tongue**: imagine moving your tongue left-right
- For **rest**: just relax and breathe normally

#### Step 2: Train a Classifier
```bash
python scripts/train_model.py \
    --data-path data/raw/session_YYYYMMDD_HHMMSS.npz \
    --model-type csp_lda \
    --verbose
```

**Model options:**
| Flag | Classifier | When to Use |
|------|-----------|-------------|
| `--model-type csp_lda` | CSP + LDA | **Start here.** Fast, works with 40+ trials/class |
| `--model-type riemannian` | Riemannian MDM | Robust to session drift, good for follow-up sessions |
| `--model-type eegnet` | EEGNet (CNN) | Deep learning, needs 100+ trials/class for best results |

**Output:**
- Prints cross-validation accuracy (should be above 30% for 5 classes, chance = 20%)
- Saves model to `models/csp_lda_YYYYMMDD_HHMMSS.pkl`
- Saves label mapping to `models/csp_lda_YYYYMMDD_HHMMSS.labels.json`

#### Step 3: Run the Cursor
```bash
python scripts/run_eeg_cursor.py \
    --model models/csp_lda_YYYYMMDD_HHMMSS.pkl \
    --verbose
```

**What happens:**
- Connects to your board and starts classifying at 16 Hz
- Moves your actual mouse cursor based on your motor imagery
- Prints status every 3 seconds (class, direction, confidence, latency)
- **Ctrl+C** to stop

---

### Path C: Use the GUI

```bash
python scripts/gui.py
```

A full graphical interface with:
- System status (board, model, GPU detection)
- One-click buttons for every operation
- Real-time EEG signal display (if pyqtgraph is installed)
- Live log output
- Model type selector (dropdown)

---

### Quick Reference

| What you want | Command |
|---------------|---------|
| Test without hardware | `python scripts/test_synthetic.py` |
| Explore your brain signals | `python scripts/erp_trainer.py` |
| Review saved ERP data | `python scripts/erp_trainer.py --review data/raw/FILE.npz` |
| Collect calibration data | `python scripts/collect_training_data.py` |
| Train a model | `python scripts/train_model.py --data-path FILE.npz` |
| Control the cursor | `python scripts/run_eeg_cursor.py --model FILE.pkl` |
| Everything via GUI | `python scripts/gui.py` |
| Everything via menu | `bash boot.sh` |
| Run unit tests | `python -m pytest tests/ -v` |

### Configuration

All parameters are in `config/settings.yaml`. Key things you might want to change:

```yaml
board:
  board_id: 2          # Change to 2 for real Cyton+Daisy (default: -1 for synthetic)
  serial_port: "COM3"  # Your board's serial port

control:
  max_velocity: 25     # Cursor speed (pixels per frame)
  confidence_threshold: 0.5  # Lower = more responsive, higher = more precise

training:
  n_trials_per_class: 40  # More trials = better model, but longer session
```

---

## Design Attributes & Innovations

### 1. Training/Inference Parity
The #1 failure mode in EEG BCIs is preprocessing mismatch between training and deployment. This project enforces **identical preprocessing** in both paths:
- Same bandpass (8-30 Hz, mi_bandpass_low/mi_bandpass_high)
- Same spatial reference (CAR)
- Same window size (classification_window_start to classification_window_end)
- Non-causal per-window filtering at inference (not stateful causal)

### 2. Label Map Persistence
The label-to-integer mapping is saved as a `.labels.json` file alongside each trained model. This prevents the catastrophic bug where alphabetical sorting of class names during training produces a different mapping than config file ordering at inference.

### 3. Confidence-Gated Control
Rather than mapping every classification output to movement, the system requires a **minimum confidence threshold** (default 0.5) before moving. This eliminates the constant drift that plagues naive BCI cursor systems.

### 4. Sustained-Imagery Click
Instead of requiring additional hardware (EMG electrodes on the jaw), click events are detected purely from EEG by monitoring for **sustained high-confidence classification** of any directional class. This is a deliberate trade-off: slightly slower clicks, but zero additional hardware.

### 5. Pluggable Classifier Factory
All classifiers implement `BaseClassifier` and are instantiated via `ClassifierFactory.create(config)`. Switching from CSP+LDA to EEGNet or Riemannian MDM requires changing one line in `settings.yaml`.

### 6. Graceful Degradation
- antropy not installed → chaos features return empty arrays (no crash)
- PyTorch not installed → EEGNet unavailable, CSP+LDA still works
- No GPU → PyTorch falls back to CPU automatically
- Synthetic board → full pipeline works without hardware

---

## Open Source Foundations

This project builds on and integrates these open source libraries:

### Core Libraries

| Library | Version | Purpose | License |
|---------|---------|---------|---------|
| [**BrainFlow**](https://github.com/brainflow-dev/brainflow) | ≥5.0 | EEG data acquisition, board abstraction | MIT |
| [**MNE-Python**](https://github.com/mne-tools/mne-python) | ≥1.6 | CSP implementation, EEG processing utilities | BSD-3 |
| [**scikit-learn**](https://github.com/scikit-learn/scikit-learn) | ≥1.3 | LDA classifier, cross-validation, metrics | BSD-3 |
| [**PyTorch**](https://github.com/pytorch/pytorch) | ≥2.0 | EEGNet deep learning classifier | BSD-3 |
| [**pyRiemann**](https://github.com/pyRiemann/pyRiemann) | ≥0.9 | Riemannian geometry classifiers (MDM) | BSD-3 |
| [**antropy**](https://github.com/raphaelvallat/antropy) | ≥0.1.6 | Entropy and fractal dimension features | BSD-3 |
| [**SciPy**](https://github.com/scipy/scipy) | ≥1.10 | Butterworth filters, Welch PSD, integration | BSD-3 |
| [**NumPy**](https://github.com/numpy/numpy) | ≥1.24 | Array computation foundation | BSD-3 |
| [**pyautogui**](https://github.com/asweigart/pyautogui) | ≥0.9 | Cross-platform cursor control | BSD-3 |
| [**pygame**](https://github.com/pygame/pygame) | ≥2.5 | Training paradigm visual cues + audio | LGPL |
| [**PyQt5**](https://riverbankcomputing.com/software/pyqt/) | ≥5.15 | GUI framework | GPL-v3 |
| [**pyqtgraph**](https://github.com/pyqtgraph/pyqtgraph) | ≥0.13 | Real-time signal plotting | MIT |

---

## License

This project is provided as-is for research and educational purposes.

Open source. Contributions welcome.
