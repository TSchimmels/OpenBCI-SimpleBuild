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
- [References](#references)
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

| Feature Type | Method | Output | Source |
|-------------|--------|--------|--------|
| **CSP** | Common Spatial Patterns via MNE | 12 log-variance components | Blankertz et al. (2008) |
| **Band Power** | Welch PSD + Simpson integration | mu/beta power per channel + ratio | Pfurtscheller & Lopes da Silva (1999) |
| **Chaos/Nonlinear** | Hjorth, permutation entropy, fractal dimensions via antropy | 6 features × 7 channels | Lotte et al. (2018) |

---

## Classification System

Three pluggable classifiers implement a common `BaseClassifier` interface:

### 1. CSP + Shrinkage LDA (Default, Recommended)

- **Spatial filtering:** CSP learns 12 spatial filters maximizing class-conditional variance ratios
- **Classification:** Linear Discriminant Analysis with Ledoit-Wolf automatic shrinkage
- **Strengths:** Fast (< 1ms inference), works well with small datasets (40+ trials/class), interpretable
- **Architecture source:** Ramoser et al. (2000), Blankertz et al. (2008)

### 2. EEGNet (Deep Learning)

- **Architecture:** Compact CNN with temporal → depthwise spatial → separable convolutions
- **Training:** Adam optimizer with early stopping on 10% validation hold-out
- **Strengths:** Learns spatial and temporal features end-to-end, no manual feature engineering
- **Architecture source:** Lawhern et al. (2018) — *Journal of Neural Engineering*

### 3. Riemannian MDM (Geometry-Aware)

- **Approach:** Maps EEG epochs to SPD covariance matrices, classifies via geodesic distances on the Riemannian manifold
- **Strengths:** Naturally robust to non-stationarity, no explicit feature extraction
- **Architecture source:** Barachant et al. (2012) — *IEEE Trans. Biomed. Eng.*

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

Implements the **Graz Motor Imagery Protocol** (Pfurtscheller & Neuper, 2001):

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

**Default session:** 5 classes × 40 trials/class × 2 runs = 400 total trials (~45 minutes)

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
│   │   └── pipeline.py             # Riemannian MDM + ClassifierFactory
│   ├── control/
│   │   ├── mouse.py                # pyautogui cursor driver
│   │   ├── mapping.py              # Signal normalization + velocity mapping
│   │   └── cursor_control.py       # EEG-driven cursor state machine (NEW)
│   └── training/
│       ├── paradigm.py             # Graz MI protocol (pygame, 5-class)
│       ├── recorder.py             # Data recording with event markers
│       └── trainer.py              # Offline training + cross-validation
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
│   └── test_classification.py      # Classifier & factory tests
│
├── data/                           # Recorded EEG sessions
│   ├── raw/                        # .npz files from calibration
│   └── processed/                  # Processed epochs
├── models/                         # Trained classifier files
└── notebooks/                      # Jupyter analysis notebooks
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/TSchimmels/OpenBCI-SimpleBuild.git
cd OpenBCI-SimpleBuild

# Run the automated installer (creates venv, installs all deps, validates)
bash install.sh

# Or use the interactive launcher
bash boot.sh
```

### Quick Test (No Hardware)

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the synthetic pipeline test
python scripts/test_synthetic.py --verbose

# Run unit tests
python -m pytest tests/ -v
```

---

## Usage

### 1. Collect Training Data
```bash
python scripts/collect_training_data.py --verbose
```
Follow the on-screen arrows and imagine the indicated movements.

### 2. Train a Classifier
```bash
python scripts/train_model.py --data-path data/raw/session_YYYYMMDD_HHMMSS.npz --verbose
```

### 3. Run the EEG Cursor
```bash
python scripts/run_eeg_cursor.py --model models/csp_lda_YYYYMMDD_HHMMSS.pkl --verbose
```

### 4. Or Use the GUI
```bash
python scripts/gui.py
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

This project builds on and integrates these open source libraries and research:

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

### Research Architecture Sources

| Component | Based On | Citation |
|-----------|----------|----------|
| CSP spatial filtering | Blankertz et al. (2008) | "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis." *IEEE Signal Processing Magazine*, 25(1), 41-56. |
| Graz MI paradigm | Pfurtscheller & Neuper (2001) | "Motor imagery and direct brain-computer communication." *Proceedings of the IEEE*, 89(7), 1123-1134. |
| EEGNet architecture | Lawhern et al. (2018) | "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces." *Journal of Neural Engineering*, 15(5), 056013. |
| Riemannian classification | Barachant et al. (2012) | "Multiclass brain-computer interface classification by Riemannian geometry." *IEEE Trans. Biomed. Eng.*, 59(4), 920-928. |
| ERD/ERS band power | Pfurtscheller & Lopes da Silva (1999) | "Event-related EEG/MEG synchronization and desynchronization: basic principles." *Clinical Neurophysiology*, 110(11), 1842-1857. |
| Nonlinear EEG features | Lotte et al. (2018) | "A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update." *Journal of Neural Engineering*, 15(3), 031005. |
| Shrinkage LDA | Ledoit & Wolf (2004) | "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*, 88(2), 365-411. |
| CSP+LDA for MI-BCI | Ramoser et al. (2000) | "Optimal Spatial Filtering of Single Trial EEG During Imagined Hand Movement." *IEEE Trans. Rehab. Eng.*, 8(4), 441-446. |

### Derived From

This project is a **pure-EEG adaptation** of the [Mental Mouse](https://github.com/TSchimmels/OPEN_BCI_BUILD) hybrid BCI project, which combined EEG motor imagery with MediaPipe eye tracking and EMG jaw-clench click detection. The adaptation removes all non-EEG modalities and replaces them with:
- **4-directional MI control** (replacing eye tracking for cursor positioning)
- **Sustained-imagery click detection** (replacing EMG jaw clench)
- **5-class paradigm** (up from 3-class in the original)

---

## Known Issues & Audit

The original project underwent a comprehensive audit that identified 8 critical bugs. All have been addressed in this build:

| ID | Issue | Status |
|----|-------|--------|
| C-1 | Training/inference preprocessing mismatch | **Fixed** — Both paths use mi_bandpass + CAR |
| C-2 | Window size mismatch (training vs runtime) | **Fixed** — Both use classification_window_start/end |
| C-3 | Label encoding order mismatch | **Fixed** — Label map saved with model |
| C-4 | Causal filter state corruption | **Fixed** — Per-window non-causal filtering |
| C-5 | DataRecorder.save() loses data after stop() | **Fixed** — Cached in _last_raw_data |
| C-6 | Synthetic board channel index mismatch | **Fixed** — Runtime validation + clamping |
| C-7 | Chaos/bandpower features computed but unused | **Acknowledged** — Available for fusion classifiers |
| C-8 | Cross-validation crashes on unfitted classifier | **Fixed** — Uses _pipeline attribute for CV |

---

## References

1. Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Muller, K.-R. (2008). Optimizing Spatial Filters for Robust EEG Single-Trial Analysis. *IEEE Signal Processing Magazine*, 25(1), 41-56.
2. Pfurtscheller, G. & Neuper, C. (2001). Motor imagery and direct brain-computer communication. *Proceedings of the IEEE*, 89(7), 1123-1134.
3. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
4. Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2012). Multiclass brain-computer interface classification by Riemannian geometry. *IEEE Trans. Biomed. Eng.*, 59(4), 920-928.
5. Pfurtscheller, G. & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11), 1842-1857.
6. Lotte, F., et al. (2018). A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update. *Journal of Neural Engineering*, 15(3), 031005.
7. Ramoser, H., Muller-Gerking, J., & Pfurtscheller, G. (2000). Optimal Spatial Filtering of Single Trial EEG During Imagined Hand Movement. *IEEE Trans. Rehab. Eng.*, 8(4), 441-446.
8. Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.

---

## License

This project is provided as-is for research and educational purposes.

Built with [Claude Code](https://claude.ai/claude-code) by Anthropic.
