# Architecture Overview

## System Design

```
                    +-----------------+
                    |  OpenBCI Board  |
                    |  Cyton+Daisy    |
                    |  16ch @ 125Hz   |
                    +--------+--------+
                             |
                      Bluetooth LE
                             |
                    +--------v--------+
                    |    BrainFlow    |
                    |  Ring Buffer    |
                    |  (45K samples)  |
                    +--------+--------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |   CALIBRATION   |          |    INFERENCE     |
     |   (offline)     |          |   (real-time)    |
     +--------+--------+          +--------+--------+
              |                             |
     +--------v--------+          +--------v--------+
     |  Preprocessing   |          |  Preprocessing   |
     |  BP 8-30Hz + CAR |          |  BP 8-30Hz + CAR |
     |  (or Laplacian)  |          |  (or Laplacian)  |
     +--------+--------+          +--------+--------+
              |                             |
     +--------v--------+          +--------v--------+
     | Epoch Extraction |          |  Sliding Window  |
     | (1.5-4.0s post   |          |  (latest 2.5s)   |
     |  cue onset)      |          |                  |
     +--------+--------+          +--------+--------+
              |                             |
     +--------v--------+          +--------v--------+
     |  Feature Extract |          |   Classifier     |
     |  CSP + FBCSP +   |          |   (trained)      |
     |  chaos + bandpwr |          |                  |
     +--------+--------+          +--------+--------+
              |                             |
     +--------v--------+          +--------v--------+
     | Multi-Model Train|          | Adaptive Router  |
     | CSP+LDA, EEGNet, |          | (optional MoE)   |
     | Riemannian, SDE  |          |                  |
     +--------+--------+          +--------v--------+
              |                             |
     +--------v--------+          +--------v--------+
     | Ensemble Builder |          |  Cursor Control  |
     | (best or voting) |          |  5-class -> 4dir |
     +--------+--------+          +--------+--------+
              |                             |
     +--------v--------+          +--------v--------+
     |   Save Model     |          |  ErrP/P300       |
     |  .pkl or .pt     |          |  Detection       |
     +------------------+          +--------+--------+
                                            |
                                   +--------v--------+
                                   |  SEAL Adaptation |
                                   |  + GFlowNet      |
                                   +-----------------+
```

## Module Dependency Graph

```
acquisition/board.py
    └── (BrainFlow)

preprocessing/
    ├── filters.py         (scipy.signal)
    ├── artifacts.py       (numpy)
    └── laplacian.py       (scipy.special)

features/
    ├── csp.py             (mne.decoding)
    ├── chaos.py           (antropy)
    ├── bandpower.py       (scipy.signal)
    ├── jacobian_features.py (scipy.linalg)
    └── variable_selector.py (torch) [optional]

classification/
    ├── base.py            (joblib)
    ├── csp_lda.py         (mne, sklearn)
    ├── eegnet.py          (torch) [optional]
    ├── neural_sde.py      (torch) [optional]
    ├── adaptive_router.py (torch for gating) [optional]
    └── pipeline.py        (factory + RiemannianClassifier)

training/
    ├── paradigm.py        (pygame)
    ├── recorder.py        (numpy)
    ├── trainer.py         (sklearn)
    ├── pretrain.py        (torch) [optional]
    ├── advanced_pipeline.py (all above)
    └── uncertainty_weights.py (torch) [optional]

analysis/
    ├── erp.py             (numpy)
    ├── time_frequency.py  (scipy)
    ├── topography.py      (numpy)
    ├── state_monitor.py   (scipy)
    ├── koopman_decomposition.py (numpy)
    ├── ftle_analysis.py   (scipy)
    └── causal_channels.py (scipy.linalg)

adaptation/
    ├── errp_detector.py   (scipy.signal)
    ├── seal_engine.py     (numpy)
    └── gflownet_strategy.py (torch) [optional]

control/
    ├── mapping.py         (numpy)
    ├── cursor_control.py  (numpy)
    └── mouse.py           (pyautogui)
```

## Data Flow Formats

| Stage | Format | Shape | Location |
|-------|--------|-------|----------|
| Raw acquisition | numpy float64 | (32, N) | BrainFlow buffer |
| Saved recording | .npz (uncompressed) | data + events_json + sf + eeg_channels | data/raw/ |
| Preprocessed | numpy float64 | (n_ch, N) | In memory |
| Epochs | numpy float64 | (n_trials, n_ch, n_samples) | In memory |
| CSP features | numpy float64 | (n_trials, n_components) | In memory |
| Trained model | .pkl (joblib) or .pt (torch) | Serialized object | models/ |
| Label mapping | .labels.json | {"class_name": int_label} | models/ |
| Session log | .json | Stats dict | data/sessions/ |

## Configuration

All tunable parameters live in `config/settings.yaml`. No magic numbers in code.

Key sections:
- `board` — hardware settings (board_id, serial_port, channel_count)
- `preprocessing` — filter parameters (bandpass, notch, CAR)
- `features` — CSP components, chaos features, band power
- `classification` — model type + per-classifier hyperparameters
- `control` — cursor velocity, dead zone, click detection
- `training` — paradigm timing, trial counts, classes
- `adaptation` — SEAL update interval, ErrP thresholds, buffer sizes
- `advanced` — SOTA module toggles and parameters

## Threading Model

- **Main thread**: GUI (PyQt5) or control loop
- **BrainFlow thread**: Hardware data acquisition (C++ internal)
- **WorkerThread** (GUI only): Subprocess execution for scripts
- **EEGAcquisitionThread** (cursor control): Polls board data with Lock

All Python-side shared state is protected by `threading.Lock` or `deque(maxlen=N)`.
