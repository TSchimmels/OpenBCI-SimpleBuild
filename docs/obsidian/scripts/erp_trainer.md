---
title: erp_trainer.py
tags: [script, training, erp, feedback]
aliases: [erp_trainer, ERP Trainer, Signal Trainer]
---

# erp_trainer.py

> [!info] File Location
> `scripts/erp_trainer.py`

## Purpose

Standalone tool for collecting motor imagery EEG data while providing real-time visual feedback of the subject's ERPs, ERDS% maps, and signal quality -- WITHOUT requiring model training. Helps subjects learn to produce consistent, detectable motor imagery signals.

## Two Modes

| Mode | Command | Purpose |
|------|---------|---------|
| **Collection** (default) | `python scripts/erp_trainer.py` | Run Graz paradigm + live ERP feedback |
| **Review** | `python scripts/erp_trainer.py --review data/raw/session.npz` | Offline analysis of a previous recording |

## ERP Display Dashboard

```mermaid
flowchart TD
    subgraph Dashboard["4-Panel Display (ERPDisplay)"]
        P1["Panel 1: ERP Waveforms\nC3, C4, Cz\nmean +/- std per class"]
        P2["Panel 2: ERDS% Spectrogram\ntime x frequency\nMorlet wavelet"]
        P3["Panel 3: Band Power\nmu [8-12] + beta [13-30]\ntimecourse"]
        P4["Panel 4: Scalp Topo Map\ninterpolated r2 or ERDS\nat selected timepoint"]
    end

    Trial["New Trial Epoch"] --> Acc["ERPAccumulator\nadd_epoch()"]
    Acc --> P1
    Trial --> ERDS["ERDSComputer\ncompute_erds()"]
    ERDS --> P2
    ERDS --> P3
    Acc -->|"signed r2"| Topo["TopoMapper\nplot()"]
    Topo --> P4
```

## Collection Mode Flow

```mermaid
sequenceDiagram
    participant User
    participant Script
    participant Board as BoardManager
    participant Paradigm as GrazParadigm (pygame)
    participant Display as ERPDisplay (matplotlib)
    participant Accum as ERPAccumulator

    Script->>Board: connect()
    Script->>Display: create dashboard
    Script->>Paradigm: setup

    loop For each trial
        Paradigm->>User: Show fixation + cue arrow
        Note over Board: Recording EEG continuously
        Paradigm->>Script: Trial complete (class_name)
        Script->>Board: get_data(epoch_samples)
        Script->>Script: bandpass + CAR filter
        Script->>Accum: add_epoch(filtered, class_name)
        Script->>Display: update all 4 panels
        Display-->>User: Real-time ERP feedback
    end

    Script->>Script: Save recording (.npz)
    Script->>Display: Show final summary
```

## Analysis Components Used

| Component | Class | Purpose |
|-----------|-------|---------|
| [[ERPAccumulator]] | `src.analysis.erp` | Running ERP averages, signed-r2, SNR |
| [[ERDSComputer]] | `src.analysis.time_frequency` | ERDS% spectrograms, band power timecourses |
| `TopoMapper` | `src.analysis.topography` | Scalp topographic maps |
| `bandpass_filter` | `src.preprocessing.filters` | 1-40 Hz broadband filtering |
| `common_average_reference` | `src.preprocessing.filters` | Spatial filtering |

## Usage

```bash
# Live collection with real-time feedback
python scripts/erp_trainer.py

# Review a previous recording
python scripts/erp_trainer.py --review data/raw/session_20260326.npz

# Custom classes and trial count
python scripts/erp_trainer.py --classes left_hand right_hand --n-trials 30 --debug
```

## Related Pages

- [[Training]] -- Module overview
- [[Analysis]] -- ERP and ERDS analysis tools
- [[ERPAccumulator]] -- Running ERP computation
- [[ERDSComputer]] -- Time-frequency decomposition
- [[ERP Analysis Pipeline]] -- Detailed flow
- [[collect_training_data]] -- Simpler alternative without feedback
- [[Channel Layout]] -- Electrode positions for topo maps
