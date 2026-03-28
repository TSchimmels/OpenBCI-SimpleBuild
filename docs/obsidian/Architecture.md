---
title: System Architecture
tags: [architecture, overview]
aliases: [Architecture Overview, System Design]
---

# System Architecture

> [!info] OpenBCI SimpleBuild
> A pure-EEG motor imagery brain-computer interface that translates 5-class imagined movement into 4-directional cursor control with click via sustained imagery. Built on OpenBCI Cyton+Daisy (16 channels) and BrainFlow.

## High-Level Data Flow

```mermaid
flowchart LR
    Board["Board\n(BrainFlow)"] -->|raw EEG| Preproc["Preprocessing\n(filters + CAR)"]
    Preproc -->|filtered EEG| Features["Features\n(CSP / bandpower / chaos)"]
    Features -->|feature vector| Classify["Classification\n(LDA / EEGNet / MDM)"]
    Classify -->|probabilities + scores| Control["Control\n(velocity map + smoothing)"]
    Control -->|dx, dy| Cursor["Cursor\n(pyautogui)"]
```

## Real-Time Pipeline Detail

```mermaid
flowchart TD
    subgraph Acquisition
        A1[BoardManager] -->|"get_data(n)"| A2[Ring Buffer]
        A2 --> A3[EEGAcquisitionThread]
    end
    subgraph Preprocessing
        A3 -->|"raw (16ch x 625 samples)"| P1[Bandpass 8-30 Hz]
        P1 --> P2[Common Average Reference]
    end
    subgraph Classification
        P2 -->|"filtered (16ch x 625 samples)"| C1{ClassifierFactory}
        C1 --> C2[CSPLDAClassifier]
        C1 --> C3[EEGNetClassifier]
        C1 --> C4[RiemannianClassifier]
        C2 --> C5[predict_all]
        C3 --> C5
        C4 --> C5
    end
    subgraph Control
        C5 -->|"proba (5,)"| D1[EEGCursorController.update]
        D1 --> D2[mi_to_direction]
        D2 --> D3[EMA Smoothing]
        D3 --> D4[MouseController.move_relative]
        D1 --> D5[_check_click]
        D5 -->|sustained| D6[MouseController.click]
    end
```

## Module Hierarchy

```mermaid
classDiagram
    direction TB

    class src {
        config.py
    }

    class acquisition {
        board.py
    }

    class preprocessing {
        filters.py
        artifacts.py
    }

    class features {
        csp.py
        chaos.py
        bandpower.py
    }

    class classification {
        base.py
        csp_lda.py
        eegnet.py
        pipeline.py
    }

    class control {
        mouse.py
        mapping.py
        cursor_control.py
    }

    class training {
        paradigm.py
        recorder.py
        trainer.py
    }

    class analysis {
        erp.py
        time_frequency.py
        topography.py
    }

    src --> acquisition
    src --> preprocessing
    src --> features
    src --> classification
    src --> control
    src --> training
    src --> analysis

    acquisition <.. preprocessing : feeds data
    preprocessing <.. features : filtered epochs
    features <.. classification : feature vectors
    classification <.. control : predictions
    acquisition <.. training : records data
    preprocessing <.. training : filters epochs
    classification <.. training : fits models
    acquisition <.. analysis : raw epochs
    preprocessing <.. analysis : filtered epochs
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure EEG (no eye tracking) | Simpler, more accessible, demonstrates raw MI capability |
| 5 classes (rest + 4 directions) | Matches the 4-directional paradigm of BCI Competition IV 2a |
| CSP+LDA as default | Works well with small training sets (40 trials/class) |
| 2.5s classification window | 1.5-4.0s post-cue captures the strongest ERD/ERS |
| 16 Hz update rate | Balance between responsiveness and classification stability |
| Sustained imagery for click | No additional hardware needed; uses existing MI pipeline |

## Related Pages

- [[Acquisition]] -- Board connection and data streaming
- [[Preprocessing]] -- Filter chain details
- [[Features]] -- Feature extraction methods
- [[Classification]] -- Classifier implementations
- [[Control]] -- Cursor movement and click logic
- [[Training]] -- Calibration and model fitting
- [[Analysis]] -- ERP and ERDS analysis tools
- [[Configuration]] -- All tunable parameters
- [[Real-Time Control Loop]] -- Sequence diagram of the main loop
- [[Signal Processing Chain]] -- Detailed filter specifications
