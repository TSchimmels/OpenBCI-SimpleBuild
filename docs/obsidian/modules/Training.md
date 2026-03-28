---
title: Training Module
tags: [module, training, calibration, paradigm]
aliases: [Calibration Module]
---

# Training Module

> [!info] Purpose
> Handles the full offline training pipeline: visual cue presentation (Graz paradigm), EEG data recording with event markers, epoch extraction, preprocessing, artifact rejection, classifier fitting, and cross-validated evaluation.

## Files

- `src/training/paradigm.py` -- `GrazParadigm` (visual cueing protocol)
- `src/training/recorder.py` -- `DataRecorder` (EEG + event recording)
- `src/training/trainer.py` -- `ModelTrainer` (preprocessing + fitting + evaluation)

## Calibration Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant Script as collect_training_data.py
    participant Board as BoardManager
    participant Rec as DataRecorder
    participant Para as GrazParadigm
    participant Pygame

    Script->>Board: connect()
    Script->>Rec: DataRecorder(board)
    Script->>Para: GrazParadigm(config)
    Script->>Rec: start()
    Rec->>Board: get_board_data() [flush buffer]

    loop For each trial (200 total)
        Para->>Pygame: draw fixation cross (2s)
        Para->>Pygame: play beep + draw arrow
        Para->>Rec: add_event("left_hand")
        Note over Rec: timestamp + sample_index
        Para->>Pygame: show cue (1.25s + 4.0s imagery)
        Para->>Pygame: blank rest (1.5-3.0s random)
    end

    Script->>Rec: stop()
    Rec->>Board: get_board_data() [all accumulated data]
    Rec-->>Script: (raw_data, events)
    Script->>Rec: save("data/raw/session_YYYYMMDD.npz")
```

## Training Pipeline (train_model.py)

```mermaid
flowchart TD
    Load["Load .npz session\nraw_data + events"] --> Extract["Extract Epochs\ntmin=1.5s, tmax=4.0s"]
    Extract --> BP["Bandpass Filter\n8-30 Hz (zero-phase)"]
    BP --> CAR["Common Average Reference"]
    CAR --> Reject["Artifact Rejection\npeak-to-peak > 100 uV"]
    Reject --> CV["Cross-Validation\n10-fold stratified"]
    CV --> Train["Train on Full Dataset"]
    Train --> Save["Save Model\nmodels/csp_lda_YYYYMMDD.pkl"]
    Save --> Labels["Save Label Map\n.labels.json"]
```

## GrazParadigm Trial Structure

| Phase | Duration | Visual | Audio |
|-------|----------|--------|-------|
| Fixation | 2.0s | White `+` on black | None |
| Cue onset | instant | Arrow appears | 1000 Hz beep (70ms) |
| Cue + imagery | 5.25s (1.25 + 4.0) | Arrow sustained | None |
| Rest | 1.5-3.0s (random) | Blank screen | None |

- Trials are pseudo-randomized in blocks (each class appears once per block, shuffled)
- 40 trials/class x 5 classes = 200 total trials
- 2 runs with break screen between them
- ESC aborts at any time

## ModelTrainer Pipeline

1. `prepare_data()` -- epochs -> bandpass -> CAR -> artifact rejection
2. `train()` -- fits classifier, returns training accuracy
3. `cross_validate()` -- stratified K-fold, reports mean/std accuracy and chance level
4. `evaluate()` -- accuracy, confusion matrix, classification report, Cohen's kappa

## Related Pages

- [[Acquisition]] -- [[BoardManager]] provides data to [[DataRecorder]]
- [[Preprocessing]] -- Filters applied during `prepare_data()`
- [[Classification]] -- [[ClassifierFactory]] creates the classifier for training
- [[collect_training_data]] -- Script that runs the full calibration
- [[train_model]] -- Script that runs the training pipeline
- [[erp_trainer]] -- Alternative with real-time ERP feedback
- [[Training Pipeline]] -- Detailed flow diagram
- [[Configuration]] -- Training config keys
