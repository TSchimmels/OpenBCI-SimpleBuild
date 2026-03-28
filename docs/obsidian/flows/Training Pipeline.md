---
title: Training Pipeline
tags: [flow, training, pipeline]
aliases: [Training Flow, Offline Training]
---

# Training Pipeline

> [!info] Overview
> The complete offline training workflow from raw EEG recording to a saved, cross-validated classifier. Orchestrated by [[collect_training_data]] -> [[train_model]], with [[ModelTrainer]] handling the preprocessing and fitting.

## End-to-End Flow

```mermaid
flowchart TD
    subgraph "Step 1: Data Collection (collect_training_data.py)"
        S1A["Connect Board"] --> S1B["Start Recording"]
        S1B --> S1C["Run Graz Paradigm\n200 trials (40/class x 5)"]
        S1C --> S1D["Stop Recording"]
        S1D --> S1E["Save .npz\ndata + events + metadata"]
    end

    subgraph "Step 2: Model Training (train_model.py)"
        S2A["Load .npz Session"] --> S2B["Extract Epochs\ntmin=1.5s, tmax=4.0s\nrelative to cue onset"]
        S2B --> S2C["Bandpass Filter\n8-30 Hz, order 4\nzero-phase (sosfiltfilt)"]
        S2C --> S2D["Common Average Reference\nsubtract channel mean"]
        S2D --> S2E["Artifact Rejection\npeak-to-peak > 100 uV\nreject noisy epochs"]
        S2E --> S2F{"Clean epochs\navailable?"}
        S2F -->|No| S2G["ERROR: All rejected"]
        S2F -->|Yes| S2H["Cross-Validation\n10-fold stratified"]
        S2H --> S2I["Train on Full Dataset\nClassifierFactory.create()\nclassifier.fit(X, y)"]
        S2I --> S2J["Save Model\n.pkl + .labels.json"]
    end

    S1E --> S2A
```

## Epoch Extraction Detail

```mermaid
flowchart LR
    Continuous["Continuous Recording\n(16 ch x ~75000 samples)\n~5 min at 250 Hz"] --> Events["Event Markers\n200 events with\nsample_index"]
    Events --> Window["For each event:\nstart = sample_index + 1.5*sf\nend = sample_index + 4.0*sf"]
    Window --> Epochs["Extracted Epochs\n(200 x 16 x 625)\n2.5s windows"]
```

## Artifact Rejection Statistics

Typical session with good electrode contact:

| Metric | Value |
|--------|-------|
| Total epochs extracted | 200 |
| Rejected (ptp > 100 uV) | 10-30 (5-15%) |
| Clean epochs for training | 170-190 |
| Threshold | 100 uV (configurable) |

> [!warning] All-Rejection Scenario
> If ALL epochs are rejected, the script exits with an error. This indicates either: bad electrode impedances, threshold too low, or excessive artifacts. Increase `preprocessing.artifact_threshold_uv` or fix electrode contact.

## Cross-Validation

- **Method**: Stratified K-fold (preserves class ratios in each fold)
- **Default K**: 10 (reduced if smallest class has fewer samples)
- **Estimator**: The full sklearn Pipeline (CSP + LDA) is re-fitted on each fold's training set, ensuring no data leakage from CSP spatial filters
- **Metrics reported**: Mean accuracy, std accuracy, chance level (1/n_classes)

## Related Pages

- [[collect_training_data]] -- Step 1: record calibration data
- [[train_model]] -- Step 2: train and save model
- [[Training]] -- Module overview
- [[Preprocessing]] -- Filters applied during prepare_data()
- [[Classification]] -- ClassifierFactory creates the classifier
- [[run_eeg_cursor]] -- Step 3: use the model for cursor control
- [[Configuration]] -- Training, preprocessing, and classification keys
