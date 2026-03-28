---
title: EEGNetClassifier Class
tags: [class, classification, eegnet, deep-learning, pytorch]
aliases: [EEGNetClassifier, EEGNet]
---

# EEGNetClassifier

> [!info] File Location
> `src/classification/eegnet.py`

## Purpose

Compact convolutional neural network designed for EEG-based BCIs. Learns spatial and temporal features end-to-end from raw (bandpass-filtered) epochs using depthwise and separable convolutions. Implements the architecture from Lawhern et al. (2018).

## Class Diagram

```mermaid
classDiagram
    class BaseClassifier {
        <<abstract>>
    }

    class EEGNetClassifier {
        -_model: EEGNetModel
        -_training_history: list
        +n_channels: int
        +n_samples: int
        +n_classes: int
        +device: torch.device
        +F1: int
        +D: int
        +F2: int
        +kernel_length: int
        +dropout: float
        +epochs: int
        +batch_size: int
        +learning_rate: float
        +patience: int
        +fit(X, y) EEGNetClassifier
        +predict(X) ndarray
        +predict_proba(X) ndarray
        +decision_function(X) ndarray
        +predict_all(X) tuple
        +save(path)
        +load(path) EEGNetClassifier
    }

    class EEGNetModel {
        <<nn.Module>>
        +conv1: Conv2d
        +bn1: BatchNorm2d
        +depthwise: Conv2d
        +bn2: BatchNorm2d
        +pool1: AvgPool2d
        +separable_depthwise: Conv2d
        +separable_pointwise: Conv2d
        +bn3: BatchNorm2d
        +pool2: AvgPool2d
        +classifier: Linear
        +forward(x) Tensor
    }

    BaseClassifier <|-- EEGNetClassifier
    EEGNetClassifier --> EEGNetModel : wraps
```

## EEGNet Architecture (3 Blocks)

```mermaid
flowchart TD
    Input["Input\n(batch, 1, 16, 625)"] --> B1["Block 1: Temporal Conv\nConv2d(1, F1=8, 1x64)\nBatchNorm"]
    B1 --> B2["Block 2: Depthwise Spatial\nConv2d(8, 16, 16x1, groups=8)\nBatchNorm + ELU\nAvgPool(1,4) + Dropout"]
    B2 --> B3["Block 3: Separable Conv\nDepthwise Conv2d(16, 16, 1x16)\nPointwise Conv2d(16, 16, 1x1)\nBatchNorm + ELU\nAvgPool(1,8) + Dropout"]
    B3 --> Flat["Flatten\n(batch, F2 * n_samples/32)"]
    Flat --> FC["Linear\n(batch, n_classes)"]
    FC --> Out["Logits\n(batch, 5)"]
```

## Constructor

```python
EEGNetClassifier(
    n_channels=16, n_samples=625, n_classes=5, device="auto",
    F1=8, D=2, F2=16, kernel_length=64, dropout=0.5,
    epochs=300, batch_size=32, learning_rate=1e-3,
    weight_decay=1e-3, patience=50, validation_fraction=0.1,
)
```

## Training

- Optimizer: Adam with weight decay
- Loss: CrossEntropyLoss
- Early stopping: patience=50 epochs, restores best weights
- Max-norm constraint on depthwise conv weights (max_norm=1, per Lawhern 2018)
- 10% validation split for early stopping

## Save/Load

Uses PyTorch `state_dict` checkpoint (not joblib):

```python
# Saves architecture params + weights + training history
clf.save("models/eegnet.pt")

# Reconstructs model from checkpoint
clf = EEGNetClassifier.load("models/eegnet.pt", device="auto")
```

> [!warning] Data Requirements
> EEGNet has ~4,000 parameters and typically needs 100+ trials/class (500+ total) for reliable training. With the default 40 trials/class, it often underfits. CSP+LDA is recommended for small datasets. See [[Limitations]].

## References

> Lawhern, V. J., et al. (2018). "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces." Journal of Neural Engineering, 15(5), 056013.

## Related Pages

- [[Classification]] -- Module overview
- [[CSPLDAClassifier]] -- Recommended alternative for small datasets
- [[Configuration]] -- EEGNet config keys under `classification.eegnet`
- [[train_model]] -- Script that trains EEGNet
- [[Research Papers]] -- Lawhern et al. (2018)
