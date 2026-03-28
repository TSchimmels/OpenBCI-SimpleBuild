---
title: Real-Time Control Loop
tags: [flow, real-time, control-loop]
aliases: [Control Loop, Main Loop]
---

# Real-Time Control Loop

> [!info] Overview
> The core loop running at 16 Hz in [[run_eeg_cursor]]. Each iteration acquires the latest EEG window, preprocesses it identically to training, classifies, and updates the cursor position. Total latency per iteration: 10-50ms.

## Sequence Diagram

```mermaid
sequenceDiagram
    participant Acq as EEGAcquisitionThread
    participant Board as BoardManager
    participant Main as Main Loop (16 Hz)
    participant BP as bandpass_filter
    participant CAR as common_average_reference
    participant Clf as Classifier
    participant Ctrl as EEGCursorController
    participant Mouse as MouseController

    loop Every 20ms (background)
        Acq->>Board: get_data(625)
        Board-->>Acq: raw (32ch x 625)
        Acq->>Acq: Update rolling buffer
    end

    loop Every ~62.5ms (main thread)
        Main->>Acq: get_window()
        Acq-->>Main: raw_window (32ch x 625)
        Main->>Main: Extract EEG channels (16ch x 625)
        Main->>BP: bandpass_filter(eeg, sf, 8, 30, order=4, causal=False)
        BP-->>Main: filtered (16ch x 625)
        Main->>CAR: common_average_reference(filtered)
        CAR-->>Main: rereferenced (16ch x 625)
        Main->>Clf: predict_all(trial)
        Clf-->>Main: (labels, proba[5], decision)
        Main->>Ctrl: update(proba[0], class_names)
        Ctrl->>Ctrl: mi_to_direction(proba, names, map, 0.5)
        Ctrl->>Ctrl: Scale velocity by confidence
        Ctrl->>Ctrl: EMA smooth (alpha=0.3)
        alt |vx| > 0.5 or |vy| > 0.5
            Ctrl->>Mouse: move_relative(vx, vy)
        end
        Ctrl->>Ctrl: _check_click(proba, names)
        alt Sustained high-confidence >= 0.8s
            Ctrl->>Mouse: click()
        end
        Ctrl-->>Main: {direction, confidence, velocity, click_event}
        Main->>Main: Sleep remainder of 62.5ms interval
    end
```

## Timing Budget

| Stage | Typical Duration | Notes |
|-------|-----------------|-------|
| `get_window()` | < 1ms | Lock + array copy |
| Channel extraction | < 1ms | Numpy indexing |
| Bandpass filter | 2-5ms | 4th-order Butterworth, `sosfiltfilt` on 625 samples |
| CAR | < 1ms | Mean subtraction |
| Classification (CSP+LDA) | 1-3ms | CSP transform + LDA predict |
| Classification (EEGNet) | 5-20ms | PyTorch forward pass (CPU/GPU) |
| Cursor update | < 1ms | Velocity calc + pyautogui move |
| **Total per iteration** | **10-30ms** | Well within 62.5ms budget |

## Data Shape Through Pipeline

```mermaid
flowchart LR
    A["Board Ring Buffer\n(32 x 45000)"] -->|get_data| B["Raw Window\n(32 x 625)"]
    B -->|channel select| C["EEG Window\n(16 x 625)"]
    C -->|bandpass| D["Filtered\n(16 x 625)"]
    D -->|CAR| E["Clean\n(16 x 625)"]
    E -->|expand dim| F["Trial\n(1 x 16 x 625)"]
    F -->|predict_all| G["Proba (1 x 5)\nDecision (1 x 5)"]
    G -->|squeeze| H["Proba (5,)"]
    H -->|update| I["dx, dy pixels"]
```

## Graceful Shutdown

- `SIGINT` / `SIGTERM` sets `shutdown_event`
- Main loop checks `shutdown_event.is_set()` each iteration
- `shutdown_event.wait(timeout=sleep_time)` replaces `time.sleep()` for responsive exit
- Finally block: stop acquisition thread -> disconnect board -> print stats

## Related Pages

- [[run_eeg_cursor]] -- Script implementing this loop
- [[Architecture]] -- System-level view
- [[Preprocessing]] -- Filter details
- [[Classification]] -- Classifier types
- [[Control]] -- Cursor control state machine
- [[EEGCursorController]] -- Update method details
- [[Limitations]] -- 2.5s window lag, 16 Hz update rate constraints
