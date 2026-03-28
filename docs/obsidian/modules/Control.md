---
title: Control Module
tags: [module, control, cursor, mouse, click]
aliases: [Cursor Control Module]
---

# Control Module

> [!info] Purpose
> Translates classifier output into physical cursor movement and click actions. Three layers: `MouseController` (low-level pyautogui), `ControlMapper` (signal processing), and `EEGCursorController` (state machine orchestrator).

## Files

- `src/control/mouse.py` -- `MouseController` (pyautogui wrapper)
- `src/control/mapping.py` -- [[ControlMapper]] (normalize, smooth, velocity map)
- `src/control/cursor_control.py` -- [[EEGCursorController]] (state machine)

## Control Stack

```mermaid
flowchart TD
    Clf["Classifier\npredict_proba() -> (5,)"] --> ECC["EEGCursorController.update()"]
    ECC --> MTD["mi_to_direction()\nproba -> direction + confidence"]
    MTD --> Vel["Velocity Scaling\n(confidence - threshold) * max_vel"]
    Vel --> EMA["EMA Smoothing\nalpha=0.3"]
    EMA --> MR["MouseController.move_relative(dx, dy)"]
    ECC --> Click["_check_click()\nsustained imagery detection"]
    Click -->|"hold > 0.8s at p > 0.7"| MC["MouseController.click()"]
```

## Click Detection State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Tracking: directional class p >= 0.7
    Tracking --> Tracking: same class sustained
    Tracking --> Idle: class changes or p drops
    Tracking --> ClickFired: elapsed >= 0.8s AND cooldown passed
    ClickFired --> CheckDouble: within 1.5s of first click?
    CheckDouble --> DoubleClick: yes, 2nd click in window
    CheckDouble --> SingleClick: no, start new window
    SingleClick --> Idle
    DoubleClick --> Idle
```

## Direction Mapping

| MI Class | Direction | Velocity Axis |
|----------|-----------|---------------|
| `left_hand` | Left | dx = -speed |
| `right_hand` | Right | dx = +speed |
| `feet` | Down | dy = +speed |
| `tongue` | Up | dy = -speed |
| `rest` | None | dx=0, dy=0 |

## ControlMapper Processing Pipeline

1. **Normalize** (Welford's online algorithm) -- z-score with running mean/variance, clip to [-1, 1]
2. **Smooth** (EMA) -- `result = (1-alpha)*prev + alpha*value`
3. **Velocity Map** -- Dead zone (0.15) + linear scaling to `max_velocity` (25 px/frame)

## Key Parameters

| Parameter | Default | Config Key | Effect |
|-----------|---------|------------|--------|
| Dead zone | 0.15 | `control.dead_zone` | Minimum signal to register as movement |
| Max velocity | 25 px/frame | `control.max_velocity` | At 16 Hz = 400 px/sec max |
| Smoothing alpha | 0.3 | `control.smoothing_alpha` | Lower = smoother, slower response |
| Confidence threshold | 0.5 | `control.confidence_threshold` | Min probability for movement |
| Click hold duration | 0.8s | `control.click.hold_duration_s` | Time of sustained imagery for click |
| Click confidence | 0.7 | `control.click.confidence_threshold` | Higher bar for click vs movement |
| Click cooldown | 0.5s | `control.click.cooldown_s` | Min time between successive clicks |

## Related Pages

- [[Classification]] -- Provides probabilities and decision scores
- [[EEGCursorController]] -- Full class reference
- [[ControlMapper]] -- Full class reference
- [[Real-Time Control Loop]] -- Sequence diagram including control
- [[Configuration]] -- All control config keys
- [[Limitations]] -- Click auto-repeat, sub-pixel truncation, no diagonal movement
