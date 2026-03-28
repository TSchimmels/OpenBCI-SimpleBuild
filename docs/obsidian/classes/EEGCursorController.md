---
title: EEGCursorController Class
tags: [class, control, cursor, state-machine]
aliases: [EEGCursorController, Cursor Controller]
---

# EEGCursorController

> [!info] File Location
> `src/control/cursor_control.py`

## Purpose

State machine that combines 5-class motor imagery classification output with velocity mapping, EMA smoothing, and sustained-imagery click detection into a single `update()` call per control loop iteration (~16 Hz).

## Class Diagram

```mermaid
classDiagram
    class EEGCursorController {
        -_mouse: MouseController
        -_mapper: ControlMapper
        -_direction_map: Dict
        -_move_threshold: float
        -_click_hold_duration: float
        -_click_threshold: float
        -_sustained_class: Optional[str]
        -_sustained_start: float
        -_vx: float
        -_vy: float
        +total_movements: int
        +total_clicks: int
        +__init__(config: Dict)
        +update(proba, class_names) Dict
        +reset()
        +position: Tuple[int, int]
        +screen_size: Tuple[int, int]
    }

    class MouseController {
        +move_to(x, y)
        +move_relative(dx, dy)
        +click(button)
        +double_click()
        +get_position() Tuple
        +get_screen_size() Tuple
    }

    class ControlMapper {
        +process(raw) float
        +mi_to_direction(proba, names, map, thresh) tuple
        +mi_to_command(proba, names, thresh) str
        +normalize(raw) float
        +smooth(value) float
        +velocity_map(signal) float
    }

    EEGCursorController --> MouseController : cursor actions
    EEGCursorController --> ControlMapper : signal processing
```

## Constructor

```python
EEGCursorController(config: Dict) -> None
```

Reads from `config["control"]` and `config["control"]["click"]`.

## update() Method

```python
def update(self, class_probabilities: ndarray, class_names: List[str]) -> Dict
```

Called once per control loop iteration. Returns:

```python
{
    "direction": "left" | "right" | "up" | "down" | None,
    "confidence": 0.0-1.0,
    "velocity": (dx, dy),
    "click_event": "click" | "double_click" | None,
    "predicted_class": "left_hand" | "right_hand" | ... ,
}
```

## Update Flow

```mermaid
flowchart TD
    P["class_probabilities (5,)"] --> D["mi_to_direction()"]
    D --> V{"direction?"}
    V -->|Yes| Scale["Scale by confidence\nspeed = (p - thresh) / (1 - thresh) * max_vel"]
    V -->|No| Zero["dx=0, dy=0"]
    Scale --> EMA["EMA: vx = 0.7*vx + 0.3*dx"]
    Zero --> EMA
    EMA --> Move{"|vx| > 0.5 or |vy| > 0.5?"}
    Move -->|Yes| Mouse["mouse.move_relative(vx, vy)"]
    Move -->|No| Skip["No movement"]
    P --> Click["_check_click()"]
    Click --> Ret["Return status dict"]
```

## Statistics

| Property | Type | Description |
|----------|------|-------------|
| `total_movements` | `int` | Count of frames where cursor actually moved |
| `total_clicks` | `int` | Total clicks fired (single + double) |
| `position` | `(int, int)` | Current cursor pixel position |
| `screen_size` | `(int, int)` | Screen resolution |

## Related Pages

- [[Control]] -- Module overview with click state machine diagram
- [[ControlMapper]] -- Signal processing component
- [[Classification]] -- Provides input probabilities
- [[run_eeg_cursor]] -- Script that creates and drives this controller
- [[Real-Time Control Loop]] -- Sequence diagram
- [[Configuration]] -- Control config keys
