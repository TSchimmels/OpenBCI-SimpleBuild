# Limitations — Complete Technical Documentation

> Every BCI system has hard boundaries. This document maps them honestly
> so users, researchers, and developers know exactly what to expect
> and where the ceilings are.

---

## Table of Contents

1. [Hardware Limitations](#1-hardware-limitations)
2. [Signal Processing Limitations](#2-signal-processing-limitations)
3. [Classification Limitations](#3-classification-limitations)
4. [Real-Time Control Limitations](#4-real-time-control-limitations)
5. [Training & Calibration Limitations](#5-training--calibration-limitations)
6. [ERP Trainer Limitations](#6-erp-trainer-limitations)
7. [Software Implementation Limitations](#7-software-implementation-limitations)
8. [Usability & Human Factors](#8-usability--human-factors)
9. [What This System Cannot Do](#9-what-this-system-cannot-do)
10. [Comparison to Research-Grade Systems](#10-comparison-to-research-grade-systems)

---

## 1. Hardware Limitations

### 1.1 Channel Count (16 channels)

The OpenBCI Cyton+Daisy provides 16 EEG channels. Research-grade motor
imagery BCIs typically use **64 to 256 channels**.

**Impact:**
- Fewer spatial samples means lower spatial resolution
- CSP spatial filters have less information to work with (max 16 components vs 64+)
- Cannot resolve fine-grained cortical patterns that distinguish subtle MI tasks
- Topographic maps have only 16 data points to interpolate from — spatial precision
  is coarse compared to high-density arrays

**Mitigation:** 16 channels covering the standard 10-20 motor cortex positions
(C3, C4, Cz, F3, F4, P3, P4) is sufficient for the 4-direction MI paradigm.
The BCI Competition IV Dataset 2a achieved ~70% 4-class accuracy with 22 channels.

### 1.2 Sampling Rate (125 Hz for Cyton+Daisy)

The Cyton+Daisy combined sampling rate is **125 Hz**, giving a Nyquist limit of
**62.5 Hz**. The synthetic board runs at 250 Hz.

**Impact:**
- Cannot analyze high-gamma activity (70-150 Hz), which carries useful motor
  information in some subjects
- Frequency resolution of the mu band (8-12 Hz) is adequate but not fine-grained
- Time-frequency analysis above 40 Hz is unreliable (bandpass capped at 40 Hz)

**Mitigation:** The primary MI signals (mu: 8-12 Hz, beta: 13-30 Hz) are well
within the 62.5 Hz Nyquist limit. High-gamma is primarily useful with invasive
(ECoG) recordings, not scalp EEG.

### 1.3 Amplifier Noise Floor

The OpenBCI Cyton uses the TI ADS1299 ADC (24-bit, 0.5 µVpp input-referred
noise). While this is excellent for a consumer device, it is noisier than
clinical amplifiers (g.USBamp: 0.1 µVpp).

**Impact:**
- Lower signal-to-noise ratio (SNR) means more trials needed for reliable ERPs
- Single-trial classification accuracy is lower than clinical-grade systems
- Subjects with weak mu modulation may produce signals below the noise floor

### 1.4 Event Timing Precision

Events (cue onset markers) are timestamped using **wall-clock time** (`time.time()`),
not hardware triggers.

**Impact:**
- Timing jitter of **1-10 milliseconds** between the visual cue appearing on
  screen and the event marker being recorded
- For ERP analysis: this jitter smears the averaged waveform by a few ms
- For MI-BCI: negligible impact (MI signals evolve over hundreds of ms)

**Mitigation:** For millisecond-precise ERP research, use a parallel port or
TTL trigger synchronized to the display refresh. This system is designed for
MI-BCI, where 10ms jitter is irrelevant.

### 1.5 USB Dongle Latency

The OpenBCI USB dongle communicates over serial at 115200 baud. The BrainFlow
ring buffer adds an additional software buffering layer.

**Impact:**
- End-to-end latency from electrode to Python: **~30-80 ms**
- Combined with classification latency (~10-50ms), total thought-to-action
  delay is **80-200 ms**

---

## 2. Signal Processing Limitations

### 2.1 Zero-Phase Filter Edge Effects

The bandpass filter uses `sosfiltfilt` (zero-phase, non-causal), which applies
the filter forward and backward. On a finite window, this creates **transient
artifacts at both edges**.

**Impact:**
- For a 4th-order Butterworth (effective 8th-order after double-pass), the
  transient is approximately `3 × (filter_order / sampling_rate)` = ~96 ms
  at each edge of a 2.5-second window
- The first and last ~100 ms of each classification window contain unreliable
  data

**Mitigation:** The classification window [1.5s, 4.0s] post-cue avoids the
cue onset transient. The window edges are the post-imagery period where the
signal is less critical. Edge effects are tolerable for the 2.5s window length.

### 2.2 Common Average Reference Assumes Uniform Noise

CAR subtracts the mean across all 16 channels at each timepoint. This assumes
all channels contribute equally to the common noise.

**Impact:**
- A single **bad channel** (high impedance, broken electrode, muscle artifact)
  will corrupt the reference for ALL channels
- If a channel contains NaN values, `np.mean()` propagates NaN to every channel
- CAR removes any truly global signal component (e.g., if all motor cortex
  channels synchronize simultaneously, CAR partially cancels it)

**Mitigation:**
- Bad channel detection (`detect_bad_channels()`) is available but not
  automatically applied in the real-time pipeline
- The `car_enabled` config flag exists but is currently dead code — CAR is
  always applied unconditionally
- Future improvement: run bad channel detection before CAR and exclude outliers

### 2.3 Fixed Frequency Band (8-30 Hz)

The MI-specific bandpass is hardcoded to 8-30 Hz (`mi_bandpass_low`,
`mi_bandpass_high`).

**Impact:**
- Subjects whose mu rhythm is centered at 7 Hz or 13 Hz (outside the standard
  8-12 Hz) will have their primary signal attenuated by the filter edges
- The beta band upper limit of 30 Hz excludes the high-beta range (30-40 Hz)
  which some subjects use for MI
- No per-subject frequency optimization

**Mitigation:** The config values are tunable in `settings.yaml`. The ERP
trainer's ERDS% spectrogram shows the subject's actual frequency profile,
allowing manual adjustment of the bandpass limits.

### 2.4 No Artifact Rejection in Real-Time

Artifact rejection (`reject_epochs()`) is applied during offline training
but **NOT during real-time cursor control**.

**Impact:**
- Eye blinks, jaw clenches, head movements, and EMG bursts during live use
  corrupt the classification input
- The classifier will produce random outputs during artifact-contaminated windows
- No mechanism to detect "this window is garbage, skip it"

**Mitigation:** The EMA smoothing on cursor velocity dampens spurious movements
from occasional noisy classifications. The confidence threshold (0.5) also
filters out low-confidence (likely artifact-driven) predictions.

### 2.5 No Independent Component Analysis (ICA)

ICA can separate EEG into independent neural and artifactual sources (e.g.,
removing eye blink components while preserving motor cortex signals). This
system does not use ICA.

**Impact:**
- Systematic artifacts (e.g., a subject who blinks every 3 seconds) are not
  removed and will contaminate both training data and real-time classification
- Subjects must maintain good artifact discipline during recording

---

## 3. Classification Limitations

### 3.1 Five-Class Accuracy Ceiling

Five-class motor imagery classification is at the **practical limit** of what
non-invasive BCI can achieve with 16 channels.

**Realistic accuracy expectations:**

| Metric | Value | Meaning |
|--------|-------|---------|
| Chance level | 20% | Random guessing |
| Typical untrained subject | 25-35% | Slightly above chance |
| After 2-3 training sessions | 35-55% | Usable with patience |
| Skilled BCI user (10+ sessions) | 50-70% | Comfortable control |
| Competition-winning systems (22ch) | 60-80% | Optimized, per-subject |

**Impact:**
- Even a "good" accuracy of 60% means **40% of cursor commands are wrong**
- The cursor will frequently move in unintended directions
- Click detection (sustained imagery) amplifies errors — a misclassification
  during a hold attempt triggers a wrong-direction click

**Context:** The BCI Competition IV Dataset 2a (4-class MI, 22 channels, 9
subjects) reported mean accuracies of 65-75% with optimized algorithms. Our
5-class system with 16 channels should expect 5-15% lower accuracy.

### 3.2 No Online Adaptation

The classifier is trained once and deployed frozen. It does not adapt to
the subject's changing brain state during a session.

**Impact:**
- Brain signals drift over time (electrode gel dries, attention fluctuates,
  fatigue accumulates). A classifier trained on data from minute 1-25 may
  perform worse by minute 45
- Session-to-session variability is even larger — a model trained on Tuesday
  may work poorly on Wednesday

**Mitigation:**
- Riemannian MDM is inherently more robust to non-stationarity than CSP+LDA
- Recalibrating (collecting new training data) before each session helps
- Future improvement: implement online adaptation (e.g., running mean update
  to covariance matrices)

### 3.3 No Transfer Learning

Each subject must undergo a full calibration session (200+ trials, ~25 minutes)
before the system works for them.

**Impact:**
- Cannot use pre-trained models from other subjects
- High barrier to first use — 25 minutes of calibration before any cursor control
- Calibration fatigue: subjects perform worse in later trials, degrading
  training data quality

### 3.4 CSP Assumptions

CSP assumes that motor imagery classes differ primarily in **band power variance**
across spatial locations. This is generally true for left/right hand imagery but
less reliable for feet and tongue.

**Impact:**
- Feet imagery produces bilateral activity on the midline (Cz), which is harder
  to separate from rest than lateralized hand imagery
- Tongue imagery activates frontal areas that overlap with eye movement artifacts
- Discriminability: left_hand vs right_hand > feet vs tongue > feet vs rest

### 3.5 EEGNet Sample Size Requirements

EEGNet (deep learning) requires substantially more training data than CSP+LDA.

**Impact:**
- With the default 40 trials/class = 200 total trials, EEGNet typically
  underfits (too few samples for the ~4,000 parameters)
- Recommended minimum for EEGNet: 100+ trials/class = 500+ total trials,
  requiring ~60 minutes of calibration
- CSP+LDA works well with 40 trials/class and should be the default choice

---

## 4. Real-Time Control Limitations

### 4.1 Classification Latency (2.5-Second Window)

Each classification uses a **2.5-second window** of EEG data (from 1.5s to
4.0s post-cue in training, matching the real-time window).

**Impact:**
- The cursor's response reflects brain activity from **1-3 seconds ago**, not
  the current thought
- Rapid direction changes are impossible — switching from "think left" to
  "think right" takes at least 2.5 seconds to register
- This is a fundamental limitation of MI-based BCI, not a software bug

**Context:** Eye-tracking systems respond in ~50ms. EMG-based systems respond
in ~100ms. EEG MI-BCI is inherently 10-50x slower.

### 4.2 Update Rate (16 Hz)

The control loop runs at 16 Hz = one classification every 62.5 ms.

**Impact:**
- At maximum velocity (25 px/frame × 16 fps), the cursor moves **400 pixels/second**
- On a 1920px-wide screen, crossing the full width takes ~4.8 seconds at max speed
- Combined with the 2.5s window lag, moving the cursor from one side of the screen
  to the other requires ~7 seconds of sustained imagery

### 4.3 Sub-Pixel Velocity Truncation

The `MouseController.move_relative()` uses `int()` truncation, not rounding.

**Impact:**
- Velocities between 0.5 and 1.0 pixels/frame pass the threshold check in
  `EEGCursorController` but produce **zero actual pixel movement** after `int()` 
  truncation
- Very low-confidence classifications (near threshold) may show "active" status
  in the log but produce no visible cursor movement
- No sub-pixel accumulation across frames — fractional remainder is lost each frame

**Workaround:** Increase `max_velocity` or lower `confidence_threshold` to ensure
velocities consistently exceed 1.0 px/frame when movement is intended.

### 4.4 Click Auto-Repeat

When a directional class is sustained above the click confidence threshold (0.7)
for longer than `hold_duration_s` (0.8s), a click fires. The sustained state then
resets and **restarts the timer**, causing another click after another 0.8s.

**Impact:**
- A 5-second sustained left-hand imagery at high confidence produces approximately
  **5 clicks** (one every 0.8s + 0.5s cooldown ≈ 1.3s per click)
- There is no "single click then stop" behavior — the user must actively switch
  to rest or another class to prevent repeated clicks

**Mitigation:** The 0.5s cooldown between clicks prevents machine-gun rapid fire.
For single-click behavior, the user should briefly relax after feeling the click
trigger.

### 4.5 No Diagonal Movement

The system maps each MI class to a single axis direction. There is no
simultaneous two-class classification for diagonal movement.

**Impact:**
- Moving to a target at 45 degrees requires alternating between horizontal
  and vertical imagery, producing a staircase trajectory
- This approximately doubles the time to reach diagonal targets compared to
  a direct-path controller

---

## 5. Training & Calibration Limitations

### 5.1 Approximate Event Timestamps

Event markers use `time.time()` (wall clock), not hardware-synchronized triggers.

**Impact:**
- Timing jitter of **1-10 ms** between actual cue display and recorded event
- Over a 25-minute session, clock drift could accumulate to **tens of milliseconds**
- For MI-BCI (signals evolve over 500ms+), this is negligible
- For millisecond-precise ERP studies, this timing is insufficient

### 5.2 No Feedback During Calibration

The standard Graz paradigm (`collect_training_data.py`) shows only arrow cues —
it does NOT show the subject their brain signals or classification results.

**Impact:**
- Subjects cannot tell if they are producing good imagery or poor imagery
- No reinforcement learning signal — the subject may develop ineffective
  strategies without knowing

**Mitigation:** Use the **ERP trainer** (`scripts/erp_trainer.py`) instead, which
provides real-time ERDS% and ERP feedback after each trial.

### 5.3 Label Encoding Determinism

Labels are encoded as integers using `sorted(unique_labels)`, producing
alphabetical ordering. The label map is saved alongside the model.

**Impact:**
- If a class name changes (e.g., "feet" → "foot"), the entire model must be
  retrained — you cannot rename classes post-training
- If a session has missing classes (e.g., subject only practiced 3 of 5), the
  label encoding will differ from a full-class session
- The `.labels.json` file MUST accompany the `.pkl` model file — losing it
  makes the model unusable (predictions map to wrong classes)

### 5.4 No Cross-Session Model Persistence

Models are specific to a single subject AND a single recording session.

**Impact:**
- A model trained on Monday's data may perform significantly worse on Tuesday
  due to: electrode placement variation, impedance changes, subject state
  differences, and cortical plasticity
- Best practice: recalibrate at the start of each session

---

## 6. ERP Trainer Limitations

### 6.1 Single-Channel ERDS Display

The time-frequency spectrogram shows only one channel at a time (default: C3).

**Impact:**
- Cannot simultaneously compare left-hemisphere and right-hemisphere ERDS
- The subject must manually switch channels to see the full spatial picture

### 6.2 Epoch Extraction in Collection Mode

The ERP trainer extracts epochs by reading the most recent `n_epoch_samples`
from the BrainFlow ring buffer after each trial.

**Impact:**
- If the trial timing varies (pygame event loop latency, system load), the
  extracted epoch may not perfectly align with the cue onset
- Unlike the offline `DataRecorder.extract_epochs()` which uses precise sample
  indices, the live extraction uses an approximate buffer tail

### 6.3 No Automatic Artifact Rejection

The ERP trainer displays all epochs, including artifact-contaminated ones.

**Impact:**
- Blink artifacts show as large spikes in the ERP waveforms
- ERDS% maps can be dominated by a single noisy trial
- The running average improves over many trials but early estimates are fragile

---

## 7. Software Implementation Limitations

### 7.1 Dead Configuration Keys

Several `settings.yaml` keys are declared but never read by any code:

| Key | Purpose | Status |
|-----|---------|--------|
| `control.mode` | Control mode selector | Hardcoded to pure_eeg |
| `control.click.method` | Click detection method selector | Hardcoded to sustained_mi |
| `training.paradigm` | Paradigm type selector | Hardcoded to Graz |
| `preprocessing.car_enabled` | Toggle CAR on/off | CAR always applied |
| `preprocessing.laplacian_enabled` | Toggle Laplacian reference | Never applied |
| `features.chaos_enabled` | Toggle chaos feature extraction | Never checked |
| `features.bandpower_enabled` | Toggle band power extraction | Never checked |
| `ui.signal_window_s` | Live signal display window | GUI does not read this |
| `ui.plot_update_ms` | Plot refresh rate | GUI does not read this |
| `ui.show_fft` | Toggle FFT display | GUI does not read this |
| `ui.show_classifier_output` | Toggle classifier output | GUI does not read this |
| `ui.feedback_type` | Feedback visualization type | GUI does not read this |

These keys exist for future extensibility but currently have no effect.

### 7.2 Chaos/Bandpower Features Computed But Unused

The `test_synthetic.py` script computes chaos features (Hjorth, entropy, fractal
dimensions) and band power features, verifying they work. However, **no classifier
actually uses these features**.

**Impact:**
- CSP+LDA uses only CSP log-variance features (12 components)
- EEGNet learns its own features end-to-end from raw (filtered) data
- Riemannian MDM uses covariance matrices directly
- The chaos and bandpower extractors exist and work but are not part of any
  classification pipeline

**Future use:** A hybrid classifier combining CSP + chaos + bandpower features
could potentially outperform CSP-only, but this requires a different pipeline
architecture (feature concatenation → classifier).

### 7.3 pyautogui X11 Dependency

`MouseController` imports `pyautogui` at module load time, which requires a
running X11 or Wayland display server.

**Impact:**
- Cannot import `src.control.mouse` in a headless environment (CI/CD, SSH)
- The lazy import in `src/control/__init__.py` mitigates this for `ControlMapper`
  and test code, but `EEGCursorController` and `run_eeg_cursor.py` still require
  a display

### 7.4 No Windows-Specific Testing

The project is developed and tested on WSL2/Linux. While the Python code is
cross-platform, several components have OS-specific behavior:

- `pyautogui` cursor control uses different backends on Windows/macOS/Linux
- `pygame` audio initialization may fail differently across platforms
- Shell scripts (`install.sh`, `boot.sh`) are bash-only — no PowerShell/cmd equivalents

### 7.5 Band Power Ratio Is Beta/Mu, Not Mu/Beta

The `BandPowerExtractor` sorts band names alphabetically (["beta", "mu"]) and
computes the ratio as `first / second = beta / mu`. This is the inverse of the
canonical ERD/ERS ratio (mu/beta).

**Impact:**
- The feature still works for classification (the classifier learns the correct
  direction), but the ratio values are inverted relative to the literature
- A ratio > 1.0 means beta dominates (typical during rest), not mu

---

## 8. Usability & Human Factors

### 8.1 BCI Illiteracy

Approximately **15-30% of the population** cannot produce detectable motor imagery
signals even after extensive training. This is called "BCI illiteracy" or
"BCI inefficiency."

**Impact:**
- Some users will never achieve above-chance accuracy regardless of calibration,
  training, or algorithm choice
- There is no way to predict BCI illiteracy before trying
- The ERP trainer can help identify illiterate users early (flat ERDS% maps,
  low r² values across all channels)

### 8.2 Cognitive Load

Controlling a cursor via motor imagery while simultaneously:
- Attending to a task on screen
- Suppressing eye movements
- Maintaining electrode contact quality
- Imagining specific movements

...is mentally exhausting. Most subjects cannot sustain effective BCI control
for more than **20-30 minutes** before performance degrades.

### 8.3 Learning Curve

Motor imagery BCI skill improves with practice but requires multiple sessions:

| Session | Expected Performance |
|---------|---------------------|
| 1 (first calibration) | Near chance, establishing baseline |
| 2-3 | Beginning to produce consistent ERD |
| 4-6 | Accuracy plateaus around 40-60% |
| 7-10 | Skilled control (if subject is not BCI-illiterate) |
| 10+ | Diminishing returns, stable performance |

### 8.4 Fatigue Effects

Within a single 25-minute calibration session:
- Trials 1-50: Subject is focused, signals are strong
- Trials 50-100: Attention begins to wane
- Trials 100-150: Fatigue sets in, mu modulation decreases
- Trials 150-200: Signals degrade, classifier trains on fatigued data

**Impact:** The classifier learns from a mixture of "good" and "fatigued" data,
which may not represent the subject's peak performance.

---

## 9. What This System Cannot Do

To set clear expectations, here is what this BCI system **does not and cannot** do:

| Capability | Status | Why |
|------------|--------|-----|
| Read thoughts | No | EEG measures bulk electrical activity, not individual thoughts |
| Control with sub-second precision | No | MI signals evolve over 1-3 seconds |
| Work without calibration | No | Each brain is unique; requires per-subject training |
| Work without electrodes | No | EEG requires scalp contact |
| Replace a mouse for daily use | No | Too slow and inaccurate for productive computing |
| Detect emotions | No | Not designed for affective computing |
| Work during heavy movement | No | Movement artifacts overwhelm EEG signals |
| Classify more than 5 states | Theoretical limit | 5-class is already pushing practical limits |
| Achieve > 90% accuracy | Extremely unlikely | Non-invasive MI-BCI has a ~80% ceiling with optimal hardware |

---

## 10. Comparison to Research-Grade Systems

| Feature | This System | BCI Competition Winners | Clinical BCI (g.tec) |
|---------|-------------|------------------------|---------------------|
| Channels | 16 | 22-64 | 16-256 |
| Sampling rate | 125 Hz | 250-1000 Hz | 256-2400 Hz |
| Amplifier | Consumer (ADS1299) | Research (BrainAmp, g.USBamp) | Clinical-grade |
| Electrode type | Wet gel | Wet gel (active) | Active wet/dry |
| Event timing | Wall clock (~10ms jitter) | Hardware TTL (~1ms) | Hardware TTL |
| Artifact rejection | Offline only | Real-time (ICA, regression) | Real-time adaptive |
| Online adaptation | None | Supervised/unsupervised | Continuous |
| Transfer learning | None | Subject-to-subject | Preloaded models |
| Typical 4-class accuracy | 35-60% | 65-80% | 70-85% |
| Setup time | 15 min (gel electrodes) | 30-60 min (active electrodes) | 5-15 min (dry cap) |
| Cost | ~$1,000 (board + cap) | $10,000-50,000 | $20,000-100,000 |

**Bottom line:** This system trades accuracy and features for accessibility and
cost. It is a research/educational tool, not a clinical assistive device.

---

*This document reflects the state of the system as of the deep validation audit
conducted on 2026-03-27. Limitations may be addressed in future versions.*
