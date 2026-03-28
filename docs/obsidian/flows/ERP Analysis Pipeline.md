---
title: ERP Analysis Pipeline
tags: [flow, analysis, erp, erds]
aliases: [ERP Flow, Analysis Pipeline]
---

# ERP Analysis Pipeline

> [!info] Overview
> The analysis flow used by [[erp_trainer]] for real-time neurofeedback during motor imagery training. Shows how raw epochs are transformed into ERP averages, ERDS% spectrograms, band power timecourses, and scalp topographic maps.

## Full Pipeline

```mermaid
flowchart TD
    subgraph Input
        Raw["Raw Epoch\n(16ch x 750 samples)\n6s at 125 Hz"]
    end

    subgraph Preprocessing
        Raw --> BP["Bandpass Filter\n1-40 Hz broadband"]
        BP --> CAR["Common Average Reference"]
    end

    subgraph ERP Path
        CAR --> BL["Baseline Correction\nsubtract mean of\npre-stimulus period (1s)"]
        BL --> Add["ERPAccumulator.add_epoch()\nper-class buffer"]
        Add --> Avg["get_erp(class)\nmean +/- std"]
        Add --> Grand["get_grand_average()\nacross all classes"]
        Add --> R2["compute_signed_r2(A, B)\nwhere/when classes differ"]
        Add --> SNR["compute_erp_snr(class)\nsignal quality per channel"]
    end

    subgraph ERDS Path
        CAR --> TFR["ERDSComputer.compute_tfr()\nMorlet wavelet convolution"]
        TFR --> Norm["Baseline Normalize\nERDS% = (power-bl)/bl * 100"]
        Norm --> AvgERDS["compute_erds_average()\nmulti-trial mean"]
        TFR --> Band["compute_band_power()\nmu [8-12], beta [13-30]"]
    end

    subgraph Display
        Avg --> P1["Panel 1: ERP Waveforms\nC3, C4, Cz"]
        AvgERDS --> P2["Panel 2: ERDS% Spectrogram\ntime x frequency"]
        Band --> P3["Panel 3: Band Power\nmu + beta timecourses"]
        R2 --> Topo["TopoMapper.plot()"]
        Topo --> P4["Panel 4: Scalp Topo Map"]
    end
```

## Baseline Correction

```mermaid
flowchart LR
    Epoch["Full Epoch\n[-1.0, 5.0]s"] --> Pre["Pre-stimulus\n[-1.0, 0.0]s\n= baseline"]
    Pre --> Mean["Mean of baseline\nper channel"]
    Mean --> Sub["Subtract from\nentire epoch"]
    Sub --> Corrected["Baseline-corrected\nepoch"]
```

**Purpose**: Removes slow drifts and DC offset so that post-stimulus changes are visible relative to the pre-stimulus state.

## ERDS% Interpretation

| ERDS% Value | Color (RdBu_r) | Meaning | MI Significance |
|-------------|-----------------|---------|-----------------|
| -50% to -100% | Blue | Strong ERD | Mu desynchronization during imagery |
| -10% to -50% | Light blue | Mild ERD | Weak but present imagery signal |
| -10% to +10% | White | No change | Baseline level |
| +10% to +50% | Light red | Mild ERS | Post-imagery beta rebound |
| +50% to +100% | Red | Strong ERS | Strong beta rebound |

## Signed-r2 Discriminability

The signed-r2 map shows where and when two MI classes produce different brain signals:

```
signed_r2 = sign(mean_A - mean_B) * (SS_between / SS_total)
```

- **High r2 at C3** for left vs right hand: C3 (left hemisphere) shows different activation -- good sign
- **Low r2 everywhere**: Classes are not distinguishable -- subject needs more practice or different strategy

## Related Pages

- [[erp_trainer]] -- Script that implements this pipeline
- [[Analysis]] -- Module overview
- [[ERPAccumulator]] -- Running ERP computation
- [[ERDSComputer]] -- Time-frequency decomposition
- [[Channel Layout]] -- 10-20 positions for topographic maps
- [[Research Papers]] -- Pfurtscheller (1999), Luck (2014)
