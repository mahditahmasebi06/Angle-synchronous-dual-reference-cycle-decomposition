# Angle-synchronous dual-reference cycle decomposition (AS-DRCD)

Code for the manuscript *"Angle-synchronous dual-reference cycle decomposition with line-voltage
self-calibration for incipient fault detection and classification in distribution networks"*
(submitted to the International Journal of Electrical Power & Energy Systems).

AS-DRCD detects and classifies incipient faults in substation waveform records:

| Stage | Step | Settings used in the paper |
| --- | --- | --- |
| I. Signal conditioning | Positive-sequence frequency tracking | 3 iterations, accepted 47.5–52.5 Hz |
| | Angle-synchronous resampling | M = 128 samples per cycle, K = 16 cycles (input fs = 4096 Hz, 81.92 samples/cycle) |
| | Line-voltage self-calibration of the VT gains | L-BFGS-B, 10 starts, gain ratios ≤ 4:1, V0 preserved |
| II. Detection | C1: current novelty contrast and amplitude | R ≥ 3 and A ≥ 0.2 (pu of load current) |
| | C2: neutral-voltage displacement | max \|V0\|/\|V1\| ≥ 0.3 |
| | Onset (MAD change point) | reference = novelty of cycles 1–3 |
| III. Representation | Novelty map N = C(k) − C(k−1), superimposed map S = C(k) − T | noise floor from orders 30–64 |
| | 91 phase-invariant descriptors + 63 phase-sorted spectral statistics | 154 features per record |
| IV. Classification | Extra-Trees | 300 trees, √d features per split, balanced class weights |

## Layout

```
src/        method, baselines, evaluation protocols and experiment scripts
charts/     figure scripts (SVG charts rendered to PNG with Playwright/Chromium) and the rendered figures in charts/png/
results/    JSON files with every number reported in the paper
data/       downloaded datasets (not part of the repository; created by the preparation scripts)
cache/      cached baseline features (created by src/make_features.py)
```

Core modules: `carm.py` (frequency tracking, angle resampling, onset, N/S maps, line-voltage self-calibration),
`features.py` (91 descriptors), `baselines.py` (B1–B5), `invariant.py` (phase sorting), `evaluate.py`
(stratified and event-grouped cross-validation), `scenarios.py` (noise and off-nominal frequency),
`mini7z.py` (standard-library 7z extractor for the OscGrid archive).

## Requirements

Python ≥ 3.11 with NumPy, SciPy, scikit-learn and Matplotlib (`pip install -r requirements.txt`).
The results were produced with Python 3.13, NumPy 2.5, SciPy 1.18, scikit-learn 1.9 and Matplotlib 3.11.
The SVG charts are rendered with Playwright (`pip install playwright && playwright install chromium`).

## Data

1. **Field incipient-fault dataset** (Xiong et al.; 236 records, SIF/MIF/PF/TD, 12 subtypes):
   clone <https://github.com/xiongsiheng/Incipient-fault-waveform-dataset> into `data/`
   (IEEE DataPort: <https://doi.org/10.21227/bwjy-7e05>).
2. **OscGrid** (independent recordings, used for the calibration and false-detection checks): download
   `unlabeled_50_6400.7z` from <https://doi.org/10.6084/m9.figshare.28465427> into `data/oscgrid/`.
   OscGrid is distributed under CC BY-NC-ND 4.0; this repository contains no OscGrid data.

Paths can be changed with the environment variables `XIONG_DIR`, `OSCGRID_7Z`, `ASDRCD_DATA`,
`ASDRCD_RESULTS` and `ASDRCD_CACHE` (see `src/paths.py`).

## Reproducing the paper

```bash
cd src
python prepare_xiong.py        # data/xiong.npz
python prepare_oscgrid.py      # data/oscgrid/u6400_recs.npz
python make_features.py        # cache/: baseline features B1-B5
python run_detection.py        # Table 6 (detection), pickup sensitivity
python run_final.py            # Tables 7, 9, 10, Figs. 9 and 13 (about 30 min on 2 cores)
python run_extra.py            # Table 8, Table 11, Figs. 11, 12 and 14
python paper_stats.py          # Figs. 7 and 10
python oscgrid_calib.py        # Fig. 15
python figure_data.py          # data of Figs. 5, 8 and 12
python figs_waveforms.py       # Figs. 1, 2, 4 and 6
cd ../charts
python charts_a.py && python charts_b.py && python charts_c.py && python render.py   # Figs. 3, 5, 7-15
```

| Paper item | Script | Output |
| --- | --- | --- |
| Table 6 | `src/run_detection.py` | `results/detection_summary.json`, `results/detect_stats2.json` |
| Tables 7, 9, 10; Figs. 9, 13 | `src/run_final.py` | `results/final.json` |
| Tables 8, 11; Figs. 11, 12, 14 | `src/run_extra.py` | `results/extra.json` |
| Figs. 7, 10 | `src/paper_stats.py` | `results/paper_stats.json` |
| Fig. 15 | `src/oscgrid_calib.py` | `results/oscgrid_calibration.json` |
| Figs. 5, 8, 12 | `src/figure_data.py` | `results/descriptor_dump.json`, `results/detect_example.json` |
| Figs. 1, 2, 4, 6 | `src/figs_waveforms.py` | `charts/png/fig_samples.png`, `fig_scenarios.png`, `fig_calibration.png`, `fig_maps.png` |
| Figs. 3, 5, 7–15 | `charts/*.py`, `charts/render.py` | `charts/png/fig_*.png` |

Figure files: 1 `fig_samples`, 2 `fig_scenarios`, 3 `fig_framework`, 4 `fig_calibration`, 5 `fig_detection`,
6 `fig_maps`, 7 `fig_v0traj`, 8 `fig_descriptors`, 9 `fig_confusion`, 10 `fig_recall`, 11 `fig_tsne`,
12 `fig_importance`, 13 `fig_robustness`, 14 `fig_sensitivity`, 15 `fig_oscgrid`.

## Use of AI-assisted tools

Parts of this code and the plotting scripts were written with the assistance of Claude Opus 5.5 (Anthropic).
The authors specified the methods, checked the code and its outputs, and take responsibility for them.

## Citation

If you use this code, please cite the article (reference to be added on publication).
