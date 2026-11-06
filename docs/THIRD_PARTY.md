# Third-party code, data and models

None of the items below is redistributed here. Obtain them from their official sources and follow their licences.

## Dataset

* **V2X-Seq (SPD) / DAIR-V2X** — https://github.com/AIR-THU/DAIR-V2X . Raw point clouds, labels, calibrations and the per-pair `system_error_offset` are read from the official release.

## Frameworks

* **OpenCOOD** — https://github.com/DerrickXuNu/OpenCOOD (V2X-ViT, F-Cooper and the intermediate-fusion training stack).
* **OpenCDA** (vendored in the original workspace; not needed for the results here).

## Detector and fusion back-ends

| Name in result files | Source | Use |
|---|---|---|
| `author` | [TraF-Align V2X-Seq release](https://github.com/zhyingS/TraF-Align) (`traf_align_v2xseq_0f6f98de`, checkpoint `net_epoch60.pth`) | single-agent detections for late fusion and initial weights for the second tier |
| `strict60` | TraF architecture trained in this project under the strict arrival protocol | late fusion, cross-detector check |
| `v2xvit_seed2024`, `v2xvit_seed2025` | V2X-ViT (OpenCOOD) detectors trained in this project, two seeds | late fusion, native baseline |
| `where2comm` | Where2comm (https://github.com/MediaBrain-SJTU/Where2comm), weights trained in this project | late fusion, native baseline, second tier |
| CoAlign | https://github.com/yifanlu0227/CoAlign (commit `64d1987`) | pose-graph baseline, second tier |
| F-Cooper, TraF native | OpenCOOD / TraF release | native baselines |

## External registration estimators (R4)

FreeAlign, CBM and RoCo, run from their official code as box-level per-frame estimators; parameters of our temporal smoothing chosen on calibration8.

## Not used in the paper tables

FFNet (VIC3D), CoSDH, ERMVP, MRCNet and V2I-Calib were examined in the original workspace and are not part of the accepted results.
