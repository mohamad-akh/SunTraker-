# SunTraker

SunTraker is an OpenCV-based project for detecting and tracking the sun in camera frames and test images.

## Project map

- `V3/sun_tracker_v3.py` — stable runnable baseline: camera capture, sun tracking, and recording.
- `Refactor/TestSunTracker.ipynb` — notebook for testing and iterating on the V3 approach.
- `Refactor/redesigned.ipynb` — experiments for day/night handling and white/dark-cloud detection.
- `Refactor/segmentation.py` — standalone LAB + flood-fill segmentation experiment with timing/FPS output.
- `Refactor/LAB_HSV.py` — pixel and colour-space inspection utility for calibrating thresholds.
- `Refactor/Images/` — reference and test images.

## Workflow

1. Experiment in notebooks or standalone scripts.
2. Test candidate approaches on the same reference images.
3. Move the verified approach into a new runnable Python version (for example, V4).
4. Keep V3 unchanged as the current baseline until the new version is validated.

Jupyter checkpoint files and runtime recordings are ignored because they are generated artifacts, not source files.
