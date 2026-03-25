# Simple NanoGS GUI

Simple NanoGS GUI is a Windows-focused fork of the original NanoGS project for training-free Gaussian splat simplification.

Original upstream project: https://github.com/saliteta/NanoGS

Project page: https://saliteta.github.io/NanoGS/

This repository keeps the upstream simplification core, then adds a practical desktop workflow and format-conversion tooling for local batch use.

![NanoGS teaser](teaser.png)

## What changed compared to the original repo

- Added a Tkinter desktop GUI for single-file and folder-based batch processing.
- Added saved parameter profiles and persisted local GUI settings.
- Added support for `.sog` input by decoding it to a temporary `.ply` file in Python.
- Added `.spz` and `.sog` round-trip export support through `gsbox.exe`.
- Added a quality-test mode that generates about 40 nearby parameter variants and writes a CSV manifest.
- Added CPU/GPU selection for cost evaluation with automatic backend choice.
- Added automatic `block_edges` tuning plus a manual override for large scenes.
- Added benchmark scripts for backend and block-size tuning.
- Added a Windows launcher batch file for the GUI.
- Extended the CLI with `--device` and `--block_edges`.

## Repository contents

- `nanogs_gui.py`: desktop GUI for running simplification jobs.
- `simplification.py`: CLI entry point and simplification pipeline.
- `utils/sog_utils.py`: `.sog` decoder and conversion to `.ply`.
- `benchmark_block_edges.py`: compare runtime across block sizes and devices.
- `benchmark_device_threshold.py`: compare CPU and GPU behavior across scene sizes.
- `start_nanogs_gui.bat`: Windows launcher that runs the GUI from `.venv`.

## Requirements

- Python 3.10+ on Windows.
- `numpy`, `scipy`, `tqdm`, and `Pillow`.
- Optional: a CUDA-capable GPU plus a matching CuPy build for GPU cost evaluation.
- Optional: `gsbox.exe` if you want `.spz` conversion or `.sog`/`.spz` export.

Example environment setup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy tqdm pillow
```

Optional GPU backend:

```powershell
python -m pip install cupy-cuda12x
```

Use a CuPy package that matches your CUDA runtime.

## GUI workflow

Start the GUI with either:

```powershell
.venv\Scripts\python.exe nanogs_gui.py
```

or:

```powershell
start_nanogs_gui.bat
```

The GUI supports:

- Single-file processing for `.ply`, `.sog`, and `.spz`.
- Folder processing with optional recursion.
- Custom output folders and suffixes.
- Parameter profiles for different compression presets.
- Automatic or explicit CPU/GPU execution.
- Quality-test sweeps around the current ratio, `k`, opacity threshold, and `lam_sh`.

### Format handling

- `.ply` files are processed directly.
- `.sog` files are converted to temporary `.ply` files inside the app, then can be exported back to `.sog` with GSBox.
- `.spz` files are converted through GSBox. A fallback converter command can also be configured in the GUI.

`gsbox.exe` is not stored in this repository. Put it next to `nanogs_gui.py` or configure its path in the GUI.

## CLI usage

The command-line flow is still available:

```powershell
python simplification.py --ply scene.ply -r 0.3 --k 8 --opacity_threshold 0.18 --lam_geo 1.0 --lam_sh 0.5 --device auto --block_edges 0
```

Key fork-specific CLI additions:

- `--device {auto,cpu,gpu}` chooses the edge-cost backend.
- `--block_edges N` overrides the auto-tuned edge block size.

## Benchmarks

Two helper scripts are included for tuning and regression checks:

```powershell
python benchmark_block_edges.py
python benchmark_device_threshold.py
```

They expect local benchmark `.ply` files and are intended as developer utilities rather than end-user entry points.

## Notes

- `nanogs_gui_settings.json` is generated locally and is intentionally ignored.
- Benchmark logs, virtual environments, and workspace files are also ignored.
- This repository is intended as a practical GUI fork of NanoGS rather than a mirror of the upstream paper repository.

## Citation

If the original NanoGS project helps your work, please cite the paper in `CITATION.bib`.

## License

See `LICENSE` in this repository.
