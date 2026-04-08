# ThereCam

ThereCam is a real-time hand keypoint tracking project.

It includes:
- A C++ desktop application that captures webcam frames, runs ONNX inference, and draws hand boxes + 21 keypoints.
- Python scripts for model training, validation, and ONNX export.
- Ultralytics hand-keypoints dataset metadata and training artifacts.

## What the app does

The runtime app in `mainProgram/`:
- Opens the default camera.
- Runs a hand pose model (`epoch60.onnx`) with ONNX Runtime.
- Draws up to 2 detected hands with confidence, side (LEFT/RIGHT), and 21 keypoints.

Core implementation files:
- `mainProgram/src/main.cpp`
- `mainProgram/src/handTracking/inference.cpp`
- `mainProgram/src/utilities/utils.cpp`

## Repository layout

- `mainProgram/`: C++ application (CMake project).
- `mainProgram/src/AIModel/`: ONNX model files used by the app.
- `ModelProcurement/`: Python training and validation scripts.
- `datasets/hand-keypoints/`: dataset config and labels/images structure.
- `runs/pose/`: Ultralytics training outputs/checkpoints.

## Prerequisites

This project is currently configured for macOS (Apple Silicon paths are hardcoded in CMake and VS Code config).

Required tools/libraries:
- CMake >= 3.16
- A C++17 compiler (Apple Clang on macOS)
- OpenCV (detectable via `find_package(OpenCV REQUIRED)`)
- ONNX Runtime C++ SDK
- RtMidi
- Python 3.10+ (for training utilities)

Expected local library paths from current config:
- ONNX Runtime root: `$HOME/libs/onnxruntime`
- RtMidi: `/opt/homebrew/opt/rtmidi`

If your machine uses different install locations, update `mainProgram/CMakeLists.txt` accordingly.

## Build and run (C++)

From repository root:

```bash
cd mainProgram
cmake -S . -B build
cmake --build build
./build/ThereCam
```

The executable is compiled with:
- `THERECAM_MODEL_PATH="mainProgram/src/AIModel/epoch60.onnx"` (resolved from CMake source dir)

To use a different model, either:
- Replace the file at `mainProgram/src/AIModel/epoch60.onnx`, or
- Change the compile definition in `mainProgram/CMakeLists.txt`.

## Model training and export (Python)

Scripts are under `ModelProcurement/`.

### Train and export

`ModelProcurement/main.py` uses Ultralytics YOLO pose training and then exports ONNX.

Install dependencies:

```bash
pip install ultralytics onnx onnxruntime numpy pandas matplotlib
```

Run training/export:

```bash
python ModelProcurement/main.py
```

### Validate ONNX output

```bash
python ModelProcurement/modelValidation.py
```

### Plot training vs validation loss

```bash
python ModelProcurement/epochValidation.py
```

## Dataset notes

Dataset YAML files:
- `ModelProcurement/hand-keypoints.yaml`
- `datasets/hand-keypoints/data.yaml`

Current configuration targets Ultralytics Hand Keypoints format with:
- `kpt_shape: [21, 3]`
- single class: `hand`

## Current implementation notes

- Inference threshold is controlled in `mainProgram/src/utilities/constants.h` (`CONFIDENCE_THRESHOLD`).
- Max tracked hands is set to 2 (`MAX_HANDS`).
- Hand side is estimated from wrist/index/pinky orientation determinant.
- `mainProgram/src/handTracking/handTracker.*` appears to be an older/incomplete path; active runtime uses `HandInference`.

## Troubleshooting

- Build cannot find ONNX Runtime headers/libs:
	- Verify `$HOME/libs/onnxruntime/include` and `$HOME/libs/onnxruntime/lib/libonnxruntime.dylib` exist.
- Build cannot find RtMidi:
	- Verify Homebrew RtMidi install at `/opt/homebrew/opt/rtmidi`.
- Camera opens but no frames:
	- Check camera permissions for your terminal/IDE and ensure no other app is locking the webcam.
- Model path error at startup:
	- Confirm `mainProgram/src/AIModel/epoch60.onnx` exists or update `THERECAM_MODEL_PATH` in CMake.

## License

No project license file is currently included in this repository.
