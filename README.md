# FACIAL EXPRESSION RECOGNITION

![Alt text for your screenshot](https://recfaces.com/wp-content/uploads/2021/03/rf-emotion-recognition-rf-830x495-1.jpeg)

This repository contains a real-time FER inference pipeline tailored for models trained on RAF-DB. It performs face detection, 5‑point alignment, feature extraction, calibrated classification, and on‑frame visualization with performance/debug overlays.

> **Highlights**: RetinaFace detection, 5‑point similarity alignment, RAF‑DB prior correction + temperature scaling, class-specific thresholds, optional class bias tuning (anger/fear/surprise), temporal EMA smoothing, and timing/quality diagnostics.

---

## Features

* **Face Detection**: RetinaFace for robust multi‑face detection and landmarks.
* **5‑Point Alignment**: Similarity transform to a canonical 224×224 face crop (RAF‑style).
* **Consistent Preprocessing**: RGB conversion + ImageNet normalization (mean/std) matching training.
* **Feature Head**: `Feature_Orthogonal` module (external) to build a 4‑stack representation.
* **Calibrated Classification**:

  * **Prior correction** using **actual RAF‑DB class distribution**.
  * **Temperature scaling (T)** to soften/sharpen logits.
  * Optional **bias tuning** (boost `anger/surprise`, slight penalty `fear`) to mitigate common confusions.
  * **Class‑wise thresholds** to control label assertiveness per class.
* **Temporal Smoothing**: EMA over per‑frame probabilities for stable video predictions.
* **Debug Overlays**:

  * `frame_id`, `pred_id`, processing time (ms), effective FPS.
  * Thumbnail of aligned 224×224 face with 5 landmarks.
  * Geometry metrics: eye line angle, eye and eye‑mouth ratios.
  * (Optional) full probability vector per face.
* **Real‑time Ready**: Designed for ~30+ FPS on common GPUs.

---

## Requirements

* Python 3.8+
* **Core**: `numpy`, `opencv-python`, `torch`, `torchvision`, `tensorflow`/`keras`, `retinaface`
* (Optional) your GPU stack (CUDA/cuDNN) for speed

Install example:

```bash
pip install numpy opencv-python torch torchvision tensorflow retinaface
```

> Ensure your installed `torch`/`tensorflow` match your CUDA runtime when using GPU.

---

## Model & Files

* **Emotion classifier** (Keras Functional): loaded via `load_emotin_model()`
* **Feature head / ORF module** (often PyTorch): loaded via `load_Orf_model()`
* **Landmarks** from RetinaFace are mapped per face crop to 224×224 coordinates.

> Use this [link](https://drive.google.com/drive/folders/1V5ekizaf0Aitx08z4Xggva2DNS77_6me?usp=sharing) and replace the loader implementations to point to your trained checkpoints.

---

## Quick Start

Minimal example from a notebook / script:

```python
Run FER-Inference.ipynb

# Output video will be saved as result.mp4 next to the script/notebook
```

---

## Desktop GUI

An interactive PySide6 desktop application is available in `gui/` for rapid experimentation.

1. Install the optional dependencies:

   ```bash
   pip install PySide6 opencv-python
   ```

2. Launch the interface:

   ```bash
   python -m gui.main
   ```

3. Use the sidebar to open still images, video files, or start the webcam. Load a trained
   PyTorch checkpoint to replace the built-in demo classifier.

The GUI ships with a colourful Aurora-inspired theme, real-time statistics cards (top expression,
FPS, faces detected), a probability table, and an event log to trace pipeline actions. Without a
custom model the interface uses a light-weight dummy classifier so you can explore the layout
before wiring your actual weights.

---

## Troubleshooting

* **Everything becomes `happy`**: missing alignment; wrong priors; T too low; thresholds too low for `happy`.
* **Many `Uncertain` labels**: thresholds too strict; `T` too high; EMA alpha too high.
* **Anger ↔ Fear confusion**: raise `ANGER_BOOST`, apply `FEAR_PENALTY`, lower `anger` threshold slightly; or fine‑tune on curated samples.
* **Laggy overlays**: `pred_id` ≪ `frame_id` → reduce resolution, enable frame skipping, or add a tracker.

---

## License & Attribution

* Face detection by **RetinaFace** (see its license).

* This inference pipeline includes concepts commonly used in FER research (alignment, calibration, smoothing). Adapt thresholds and priors to RAF data.

---

##  Acknowledgements

This work builds upon the concepts introduced in  
**“Quaternion Orthogonal Transformer for Facial Expression Recognition in the Wild”**,  
which inspired the architecture and feature orthogonalization modules used here.  
We thank the original authors for their contribution to advancing FER research.

Also, thanks to RAF‑DB and open‑source contributors of RetinaFace, PyTorch, TensorFlow/Keras, and OpenCV.

---

## Roadmap

* Adaptive bias correction from confusion matrices
* Async/threaded capture + inference queue
* Export to ONNX / TensorRT / TFLite for edge deployment
* Multi‑person tracking and ID‑consistent temporal fusion

