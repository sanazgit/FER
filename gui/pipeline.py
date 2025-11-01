"""Reusable FER inference helpers used by the GUI."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Optional heavy dependencies are imported lazily in load_classification_model
try:  # pragma: no cover - optional import
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch may not be installed on CPU-only setups
    torch = None
    nn = None


EmotionScores = Dict[str, float]


@dataclass
class ExpressionPrediction:
    """Container for a single face prediction."""

    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    probabilities: EmotionScores = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration values exposed in the GUI."""

    camera_index: int = 0
    smoothing_alpha: float = 0.6
    min_confidence: float = 0.3
    detection_scale_factor: float = 1.2
    detection_min_neighbors: int = 5


class EmotionSmoother:
    """Simple exponential smoothing over probabilities."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.state: Optional[EmotionScores] = None

    def reset(self) -> None:
        self.state = None

    def __call__(self, scores: EmotionScores) -> EmotionScores:
        if self.state is None:
            self.state = scores
            return scores

        blended: EmotionScores = {}
        for key in scores.keys():
            prev = self.state.get(key, 0.0)
            blended[key] = self.alpha * scores[key] + (1 - self.alpha) * prev
        self.state = blended
        return blended


class HaarCascadeDetector:
    """Fallback face detector based on Haar cascades."""

    def __init__(self, scale_factor: float = 1.2, min_neighbors: int = 5) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def __call__(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detections]


class DummyClassifier:
    """Reasonable stand-in when no trained model is supplied."""

    EMOTIONS: Sequence[str] = (
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Sad",
        "Surprise",
    )

    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    def __call__(self, face: np.ndarray) -> EmotionScores:
        probabilities = self._rng.dirichlet(np.ones(len(self.EMOTIONS)))
        return {emotion: float(prob) for emotion, prob in zip(self.EMOTIONS, probabilities)}


class FERPipeline:
    """High-level orchestrator for video frame inference."""

    def __init__(
        self,
        detector: Optional[Callable[[np.ndarray], List[Tuple[int, int, int, int]]]] = None,
        classifier: Optional[Callable[[np.ndarray], EmotionScores]] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.detector = detector or HaarCascadeDetector(
            scale_factor=self.config.detection_scale_factor,
            min_neighbors=self.config.detection_min_neighbors,
        )
        self.classifier = classifier or DummyClassifier()
        self.smoothers: Dict[int, EmotionSmoother] = {}
        self.last_prediction_time: Optional[float] = None
        self.fps: float = 0.0

    def set_classifier(self, classifier: Callable[[np.ndarray], EmotionScores]) -> None:
        self.classifier = classifier
        self.reset_temporal_state()

    def reset_temporal_state(self) -> None:
        """Clear temporal memory such as smoothing buffers and FPS."""

        self.smoothers.clear()
        self.last_prediction_time = None
        self.fps = 0.0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[ExpressionPrediction]]:
        """Return an annotated frame and the predictions for each detected face."""

        start = time.perf_counter()
        predictions: List[ExpressionPrediction] = []
        next_smoothers: Dict[int, EmotionSmoother] = {}

        boxes = self.detector(frame)
        for x, y, w, h in boxes:
            face = self._extract_face(frame, x, y, w, h)
            scores = self.classifier(face)
            key = self._box_key(x, y, w, h)
            smoother = self.smoothers.get(key) or EmotionSmoother(self.config.smoothing_alpha)
            smoother.alpha = self.config.smoothing_alpha
            smoothed = smoother(scores)
            next_smoothers[key] = smoother
            label, confidence = max(smoothed.items(), key=lambda item: item[1])
            if confidence < self.config.min_confidence:
                label = "Uncertain"
            predictions.append(
                ExpressionPrediction(
                    label=label,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    probabilities=smoothed,
                )
            )
            self._draw_overlay(frame, x, y, w, h, label, confidence)

        self.smoothers = next_smoothers
        self._update_fps(start)
        return frame, predictions

    def _update_fps(self, start_time: float) -> None:
        now = time.perf_counter()
        if self.last_prediction_time is not None:
            delta = now - self.last_prediction_time
            if delta > 0:
                self.fps = 1.0 / delta
        self.last_prediction_time = now

    @staticmethod
    def _box_key(x: int, y: int, w: int, h: int) -> int:
        return hash((x // 10, y // 10, w // 10, h // 10))

    @staticmethod
    def _extract_face(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        crop = frame[y : y + h, x : x + w]
        return cv2.resize(crop, (224, 224))

    @staticmethod
    def _draw_overlay(
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        label: str,
        confidence: float,
    ) -> None:
        colour = (56, 182, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), colour, cv2.FILLED)
        text = f"{label}: {confidence * 100:0.1f}%"
        cv2.putText(frame, text, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 23, 42), 2)


def load_torch_classifier(model_path: str, device: Optional[str] = None) -> Callable[[np.ndarray], EmotionScores]:
    """Load a PyTorch checkpoint and return a callable classifier."""

    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required to load the supplied checkpoint.")

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device_name)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Lazy import to avoid circular references
    from resnet_pose_attention_v2 import ResNet_Pose, BasicBlock

    model = ResNet_Pose(BasicBlock, [2, 2, 2, 2], num_classes=7)
    model.load_state_dict(state_dict)
    model.to(device_name)
    model.eval()

    emotions = DummyClassifier.EMOTIONS

    @torch.no_grad()
    def _predict(face: np.ndarray) -> EmotionScores:
        tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device_name)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return {emotion: float(prob) for emotion, prob in zip(emotions, probs)}

    return _predict
