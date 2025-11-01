"""Entry point for the PySide6 based FER GUI."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from .pipeline import FERPipeline, ExpressionPrediction, load_torch_classifier
from .theme import build_stylesheet, format_percentage
from .widgets import EventLog, ProbabilityTable, StatCard, VideoDisplay


class FERWindow(QtWidgets.QMainWindow):
    """Main application window that wires the GUI with the inference pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FER Studio")
        self.resize(1320, 780)
        self.setStyleSheet(build_stylesheet())

        self.pipeline = FERPipeline()
        self.capture: Optional[cv2.VideoCapture] = None
        self.active_source: Optional[str] = None

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._on_tick)

        self._build_ui()

    # ------------------------------------------------------------------ UI SETUP
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)

        layout.addLayout(self._build_left_panel(), 3)
        layout.addLayout(self._build_right_panel(), 5)

    def _build_left_panel(self) -> QtWidgets.QVBoxLayout:
        panel = QtWidgets.QVBoxLayout()
        panel.setSpacing(16)

        title = QtWidgets.QLabel("FER Studio")
        title.setObjectName("titleLabel")
        subtitle = QtWidgets.QLabel("Real-time facial expression insights")
        subtitle.setObjectName("subtitleLabel")
        panel.addWidget(title)
        panel.addWidget(subtitle)

        panel.addSpacing(12)
        panel.addLayout(self._build_button_row())
        panel.addLayout(self._build_settings_card())
        panel.addWidget(self._build_stats_card())
        panel.addWidget(self._build_log_card())
        panel.addStretch(1)
        return panel

    def _build_button_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)

        self.btn_open_image = QtWidgets.QPushButton("Open Image…")
        self.btn_open_image.clicked.connect(self.open_image)
        self.btn_open_video = QtWidgets.QPushButton("Open Video…")
        self.btn_open_video.clicked.connect(self.open_video)
        self.btn_toggle_camera = QtWidgets.QPushButton("Start Webcam")
        self.btn_toggle_camera.clicked.connect(self.toggle_camera)
        self.btn_stop_stream = QtWidgets.QPushButton("Stop Stream")
        self.btn_stop_stream.clicked.connect(self._handle_stop_button)

        row.addWidget(self.btn_open_image)
        row.addWidget(self.btn_open_video)
        row.addWidget(self.btn_toggle_camera)
        row.addWidget(self.btn_stop_stream)
        return row

    def _build_settings_card(self) -> QtWidgets.QVBoxLayout:
        card = QtWidgets.QVBoxLayout()
        frame = QtWidgets.QFrame()
        frame.setObjectName("card")
        frame.setLayout(card)
        card.setContentsMargins(18, 18, 18, 18)
        card.setSpacing(14)

        # Model loader
        self.btn_load_model = QtWidgets.QPushButton("Load PyTorch Model…")
        self.btn_load_model.clicked.connect(self.load_model)
        card.addWidget(self.btn_load_model)

        # Smoothing
        smoothing_layout = QtWidgets.QHBoxLayout()
        smoothing_label = QtWidgets.QLabel("Temporal smoothing")
        self.smoothing_spin = QtWidgets.QDoubleSpinBox()
        self.smoothing_spin.setRange(0.0, 0.95)
        self.smoothing_spin.setSingleStep(0.05)
        self.smoothing_spin.setValue(self.pipeline.config.smoothing_alpha)
        self.smoothing_spin.valueChanged.connect(self._update_smoothing)
        smoothing_layout.addWidget(smoothing_label)
        smoothing_layout.addStretch(1)
        smoothing_layout.addWidget(self.smoothing_spin)
        card.addLayout(smoothing_layout)

        # Confidence
        confidence_layout = QtWidgets.QHBoxLayout()
        confidence_label = QtWidgets.QLabel("Min confidence")
        self.confidence_spin = QtWidgets.QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 0.99)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(self.pipeline.config.min_confidence)
        self.confidence_spin.valueChanged.connect(self._update_confidence)
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addStretch(1)
        confidence_layout.addWidget(self.confidence_spin)
        card.addLayout(confidence_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(frame)
        return layout

    def _build_stats_card(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("card")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        self.expression_card = StatCard("Top Expression")
        self.fps_card = StatCard("Pipeline FPS")
        self.faces_card = StatCard("Faces Detected")

        layout.addWidget(self.expression_card)
        layout.addWidget(self.fps_card)
        layout.addWidget(self.faces_card)
        return frame

    def _build_log_card(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("card")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        label = QtWidgets.QLabel("Event log")
        label.setObjectName("subtitleLabel")
        self.log_widget = EventLog()

        layout.addWidget(label)
        layout.addWidget(self.log_widget)
        return frame

    def _build_right_panel(self) -> QtWidgets.QVBoxLayout:
        panel = QtWidgets.QVBoxLayout()
        panel.setSpacing(18)

        self.video_widget = VideoDisplay()
        panel.addWidget(self.video_widget, stretch=4)

        prob_frame = QtWidgets.QFrame()
        prob_frame.setObjectName("card")
        prob_layout = QtWidgets.QVBoxLayout(prob_frame)
        prob_layout.setContentsMargins(18, 18, 18, 18)
        prob_layout.setSpacing(12)

        prob_title = QtWidgets.QLabel("Emotion distribution")
        prob_title.setObjectName("subtitleLabel")
        self.probabilities_table = ProbabilityTable()

        prob_layout.addWidget(prob_title)
        prob_layout.addWidget(self.probabilities_table)
        panel.addWidget(prob_frame)
        return panel

    # ------------------------------------------------------------------ ACTIONS
    def open_image(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select image", str(Path.home()), "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_name:
            return
        frame = cv2.imread(file_name)
        if frame is None:
            self.log_widget.push_message(f"Failed to load image: {file_name}")
            return

        self._stop_stream("Switched to still image")
        self.pipeline.reset_temporal_state()
        annotated, predictions = self.pipeline.process_frame(frame.copy())
        self._update_ui_from_predictions(annotated, predictions)
        self.log_widget.push_message(f"Loaded still image: {Path(file_name).name}")

    def open_video(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video", str(Path.home()), "Videos (*.mp4 *.mov *.avi *.mkv)"
        )
        if not file_name:
            return
        self._open_capture(cv2.VideoCapture(file_name), f"Playing {Path(file_name).name}", source="video")

    def toggle_camera(self) -> None:
        if self.active_source == "camera":
            self._stop_stream("Webcam stopped")
            return

        index = int(self.pipeline.config.camera_index)
        self._open_capture(cv2.VideoCapture(index), "Webcam started", source="camera")

    def _handle_stop_button(self) -> None:
        self._stop_stream("Stream stopped")

    def load_model(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select checkpoint", str(Path.home()), "PyTorch (*.pth *.pt *.pth.tar *.pt.tar)"
        )
        if not file_name:
            return
        try:
            classifier = load_torch_classifier(file_name)
        except Exception as exc:  # pragma: no cover - depends on runtime env
            QtWidgets.QMessageBox.critical(self, "Model load failed", str(exc))
            self.log_widget.push_message("Model load failed")
            return

        self.pipeline.set_classifier(classifier)
        self.log_widget.push_message(f"Loaded classifier: {Path(file_name).name}")

    def _open_capture(self, capture: cv2.VideoCapture, message: str, *, source: str) -> None:
        if not capture.isOpened():
            self.log_widget.push_message("Unable to open video source")
            return

        self._release_capture()
        self.capture = capture
        self.active_source = source
        self.pipeline.reset_temporal_state()
        self.timer.start()
        if source == "camera":
            self.btn_toggle_camera.setText("Stop Webcam")
        self.log_widget.push_message(message)

    def _stop_stream(self, message: Optional[str] = None) -> None:
        self.timer.stop()
        self._release_capture()
        self.active_source = None
        self.pipeline.reset_temporal_state()
        self.btn_toggle_camera.setText("Start Webcam")
        if message:
            self.log_widget.push_message(message)

    def _release_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _update_smoothing(self, value: float) -> None:
        self.pipeline.config.smoothing_alpha = float(value)
        self.pipeline.reset_temporal_state()
        self.log_widget.push_message(f"Smoothing updated → {value:.2f}")

    def _update_confidence(self, value: float) -> None:
        self.pipeline.config.min_confidence = float(value)
        self.log_widget.push_message(f"Confidence threshold → {value:.2f}")

    # ------------------------------------------------------------------ LOOP
    def _on_tick(self) -> None:
        if self.capture is None:
            return

        grabbed, frame = self.capture.read()
        if not grabbed:
            self._stop_stream("Stream ended")
            return

        annotated, predictions = self.pipeline.process_frame(frame)
        self._update_ui_from_predictions(annotated, predictions)

    def _update_ui_from_predictions(
        self, frame: cv2.Mat, predictions: list[ExpressionPrediction]
    ) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        bytes_per_line = channel * width
        q_image = QtGui.QImage(rgb.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()
        self.video_widget.update_frame(q_image)

        if predictions:
            top_prediction = max(predictions, key=lambda pred: pred.confidence)
            self.expression_card.set_value(
                f"{top_prediction.label}\n{format_percentage(top_prediction.confidence)}"
            )
            self.probabilities_table.update_probabilities(top_prediction.probabilities)
        else:
            self.expression_card.set_value("No face\n—")
            self.probabilities_table.update_probabilities({})

        self.faces_card.set_value(str(len(predictions)))
        self.fps_card.set_value(f"{self.pipeline.fps:0.1f}")

    # ------------------------------------------------------------------ LIFECYCLE
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt naming convention
        self._stop_stream()
        super().closeEvent(event)


def launch_app() -> None:
    """Launch the FER Studio application."""

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("FER Studio")
    app.setOrganizationName("FER")

    window = FERWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    launch_app()
