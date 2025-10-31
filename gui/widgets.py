"""Custom Qt widgets used in the FER GUI."""
from __future__ import annotations

from typing import Dict

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import format_percentage, probability_to_colour


class VideoDisplay(QtWidgets.QLabel):
    """A QLabel configured for high-quality frame rendering."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border-radius: 18px; background-color: rgba(15, 23, 42, 180);")

    def update_frame(self, frame: QtGui.QImage) -> None:
        pixmap = QtGui.QPixmap.fromImage(frame).scaled(
            self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)


class StatCard(QtWidgets.QFrame):
    """Small rounded info card for a single metric."""

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(16, 16, 16, 16)
        self.layout().setSpacing(4)

        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("subtitleLabel")
        self.value_label = QtWidgets.QLabel("—")
        self.value_label.setObjectName("titleLabel")
        self.value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.layout().addWidget(self.title_label)
        self.layout().addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class ProbabilityTable(QtWidgets.QTableWidget):
    """Displays the probability distribution for each detected emotion."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Emotion", "Confidence"])
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setMinimumHeight(220)

    def update_probabilities(self, probabilities: Dict[str, float]) -> None:
        emotions = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        self.setRowCount(len(emotions))
        self.clearContents()
        for row, (emotion, probability) in enumerate(emotions):
            emotion_item = QtWidgets.QTableWidgetItem(emotion)
            probability_item = QtWidgets.QTableWidgetItem(format_percentage(probability))
            probability_item.setForeground(QtGui.QBrush(QtGui.QColor(probability_to_colour(probability))))
            self.setItem(row, 0, emotion_item)
            self.setItem(row, 1, probability_item)
        self.resizeRowsToContents()


class EventLog(QtWidgets.QListWidget):
    """Scrollable list of textual events."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWordWrap(True)
        self.setMinimumHeight(160)

    def push_message(self, message: str) -> None:
        timestamp = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self.insertItem(0, f"[{timestamp}] {message}")

        # Keep the list compact
        while self.count() > 100:
            item = self.takeItem(self.count() - 1)
            del item
