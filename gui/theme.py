"""Shared theming utilities for the FER GUI."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """Centralised colour palette used across the GUI."""

    name: str = "Aurora"
    bg_primary: str = "#0f172a"
    bg_secondary: str = "#111c34"
    accent: str = "#38bdf8"
    accent_soft: str = "#1e293b"
    text_primary: str = "#f8fafc"
    text_secondary: str = "#cbd5f5"
    positive: str = "#4ade80"
    warning: str = "#facc15"
    danger: str = "#fb7185"


PALETTE = Palette()


def build_stylesheet(palette: Palette = PALETTE) -> str:
    """Return a rich QSS stylesheet for a cohesive appearance."""

    return f"""
        QWidget {{
            background-color: {palette.bg_primary};
            color: {palette.text_primary};
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        }}

        QPushButton {{
            background-color: {palette.accent_soft};
            border-radius: 10px;
            border: 1px solid {palette.accent};
            padding: 8px 16px;
            color: {palette.text_primary};
        }}

        QPushButton:hover {{
            background-color: {palette.accent};
            color: {palette.bg_primary};
        }}

        QPushButton:pressed {{
            background-color: {palette.accent_soft};
            border-color: {palette.text_secondary};
        }}

        QLabel#titleLabel {{
            font-size: 32px;
            font-weight: 600;
            color: {palette.text_primary};
        }}

        QLabel#subtitleLabel {{
            font-size: 18px;
            color: {palette.text_secondary};
        }}

        QFrame#card {{
            background-color: {palette.bg_secondary};
            border-radius: 16px;
            border: 1px solid {palette.accent_soft};
        }}

        QListWidget {{
            background-color: {palette.bg_secondary};
            border-radius: 12px;
            border: 1px solid {palette.accent_soft};
            padding: 12px;
        }}

        QTableWidget {{
            background-color: {palette.bg_secondary};
            border-radius: 12px;
            gridline-color: {palette.accent_soft};
        }}

        QHeaderView::section {{
            background-color: {palette.accent_soft};
            color: {palette.text_secondary};
            padding: 6px;
            border: none;
        }}
    """


def probability_to_colour(probability: float, palette: Palette = PALETTE) -> str:
    """Return a hex colour to indicate probability strength."""

    if probability > 0.7:
        return palette.positive
    if probability > 0.4:
        return palette.warning
    return palette.danger


def format_percentage(probability: float) -> str:
    return f"{probability * 100:.1f}%"
