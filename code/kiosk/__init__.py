"""Pi-local kiosk touchscreen UI (ADR 0004).

A second renderer of the in-process ``SessionWorkflow`` beside
``session_console`` -- serves extracted wireframe screens to Chromium
``--kiosk`` over ``localhost`` and routes taps back through the existing
``dispatch`` verbs. Presentation/transport only; no domain logic lives here.
"""
from __future__ import annotations

from kiosk.relay import KioskRelay

__all__ = ["KioskRelay"]
