# Empty-Backlight Setup runs via widened confirm_empty

Camera Calibration no longer runs in SessionWorkflow construction or poll_drain.
Engage, slide start, and Calibration Failure RETRY all dispatch confirm_empty,
which runs activate_mode then installs the locked Presence Baseline. Failures
set CALIBRATION_FAILED and show kiosk screen 04. This keeps one operator verb,
avoids silent Chromium death on occupied startup, and leaves screen 19 as
Capture Error only.
