"""Console prototype for the shared automatic capture session."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from capture_runtime import CaptureConfiguration, CaptureController
from capture_session import CaptureSession, CaptureState, FrameResult
from capture_storage import CaptureStore
from capture_telemetry import StatusReporter, TelemetryWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("block", "slide"), required=True)
    parser.add_argument("--output", type=Path, default=Path("captures"))
    parser.add_argument(
        "--telemetry", type=Path, default=Path("outputs/capture_telemetry.csv")
    )
    return parser


def create_session(
    mode: str, configuration: CaptureConfiguration | None = None
) -> CaptureSession:
    config = configuration or CaptureConfiguration()
    return CaptureSession(config.session, mode=mode)


def _prepare_initial_baseline(session: CaptureSession) -> None:
    """Start empty-backlight calibration before any scan or placement."""
    if session.state is CaptureState.AWAITING_BASELINE_CONFIRMATION:
        input("Leave the backlight empty, then press Enter to build the baseline.")
        session.confirm_empty()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Delayed import keeps laptop imports/tests free of Pi-only dependencies.
    from picamera2_adapter import Picamera2Adapter

    configuration = CaptureConfiguration()
    session = create_session(args.mode, configuration)
    camera = Picamera2Adapter()
    controller = CaptureController(
        session=session,
        camera=camera,
        store=CaptureStore(args.output),
        working_dir=args.output / ".pending",
        configuration=configuration,
    )
    reporter = StatusReporter(print)
    _prepare_initial_baseline(session)
    controller.start()

    try:
        with TelemetryWriter(args.telemetry) as telemetry:
            while True:
                now = time.monotonic()
                captured_at = datetime.now(timezone.utc)
                frame = camera.preview_frame()
                record = controller.handle_frame(
                    frame, now=now, captured_at=captured_at
                )
                result = controller.last_frame_result or FrameResult()
                event = "capture_saved" if record is not None else ""
                if session.state is CaptureState.CAPTURE_ERROR:
                    event = "capture_error"
                telemetry.record(
                    timestamp=captured_at.isoformat().replace("+00:00", "Z"),
                    session=session,
                    frame=result,
                    settings=controller.settings,
                    capture_path=str(record.path) if record is not None else "",
                    event=event,
                )
                reporter.publish(now=now, session=session, frame=result)

                if session.state is CaptureState.CAPTURE_ERROR:
                    if input("Capture failed. Type R to Retry: ").strip().lower() == "r":
                        controller.retry(captured_at=datetime.now(timezone.utc))
                elif session.state is CaptureState.WAITING_FOR_SCAN:
                    while not session.submit_scan(
                        input("Scan next block ID: ").strip()
                    ).accepted:
                        reporter.publish(
                            now=time.monotonic(), session=session, frame=result
                        )
    except KeyboardInterrupt:
        return 0
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
