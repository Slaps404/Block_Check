"""Success/failure coverage for tools/setup_check.py's exit-code contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from setup_check import CV2_GUIDANCE, ZXING_GUIDANCE, check_dependencies  # noqa: E402


def _ok(_module_name):
    return True, ""


def _fail(module_name):
    return False, f"ImportError: no module named {module_name}"


def test_check_dependencies_passes_when_all_decoders_import():
    assert check_dependencies(importer=_ok) == 0


def test_check_dependencies_fails_when_a_decoder_is_missing():
    assert check_dependencies(importer=_fail) == 1


def test_check_dependencies_reports_only_the_missing_decoder(capsys):
    def _zxing_only(module_name):
        return (True, "") if module_name == "zxingcpp" else _fail(module_name)

    assert check_dependencies(importer=_zxing_only) == 1
    out = capsys.readouterr().out
    assert "[OK] zxingcpp" in out
    assert "[FAIL] cv2" in out
    assert CV2_GUIDANCE in out
    assert ZXING_GUIDANCE not in out


def test_check_dependencies_uses_live_imports_by_default():
    assert check_dependencies() == 0
