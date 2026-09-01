from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from labelone.system_picker import DirectoryPickerError, _finish, _macos


def test_picker_result_returns_existing_directory_and_handles_cancel(tmp_path: Path) -> None:
    selected = _finish(SimpleNamespace(returncode=0, stdout=f"{tmp_path}\n", stderr=""))
    canceled = _finish(SimpleNamespace(returncode=1, stdout="", stderr="User canceled. (-128)"))

    assert selected == tmp_path.resolve()
    assert canceled is None


def test_picker_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(DirectoryPickerError, match="no longer available"):
        _finish(SimpleNamespace(returncode=0, stdout=str(tmp_path / "missing"), stderr=""))


def test_macos_picker_invokes_native_osascript_dialog(tmp_path: Path, monkeypatch) -> None:
    captured: list[list[str]] = []

    def run(command, **_kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=0, stdout=f"{tmp_path}\n", stderr="")

    monkeypatch.setattr("labelone.system_picker.subprocess.run", run)

    assert _macos("选择图像数据集文件夹", tmp_path) == tmp_path.resolve()
    assert captured[0][0] == "/usr/bin/osascript"
    assert "choose folder" in captured[0][2]
    assert "activate" not in captured[0][2]
