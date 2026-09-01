from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

from labelone.errors import LabelOneError


class DirectoryPickerError(LabelOneError):
    code = "directory_picker_error"


def _finish(result: subprocess.CompletedProcess[str]) -> Path | None:
    if result.returncode != 0:
        message = (result.stderr or "").strip()
        if "User canceled" in message or "-128" in message or result.returncode == 1:
            return None
        raise DirectoryPickerError("System directory picker failed", details={"error": message})
    value = (result.stdout or "").strip()
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise DirectoryPickerError("Selected directory is no longer available", details={"path": str(path)})
    return path


def _macos(title: str, initial_dir: Path | None) -> Path | None:
    script = """
on run argv
  set promptText to item 1 of argv
  set initialPath to item 2 of argv
  if initialPath is "" then
    set selectedFolder to choose folder with prompt promptText
  else
    set selectedFolder to choose folder with prompt promptText default location POSIX file initialPath
  end if
  return POSIX path of selectedFolder
end run
"""
    result = subprocess.run(
        ["/usr/bin/osascript", "-e", script, title, str(initial_dir) if initial_dir else ""],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return _finish(result)


def _linux(title: str, initial_dir: Path | None) -> Path | None:
    picker = shutil.which("zenity") or shutil.which("kdialog")
    if picker is None:
        raise DirectoryPickerError("No supported system directory picker is installed")
    if Path(picker).name == "zenity":
        command = [picker, "--file-selection", "--directory", f"--title={title}"]
        if initial_dir:
            command.append(f"--filename={initial_dir}/")
    else:
        command = [picker, "--getexistingdirectory", str(initial_dir or Path.home()), "--title", title]
    return _finish(subprocess.run(command, capture_output=True, text=True, timeout=300, check=False))


def _windows(title: str, initial_dir: Path | None) -> Path | None:
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if powershell is None:
        raise DirectoryPickerError("PowerShell is required for the system directory picker")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d=New-Object System.Windows.Forms.FolderBrowserDialog; "
        "$d.Description=$env:LABELONE_PICKER_TITLE; "
        "if($env:LABELONE_PICKER_INITIAL){$d.SelectedPath=$env:LABELONE_PICKER_INITIAL}; "
        "if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){Write-Output $d.SelectedPath}else{exit 1}"
    )
    environment = os.environ.copy()
    environment["LABELONE_PICKER_TITLE"] = title
    environment["LABELONE_PICKER_INITIAL"] = str(initial_dir) if initial_dir else ""
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=environment,
    )
    return _finish(result)


def pick_directory(title: str = "选择文件夹", initial_dir: Path | None = None) -> Path | None:
    safe_title = title.strip()[:160] or "选择文件夹"
    initial = initial_dir.expanduser().resolve() if initial_dir and initial_dir.expanduser().is_dir() else None
    try:
        if sys.platform == "darwin":
            return _macos(safe_title, initial)
        if sys.platform.startswith("win"):
            return _windows(safe_title, initial)
        return _linux(safe_title, initial)
    except subprocess.TimeoutExpired as exc:
        raise DirectoryPickerError("System directory picker timed out") from exc
