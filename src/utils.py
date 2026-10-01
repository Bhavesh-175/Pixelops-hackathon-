"""
utils.py
========
Shared helpers used by every stage. Centralising subprocess handling here
means every COLMAP call is checked the same way - a non-zero exit code
always raises, it is never silently swallowed.
"""

from __future__ import annotations
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import config


class PipelineStageError(RuntimeError):
    """Raised when a pipeline stage cannot honestly proceed."""


def setup_logging() -> logging.Logger:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pixelops")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(config.LOG_PATH, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = setup_logging()


def ensure_all_dirs() -> None:
    for d in config.ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def check_colmap_installed() -> str:
    """Returns the resolved COLMAP executable path or raises PipelineStageError."""
    exe = shutil.which(config.COLMAP_BIN) or shutil.which("colmap")
    if exe is None:
        raise PipelineStageError(
            "COLMAP executable not found on PATH. Install COLMAP and either add it "
            "to PATH or set config.COLMAP_BIN to the full path of colmap.exe / colmap.bat."
        )
    try:
        result = subprocess.run([exe, "-h"], capture_output=True, text=True, timeout=30)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError(f"COLMAP found at {exe} but failed to execute: {exc}") from exc
    if result.returncode not in (0, 1):  # colmap -h sometimes exits 1
        raise PipelineStageError(f"COLMAP at {exe} returned unexpected exit code {result.returncode}")
    log.info(f"COLMAP OK: {exe}")
    return exe


def gpu_available() -> bool:
    """Best-effort NVIDIA GPU detection. Returns False (never guesses True)."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        result = subprocess.run([nvidia_smi], capture_output=True, text=True, timeout=15)
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def run_cmd(cmd: Sequence[str], stage: str, cwd: Path | None = None,
            timeout: int | None = None) -> subprocess.CompletedProcess:
    """
    Run an external command (COLMAP, ffprobe, ...) and raise a clear,
    diagnosable error on failure instead of continuing with bad state.
    """
    printable = " ".join(str(c) for c in cmd)
    log.debug(f"[{stage}] RUN: {printable}")
    start = time.time()
    try:
        result = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise PipelineStageError(f"[{stage}] executable not found: {cmd[0]} ({exc})") from exc
    except subprocess.TimeoutExpired as exc:
        raise PipelineStageError(f"[{stage}] timed out after {timeout}s: {printable}") from exc

    elapsed = time.time() - start
    log.debug(f"[{stage}] finished in {elapsed:.1f}s, exit={result.returncode}")

    if result.returncode != 0:
        tail_out = "\n".join(result.stdout.strip().splitlines()[-25:])
        tail_err = "\n".join(result.stderr.strip().splitlines()[-25:])
        raise PipelineStageError(
            f"[{stage}] command failed (exit {result.returncode}): {printable}\n"
            f"--- stdout (tail) ---\n{tail_out}\n"
            f"--- stderr (tail) ---\n{tail_err}"
        )
    return result


class Timer:
    """Context manager that logs and records elapsed wall time for a stage."""

    def __init__(self, name: str, timings: dict):
        self.name = name
        self.timings = timings

    def __enter__(self):
        self._t0 = time.time()
        log.info(f"=== STAGE START: {self.name} ===")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self._t0
        self.timings[self.name] = round(elapsed, 2)
        if exc_type is None:
            log.info(f"=== STAGE OK: {self.name} ({elapsed:.1f}s) ===")
        else:
            log.error(f"=== STAGE FAILED: {self.name} after {elapsed:.1f}s: {exc} ===")
        return False  # never swallow the exception
