"""Render a framed still into a vertical video via ffmpeg.

Single static frame, no motion, no audio -- the "genuinely still" format. On a
truly static frame the loop point is invisible, so a short duration simply
means more loops inside the same dwell time.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)


class VideoError(RuntimeError):
    """ffmpeg was unavailable or failed to produce a valid file."""


def ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise VideoError("ffmpeg not found on PATH")
    return path


def render_still(
    image: Image.Image,
    out_path: Path,
    *,
    duration_seconds: float,
    fps: int = 30,
    crf: int = 18,
) -> Path:
    """Encode ``image`` as a still video of ``duration_seconds``."""
    ffmpeg = ensure_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_path = out_path.with_suffix(".frame.png")
    image.save(frame_path, format="PNG")

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-loop", "1",
        "-i", str(frame_path),
        "-t", f"{duration_seconds}",
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        # yuv420p + even dimensions is what every player and YouTube's
        # ingest actually accepts; without it some encoders emit yuv444p.
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        # Keyframe on every frame so a 1-2s clip seeks and loops cleanly.
        "-g", str(fps),
        "-movflags", "+faststart",
        "-an",
        str(out_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        raise VideoError("ffmpeg timed out") from exc
    finally:
        frame_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise VideoError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size < 1024:
        raise VideoError("ffmpeg produced no usable output file")

    log.info("rendered %s (%.1fs, %d bytes)", out_path.name, duration_seconds, out_path.stat().st_size)
    return out_path
