#!/usr/bin/env python3
"""Capture Reddit posts via GroundingDINO when hotkeys are pressed.

Workflow
--------
1. Press `s` to start the capture loop. The current fully-visible post (if any)
   is located, cropped, and saved into the `temporary/` directory.
2. Press `n` to advance to the next post. The script scrolls as needed until it
   finds the next fully-visible post that also contains both the upvote and
   downvote buttons inside the detected `postCard` region.

Press `q` or `Esc` at any time to quit. Monitor-selection hotkeys from the
original screenshot helper (`a`, `1`..`9`) are still available.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from pynput import keyboard  # type: ignore
from pynput import mouse
import mss
from mss.base import MSSBase


# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "temporary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAILED_OUTPUT_DIR = OUTPUT_DIR / "failed_attempts"
FAILED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUNDING_ROOT = REPO_ROOT / "GroundingDino"
GROUNDING_OUTPUT_DIR = GROUNDING_ROOT / "outputs"
GROUNDING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Make GroundingDINO tools importable.
GROUNDING_TOOLS = GROUNDING_ROOT / "tools"
if str(GROUNDING_TOOLS) not in sys.path:
    sys.path.insert(0, str(GROUNDING_TOOLS))

from grounding_dino_runner import (  # type: ignore  # noqa: E402
    crop_or_draw,
    run_grounding_dino,
)


# ---------------------------------------------------------------------------
# Constants and global state
# ---------------------------------------------------------------------------
PROMPT = "postCard . upvote . downvote"
LABEL_POST = "postcard"
LABEL_UPVOTE = "upvote"
LABEL_DOWNVOTE = "downvote"

SCROLL_STEP = -3  # negative scrolls downward (showing content further down)
SCROLL_DELAY = 0.75  # seconds between scroll attempts
MAX_SCROLL_ATTEMPTS = 12
POST_MARGIN = 4  # px margin used when checking visibility
NEW_POST_GAP = 12  # px gap required between previously saved post and next one

current_monitor_index: int = 0  # 0 = virtual desktop (all monitors)
process_active: bool = False
stop_requested: bool = False
last_captured_bottom: int = -NEW_POST_GAP
capture_thread: Optional[threading.Thread] = None
mouse_controller = mouse.Controller()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    label: str
    box: Tuple[int, int, int, int]  # x0, y0, x1, y1 in pixel space
    score: Optional[float]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def timestamped_name(prefix: str = "post", ext: str = "png") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{prefix}_{ts}.{ext}"


def describe_monitors(monitors: Sequence[dict]) -> str:
    lines = ["Detected monitors:"]
    for i, mon in enumerate(monitors):
        tag = "ALL" if i == 0 else f"{i}"
        lines.append(
            f"  {tag}: {mon['width']}x{mon['height']} at ({mon['left']},{mon['top']})"
        )
    return "\n".join(lines)


def grab_as_pil(sct: MSSBase, monitor_box: dict) -> Image.Image:
    shot = sct.grab(monitor_box)
    return Image.frombytes("RGB", shot.size, shot.rgb)


def capture_monitor_image() -> Tuple[Image.Image, Path]:
    """Capture the currently selected monitor and persist it as PNG."""
    global current_monitor_index

    out_path = timestamped_name(prefix="screen")

    with mss.mss() as sct:
        monitors = sct.monitors
        idx = current_monitor_index
        if idx < 0 or idx >= len(monitors):
            print(
                f"[warn] Monitor index {idx} invalid. Falling back to ALL (0).",
                file=sys.stderr,
            )
            idx = 0

        img = grab_as_pil(sct, monitors[idx])
        img.save(out_path, format="PNG", optimize=False)

    return img, out_path


def print_help(monitors: Sequence[dict]) -> None:
    print(
        "\nHotkeys:\n"
        "  • s : start capture loop (grab first visible post)\n"
        "  • n : capture next visible post\n"
        "  • a : capture ALL monitors (virtual desktop)\n"
        "  • 1..9 : choose specific monitor\n"
        "  • q or Esc : quit\n"
    )
    print(describe_monitors(monitors))
    print(
        f"\nCurrent target: {'ALL' if current_monitor_index == 0 else current_monitor_index}\n"
    )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def _normalize_label(raw: str) -> Tuple[str, Optional[float]]:
    raw = raw.strip()
    if "(" in raw and raw.endswith(")"):
        base, score_part = raw.rsplit("(", 1)
        base = base.strip().lower()
        try:
            return base, float(score_part[:-1])
        except ValueError:
            return base, None
    return raw.lower(), None


def boxes_to_xyxy(
    boxes: Sequence[Sequence[float]], width: int, height: int
) -> torch.Tensor:
    tensor_boxes = (
        boxes
        if isinstance(boxes, torch.Tensor)
        else torch.tensor(boxes, dtype=torch.float32)
    )
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    boxes_abs = tensor_boxes * scale
    boxes_abs[:, :2] -= boxes_abs[:, 2:] / 2.0
    boxes_abs[:, 2:] += boxes_abs[:, :2]

    x0y0 = torch.maximum(boxes_abs[:, :2], torch.tensor([0.0, 0.0]))
    x1y1 = torch.minimum(
        boxes_abs[:, 2:], torch.tensor([float(width - 1), float(height - 1)])
    )
    return torch.cat([x0y0, x1y1], dim=1).round().to(torch.int64)


def gather_detections(
    image: Image.Image, inference: Dict[str, Sequence]
) -> List[Detection]:
    width, height = image.size
    boxes_xyxy = boxes_to_xyxy(inference["boxes"], width, height)

    detections: List[Detection] = []
    for i, raw_label in enumerate(inference["labels"]):
        label, score = _normalize_label(str(raw_label))
        x0, y0, x1, y1 = boxes_xyxy[i].tolist()
        detections.append(Detection(label=label, score=score, box=(x0, y0, x1, y1)))

    print("detections: ", detections)
    return detections


def box_contains(
    outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]
) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def post_fully_visible(post_box: Tuple[int, int, int, int], height: int) -> bool:
    isVisible = post_box[1] >= POST_MARGIN and post_box[3] <= height - POST_MARGIN
    print("isVisible: ", isVisible)
    return isVisible


def select_next_post(
    detections: Iterable[Detection],
    *,
    image_height: int,
    min_top: int,
) -> Optional[Detection]:
    posts = [d for d in detections if d.label == LABEL_POST]
    if not posts:
        return None

    upvotes = [d for d in detections if d.label == LABEL_UPVOTE]
    downvotes = [d for d in detections if d.label == LABEL_DOWNVOTE]

    posts.sort(key=lambda d: d.box[1])  # top-most first

    for post in posts:
        if post.box[1] <= min_top:
            continue
        if not post_fully_visible(post.box, image_height):
            continue

        up_inside = any(box_contains(post.box, up.box) for up in upvotes)
        down_inside = any(box_contains(post.box, down.box) for down in downvotes)

        if up_inside and down_inside:
            return post

    return None


def run_grounding(image_path: Path) -> Dict[str, Sequence]:
    return run_grounding_dino(
        image_path=image_path,
        text_prompt=PROMPT,
        output_dir=GROUNDING_OUTPUT_DIR,
    )


# ---------------------------------------------------------------------------
# Core capture logic
# ---------------------------------------------------------------------------
def capture_and_save_post(*, reset_position: bool = False) -> bool:
    global last_captured_bottom, stop_requested

    if reset_position:
        last_captured_bottom = -NEW_POST_GAP

    for attempt in range(1, MAX_SCROLL_ATTEMPTS + 1):
        if stop_requested:
            print("Capture interrupted.")
            return False
        image, screenshot_path = capture_monitor_image()

        try:
            inference = run_grounding(screenshot_path)
        except Exception as err:  # pragma: no cover - provides user feedback
            print(f"[grounding] failed: {err}")
            screenshot_path.unlink(missing_ok=True)
            return False

        if stop_requested:
            screenshot_path.unlink(missing_ok=True)
            print("Capture interrupted.")
            return False

        detections = gather_detections(image, inference)
        candidate = select_next_post(
            detections,
            image_height=image.size[1],
            min_top=last_captured_bottom + NEW_POST_GAP,
        )

        if candidate:
            crop = image.crop(candidate.box)
            out_path = timestamped_name(prefix="post")
            crop.save(out_path, format="PNG", optimize=False)
            last_captured_bottom = candidate.box[3]

            print(
                f"Saved post @ y={candidate.box[1]}..{candidate.box[3]} to {out_path.name}"
            )

            screenshot_path.unlink(missing_ok=True)
            return True

        # No candidate — clean up and scroll for another attempt.
        try:
            crop_or_draw(
                image_path=str(screenshot_path),
                tgt=inference,
                mode="draw",
                output_dir=FAILED_OUTPUT_DIR,
            )
            print(
                f"Attempt {attempt}: saved debug annotation to {FAILED_OUTPUT_DIR.name}."
            )
        except Exception as dbg_err:
            print(f"[debug] failed to save annotation: {dbg_err}")

        screenshot_path.unlink(missing_ok=True)
        if stop_requested:
            print("Capture interrupted.")
            return False
        mouse_controller.scroll(0, SCROLL_STEP)
        print(
            f"Attempt {attempt}: no fully visible post. Scrolling and retrying in {SCROLL_DELAY}s…"
        )
        time.sleep(SCROLL_DELAY)

    print("Exhausted scroll attempts without finding a valid post.")
    return False


def _capture_worker(reset_position: bool) -> None:
    global capture_thread
    try:
        capture_and_save_post(reset_position=reset_position)
    finally:
        capture_thread = None


def launch_capture(reset_position: bool) -> None:
    global capture_thread

    if capture_thread and capture_thread.is_alive():
        print("Capture already in progress; wait for it to finish.")
        return

    thread = threading.Thread(
        target=_capture_worker,
        args=(reset_position,),
        name="post_capture",
        daemon=True,
    )
    capture_thread = thread
    thread.start()


# ---------------------------------------------------------------------------
# Hotkey handling
# ---------------------------------------------------------------------------
def on_press(key):
    global current_monitor_index, process_active, stop_requested

    try:
        if key.char:
            c = key.char.lower()

            if c == "s":
                stop_requested = False
                process_active = True
                print("Starting capture loop…")
                launch_capture(reset_position=True)
                return

            if c == "n":
                if not process_active:
                    print("Press 's' first to start the capture loop.")
                    return
                stop_requested = False
                launch_capture(reset_position=False)
                return

            if c == "q":
                print("Quitting…")
                stop_requested = True
                return None

            if c == "a":
                current_monitor_index = 0
                with mss.mss() as sct:
                    mon = sct.monitors[0]
                    print(f"Switched to ALL monitors ({mon['width']}x{mon['height']}).")
                return

            if c.isdigit() and c != "0":
                new_idx = int(c)
                with mss.mss() as sct:
                    if new_idx < len(sct.monitors):
                        current_monitor_index = new_idx
                        mon = sct.monitors[new_idx]
                        process_active = False
                        stop_requested = False
                        print(
                            f"Switched to monitor {new_idx}: {mon['width']}x{mon['height']}"
                            f" at ({mon['left']},{mon['top']}). Capture loop paused."
                        )
                    else:
                        print(
                            f"No monitor {new_idx}. Available: 1..{len(sct.monitors) - 1} or 'a'."
                        )
                return

    except AttributeError:
        pass

    if key == keyboard.Key.esc:
        stop_requested = True
        print("Quitting…")
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    global stop_requested
    with mss.mss() as sct:
        print(
            "Ready. Press 's' to start capturing posts, 'n' for the next one, 'q' to quit."
        )
        print_help(sct.monitors)

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    if capture_thread and capture_thread.is_alive():
        stop_requested = True
        capture_thread.join()


if __name__ == "__main__":
    main()
