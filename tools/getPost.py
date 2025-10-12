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

import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from difflib import SequenceMatcher

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
TEMPORARY_DIR = REPO_ROOT / "tools" / "temporary"
OUTPUT_DIR = TEMPORARY_DIR / "posts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAILED_OUTPUT_DIR = TEMPORARY_DIR / "failed_attempts"
FAILED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_FAILED_ATTEMPTS = False

LAST_TITLE_PATH = TEMPORARY_DIR / "lastPostTitle.txt"

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
PROMPT = "postCard . upvote . downvote . postTitle"
LABEL_POST = "postcard"
LABEL_UPVOTE = "upvote"
LABEL_DOWNVOTE = "downvote"
LABEL_POSTITLE = "posttitle"

SCROLL_STEP = -2  # negative scrolls downward (showing content further down)
SCROLL_DELAY = 0.05  # seconds between scroll attempts
MAX_SCROLL_ATTEMPTS = 50
POST_MARGIN = 4  # px margin used when checking visibility

current_monitor_index: int = 0  # 0 = virtual desktop (all monitors)
process_active: bool = False
stop_requested: bool = False
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


def clean_title(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_title(text: str) -> str:
    return clean_title(text).casefold()


def read_last_post_title() -> str:
    if not LAST_TITLE_PATH.exists():
        return ""
    try:
        return clean_title(LAST_TITLE_PATH.read_text(encoding="utf-8"))
    except OSError as err:
        print(f"[title-cache] failed to read last title: {err}")
        return ""


def write_last_post_title(title: str) -> None:
    try:
        LAST_TITLE_PATH.write_text(clean_title(title), encoding="utf-8")
    except OSError as err:
        print(f"[title-cache] failed to write last title: {err}")


def extract_title_text(image: Image.Image, box: Tuple[int, int, int, int]) -> str:
    crop = image.crop(box)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        crop.save(tmp_path, format="PNG", optimize=False)

    try:
        result = subprocess.run(
            ["tesseract", str(tmp_path), "stdout"],
            check=True,
            capture_output=True,
            text=True,
        )
        text = result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        print(f"[ocr] failed to read post title: {err}")
        text = ""
    finally:
        tmp_path.unlink(missing_ok=True)

    return clean_title(text)


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


def find_title_in_post(
    post: Detection, detections: Iterable[Detection]
) -> Optional[Detection]:
    titles = [d for d in detections if d.label == LABEL_POSTITLE]
    titles_in_post = [d for d in titles if box_contains(post.box, d.box)]
    if len(titles_in_post) != 1:
        if SAVE_FAILED_ATTEMPTS and titles_in_post:
            print(
                f"post @ y={post.box[1]}..{post.box[3]} has {len(titles_in_post)} title detections; expected exactly 1."
            )
        return None
    return titles_in_post[0]


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
) -> Optional[Detection]:
    posts = [d for d in detections if d.label == LABEL_POST]
    if not posts:
        return None

    upvotes = [d for d in detections if d.label == LABEL_UPVOTE]
    downvotes = [d for d in detections if d.label == LABEL_DOWNVOTE]

    posts.sort(key=lambda d: d.box[1])  # top-most first

    for post in posts:
        if not post_fully_visible(post.box, image_height):
            if SAVE_FAILED_ATTEMPTS:
                print(
                    f"skip post @ y={post.box[1]}..{post.box[3]} because not fully visible."
                )
            continue

        ups_in_post = [up for up in upvotes if box_contains(post.box, up.box)]
        downs_in_post = [dn for dn in downvotes if box_contains(post.box, dn.box)]

        if SAVE_FAILED_ATTEMPTS:
            print(
                f"post @ y={post.box[1]}..{post.box[3]} has {len(ups_in_post)} upvote(s) and {len(downs_in_post)} downvote(s)."
            )

        if len(ups_in_post) != 1 or len(downs_in_post) != 1:
            continue

        title_detection = find_title_in_post(post, detections)
        if not title_detection:
            continue

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
def capture_and_save_post() -> bool:
    global stop_requested

    last_title = read_last_post_title()
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
        candidate = select_next_post(detections, image_height=image.size[1])

        if candidate:
            title_detection = find_title_in_post(candidate, detections)
            title_text = ""
            if title_detection:
                title_text = extract_title_text(image, title_detection.box)
            else:
                print("[ocr] no postTitle detection inside the selected post.")

            normalized_title = normalize_title(title_text)
            normalized_last = normalize_title(last_title)
            if normalized_title and normalized_last:
                similarity = SequenceMatcher(
                    None, normalized_title, normalized_last
                ).ratio()
                print(f"Title similarity: {similarity:.3f}")
                if similarity >= 0.9:
                    screenshot_path.unlink(missing_ok=True)
                    mouse_controller.scroll(0, SCROLL_STEP)
                    time.sleep(SCROLL_DELAY)
                    print(
                        "Detected previously captured post. Scrolling to search again…"
                    )
                    continue

            crop = image.crop(candidate.box)
            out_path = timestamped_name(prefix="post")
            crop.save(out_path, format="PNG", optimize=False)
            print(
                f"Saved post @ y={candidate.box[1]}..{candidate.box[3]} to {out_path.name}"
            )

            if title_text:
                write_last_post_title(title_text)
                last_title = title_text
            else:
                write_last_post_title("")
                last_title = ""

            screenshot_path.unlink(missing_ok=True)
            return True

        # No candidate — clean up and scroll for another attempt.
        if SAVE_FAILED_ATTEMPTS:
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


def _capture_worker() -> None:
    global capture_thread
    try:
        capture_and_save_post()
    finally:
        capture_thread = None


def launch_capture() -> None:
    global capture_thread

    if capture_thread and capture_thread.is_alive():
        print("Capture already in progress; wait for it to finish.")
        return

    thread = threading.Thread(
        target=_capture_worker,
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
                launch_capture()
                return

            if c == "n":
                if not process_active:
                    print("Press 's' first to start the capture loop.")
                    return
                stop_requested = False
                launch_capture()
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
