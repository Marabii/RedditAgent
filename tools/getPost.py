#!/usr/bin/env python3
"""Capture Reddit posts via GroundingDINO without relying on hotkeys.

Callers can invoke the public `capture_next_post` function to locate the next
fully-visible post on screen, crop it, and persist both the image and metadata
for downstream automation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from difflib import SequenceMatcher

import torch
from PIL import Image
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

LAST_POST_INFO_PATH = TEMPORARY_DIR / "lastPostInfo.json"
RECENT_TITLES_PATH = TEMPORARY_DIR / "recentTitles.json"

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
PROMPT = "postCard . upvote . downvote . comment . postTitle"
LABEL_POST = "postcard"
LABEL_UPVOTE = "upvote"
LABEL_DOWNVOTE = "downvote"
LABEL_COMMENT = "comment"
LABEL_POSTITLE = "posttitle"

SCROLL_STEP = -2  # negative scrolls downward (showing content further down)
SCROLL_DELAY = 0.05  # seconds between scroll attempts
POST_MARGIN = 4  # px margin used when checking visibility

mouse_controller = mouse.Controller()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    label: str
    box: Tuple[int, int, int, int]  # x0, y0, x1, y1 in pixel space
    score: Optional[float]


@dataclass
class PostSelection:
    post: Detection
    upvote: Detection
    downvote: Detection
    comment: Detection


@dataclass
class PostSelectionWithTitle:
    selection: PostSelection
    title_text: str


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


def read_last_post_info() -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
    if not LAST_POST_INFO_PATH.exists():
        return "", None
    try:
        data = json.loads(LAST_POST_INFO_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        print(f"[post-cache] failed to read last post info: {err}")
        return "", None

    title = clean_title(str(data.get("title", "")))
    bbox = _bbox_from_data(data.get("bbox"))

    return title, bbox


def read_recent_titles(max_titles: int = 3) -> List[str]:
    if not RECENT_TITLES_PATH.exists():
        return []
    try:
        data = json.loads(RECENT_TITLES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []

    titles: List[str] = []
    for entry in data:
        if isinstance(entry, str) and entry.strip():
            titles.append(clean_title(entry))
        if len(titles) >= max_titles:
            break
    return titles


def push_recent_title(title: str, max_titles: int = 3) -> None:
    title = clean_title(title)
    if not title:
        return
    titles = read_recent_titles(max_titles=max_titles)
    existing = [t for t in titles if t.lower() != title.lower()]
    updated = [title] + existing
    trimmed = updated[:max_titles]
    try:
        RECENT_TITLES_PATH.write_text(
            json.dumps(trimmed, ensure_ascii=True, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _bbox_to_list(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[List[int]]:
    if not bbox:
        return None
    return [int(v) for v in bbox]


def _bbox_from_data(
    raw: object,
) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    if not all(isinstance(v, (int, float)) for v in raw):
        return None
    return tuple(int(v) for v in raw)  # type: ignore[return-value]


def write_last_post_info(
    title: str,
    bbox: Optional[Tuple[int, int, int, int]],
    *,
    upvote_bbox: Optional[Tuple[int, int, int, int]] = None,
    downvote_bbox: Optional[Tuple[int, int, int, int]] = None,
    comment_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    payload = {
        "title": clean_title(title),
        "bbox": _bbox_to_list(bbox),
        "upvote_bbox": _bbox_to_list(upvote_bbox),
        "downvote_bbox": _bbox_to_list(downvote_bbox),
        "comment_bbox": _bbox_to_list(comment_bbox),
    }
    try:
        LAST_POST_INFO_PATH.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )
    except OSError as err:
        print(f"[post-cache] failed to write last post info: {err}")


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


def grab_as_pil(sct: MSSBase, monitor_box: dict) -> Image.Image:
    shot = sct.grab(monitor_box)
    return Image.frombytes("RGB", shot.size, shot.rgb)


def capture_monitor_image(monitor_index: int = 0) -> Tuple[Image.Image, Path]:
    """Capture the requested monitor and persist it as PNG."""
    out_path = timestamped_name(prefix="screen")

    with mss.mss() as sct:
        monitors = sct.monitors
        idx = monitor_index
        if idx < 0 or idx >= len(monitors):
            print(
                f"[warn] Monitor index {idx} invalid. Falling back to ALL (0).",
                file=sys.stderr,
            )
            idx = 0

        img = grab_as_pil(sct, monitors[idx])
        img.save(out_path, format="PNG", optimize=False)

    return img, out_path


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
    image: Image.Image,
    image_height: int,
    recent_titles: Sequence[str],
) -> Optional[PostSelectionWithTitle]:
    posts = [d for d in detections if d.label == LABEL_POST]
    if not posts:
        return None

    upvotes = [d for d in detections if d.label == LABEL_UPVOTE]
    downvotes = [d for d in detections if d.label == LABEL_DOWNVOTE]
    comments = [d for d in detections if d.label == LABEL_COMMENT]

    posts.sort(key=lambda d: d.box[1])  # top-most first

    normalized_recent = [normalize_title(title) for title in recent_titles if title]

    for post in posts:
        if not post_fully_visible(post.box, image_height):
            if SAVE_FAILED_ATTEMPTS:
                print(
                    f"skip post @ y={post.box[1]}..{post.box[3]} because not fully visible."
                )
            continue

        ups_in_post = [up for up in upvotes if box_contains(post.box, up.box)]
        downs_in_post = [dn for dn in downvotes if box_contains(post.box, dn.box)]
        comments_in_post = [cm for cm in comments if box_contains(post.box, cm.box)]

        if SAVE_FAILED_ATTEMPTS:
            print(
                (
                    f"post @ y={post.box[1]}..{post.box[3]} has "
                    f"{len(ups_in_post)} upvote(s), {len(downs_in_post)} downvote(s), "
                    f"and {len(comments_in_post)} comment button(s)."
                )
            )

        if (
            len(ups_in_post) != 1
            or len(downs_in_post) != 1
            or len(comments_in_post) != 1
        ):
            continue

        title_detection = find_title_in_post(post, detections)
        if not title_detection:
            continue

        title_text = extract_title_text(image, title_detection.box)
        if not title_text.strip():
            if SAVE_FAILED_ATTEMPTS:
                print("Skipping post because extracted title was empty")
            continue
        normalized_title = normalize_title(title_text)

        duplicate_found = False
        if normalized_title:
            for candidate in normalized_recent:
                if not candidate:
                    continue
                similarity = SequenceMatcher(None, normalized_title, candidate).ratio()
                print(f"Title similarity vs recent: {similarity:.3f}")
                if similarity >= 0.9:
                    duplicate_found = True
                    break
        if duplicate_found:
            if SAVE_FAILED_ATTEMPTS:
                print("Detected recently captured post; searching for another title.")
            continue

        return PostSelectionWithTitle(
            selection=PostSelection(
                post=post,
                upvote=ups_in_post[0],
                downvote=downs_in_post[0],
                comment=comments_in_post[0],
            ),
            title_text=title_text,
        )

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
def capture_next_post(
    monitor_index: int = 0,
) -> Optional[Dict[str, object]]:
    last_title, _ = read_last_post_info()
    recent_titles = read_recent_titles()
    if last_title:
        normalized_last = clean_title(last_title)
        recent_titles = [normalized_last] + [
            t for t in recent_titles if t.lower() != normalized_last.lower()
        ]
        recent_titles = recent_titles[:3]
    attempt = 1
    while True:
        image, screenshot_path = capture_monitor_image(monitor_index)

        try:
            inference = run_grounding(screenshot_path)
        except Exception as err:  # pragma: no cover - provides user feedback
            print(f"[grounding] failed: {err}")
            screenshot_path.unlink(missing_ok=True)
            return None

        detections = gather_detections(image, inference)
        selection_with_title = select_next_post(
            detections,
            image=image,
            image_height=image.size[1],
            recent_titles=recent_titles,
        )

        if selection_with_title:
            selection = selection_with_title.selection
            title_text = selection_with_title.title_text

            crop = image.crop(selection.post.box)
            out_path = timestamped_name(prefix="post")
            crop.save(out_path, format="PNG", optimize=False)
            print(
                f"Saved post @ y={selection.post.box[1]}..{selection.post.box[3]} to {out_path.name}"
            )

            write_last_post_info(
                title_text,
                selection.post.box,
                upvote_bbox=selection.upvote.box,
                downvote_bbox=selection.downvote.box,
                comment_bbox=selection.comment.box,
            )
            last_title = title_text
            push_recent_title(title_text)
            recent_titles = read_recent_titles()

            screenshot_path.unlink(missing_ok=True)
            result = {
                "image_path": out_path,
                "title": title_text,
                "bbox": selection.post.box,
                "upvote_bbox": selection.upvote.box,
                "downvote_bbox": selection.downvote.box,
                "comment_bbox": selection.comment.box,
            }
            break

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
        mouse_controller.scroll(0, SCROLL_STEP)
        print("------- scrolling----------")
        print(
            f"Attempt {attempt}: no fully visible post. Scrolling and retrying in {SCROLL_DELAY}s…"
        )
        time.sleep(SCROLL_DELAY)

    if result is not None:
        mouse_controller.scroll(0, -SCROLL_STEP)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    result = capture_next_post()
    if result:
        print(
            f"Captured post image saved to {result['image_path']} with title '{result['title']}'."
        )
    else:
        print("No valid post captured.")


if __name__ == "__main__":
    main()
