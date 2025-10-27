#!/usr/bin/env python3
"""Automate composing and submitting a Reddit comment using GroundingDINO."""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import logging

import mss
from PIL import Image
from pynput import keyboard, mouse
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
TEMP_DIR = TOOLS_ROOT / "temporary"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

GROUNDING_ROOT = REPO_ROOT / "GroundingDino"
GROUNDING_TOOLS = GROUNDING_ROOT / "tools"
GROUNDING_OUTPUT_DIR = GROUNDING_ROOT / "outputs"
GROUNDING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(GROUNDING_TOOLS) not in sys.path:
    sys.path.insert(0, str(GROUNDING_TOOLS))

from grounding_dino_runner import run_grounding_dino  # type: ignore  # noqa: E402


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    label: str
    box: Tuple[int, int, int, int]
    score: Optional[float]


@dataclass
class ValidatedPost:
    post: Detection
    title: Detection
    comment: Detection
    upvote: Detection
    downvote: Detection


@dataclass
class CommentComposerContext:
    composer: Detection
    submit: Detection
    cancel: Detection


@dataclass
class WriteCommentResult:
    post_image: Image.Image
    post_context: ValidatedPost
    composer_context: CommentComposerContext


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROMPT_POST_DETAILED = "postCardInDetail . postTitle . comment . upvote . downvote"
PROMPT_COMMENT_COMPOSER = "commentComposer . submitComment . cancelComment"

LABEL_POST = "postcardindetail"
LABEL_TITLE = "posttitle"
LABEL_COMMENT = "comment"
LABEL_UPVOTE = "upvote"
LABEL_DOWNVOTE = "downvote"
LABEL_COMPOSER = "commentcomposer"
LABEL_SUBMIT = "submitcomment"
LABEL_CANCEL = "cancelcomment"

TITLE_RELATIVE_CUTOFF = 0.4
BOTTOM_RELATIVE_CUTOFF = 0.6


mouse_controller = mouse.Controller()
keyboard_controller = keyboard.Controller()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def timestamped_path(prefix: str, ext: str = "png") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return TEMP_DIR / f"{prefix}_{ts}.{ext}"


def capture_monitor_image(
    monitor_index: int = 0,
) -> Tuple[Image.Image, Path, Dict[str, int]]:
    """Capture a monitor screenshot and persist it for GroundingDINO."""
    out_path = timestamped_path("screen")

    with mss.mss() as sct:
        monitors = sct.monitors
        idx = monitor_index
        if idx < 0 or idx >= len(monitors):
            idx = 0
        mon = dict(monitors[idx])
        grab = sct.grab(mon)
        img = Image.frombytes("RGB", grab.size, grab.rgb)
        img.save(out_path, format="PNG", optimize=False)

    return img, out_path, mon


def run_grounding(image_path: Path, prompt: str) -> Dict[str, Sequence]:
    return run_grounding_dino(
        image_path=image_path,
        text_prompt=prompt,
        output_dir=GROUNDING_OUTPUT_DIR,
    )


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


def _boxes_to_xyxy(
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


def detections_from_inference(
    image: Image.Image, inference: Dict[str, Sequence]
) -> List[Detection]:
    width, height = image.size
    boxes_xyxy = _boxes_to_xyxy(inference["boxes"], width, height)

    detections: List[Detection] = []
    for idx, raw_label in enumerate(inference["labels"]):
        label, score = _normalize_label(str(raw_label))
        x0, y0, x1, y1 = boxes_xyxy[idx].tolist()
        detections.append(Detection(label=label, score=score, box=(x0, y0, x1, y1)))
    return detections


def _box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x0, y0, x1, y1 = box
    return (x0 + x1) // 2, (y0 + y1) // 2


def _box_height(box: Tuple[int, int, int, int]) -> int:
    return max(0, box[3] - box[1])


def _contains(
    outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]
) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _relative_center_y(
    inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]
) -> float:
    outer_top, outer_bottom = outer[1], outer[3]
    if outer_bottom == outer_top:
        return 0.0
    center_y = (inner[1] + inner[3]) / 2.0
    return (center_y - outer_top) / (outer_bottom - outer_top)


def validate_post_structure(detections: Iterable[Detection]) -> Optional[ValidatedPost]:
    posts = [d for d in detections if d.label == LABEL_POST]
    if not posts:
        return None

    titles = [d for d in detections if d.label == LABEL_TITLE]
    comments = [d for d in detections if d.label == LABEL_COMMENT]
    upvotes = [d for d in detections if d.label == LABEL_UPVOTE]
    downvotes = [d for d in detections if d.label == LABEL_DOWNVOTE]

    for post in sorted(posts, key=lambda d: d.box[1]):
        title_candidates = [t for t in titles if _contains(post.box, t.box)]
        if len(title_candidates) != 1:
            continue
        title = title_candidates[0]
        title_rel = _relative_center_y(title.box, post.box)
        if title_rel > TITLE_RELATIVE_CUTOFF:
            continue

        comment_candidates = [c for c in comments if _contains(post.box, c.box)]
        upvote_candidates = [u for u in upvotes if _contains(post.box, u.box)]
        downvote_candidates = [d for d in downvotes if _contains(post.box, d.box)]

        if not (comment_candidates and upvote_candidates and downvote_candidates):
            continue

        comment = comment_candidates[0]
        upvote = upvote_candidates[0]
        downvote = downvote_candidates[0]

        for candidate in (comment, upvote, downvote):
            rel = _relative_center_y(candidate.box, post.box)
            if rel < BOTTOM_RELATIVE_CUTOFF:
                break
        else:
            return ValidatedPost(
                post=post,
                title=title,
                comment=comment,
                upvote=upvote,
                downvote=downvote,
            )

    return None


def crop_post_image(image: Image.Image, post: Detection) -> Image.Image:
    x0, y0, x1, y1 = post.box
    # Include the bottom edge when cropping so controls remain visible.
    return image.crop((x0, y0, x1 + 1, y1 + 1))


def validate_comment_composer(
    detections: Iterable[Detection],
) -> Optional[CommentComposerContext]:
    composers = [d for d in detections if d.label == LABEL_COMPOSER]
    submits = [d for d in detections if d.label == LABEL_SUBMIT]
    cancels = [d for d in detections if d.label == LABEL_CANCEL]

    if not composers or not submits or not cancels:
        return None

    for composer in sorted(composers, key=lambda d: d.box[1]):
        submit_candidates = [s for s in submits if _contains(composer.box, s.box)]
        cancel_candidates = [c for c in cancels if _contains(composer.box, c.box)]
        if submit_candidates and cancel_candidates:
            return CommentComposerContext(
                composer=composer,
                submit=submit_candidates[0],
                cancel=cancel_candidates[0],
            )
    return None


def _adjust_point(
    point: Tuple[int, int], monitor_box: Dict[str, int]
) -> Tuple[int, int]:
    left = int(monitor_box.get("left", 0))
    top = int(monitor_box.get("top", 0))
    return point[0] + left, point[1] + top


def _click_point(point: Tuple[int, int]) -> None:
    mouse_controller.position = point
    mouse_controller.click(mouse.Button.left)


def _paste_text(text: str) -> None:
    try:
        import pyperclip  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        pyperclip = None  # type: ignore

    if pyperclip:
        try:
            pyperclip.copy(text)
        except Exception:
            pyperclip = None  # type: ignore

    if pyperclip:
        modifier = (
            keyboard.Key.cmd if platform.system() == "Darwin" else keyboard.Key.ctrl
        )
        with keyboard_controller.pressed(modifier):  # type: ignore[attr-defined]
            keyboard_controller.press("v")
            keyboard_controller.release("v")
    else:
        keyboard_controller.type(text)


def write_reddit_comment(
    comment_text: str,
    *,
    monitor_index: int = 0,
    comment_delay: float = 1.0,
    auto_submit: bool = False,
) -> WriteCommentResult:
    """Locate a Reddit post, open the composer, and paste/optionally submit a comment."""

    logger.info("Begin comment workflow (auto_submit=%s)", auto_submit)

    image, screenshot_path, monitor_box = capture_monitor_image(monitor_index)
    try:
        inference = run_grounding(screenshot_path, PROMPT_POST_DETAILED)
    finally:
        screenshot_path.unlink(missing_ok=True)

    detections = detections_from_inference(image, inference)
    post_context = validate_post_structure(detections)
    if not post_context:
        logger.warning("Post validation failed; 'postCardInDetail' structure not found")
        raise RuntimeError("Failed to validate a Reddit post in the screenshot.")

    post_image = crop_post_image(image, post_context.post)
    logger.info("Validated post; clicking comment control")

    comment_center = _box_center(post_context.comment.box)
    _click_point(_adjust_point(comment_center, monitor_box))

    time.sleep(max(0.1, comment_delay))

    second_image, second_path, second_monitor_box = capture_monitor_image(monitor_index)
    try:
        composer_inference = run_grounding(second_path, PROMPT_COMMENT_COMPOSER)
    finally:
        second_path.unlink(missing_ok=True)

    composer_detections = detections_from_inference(second_image, composer_inference)
    composer_context = validate_comment_composer(composer_detections)
    if not composer_context:
        logger.warning("Comment composer validation failed; required controls missing")
        raise RuntimeError("Failed to validate the Reddit comment composer.")

    composer_center = _box_center(composer_context.composer.box)
    _click_point(_adjust_point(composer_center, second_monitor_box))
    time.sleep(0.2)

    logger.info("Pasting comment text (%d characters)", len(comment_text))
    _paste_text(comment_text)
    time.sleep(0.1)

    submit_center = _box_center(composer_context.submit.box)
    if auto_submit:
        logger.info("Auto-submit enabled; clicking submit button")
        _click_point(_adjust_point(submit_center, second_monitor_box))
    else:
        mouse_controller.position = _adjust_point(submit_center, second_monitor_box)
        logger.info("Auto-submit disabled; leaving comment for manual review")

    return WriteCommentResult(
        post_image=post_image,
        post_context=post_context,
        composer_context=composer_context,
    )


__all__ = [
    "write_reddit_comment",
    "ValidatedPost",
    "CommentComposerContext",
    "WriteCommentResult",
]

if __name__ == "__main__":
    time.sleep(2)
    write_reddit_comment("hello world, this is a test")
