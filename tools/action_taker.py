#!/usr/bin/env python3
"""Trigger Reddit post interactions using stored bounding boxes."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Tuple
from pynput import mouse


REPO_ROOT = Path(__file__).resolve().parent
TEMPORARY_DIR = REPO_ROOT / "temporary"
LAST_POST_INFO_PATH = TEMPORARY_DIR / "lastPostInfo.json"
HYPERPARAM_X = 0
HYPERPARAM_Y = 0
mouse_controller = mouse.Controller()


def _load_last_post_info() -> dict:
    try:
        raw = LAST_POST_INFO_PATH.read_text(encoding="utf-8")
    except OSError as err:
        raise RuntimeError(f"failed to read post info: {err}") from err

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"post info is not valid JSON: {err}") from err

    return data


def _box_center(box: Sequence[int]) -> Tuple[int, int]:
    if len(box) != 4:
        raise ValueError(f"unexpected bbox length: {box}")
    x0, y0, x1, y1 = (int(v) for v in box)
    return (x0 + x1) // 2 - HYPERPARAM_X, (y0 + y1) // 2 - HYPERPARAM_Y


def _move_and_click(point: Tuple[int, int], clicks: int = 1) -> None:
    mouse_controller.position = point
    for _ in range(clicks):
        mouse_controller.click(mouse.Button.left)


def _ensure_bbox(data: dict, key: str) -> Sequence[int]:
    box = data.get(key)
    if not isinstance(box, Iterable):
        raise RuntimeError(f"missing '{key}' in post info")
    box_list = list(box)
    if len(box_list) != 4:
        raise RuntimeError(f"'{key}' must contain four numbers")
    return box_list


def upvote_post() -> None:
    """Click the stored upvote button for the last captured post."""

    info = _load_last_post_info()
    bbox = _ensure_bbox(info, "upvote_bbox")
    _move_and_click(_box_center(bbox))


def downvote_post() -> None:
    """Click the stored downvote button for the last captured post."""

    info = _load_last_post_info()
    bbox = _ensure_bbox(info, "downvote_bbox")
    _move_and_click(_box_center(bbox))


def comment_on_post() -> None:
    info = _load_last_post_info()
    bbox = _ensure_bbox(info, "comment_bbox")
    _move_and_click(_box_center(bbox))
