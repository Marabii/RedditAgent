#!/usr/bin/env python3
"""Utilities for capturing post detail images and querying Perplexity."""

from __future__ import annotations

import base64
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mss
from dotenv import load_dotenv
from PIL import Image
import torch

load_dotenv()

try:
    from perplexity import Perplexity  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    Perplexity = None  # type: ignore
    _perplexity_import_error = exc
else:
    _perplexity_import_error = None


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
TEMP_DIR = TOOLS_ROOT / "temporary"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
POST_OUTPUT_DIR = TEMP_DIR / "posts"
POST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROUNDING_ROOT = REPO_ROOT / "GroundingDino"
GROUNDING_TOOLS = GROUNDING_ROOT / "tools"
GROUNDING_OUTPUT_DIR = GROUNDING_ROOT / "outputs"
GROUNDING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(GROUNDING_TOOLS) not in sys.path:
    sys.path.insert(0, str(GROUNDING_TOOLS))

from grounding_dino_runner import run_grounding_dino  # type: ignore  # noqa: E402


PROMPT_POST_DETAIL = "postCardInDetail"
LABEL_POST_DETAIL = "postcardindetail"


@dataclass
class Detection:
    label: str
    box: Tuple[int, int, int, int]
    score: Optional[float]


class PerplexityClient:
    """Lazy Perplexity client wrapper that defers initialization until required."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        if api_key is None:
            api_key = os.getenv("PERPLEXITY_API_KEY")
        self.api_key = api_key
        self._client: Optional[Perplexity] = None

    def ensure_client(self) -> Perplexity:
        if Perplexity is None:
            raise RuntimeError(
                "Perplexity SDK not available"
            ) from _perplexity_import_error
        if not self.api_key:
            raise RuntimeError("PERPLEXITY_API_KEY environment variable is not set.")
        if self._client is None:
            self._client = Perplexity(api_key=self.api_key)
        return self._client

    def describe_image(self, prompt: str, image_path: Path) -> str:
        client = self.ensure_client()
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(image_path)}"
                    },
                },
            ],
        }
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model="sonar-pro", messages=[message]
        )
        return response.choices[0].message.content


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def timestamped_path(prefix: str, *, ext: str = "png") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return POST_OUTPUT_DIR / f"{prefix}_{ts}.{ext}"


def capture_monitor_image(
    monitor_index: int = 0,
) -> Tuple[Image.Image, Path, Dict[str, int]]:
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


def _area(box: Tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def select_postcard_detail(
    detections: Iterable[Detection],
) -> Optional[Detection]:
    cards = [d for d in detections if d.label == LABEL_POST_DETAIL]
    if not cards:
        return None
    return max(cards, key=lambda d: _area(d.box))


def crop_detection(image: Image.Image, detection: Detection, *, margin: int = 6) -> Image.Image:
    x0, y0, x1, y1 = detection.box
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(image.width - 1, x1 + margin)
    y1 = min(image.height - 1, y1 + margin)
    return image.crop((x0, y0, x1 + 1, y1 + 1))


def capture_postcard_detail(
    monitor_index: int = 0,
    *,
    retries: int = 2,
    delay: float = 0.35,
) -> Path:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 2):
        image, screen_path, _ = capture_monitor_image(monitor_index)
        try:
            inference = run_grounding(screen_path, PROMPT_POST_DETAIL)
        finally:
            screen_path.unlink(missing_ok=True)

        detections = detections_from_inference(image, inference)
        detection = select_postcard_detail(detections)
        if detection:
            crop = crop_detection(image, detection)
            out_path = timestamped_path("postdetail")
            crop.save(out_path, format="PNG", optimize=False)
            return out_path

        last_error = RuntimeError("No postCardInDetail detected in screenshot")
        time.sleep(max(0.1, delay))

    if last_error:
        raise last_error
    raise RuntimeError("Failed to capture postCardInDetail detail view")


def describe_post_image(
    image_path: Path,
    prompt: str,
    *,
    api_key: Optional[str] = None,
) -> str:
    client = PerplexityClient(api_key=api_key)
    return client.describe_image(prompt, image_path)


def run_query_with_image(query: str, image_path: Optional[str] = None) -> str:
    if not image_path:
        raise ValueError("image_path is required for Perplexity multimodal queries")
    return describe_post_image(Path(image_path), query)


def main(argv: Sequence[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Capture a postCardInDetail crop and optionally describe it via Perplexity.",
    )
    parser.add_argument("--monitor", type=int, default=0, help="Monitor index for capture")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="If set, run Perplexity description on the captured image",
    )
    parser.add_argument("--prompt", type=str, default="Explain the Reddit post.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination path for the captured crop",
    )
    args = parser.parse_args(argv)

    crop_path = capture_postcard_detail(monitor_index=args.monitor)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.open(crop_path).save(args.output)
        crop_path = args.output
    print(crop_path)

    if args.describe:
        description = describe_post_image(crop_path, args.prompt)
        print(description)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

