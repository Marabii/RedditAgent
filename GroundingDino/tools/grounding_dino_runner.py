import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, List, Tuple

import numpy as np
import torch
from PIL import ImageDraw, ImageFont, Image

from inference_on_a_image import (
    get_grounding_output,
    load_image,
    load_model,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "cfg_odvg.py"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "checkpoint0014.pth"

# Keep tokenizer single-threaded for fork safety and silence noisy warnings.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="CUDA initialization: Unexpected error from cudaGetDeviceCount",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The `device` argument is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.autocast",
    category=FutureWarning,
)


def _resolve(path: Union[str, Path, None], fallback: Path) -> Path:
    if path is None:
        return fallback
    return Path(path).expanduser().resolve()


def _safe_fragment(raw: str) -> str:
    """Turn an arbitrary string into a filesystem-friendly fragment."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "item"


def run_grounding_dino(
    image_path: Union[str, Path],
    text_prompt: str,
    output_dir: Union[str, Path],
    *,
    config_file: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    token_spans: Optional[Sequence[Sequence[Sequence[int]]]] = None,
    cpu_only: bool = False,
) -> Dict[str, Any]:
    """Run GroundingDINO on a single image and return predictions and artifacts."""

    config_file = _resolve(config_file, DEFAULT_CONFIG)
    checkpoint_path = _resolve(checkpoint_path, DEFAULT_CHECKPOINT)
    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_pil, image = load_image(str(image_path))
    model = load_model(str(config_file), str(checkpoint_path), cpu_only=cpu_only)

    if token_spans is not None:
        text_threshold = None

    boxes, labels = get_grounding_output(
        model,
        image,
        text_prompt,
        box_threshold,
        text_threshold=text_threshold,
        cpu_only=cpu_only,
        token_spans=token_spans,
    )

    return {
        "boxes": boxes,
        "labels": labels,
    }


def crop_or_draw(
    image_path: str,
    tgt: Dict[str, Union[Sequence[int], torch.Tensor, List]],
    *,
    mode: str = "draw",
    output_dir: Optional[Union[str, Path]] = None,
):
    """
    If mode == "draw" (default): draw rectangles + labels on the image, write the
        annotated image + mask to disk, and return (image_pil, mask).
    If mode == "crop": crop each box, save crops to disk, and return
        List[(crop_img, label, (x0, y0, x1, y1))].

    Parameters
    ----------
    image_path : str
        Path to the source image.
    tgt : dict
        Must contain:
          - "boxes": Tensor[N, 4] normalized xywh in [0,1]
          - "labels": List[str] of length N
        (Image size will be inferred from image itself.)
    mode : {"draw","crop"}
        Drawing vs cropping behavior.
    output_dir : str or Path, optional
        Directory where the annotated image / crops will be stored. Falls back to
        GroundingDINO's `outputs` directory when omitted.
    """
    image_path = Path(image_path).expanduser().resolve()
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (ROOT / "outputs").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    image_pil = load_image(str(image_path))[
        0
    ].copy()  # operate on a copy of the source image
    W, H = image_pil.size  # infer size: (width, height)

    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    # Ensure tensor for math
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)

    # scale normalized xywh (0..1) to pixel space (W,H)
    boxes_abs = boxes * torch.tensor([W, H, W, H], dtype=torch.float32)

    # xywh -> xyxy
    boxes_abs[:, :2] -= boxes_abs[:, 2:] / 2.0
    boxes_abs[:, 2:] += boxes_abs[:, :2]

    # Clamp to image bounds and cast to ints
    x0y0 = torch.maximum(boxes_abs[:, :2], torch.tensor([0.0, 0.0]))
    x1y1 = torch.minimum(boxes_abs[:, 2:], torch.tensor([float(W - 1), float(H - 1)]))
    boxes_xyxy = torch.cat([x0y0, x1y1], dim=1).round().to(torch.int64)

    if mode == "crop":
        crops: List[Tuple[Image.Image, str, Tuple[int, int, int, int]]] = []
        for (x0, y0, x1, y1), label in zip(boxes_xyxy.tolist(), labels):
            if x1 <= x0 or y1 <= y0:
                continue
            # PIL crop: right/lower are exclusive -> +1 to include the edge
            crop_box = (x0, y0, x1 + 1, y1 + 1)
            crop_img = image_pil.crop(crop_box)
            crops.append((crop_img, str(label), (x0, y0, x1, y1)))

            safe_label = _safe_fragment(str(label))
            crop_idx = len(crops) - 1
            crop_name = f"{image_path.stem}_crop_{crop_idx:02d}_{safe_label}.png"
            crop_path = output_root / crop_name
            crop_img.save(crop_path)
        return crops

    # ==== draw mode ====
    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for (x0, y0, x1, y1), label in zip(boxes_xyxy.tolist(), labels):
        color = tuple(np.random.randint(0, 255, size=3).tolist())

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        font = ImageFont.load_default()
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font=font)
        else:
            w, h = draw.textsize(str(label), font=font)  # type: ignore
            bbox = (x0, y0, x0 + w, y0 + h)

        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white", font=font)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    annotated_name = f"{image_path}_annotated.png"
    image_pil.save(output_root / annotated_name)

    return image_pil, mask


__all__ = ["run_grounding_dino", "crop_or_draw"]
