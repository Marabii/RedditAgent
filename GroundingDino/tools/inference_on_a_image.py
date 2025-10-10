import os
import torch
from PIL import Image

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    for flag in ("use_checkpoint", "use_transformer_ckpt"):
        if hasattr(args, flag):
            setattr(args, flag, False)
    args.device = "cuda" if (torch.cuda.is_available() and not cpu_only) else "cpu"
    model = build_model(args)

    # allow argparse.Namespace inside checkpoints
    import argparse as _argparse

    torch.serialization.add_safe_globals([_argparse.Namespace])

    checkpoint = torch.load(
        model_checkpoint_path, map_location="cpu", weights_only=False
    )
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    model.eval()
    return model


def get_grounding_output(
    model,
    image,
    caption,
    box_threshold,
    text_threshold=None,
    with_logits=True,
    cpu_only=False,
    token_spans=None,
):
    assert (text_threshold is not None) or (token_spans is not None), (
        "text_threshold and token_spans should not be None at the same time!"
    )

    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    device = "cuda" if (torch.cuda.is_available() and not cpu_only) else "cpu"
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4), cx,cy,w,h in [0,1]

    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)

        pred_phrases = []
        for logit, _box in zip(logits_filt, boxes_filt):
            phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenizer
            )
            pred_phrases.append(
                phrase + (f"({str(logit.max().item())[:4]})" if with_logits else "")
            )
    else:
        # given-phrase mode (fixed: use caption instead of undefined text_prompt)
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption), token_span=token_spans
        ).to(image.device)  # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq

        all_logits, all_phrases, all_boxes = [], [], []
        for span_list, logit_phr in zip(token_spans, logits_for_phrases):
            phrase = " ".join([caption[_s:_e] for (_s, _e) in span_list])
            filt_mask = logit_phr > box_threshold
            all_boxes.append(boxes[filt_mask])
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                all_phrases.extend(
                    [
                        phrase + f"({str(logit_value.item())[:4]})"
                        for logit_value in logit_phr[filt_mask]
                    ]
                )
            else:
                all_phrases.extend([phrase] * int(filt_mask.sum().item()))
        boxes_filt = (
            torch.cat(all_boxes, dim=0).cpu()
            if len(all_boxes) > 0
            else torch.empty((0, 4))
        )
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


def run_grounding_dino(
    image_path: str,
    text_prompt: str,
    output_dir: str,
    *,
    config_file: str = "config/cfg_odvg.py",
    checkpoint_path: str = "checkpoints/checkpoint0014.pth",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    token_spans=None,
    cpu_only: bool = False,
):
    """
    Run GroundingDINO on an image with a text prompt.

    Required:
        image_path: path to the image file.
        text_prompt: natural language prompt.
        output_dir: directory to save outputs.

    Optional (defaults mimic the CLI):
        config_file: model config path (default: "config/cfg_odvg.py")
        checkpoint_path: checkpoint path (default: "checkpoints/checkpoint0014.pth")
        box_threshold: filter threshold for boxes (default: 0.3)
        text_threshold: token activation threshold (default: 0.25). Ignored if token_spans is provided.
        token_spans: list of list of (start, end) indices into the *caption* for given-phrase mode.
        cpu_only: force CPU (default: False)

    Returns:
        dict with:
            - "boxes": Tensor [N, 4] (cx, cy, w, h) normalized to [0,1]
            - "labels": List[str] predicted phrases (optionally with logits)
    """
    os.makedirs(output_dir, exist_ok=True)

    # load image
    _, image = load_image(image_path)

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)

    # handle thresholds interplay
    if token_spans is not None:
        text_threshold = None  # given-phrase mode ignores text_threshold
        print("Using token_spans. Set the text_threshold to None.")

    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        cpu_only=cpu_only,
        token_spans=token_spans,
    )

    return {
        "boxes": boxes_filt,
        "labels": pred_phrases,
    }


# -----------------------
# Optional: keep CLI
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        default="config/cfg_odvg.py",
        help="path to config file",
    )
    parser.add_argument(
        "--checkpoint_path",
        "-p",
        type=str,
        default="checkpoints/checkpoint0014.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--image_path", "-i", type=str, required=True, help="path to image file"
    )
    parser.add_argument(
        "--text_prompt", "-t", type=str, required=True, help="text prompt"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", help="output directory"
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.3, help="box threshold"
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25, help="text threshold"
    )
    parser.add_argument(
        "--token_spans",
        type=str,
        default=None,
        help="e.g. '[[[2,5]], [[0,1],[2,5]]]' over the *caption* string.",
    )
    parser.add_argument("--cpu-only", action="store_true", help="run on CPU only")
    args = parser.parse_args()

    # parse token_spans if provided as a JSON-like string
    token_spans = None
    if args.token_spans:
        import ast

        token_spans = ast.literal_eval(args.token_spans)

    run_grounding_dino(
        image_path=args.image_path,
        text_prompt=args.text_prompt,
        output_dir=args.output_dir,
        config_file=args.config_file,
        checkpoint_path=args.checkpoint_path,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        token_spans=token_spans,
        cpu_only=args.cpu_only,
    )
