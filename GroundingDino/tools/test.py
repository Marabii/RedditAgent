from grounding_dino_runner import run_grounding_dino, crop_or_draw

res = run_grounding_dino(
    image_path="/home/hamza/Pictures/reddit.png",
    text_prompt="postCard . upvote . downvote . comment . postTitle",
    output_dir="../outputs",
    # (optional overrides)
    # config_file="config/cfg_odvg.py",
    # checkpoint_path="checkpoints/checkpoint0014.pth",
    # box_threshold=0.3,
    # text_threshold=0.25,
    # cpu_only=True,
)

crop_or_draw(image_path="/home/hamza/Pictures/reddit.png", tgt=res, mode="draw")

# crops = crop_or_draw(image_path="/home/hamza/Pictures/reddit.png", tgt=res, mode="crop")
# for i, (crop_img, label, box) in enumerate(crops):  # type: ignore
#     crop_img.save("../outputs" + "_crop_{i:02d}.jpg")

print(res)
