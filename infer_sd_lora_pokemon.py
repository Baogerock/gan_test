import argparse
import os
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.utils import make_grid, save_image


def set_hf_cache(hf_home: str):
    hf = str(Path(hf_home).resolve())
    os.environ["HF_HOME"] = hf
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(hf) / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(Path(hf) / "transformers")
    Path(hf).mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Inference with pretrained SD + LoRA")
    p.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora_dir", type=str, default="sd_checkpoints/lora_latest")
    p.add_argument("--prompt", type=str, default="a pokemon creature, official artwork")
    p.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted")
    p.add_argument("--out", type=str, default="sd_samples/infer_latest.png")
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--hf_home", type=str, default=".hf_cache")
    p.add_argument("--proxy", type=str, default="", help="e.g. http://127.0.0.1:7897")
    return p.parse_args()


def main():
    args = parse_args()
    set_hf_cache(args.hf_home)
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    from diffusers import StableDiffusionPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Using device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    lora_dir = Path(args.lora_dir)
    if lora_dir.exists():
        pipe.load_lora_weights(str(lora_dir))
        print(f"Loaded LoRA from: {lora_dir.resolve()}")
    else:
        print(f"[WARN] LoRA dir not found: {lora_dir}. Generating with base model.")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = pipe(
        [args.prompt] * args.num_images,
        negative_prompt=[args.negative_prompt] * args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = [transforms.ToTensor()(im) for im in images]
    grid = make_grid(torch.stack(tensors), nrow=4, padding=2)
    save_image(grid, out_path)
    print(f"Saved inference image: {out_path.resolve()}")


if __name__ == "__main__":
    main()
