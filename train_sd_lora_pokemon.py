import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def set_hf_cache(hf_home: str):
    hf = str(Path(hf_home).resolve())
    os.environ["HF_HOME"] = hf
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(hf) / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(Path(hf) / "transformers")
    Path(hf).mkdir(parents=True, exist_ok=True)


class PromptImageDataset(Dataset):
    def __init__(self, root: Path, prompt: str, size: int = 256):
        self.paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not self.paths:
            raise RuntimeError(f"No images found under: {root}")
        self.prompt = prompt
        self.transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            img = img.convert("RGB")
            pixel_values = self.transform(img)
        return {"pixel_values": pixel_values, "prompt": self.prompt}


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    prompts = [e["prompt"] for e in examples]
    return {"pixel_values": pixel_values, "prompts": prompts}


def save_samples(pipe, prompt: str, out_path: Path, num_images: int, steps: int, guidance: float, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(1234)
    imgs = pipe(
        [prompt] * num_images,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images
    tensors = [transforms.ToTensor()(im) for im in imgs]
    grid = make_grid(torch.stack(tensors), nrow=4, padding=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)


def parse_args():
    p = argparse.ArgumentParser(description="Finetune external pretrained SD with LoRA on Pokemon")
    p.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--dataset_dir", type=str, default="pokemon")
    p.add_argument("--prompt", type=str, default="a pokemon creature, official artwork")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max_train_steps", type=int, default=0, help="0 means disabled")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--sample_interval", type=int, default=1)
    p.add_argument("--sample_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=7.0)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--samples_dir", type=str, default="sd_samples")
    p.add_argument("--checkpoints_dir", type=str, default="sd_checkpoints")
    p.add_argument("--latest_ckpt", type=str, default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf_home", type=str, default=".hf_cache")
    p.add_argument("--proxy", type=str, default="", help="e.g. http://127.0.0.1:7897")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    set_hf_cache(args.hf_home)

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from diffusers.optimization import get_cosine_schedule_with_warmup
    from peft import LoraConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep training in float32 for stability with LoRA params on this environment.
    dtype = torch.float32
    print(f"Using device: {device}")

    data_dir = Path(args.dataset_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    samples_dir = Path(args.samples_dir)
    ckpt_dir = Path(args.checkpoints_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = Path(args.latest_ckpt) if args.latest_ckpt else ckpt_dir / "latest.pt"
    latest_lora_dir = ckpt_dir / "lora_latest"

    # Load full pipeline once, then reuse components.
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
    )
    unet.add_adapter(lora_cfg)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    dataset = PromptImageDataset(data_dir, prompt=args.prompt, size=args.resolution)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,
    )
    print(f"Loaded {len(dataset)} images from {data_dir.resolve()}")

    total_steps = args.epochs * len(loader)
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    lr_sched = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    start_epoch = 1
    global_step = 0

    if args.resume and latest_ckpt.exists() and latest_lora_dir.exists():
        state = torch.load(latest_ckpt, map_location=device)
        unet.load_attn_procs(str(latest_lora_dir))
        optimizer.load_state_dict(state["optimizer"])
        lr_sched.load_state_dict(state["lr_sched"])
        global_step = int(state.get("global_step", 0))
        start_epoch = int(state["epoch"]) + 1
        print(f"Resumed from {latest_ckpt} (next epoch={start_epoch}, step={global_step})")

    text_encoder.eval()
    vae.eval()
    unet.train()

    stop_training = False
    for epoch in range(start_epoch, args.epochs + 1):
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            prompts = batch["prompts"]

            with torch.no_grad():
                inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                encoder_hidden_states = text_encoder(inputs.input_ids.to(device))[0]
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lr_sched.step()

            running += loss.item()
            global_step += 1

            if step % 20 == 0 or step == len(loader):
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(loader)}] Loss: {running/step:.4f}")

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                stop_training = True
                break

        unet.save_attn_procs(str(latest_lora_dir))
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "lr_sched": lr_sched.state_dict(),
                "args": vars(args),
            },
            latest_ckpt,
        )

        if epoch % args.sample_interval == 0:
            unet.eval()
            save_samples(
                pipe,
                prompt=args.prompt,
                out_path=samples_dir / f"epoch_{epoch:04d}.png",
                num_images=args.num_samples,
                steps=args.sample_steps,
                guidance=args.guidance_scale,
                device=device,
            )
            unet.train()
            print(f"Saved sample and latest checkpoint for epoch {epoch}")

        if stop_training:
            print("Reached --max_train_steps, stopping early.")
            break

    print("Training finished.")
    print(f"Samples: {samples_dir.resolve()}")
    print(f"Checkpoints: {ckpt_dir.resolve()}")


if __name__ == "__main__":
    main()
