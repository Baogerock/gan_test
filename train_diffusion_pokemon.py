import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


class PokemonDataset(Dataset):
    def __init__(self, root: Path, image_size: int = 64):
        self.image_paths = list_images(root)
        if not self.image_paths:
            raise RuntimeError(f"No images found under: {root}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert("RGB")
            return self.transform(img)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        emb = math.log(10000) / max(1, (half - 1))
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.down1 = ResBlock(base, base, time_dim)
        self.downsample1 = nn.Conv2d(base, base, 4, stride=2, padding=1)

        self.down2 = ResBlock(base, base * 2, time_dim)
        self.downsample2 = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)

        self.mid1 = ResBlock(base * 2, base * 4, time_dim)
        self.mid2 = ResBlock(base * 4, base * 2, time_dim)

        self.upsample1 = nn.ConvTranspose2d(base * 2, base * 2, 4, stride=2, padding=1)
        self.up1 = ResBlock(base * 4, base, time_dim)

        self.upsample2 = nn.ConvTranspose2d(base, base, 4, stride=2, padding=1)
        self.up2 = ResBlock(base * 2, base, time_dim)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = self.time_mlp(t)

        x = self.init_conv(x)
        s1 = self.down1(x, t_emb)
        x = self.downsample1(s1)

        s2 = self.down2(x, t_emb)
        x = self.downsample2(s2)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.upsample1(x)
        x = torch.cat([x, s2], dim=1)
        x = self.up1(x, t_emb)

        x = self.upsample2(x)
        x = torch.cat([x, s1], dim=1)
        x = self.up2(x, t_emb)

        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x


class DiffusionSchedule:
    def __init__(self, timesteps: int, device: torch.device):
        self.timesteps = timesteps
        beta_start = 1e-4
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape):
        out = arr.gather(0, t)
        return out.reshape(t.size(0), *([1] * (len(x_shape) - 1)))


def q_sample(schedule: DiffusionSchedule, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    s1 = schedule.extract(schedule.sqrt_alphas_cumprod, t, x0.shape)
    s2 = schedule.extract(schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return s1 * x0 + s2 * noise


def ddim_sample(model, schedule: DiffusionSchedule, device: torch.device, n: int, sample_steps: int, image_size: int):
    model.eval()
    with torch.no_grad():
        x = torch.randn(n, 3, image_size, image_size, device=device)
        step_indices = torch.linspace(schedule.timesteps - 1, 0, sample_steps, device=device).long()

        for i, t_now in enumerate(step_indices):
            t = torch.full((n,), t_now, device=device, dtype=torch.long)
            eps = model(x, t)

            a_t = schedule.alphas_cumprod[t_now]
            if i == len(step_indices) - 1:
                a_prev = torch.tensor(1.0, device=device)
            else:
                t_prev = step_indices[i + 1]
                a_prev = schedule.alphas_cumprod[t_prev]

            x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * eps

        x = x.clamp(-1, 1)
    model.train()
    return x


def update_ema(ema_model, model, decay=0.999):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            if k in msd:
                v.copy_(v * decay + msd[k] * (1.0 - decay))


def parse_args():
    p = argparse.ArgumentParser(description="Train diffusion model on Pokemon")
    p.add_argument("--data_dir", type=str, default="pokemon")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--timesteps", type=int, default=400)
    p.add_argument("--sample_steps", type=int, default=80)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--sample_interval", type=int, default=1)
    p.add_argument("--samples_dir", type=str, default="samples")
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p.add_argument("--latest_ckpt", type=str, default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample_from_ema", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    sample_dir = Path(args.samples_dir)
    ckpt_dir = Path(args.checkpoints_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = Path(args.latest_ckpt) if args.latest_ckpt else ckpt_dir / "latest.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = PokemonDataset(data_dir, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"Loaded {len(dataset)} images from {data_dir.resolve()}")

    model = TinyUNet(in_ch=3, base=args.base_channels, time_dim=args.time_dim).to(device)
    ema_model = TinyUNet(in_ch=3, base=args.base_channels, time_dim=args.time_dim).to(device)
    ema_model.load_state_dict(model.state_dict())
    for param in ema_model.parameters():
        param.requires_grad_(False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = DiffusionSchedule(args.timesteps, device)

    start_epoch = 1
    if args.resume and latest_ckpt_path.exists():
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt.get("ema_model", ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from checkpoint: {latest_ckpt_path} (next epoch: {start_epoch})")

    for epoch in range(start_epoch, args.epochs + 1):
        running_loss = 0.0
        for step, x0 in enumerate(loader, start=1):
            x0 = x0.to(device, non_blocking=True)
            bsz = x0.size(0)

            t = torch.randint(0, args.timesteps, (bsz,), device=device).long()
            noise = torch.randn_like(x0)
            x_t = q_sample(schedule, x0, t, noise)
            pred_noise = model(x_t, t)

            loss = torch.mean((pred_noise - noise) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema_model, model, decay=args.ema_decay)

            running_loss += loss.item()
            if step % 20 == 0 or step == len(loader):
                avg = running_loss / step
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(loader)}] Loss: {avg:.4f}")

        if epoch % args.sample_interval == 0:
            sampler = ema_model if args.sample_from_ema else model
            samples = ddim_sample(
                sampler,
                schedule,
                device=device,
                n=args.num_samples,
                sample_steps=args.sample_steps,
                image_size=args.image_size,
            )
            grid = make_grid((samples.detach().cpu() + 1) / 2, nrow=4, padding=2)
            sample_path = sample_dir / f"epoch_{epoch:04d}.png"
            save_image(grid, sample_path)

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                latest_ckpt_path,
            )
            print(f"Saved sample and latest checkpoint for epoch {epoch}")

    print("Training finished.")
    print(f"Samples saved to: {sample_dir.resolve()}")
    print(f"Checkpoints saved to: {ckpt_dir.resolve()}")


if __name__ == "__main__":
    main()
