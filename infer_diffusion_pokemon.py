import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / max(1, (half - 1))
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
                a_prev = schedule.alphas_cumprod[step_indices[i + 1]]
            x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * eps
        return x.clamp(-1, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Inference for diffusion Pokemon model")
    p.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    p.add_argument("--out", type=str, default="samples/infer_latest.png")
    p.add_argument("--num_images", type=int, default=16)
    p.add_argument("--sample_steps", type=int, default=80)
    p.add_argument("--timesteps", type=int, default=400)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get("args", {})
    args.base_channels = int(saved_args.get("base_channels", args.base_channels))
    args.time_dim = int(saved_args.get("time_dim", args.time_dim))
    args.image_size = int(saved_args.get("image_size", args.image_size))
    args.timesteps = int(saved_args.get("timesteps", args.timesteps))

    model = TinyUNet(in_ch=3, base=args.base_channels, time_dim=args.time_dim).to(device)
    key = "ema_model" if args.use_ema and "ema_model" in ckpt else "model"
    model.load_state_dict(ckpt[key])

    schedule = DiffusionSchedule(args.timesteps, device)
    samples = ddim_sample(
        model,
        schedule,
        device=device,
        n=args.num_images,
        sample_steps=args.sample_steps,
        image_size=args.image_size,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid((samples.detach().cpu() + 1) / 2, nrow=4, padding=2)
    save_image(grid, out_path)
    print(f"Saved inference image: {out_path.resolve()}")


if __name__ == "__main__":
    main()
