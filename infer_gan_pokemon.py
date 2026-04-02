import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image


class Generator(nn.Module):
    def __init__(self, z_dim=256, g_base=96, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, g_base * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_base * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_base * 8, g_base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_base * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_base * 4, g_base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_base * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_base * 2, g_base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_base),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_base, out_ch, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


def parse_args():
    p = argparse.ArgumentParser(description="Inference for 0.0.2 stable GAN")
    p.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    p.add_argument("--out", type=str, default="samples/infer_latest.png")
    p.add_argument("--num_images", type=int, default=16)
    p.add_argument("--z_dim", type=int, default=256)
    p.add_argument("--g_base", type=int, default=96)
    p.add_argument("--use_ema", action="store_true", help="Use EMA generator weights if available.")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        args_dict = ckpt["args"]
        if "z_dim" in args_dict:
            args.z_dim = int(args_dict["z_dim"])
        if "g_base" in args_dict:
            args.g_base = int(args_dict["g_base"])

    G = Generator(z_dim=args.z_dim, g_base=args.g_base).to(device)
    key = "G_ema" if args.use_ema and "G_ema" in ckpt else "G"
    G.load_state_dict(ckpt[key])
    G.eval()

    with torch.no_grad():
        z = torch.randn(args.num_images, args.z_dim, 1, 1, device=device)
        generated = G(z)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid((generated.detach().cpu() + 1) / 2, nrow=4, padding=2)
    save_image(grid, out_path)
    print(f"Saved inference image: {out_path.resolve()}")


if __name__ == "__main__":
    main()
