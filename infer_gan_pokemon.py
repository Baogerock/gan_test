import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, ngf: int = 64, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


def tensor_grid(t: torch.Tensor, nrow: int = 4) -> np.ndarray:
    t = t.detach().cpu().clamp(-1, 1)
    t = (t + 1.0) / 2.0
    b, c, h, w = t.shape
    ncol = math.ceil(b / nrow)
    grid = torch.zeros(c, ncol * h, nrow * w)
    for i in range(b):
        r = i // nrow
        col = i % nrow
        grid[:, r * h : (r + 1) * h, col * w : (col + 1) * w] = t[i]
    return grid.permute(1, 2, 0).numpy()


def save_sample_image(generated: torch.Tensor, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    arr = (tensor_grid(generated, nrow=4) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(out_file)


def parse_args():
    p = argparse.ArgumentParser(description="Inference for 0.0.1 DCGAN")
    p.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    p.add_argument("--out", type=str, default="samples/infer_latest.png")
    p.add_argument("--num_images", type=int, default=16)
    p.add_argument("--z_dim", type=int, default=128)
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

    G = Generator(z_dim=args.z_dim).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    with torch.no_grad():
        z = torch.randn(args.num_images, args.z_dim, 1, 1, device=device)
        generated = G(z)

    save_sample_image(generated, Path(args.out))
    print(f"Saved inference image: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
