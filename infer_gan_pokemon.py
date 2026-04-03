import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image


class SelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(8, channels // 8)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key = nn.Conv2d(channels, hidden, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h * w)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(b, c, h * w)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return x + self.gamma * out


class Generator(nn.Module):
    def __init__(self, z_dim=256, g_base=96, out_ch=3):
        super().__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, g_base * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_base * 8),
            nn.ReLU(True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(g_base * 8, g_base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_base * 4),
            nn.ReLU(True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(g_base * 4, g_base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_base * 2),
            nn.ReLU(True),
        )
        self.attn = SelfAttention(g_base * 2)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(g_base * 2, g_base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_base),
            nn.ReLU(True),
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(g_base, out_ch, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.deconv1(z)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.attn(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x


def parse_args():
    p = argparse.ArgumentParser(description="Inference for gan_0.0.3")
    p.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    p.add_argument("--out", type=str, default="samples/infer_latest.png")
    p.add_argument("--num_images", type=int, default=16)
    p.add_argument("--z_dim", type=int, default=256)
    p.add_argument("--g_base", type=int, default=96)
    p.add_argument("--use_ema", action="store_true", help="Use EMA generator weights if available")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        saved_args = ckpt["args"]
        args.z_dim = int(saved_args.get("z_dim", args.z_dim))
        args.g_base = int(saved_args.get("g_base", args.g_base))

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
