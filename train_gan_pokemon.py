import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


class PokemonDataset(Dataset):
    def __init__(self, root: Path, image_size: int = 64):
        self.root = root
        self.image_size = image_size
        self.image_paths = [
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if not self.image_paths:
            raise RuntimeError(f"No images found under: {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            arr = np.asarray(img, dtype=np.float32)
        arr = (arr / 127.5) - 1.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


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


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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
    parser = argparse.ArgumentParser(description="Train simple DCGAN on Pokemon images")
    parser.add_argument("--data_dir", type=str, default="pokemon")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--sample_interval", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if it exists.")
    parser.add_argument(
        "--latest_ckpt",
        type=str,
        default="checkpoints/latest.pt",
        help="Path to the latest training checkpoint file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    sample_dir = Path("samples")
    ckpt_dir = Path("checkpoints")
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = Path(args.latest_ckpt)

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

    G = Generator(z_dim=args.z_dim).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    optim_g = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optim_d = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(16, args.z_dim, 1, 1, device=device)
    start_epoch = 1

    if args.resume and latest_ckpt_path.exists():
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        optim_g.load_state_dict(ckpt["opt_g"])
        optim_d.load_state_dict(ckpt["opt_d"])
        start_epoch = int(ckpt["epoch"]) + 1
        if "fixed_noise" in ckpt:
            fixed_noise = ckpt["fixed_noise"].to(device)
        print(f"Resumed from checkpoint: {latest_ckpt_path} (next epoch: {start_epoch})")

    for epoch in range(start_epoch, args.epochs + 1):
        for i, real in enumerate(loader, start=1):
            real = real.to(device)
            bsz = real.size(0)

            # Train Discriminator
            D.zero_grad(set_to_none=True)
            real_targets = torch.full((bsz,), 0.9, device=device)
            fake_targets = torch.zeros((bsz,), device=device)

            out_real = D(real)
            loss_real = criterion(out_real, real_targets)

            noise = torch.randn(bsz, args.z_dim, 1, 1, device=device)
            fake = G(noise)
            out_fake = D(fake.detach())
            loss_fake = criterion(out_fake, fake_targets)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optim_d.step()

            # Train Generator
            G.zero_grad(set_to_none=True)
            out_fake_for_g = D(fake)
            loss_g = criterion(out_fake_for_g, real_targets)
            loss_g.backward()
            optim_g.step()

            if i % 20 == 0 or i == len(loader):
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{i}/{len(loader)}] "
                    f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}"
                )

        if epoch % args.sample_interval == 0:
            with torch.no_grad():
                generated = G(fixed_noise)

            sample_path = sample_dir / f"epoch_{epoch:04d}.png"
            save_sample_image(generated, sample_path)
            torch.save(
                {
                    "epoch": epoch,
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "opt_g": optim_g.state_dict(),
                    "opt_d": optim_d.state_dict(),
                    "fixed_noise": fixed_noise.detach().cpu(),
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
