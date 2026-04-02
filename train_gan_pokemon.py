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
        self.root = root
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
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)


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


class Discriminator(nn.Module):
    def __init__(self, d_base=64, in_ch=3):
        super().__init__()

        def snconv(ic, oc, k=4, s=2, p=1):
            return nn.utils.spectral_norm(nn.Conv2d(ic, oc, k, s, p, bias=False))

        self.net = nn.Sequential(
            snconv(in_ch, d_base),
            nn.LeakyReLU(0.2, inplace=True),
            snconv(d_base, d_base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            snconv(d_base * 2, d_base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            snconv(d_base * 4, d_base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(d_base * 8, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def update_ema(ema_model, model, decay=0.999):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            if k in msd:
                v.copy_(v * decay + msd[k] * (1.0 - decay))


def save_sample_image(generated: torch.Tensor, out_file: Path, nrow: int = 4):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid((generated.detach().cpu() + 1) / 2, nrow=nrow, padding=2)
    save_image(grid, out_file)


def d_hinge_loss(real_logits, fake_logits):
    return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()


def g_hinge_loss(fake_logits):
    return -fake_logits.mean()


def compute_r1_penalty(discriminator, real_images):
    real_images = real_images.requires_grad_(True)
    real_scores = discriminator(real_images)
    grad = torch.autograd.grad(
        outputs=real_scores.sum(),
        inputs=real_images,
        create_graph=True,
    )[0]
    return grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()


def parse_args():
    p = argparse.ArgumentParser(description="Stable DCGAN-style training for Pokemon")
    p.add_argument("--data_dir", type=str, default="pokemon")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--z_dim", type=int, default=256)
    p.add_argument("--g_base", type=int, default=96)
    p.add_argument("--d_base", type=int, default=64)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=1.5e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--sample_interval", type=int, default=1)
    p.add_argument("--samples_dir", type=str, default="samples")
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--r1_gamma", type=float, default=0.1)
    p.add_argument("--r1_interval", type=int, default=32)
    p.add_argument(
        "--sample_from_ema",
        action="store_true",
        help="Use EMA generator for saving samples (default uses current generator).",
    )
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if it exists.")
    p.add_argument(
        "--latest_ckpt",
        type=str,
        default="",
        help="Path to the latest training checkpoint file. Defaults to <checkpoints_dir>/latest.pt",
    )
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

    G = Generator(z_dim=args.z_dim, g_base=args.g_base).to(device)
    D = Discriminator(d_base=args.d_base).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    G_ema = Generator(z_dim=args.z_dim, g_base=args.g_base).to(device)
    G_ema.load_state_dict(G.state_dict())
    G_ema.eval()
    for param in G_ema.parameters():
        param.requires_grad_(False)

    opt_g = optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    fixed_noise = torch.randn(16, args.z_dim, 1, 1, device=device)
    start_epoch = 1

    if args.resume and latest_ckpt_path.exists():
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        G_ema.load_state_dict(ckpt["G_ema"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = int(ckpt["epoch"]) + 1
        if "fixed_noise" in ckpt:
            fixed_noise = ckpt["fixed_noise"].to(device)
        print(f"Resumed from checkpoint: {latest_ckpt_path} (next epoch: {start_epoch})")

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        for step, real in enumerate(loader, start=1):
            global_step += 1
            real = real.to(device, non_blocking=True)

            # Train D
            z = torch.randn(real.size(0), args.z_dim, 1, 1, device=device)
            with torch.no_grad():
                fake = G(z)
            real_logits = D(real)
            fake_logits = D(fake)
            loss_d = d_hinge_loss(real_logits, fake_logits)

            # Lazy R1 regularization
            if args.r1_gamma > 0 and global_step % args.r1_interval == 0:
                r1 = compute_r1_penalty(D, real)
                loss_d = loss_d + (args.r1_gamma * 0.5 * args.r1_interval) * r1

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # Train G
            z = torch.randn(real.size(0), args.z_dim, 1, 1, device=device)
            gen = G(z)
            fake_logits_for_g = D(gen)
            loss_g = g_hinge_loss(fake_logits_for_g)

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            update_ema(G_ema, G, decay=args.ema_decay)

            if step % 20 == 0 or step == len(loader):
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(loader)}] "
                    f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}"
                )

        if epoch % args.sample_interval == 0:
            with torch.no_grad():
                sampler = G_ema if args.sample_from_ema else G
                samples = sampler(fixed_noise)
            sample_path = sample_dir / f"epoch_{epoch:04d}.png"
            save_sample_image(samples, sample_path, nrow=4)
            torch.save(
                {
                    "epoch": epoch,
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "G_ema": G_ema.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
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
