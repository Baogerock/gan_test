import argparse
from pathlib import Path

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Create GIF from epoch sample images.")
    p.add_argument("--samples_dir", type=str, default="samples", help="Directory containing epoch_*.png")
    p.add_argument("--pattern", type=str, default="epoch_*.png", help="Glob pattern for frame files")
    p.add_argument("--out", type=str, default="", help="Output GIF path. Defaults to <samples_dir>/progress.gif")
    p.add_argument("--fps", type=float, default=12.0, help="Frames per second")
    p.add_argument("--loop", type=int, default=0, help="GIF loop count. 0 means infinite loop")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be > 0")

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        raise FileNotFoundError(f"samples_dir not found: {samples_dir}")

    frames = sorted(samples_dir.glob(args.pattern))
    if not frames:
        raise RuntimeError(f"No frames found with pattern '{args.pattern}' under {samples_dir}")

    images = [Image.open(p).convert("RGB") for p in frames]
    base_size = images[0].size
    normalized = [im.resize(base_size, Image.BICUBIC) if im.size != base_size else im for im in images]

    out_path = Path(args.out) if args.out else (samples_dir / "progress.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = max(1, int(round(1000.0 / args.fps)))
    normalized[0].save(
        out_path,
        save_all=True,
        append_images=normalized[1:],
        duration=duration_ms,
        loop=args.loop,
        optimize=False,
    )

    print(f"Saved GIF: {out_path.resolve()}")
    print(f"frames={len(normalized)} fps={args.fps} duration_ms={duration_ms}")


if __name__ == "__main__":
    main()
