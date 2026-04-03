# Diffusion 0.0.2 (Pretrain + Finetune)

This branch adds transfer-style diffusion training for small target datasets.

## 1) Stage A: Pretrain on a larger source set

Example (anime faces):

```powershell
conda run -n common python train_diffusion_pokemon.py \
  --dataset_dir animefaces \
  --epochs 5 \
  --batch_size 16 \
  --timesteps 100 \
  --sample_steps 20 \
  --samples_dir diff_pre_samples \
  --checkpoints_dir diff_pre_ckpt
```

## 2) Stage B: Finetune on Pokemon

```powershell
conda run -n common python train_diffusion_pokemon.py \
  --dataset_dir pokemon \
  --epochs 50 \
  --batch_size 16 \
  --timesteps 100 \
  --sample_steps 20 \
  --samples_dir diff_ft_samples \
  --checkpoints_dir diff_ft_ckpt \
  --pretrained_ckpt diff_pre_ckpt/latest.pt \
  --pretrained_use_ema
```

## Resume finetune

```powershell
conda run -n common python train_diffusion_pokemon.py \
  --dataset_dir pokemon \
  --epochs 50 \
  --samples_dir diff_ft_samples \
  --checkpoints_dir diff_ft_ckpt \
  --resume
```

## Inference + denoise visualization

```powershell
conda run -n common python infer_diffusion_pokemon.py \
  --ckpt diff_ft_ckpt/latest.pt \
  --out diff_ft_samples/infer.png \
  --use_ema \
  --sample_steps 40 \
  --visualize_denoise \
  --denoise_frames 20 \
  --denoise_fps 12
```
