# External Pretrained Diffusion (SD LoRA) - diff_0.0.2

This workflow fine-tunes an external pretrained Stable Diffusion model with LoRA.

## Install dependencies (proxy example)

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
conda run -n common python -m pip install diffusers==0.31.0 accelerate peft -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

## Train (real pretrained model)

```powershell
conda run -n common python train_sd_lora_pokemon.py \
  --pretrained_model runwayml/stable-diffusion-v1-5 \
  --dataset_dir pokemon \
  --prompt "a pokemon creature, official artwork" \
  --resolution 256 \
  --batch_size 2 \
  --epochs 20 \
  --lr 1e-4 \
  --sample_interval 1 \
  --sample_steps 25 \
  --samples_dir sd_samples \
  --checkpoints_dir sd_checkpoints \
  --proxy http://127.0.0.1:7897
```

## Resume

```powershell
conda run -n common python train_sd_lora_pokemon.py \
  --pretrained_model runwayml/stable-diffusion-v1-5 \
  --dataset_dir pokemon \
  --samples_dir sd_samples \
  --checkpoints_dir sd_checkpoints \
  --resume \
  --proxy http://127.0.0.1:7897
```

## Inference with LoRA

```powershell
conda run -n common python infer_sd_lora_pokemon.py \
  --pretrained_model runwayml/stable-diffusion-v1-5 \
  --lora_dir sd_checkpoints/lora_latest \
  --prompt "a pokemon creature, official artwork, clean background" \
  --out sd_samples/infer_latest.png \
  --num_images 8 \
  --steps 30 \
  --guidance_scale 7.0 \
  --proxy http://127.0.0.1:7897
```
