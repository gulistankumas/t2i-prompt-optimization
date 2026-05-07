"""Adim 2/4: 5 prompt × 2 (router + baseline) = 10 gorsel uret.

Replicate ve OpenAI API kullanir. Her gorsel ayni klasore kaydedilir;
manifest CSV uretilir (path, suresi, model, prompt_id).

Cikti: data/validation/images/*.png + data/validation/generation_log.csv
"""
import base64
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "data" / "validation"
IMG_DIR = VAL_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Replicate model path haritasi
REPLICATE_PATHS = {
    "FLUX.1-schnell": "black-forest-labs/flux-schnell",
    "FLUX.1-dev": "black-forest-labs/flux-dev",
    "SDXL": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    "SD3.5-Large": "stability-ai/stable-diffusion-3.5-large",
    "SD1.5": "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
}


def save_replicate_output(output, path: Path):
    item = output[0] if isinstance(output, list) else output
    if hasattr(item, "read"):
        path.write_bytes(item.read())
    elif isinstance(item, str):
        path.write_bytes(requests.get(item).content)
    else:
        raise ValueError(f"Bilinmeyen Replicate output tipi: {type(item)}")


def generate_replicate(model_name: str, prompt: str) -> bytes:
    import replicate
    path_id = REPLICATE_PATHS[model_name]
    inp = {"prompt": prompt}
    if model_name in ("FLUX.1-schnell", "FLUX.1-dev"):
        inp.update({"num_outputs": 1, "aspect_ratio": "1:1", "output_format": "png"})
    elif model_name == "SDXL":
        inp.update({"width": 1024, "height": 1024, "num_outputs": 1})
    elif model_name == "SD3.5-Large":
        inp.update({"aspect_ratio": "1:1", "output_format": "png"})
    elif model_name == "SD1.5":
        inp.update({"width": 768, "height": 768, "num_outputs": 1})
    output = replicate.run(path_id, input=inp)
    return output


def generate_openai(model_name: str, prompt: str) -> bytes:
    """GPT-Image-1 high quality."""
    from openai import OpenAI
    client = OpenAI()
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        quality="high",
        n=1,
    )
    return base64.b64decode(response.data[0].b64_json)


def generate_one(model_name: str, api: str, prompt: str, out_path: Path):
    t0 = time.time()
    if api == "replicate":
        output = generate_replicate(model_name, prompt)
        save_replicate_output(output, out_path)
    elif api == "openai":
        img_bytes = generate_openai(model_name, prompt)
        out_path.write_bytes(img_bytes)
    else:
        raise ValueError(f"Bilinmeyen API: {api}")
    return time.time() - t0


def main():
    df = pd.read_csv(VAL_DIR / "selected_5_prompts.csv")
    print(f"5 prompt yuklendi.")

    # token kontrolleri
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("HATA: REPLICATE_API_TOKEN yok"); sys.exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        print("HATA: OPENAI_API_KEY yok"); sys.exit(1)

    log_rows = []
    for i, row in df.iterrows():
        vid = row["validation_id"]
        prompt = row["prompt"]
        for role, model_col, api_col, cost_col in [
            ("router", "router_model", "router_api", "router_cost"),
            ("baseline", "baseline_model", "baseline_api", "baseline_cost"),
        ]:
            model = row[model_col]
            api = row[api_col]
            cost = row[cost_col]
            out_path = IMG_DIR / f"{vid}_{role}_{model.replace('.', '').replace('/', '_')}.png"

            print(f"\n[{vid}] {role:8s} {model:18s} ({api}) — {prompt[:60]}...")
            try:
                elapsed = generate_one(model, api, prompt, out_path)
                size_kb = out_path.stat().st_size / 1024
                print(f"  OK ({elapsed:.1f}s, {size_kb:.0f} KB) -> {out_path.name}")
                log_rows.append({
                    "validation_id": vid, "role": role, "model": model, "api": api,
                    "track": row["track"], "prompt": prompt,
                    "image_path": str(out_path.relative_to(ROOT)),
                    "elapsed_s": round(elapsed, 1),
                    "cost_usd": cost, "status": "OK",
                })
            except Exception as e:
                print(f"  FAIL: {e}")
                log_rows.append({
                    "validation_id": vid, "role": role, "model": model, "api": api,
                    "track": row["track"], "prompt": prompt,
                    "image_path": "",
                    "elapsed_s": 0,
                    "cost_usd": cost, "status": f"ERROR: {e}",
                })

    log_df = pd.DataFrame(log_rows)
    log_path = VAL_DIR / "generation_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n{'=' * 70}\nManifest: {log_path}")
    n_ok = (log_df["status"] == "OK").sum()
    print(f"Toplam: {n_ok}/{len(log_df)} basarili")
    print(f"Toplam sure: {log_df['elapsed_s'].sum():.0f}s")
    print(f"Toplam tahmini maliyet: ${log_df[log_df['status'] == 'OK']['cost_usd'].sum():.3f}")


if __name__ == "__main__":
    main()
