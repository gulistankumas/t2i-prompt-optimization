"""Replicate API testi — tek prompt sanity check.

Para harcamadan önce: API çalışıyor mu, FLUX.1-schnell yanıt veriyor mu, görsel
disk'e kaydediliyor mu doğrulanır. Maliyet: ~$0.003.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("REPLICATE_API_TOKEN")
if not token:
    print("HATA: REPLICATE_API_TOKEN bulunamadi. .env dosyasina ekle.")
    sys.exit(1)
print(f"Token bulundu: {token[:10]}...{token[-4:]}")

import replicate

PROMPT = "a serene mountain landscape at sunset, photorealistic"
print(f"\nPrompt: {PROMPT}")
print(f"Model: black-forest-labs/flux-schnell ($0.003/img)")
print(f"Baslangic: {datetime.now().strftime('%H:%M:%S')}")

output = replicate.run(
    "black-forest-labs/flux-schnell",
    input={
        "prompt": PROMPT,
        "num_outputs": 1,
        "aspect_ratio": "1:1",
        "output_format": "png",
    },
)

print(f"Bitis: {datetime.now().strftime('%H:%M:%S')}")
print(f"Output tipi: {type(output)}")

ROOT = Path(__file__).resolve().parents[1]
out_dir = ROOT / "data" / "validation" / "test"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "replicate_test.png"

# Replicate SDK 1.x: FileOutput objesi (.read()) veya FileOutput listesi
if isinstance(output, list):
    item = output[0]
    if hasattr(item, "read"):
        out_path.write_bytes(item.read())
    else:
        # eski: URL listesi
        out_path.write_bytes(requests.get(item).content)
elif hasattr(output, "read"):
    out_path.write_bytes(output.read())
else:
    print(f"Beklenmedik output: {output}")
    sys.exit(1)

size_kb = out_path.stat().st_size / 1024
print(f"\nKaydedildi: {out_path} ({size_kb:.1f} KB)")
print("\nTEST BASARILI! Replicate API calisiyor.")
