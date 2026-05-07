"""Eksik gorselleri yeniden uret (rate limit aware)."""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 24'teki fonksiyonlari kullan
import importlib.util
spec = importlib.util.spec_from_file_location(
    "_gen", Path(__file__).parent / "24_generate_images.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "data" / "validation"
IMG_DIR = VAL_DIR / "images"

log = pd.read_csv(VAL_DIR / "generation_log.csv")
failed = log[log["status"] != "OK"].copy()
print(f"Eksik gorsel sayisi: {len(failed)}")
if len(failed) == 0:
    print("Hepsi tamam, retry gerekmiyor.")
    sys.exit(0)

DELAY_BETWEEN = 12.0  # 6 req/min = en az 10s; 12s guvenli marj

for idx, row in failed.iterrows():
    vid = row["validation_id"]
    role = row["role"]
    model = row["model"]
    api = row["api"]
    prompt = row["prompt"]
    out_path = IMG_DIR / f"{vid}_{role}_{model.replace('.', '').replace('/', '_')}.png"
    print(f"\n[{vid}] {role:8s} {model:18s}")
    for attempt in range(3):
        try:
            elapsed = gen.generate_one(model, api, prompt, out_path)
            size_kb = out_path.stat().st_size / 1024
            print(f"  OK (deneme {attempt + 1}, {elapsed:.1f}s, {size_kb:.0f} KB)")
            log.loc[idx, "status"] = "OK"
            log.loc[idx, "image_path"] = str(out_path.relative_to(ROOT))
            log.loc[idx, "elapsed_s"] = round(elapsed, 1)
            break
        except Exception as e:
            err = str(e)[:120]
            print(f"  deneme {attempt + 1} fail: {err}")
            if "429" in err or "throttl" in err.lower():
                wait = 15 + attempt * 5
                print(f"  rate limit; {wait}s bekleniyor...")
                time.sleep(wait)
            else:
                log.loc[idx, "status"] = f"ERROR: {err}"
                break
    if api == "replicate":
        print(f"  bir sonraki cagri icin {DELAY_BETWEEN}s bekleniyor...")
        time.sleep(DELAY_BETWEEN)

log.to_csv(VAL_DIR / "generation_log.csv", index=False)
n_ok = (log["status"] == "OK").sum()
print(f"\nGuncel: {n_ok}/{len(log)} basarili")
print(f"Toplam tahmini maliyet: ${log[log['status'] == 'OK']['cost_usd'].sum():.3f}")
