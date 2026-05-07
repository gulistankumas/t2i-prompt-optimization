"""
Adim 1/4: 5 validation prompt'u sec.

Strateji: Premium modda Replicate veya OpenAI'da bulunan modellerin
onerildigi track'leri sec.

Cikti: data/validation/selected_5_prompts.csv
"""

import pandas as pd
import json
import os

# Test seti ve lookup
df_test = pd.read_csv('data/processed/test_prompts.csv')
print(f"Test setinde {len(df_test)} prompt")

with open('data/processed/lookup_table_2mode.json', 'r') as f:
    lookup = json.load(f)

premium_choices = {t: lookup['premium'][t]['model'] for t in lookup['premium']}
print("\nPremium mod secimleri:")
for track, model in premium_choices.items():
    print(f"  {track:20s} -> {model}")

# Erisilebilir modeller
replicate_models = {'SD1.5', 'SDXL', 'SD3.5-Large', 'FLUX.1-schnell', 'FLUX.1-dev', 'Qwen-Image'}
openai_models = {'GPT-Image-1'}
available_models = replicate_models | openai_models

# 5 prompt secim stratejisi
target_distribution = {
    'style': 1,
    'affection': 1,
    'composition': 1,
    'text_rendering': 2  # 2 tane GPT-Image-1 testi
}

SEED = 42
selected_prompts = []
for track, count in target_distribution.items():
    track_prompts = df_test[df_test['track'] == track]
    if len(track_prompts) < count:
        print(f"UYARI: {track} icin {len(track_prompts)} prompt var, {count} istendi")
        sampled = track_prompts
    else:
        sampled = track_prompts.sample(count, random_state=SEED)
    selected_prompts.append(sampled)

df_sel = pd.concat(selected_prompts).reset_index(drop=True)
df_sel['router_model'] = df_sel['track'].map(premium_choices)
df_sel['baseline_model'] = 'FLUX.1-dev'


def get_api(m):
    return 'replicate' if m in replicate_models else ('openai' if m in openai_models else 'unknown')


df_sel['router_api'] = df_sel['router_model'].apply(get_api)
df_sel['baseline_api'] = df_sel['baseline_model'].apply(get_api)

cost_map = {
    'SD1.5': 0.003, 'SDXL': 0.030, 'SD3.5-Large': 0.065,
    'FLUX.1-schnell': 0.003, 'FLUX.1-dev': 0.030,
    'GPT-Image-1': 0.167, 'Qwen-Image': 0.025
}
df_sel['router_cost'] = df_sel['router_model'].map(cost_map)
df_sel['baseline_cost'] = df_sel['baseline_model'].map(cost_map)
df_sel['validation_id'] = ['val_' + str(i + 1).zfill(2) for i in range(len(df_sel))]

os.makedirs('data/validation', exist_ok=True)
output_path = 'data/validation/selected_5_prompts.csv'
df_sel.to_csv(output_path, index=False)

# Rapor
print(f"\n{'=' * 70}\nSECILEN 5 PROMPT\n{'=' * 70}")
for _, row in df_sel.iterrows():
    print(f"\n[{row['validation_id']}] Track: {row['track']}")
    print(f"  Prompt: {row['prompt'][:90]}...")
    print(f"  Router:   {row['router_model']:18s} ({row['router_api']:9s}) ${row['router_cost']:.3f}")
    print(f"  Baseline: {row['baseline_model']:18s} ({row['baseline_api']:9s}) ${row['baseline_cost']:.3f}")

tot_router = df_sel['router_cost'].sum()
tot_baseline = df_sel['baseline_cost'].sum()
print(f"\n{'=' * 70}\nMALIYET\n{'=' * 70}")
print(f"Router toplam:   ${tot_router:.3f}")
print(f"Baseline toplam: ${tot_baseline:.3f}")
print(f"Genel:           ${tot_router + tot_baseline:.3f}")
print(f"+ GPT-4o-vision: ~$1.00")
print(f"= Beklenen:      ~${tot_router + tot_baseline + 1:.2f}")
print(f"\nKaydedildi: {output_path}")
