#!/usr/bin/env python
"""Extract a helpfulness steering direction from Qwen2.5-0.5B-Instruct.

Contrasts 8 helpful vs 8 evasive response prompts at layer 8
(blocks.8.hook_resid_post).  Saves:
  data/steering/helpfulness_qwen_2.5_0.5b_l8.safetensors
  data/steering/helpfulness_qwen_2.5_0.5b_l8.safetensors.json

Usage:
    uv run --extra interp python scripts/extract_helpfulness_direction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

HOOK_SITE = "blocks.8.hook_resid_post"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = ROOT / "data" / "steering"
OUTPUT_PATH = OUTPUT_DIR / "helpfulness_qwen_2.5_0.5b_l8.safetensors"
SIDECAR_PATH = OUTPUT_DIR / "helpfulness_qwen_2.5_0.5b_l8.safetensors.json"

# Helpful: direct, substantive answers
HELPFUL_PROMPTS = [
    "What is the capital of France? The capital of France is Paris.",
    "How do I sort a list in Python? Use list.sort() or sorted(list).",
    "What causes rainbows? Light refracts through water droplets splitting into colours.",
    "How do I make pasta? Boil water, add salt, cook pasta 8-10 minutes, drain.",
    "What is the speed of light? Approximately 299,792,458 metres per second.",
    "How do I open a file in Python? Use open('filename', 'r') as a context manager.",
    "What is photosynthesis? Plants convert sunlight, CO2, and water into glucose.",
    "How do I reverse a string in Python? Use slicing: s[::-1].",
]

# Evasive: deflection, refusal to engage, hedging without substance
EVASIVE_PROMPTS = [
    "What is the capital of France? I'm not sure I can answer that question.",
    "How do I sort a list in Python? That depends on many factors I cannot determine.",
    "What causes rainbows? There are many perspectives and I cannot say for certain.",
    "How do I make pasta? I would recommend consulting a professional chef for advice.",
    "What is the speed of light? This is a complex topic beyond my ability to address.",
    "How do I open a file in Python? I'm unable to provide programming guidance.",
    "What is photosynthesis? I prefer not to speculate about scientific processes.",
    "How do I reverse a string in Python? I cannot assist with coding questions.",
]


def _collect_activations(model: object, tokenizer: object, prompts: list[str]) -> object:
    import torch
    acts = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]  # type: ignore[attr-defined]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=HOOK_SITE)  # type: ignore[attr-defined]
        h = cache[HOOK_SITE][0, -1, :]
        acts.append(h)
    return torch.stack(acts, dim=0)


def main() -> None:
    import torch
    from safetensors.torch import save_file
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    try:
        model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu", dtype=torch.float32)
    except Exception as exc:
        print(f"ERROR: Could not load {MODEL_NAME} via TransformerLens: {exc}")
        print("Skipping helpfulness direction extraction.")
        sys.exit(1)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Collecting activations at {HOOK_SITE}...")
    helpful_acts = _collect_activations(model, tokenizer, HELPFUL_PROMPTS)
    evasive_acts = _collect_activations(model, tokenizer, EVASIVE_PROMPTS)

    raw = helpful_acts.mean(dim=0) - evasive_acts.mean(dim=0)
    norm = float(raw.norm().item())
    direction = raw / norm

    proj_help = (helpful_acts @ direction).float()
    proj_evas = (evasive_acts @ direction).float()
    margin = float((proj_help.mean() - proj_evas.mean()).item())
    spread = float((proj_help.std() + proj_evas.std() + 1e-8).item())
    quality = margin / spread

    print(f"Direction norm (raw): {norm:.4f}")
    print(f"Extraction quality (projection margin): {quality:.4f}")

    save_file({"direction": direction.cpu()}, str(OUTPUT_PATH))
    print(f"Saved direction to {OUTPUT_PATH}")

    sidecar = {
        "name": "helpfulness-qwen-2.5-0.5b-l8",
        "model": MODEL_NAME,
        "hook_site": HOOK_SITE,
        "hidden_dim": int(direction.shape[0]),
        "direction_norm": norm,
        "extraction_quality": quality,
        "helpful_prompt_count": len(HELPFUL_PROMPTS),
        "evasive_prompt_count": len(EVASIVE_PROMPTS),
        "description": (
            "Helpfulness direction: helpful minus evasive response prompts. "
            "Positive coefficients steer toward more helpful completions."
        ),
        "license": "research-only",
        "source_paper": "Arditi et al. 2024",
    }
    SIDECAR_PATH.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved sidecar to {SIDECAR_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
