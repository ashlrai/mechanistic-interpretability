#!/usr/bin/env python
"""Extract a sentiment steering direction from gpt2-medium.

Contrasts 10 positive vs 10 negative movie-review prompts at layer 8
(blocks.8.hook_resid_pre).  Saves:
  data/steering/sentiment_gpt2_medium_l8.safetensors   -- direction tensor
  data/steering/sentiment_gpt2_medium_l8.safetensors.json  -- metadata sidecar

Usage:
    uv run --extra interp python scripts/extract_sentiment_direction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src/ is on path when run from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

HOOK_SITE = "blocks.8.hook_resid_pre"
MODEL_NAME = "gpt2-medium"
OUTPUT_DIR = ROOT / "data" / "steering"
OUTPUT_PATH = OUTPUT_DIR / "sentiment_gpt2_medium_l8.safetensors"
SIDECAR_PATH = OUTPUT_DIR / "sentiment_gpt2_medium_l8.safetensors.json"

POSITIVE_PROMPTS = [
    "This movie was absolutely wonderful and I loved every minute of it.",
    "An outstanding film — the performances were brilliant and moving.",
    "I was completely captivated from start to finish. Highly recommended!",
    "The cinematography is breathtaking and the story is deeply touching.",
    "A masterpiece of modern cinema. The direction was flawless.",
    "I left the theatre feeling uplifted and inspired. What a beautiful film.",
    "The acting was superb and the script was witty and heartfelt.",
    "One of the best films I have seen in years. Truly extraordinary.",
    "This film exceeded every expectation. A pure joy to watch.",
    "The characters felt real and I cared deeply about their journey.",
]

NEGATIVE_PROMPTS = [
    "This movie was a complete waste of time. I hated every minute.",
    "A terrible film — the acting was wooden and the plot was nonsense.",
    "I walked out after an hour. Painfully boring and predictable.",
    "The worst film I have seen all year. Avoid at all costs.",
    "The direction was amateurish and the script made no sense.",
    "I left the theatre feeling cheated. What a dreadful experience.",
    "The acting was awful and the story was both confusing and dull.",
    "One of the worst films I have seen in years. A total disaster.",
    "This film failed on every level. Deeply disappointing.",
    "The characters were unlikeable and I stopped caring immediately.",
]


def _collect_activations(model: object, tokenizer: object, prompts: list[str]) -> object:
    """Return (n_prompts, d_model) last-token activations at HOOK_SITE."""
    import torch
    acts = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]  # type: ignore[attr-defined]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=HOOK_SITE)  # type: ignore[attr-defined]
        h = cache[HOOK_SITE][0, -1, :]  # (d_model,)
        acts.append(h)
    return torch.stack(acts, dim=0)  # (n, d_model)


def main() -> None:
    import torch
    from safetensors.torch import save_file
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu", dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Collecting activations at {HOOK_SITE}...")
    pos_acts = _collect_activations(model, tokenizer, POSITIVE_PROMPTS)
    neg_acts = _collect_activations(model, tokenizer, NEGATIVE_PROMPTS)

    # Mean-difference direction
    raw = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    norm = float(raw.norm().item())
    direction = raw / norm

    # Extraction quality (projection margin)
    proj_pos = (pos_acts @ direction).float()
    proj_neg = (neg_acts @ direction).float()
    margin = float((proj_pos.mean() - proj_neg.mean()).item())
    spread = float((proj_pos.std() + proj_neg.std() + 1e-8).item())
    quality = margin / spread

    print(f"Direction norm (raw): {norm:.4f}")
    print(f"Extraction quality (projection margin): {quality:.4f}")

    save_file({"direction": direction.cpu()}, str(OUTPUT_PATH))
    print(f"Saved direction to {OUTPUT_PATH}")

    sidecar = {
        "name": "sentiment-gpt2-medium-l8",
        "model": MODEL_NAME,
        "hook_site": HOOK_SITE,
        "hidden_dim": int(direction.shape[0]),
        "direction_norm": norm,
        "extraction_quality": quality,
        "positive_prompt_count": len(POSITIVE_PROMPTS),
        "negative_prompt_count": len(NEGATIVE_PROMPTS),
        "description": (
            "Sentiment direction: positive minus negative movie-review prompts. "
            "Positive coefficients steer toward positive sentiment."
        ),
        "license": "research-only",
        "source_paper": "Zou et al. 2023",
    }
    SIDECAR_PATH.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved sidecar to {SIDECAR_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
