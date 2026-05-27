---
name: Reproduce a published finding
about: Request that the platform reproduce a canonical mech-interp finding
title: "[VALIDATE] "
labels: validation, enhancement
---

## Published finding to reproduce

Paper / blog / repo link:

One-paragraph description of the finding:

## The canonical numbers we'd be matching

Specific quantitative claims the reproduction should match (e.g., "IOI circuit recovers 12 specific heads", "SAE feature 47 fires on banana-related tokens"):

## Prerequisites

- Model: (e.g., `gpt2-small`, `Llama-3.2-1B`, `Qwen2.5-1.5B-Instruct`)
- Corpus: (e.g., 30 IOI prompts, 1000 documents from pile-1k)
- Backend: `transformerlens` / `huggingface` / either
- Compute estimate (CPU minutes / GPU minutes):

## Existing platform families to use

Which `ExperimentFamily` values would chain to produce this:

```
1. mech run --name ...
2. mech run --name ...
3. mech audit-... (compile)
```

## Success criteria

Specific assertion that would close this issue (e.g., "≥7 of 12 canonical IOI heads recovered with `acdc_edge`", "scrubbing faithfulness > 0.7 on the X hypothesis").

## Related investigations

Link to any existing `docs/investigations/*.md` that this builds on.
