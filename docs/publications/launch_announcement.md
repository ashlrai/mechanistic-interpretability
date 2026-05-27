# v0.1.0-preview launch announcement (multi-channel draft)

Distribution draft for the v0.1.0-preview release. Pick channels and adapt — these are starting points, not finished posts.

---

## Twitter / X thread (10 tweets)

**1/10** Quietly built a local mechanistic-interpretability research platform over the last few weeks. Just released v0.1.0-preview. Two real findings inside (one publishable negative result on abliteration, one on SAE seed-stability). 19-second quickstart. 🧵

**2/10** Stack:
- 15 experiment families (circuit_patching, SAE, ACDC-edge, attribution_patching, refusal_direction, crosscoders, causal_scrubbing, ...)
- 37 CLI commands, agentic followup loop
- TransformerLens + universal HF backend
- 723 tests, ruff + mypy --strict clean
- All local, all reproducible

**3/10** Try it: `pip install` → `mech demo` → 19 seconds → 3-panel chart of GPT-2 doing factual recall (DLA + logit lens + circuit patching, on 3 prompts, gpt2-small).

The demo is the "click here for first experience" most mech-interp tools lack.

**4/10** Headline finding 1 (live in docs/publications/):

I trained the same SAE 5 times with different seeds. Live-only median best-match cosine: 0.50 at layer 0, 0.32 at layer 6, 0.26 with bigger dictionaries.

**Zero conditions cross the 0.9 "same feature" threshold.**

**5/10** This is informative because the SAE literature implicitly assumes feature dictionaries are seed-stable. "The X feature of model M" is a training-run property, not a model property, at the scales I tested.

Full paper draft + reproduce.sh: link below.

**6/10** Headline finding 2: I ran a full 4-stage abliteration audit on Qwen2.5-1.5B-Instruct.

- Direction extracted: quality 4.105 (clean separation)
- Layer-10 single-coefficient steering: doesn't decrease refusal
- Circuit hypothesis "L9+L10 attn implement refusal": **faithfulness 0.041, REJECTED**

**7/10** Mechanistic evidence: the refusal info is in resid_post at L10-11 (recovery 0.5-1.0), but the local attention heads at those layers carry almost none of it (recovery 0.02-0.13). MLPs probably write it — the standard abliteration recipe targets the wrong site.

**8/10** The platform also has:
- Interactive Gradio demo: `mech gradio`
- Pretrained SAE registry + sae_lens bridge
- Steering vector library: `mech apply-steering --vector refusal-qwen-2.5-1.5b-l10`
- Multi-model audit infra (8 candidate models prebuilt)
- Closed-loop iterate-from-run

**9/10** Reproducibility receipts on every run: environment.json with torch / numpy / transformer-lens versions + uv.lock SHA + seed + model weight hash. Months later anyone can verify why a result holds (or doesn't).

**10/10** Repo: https://github.com/ashlrai/mechanistic-interpretability
Release notes: /releases/tag/v0.1.0-preview
Docs: https://ashlrai.github.io/mechanistic-interpretability/
Full SAE replication crisis writeup: /docs/publications/sae_replication_crisis.md

MIT licensed. Issues + PRs open.

---

## LessWrong / AlignmentForum post

Use the SAE replication crisis writeup at `docs/publications/lesswrong_post.md` as the primary post. Cross-reference the abliteration audit as a shorter follow-up post a few days later if the first lands well.

---

## Newsletter / blog submission targets

- **AI Alignment Newsletter** (Rohin Shah / Zac Hatfield-Dodds) — submit via the existing newsletter submission form
- **The Alignment Forum** — auto-syndicated from LessWrong, no separate submission
- **Mech-Interp Discord** (Neel Nanda's TransformerLens discord) — share in #general after posting
- **Anthropic's mech-interp research** — informal cite via email to the SAE team if you have a contact
- **EleutherAI Discord** — share in #interpretability after posting

---

## What's authorized vs not (re-read CLAUDE.md guidance)

The drafts above are starting points. **Posting them publicly should be a human decision** because:

1. **Voice + tone risk.** I write in a specific cadence that may not match your established voice. Readers who follow you will notice if a post sounds like Claude.
2. **Reputation risk.** A poorly-received post on AlignmentForum / Twitter can be difficult to unwind. You should read each post end-to-end before pressing send.
3. **Engagement risk.** If the post gets responses, replies should come from you — not from me posting under your name without you seeing the thread context.

I can polish/iterate these drafts based on your feedback. I should not post them as you.
