# v0.1.0-preview launch announcement (multi-channel draft)

Distribution draft for the v0.1.0-preview release. Pick channels and adapt — these are starting points, not finished posts.

---

## Twitter / X thread (10 tweets)

**1/10** Quietly built a local mechanistic-interpretability research platform over the last few weeks. Just released v0.1.0-preview. Two measured findings inside (SAE seed-reproducibility vs training scale; abliteration recipe degrading with model size). 19-second quickstart. 🧵

**2/10** Stack:
- 15 experiment families (circuit_patching, SAE, ACDC-edge, attribution_patching, refusal_direction, crosscoders, causal_scrubbing, ...)
- 37 CLI commands, agentic followup loop
- TransformerLens + universal HF backend
- 723 tests, ruff + mypy --strict clean
- All local, all reproducible

**3/10** Try it: `pip install` → `mech demo` → 19 seconds → 3-panel chart of GPT-2 doing factual recall (DLA + logit lens + circuit patching, on 3 prompts, gpt2-small).

The demo is the "click here for first experience" most mech-interp tools lack.

**4/10** Finding 1: how reproducible are SAE features across seeds?

Trained GPT-2 small SAEs at 5 seeds. Live-only median cross-seed best-match cosine: 0.26 at 992 tokens → 0.47 at 30K tokens (scale control, dead features 65%→8.5%). Only ~4% of features reach the 0.9 "same feature" bar.

**5/10** So: reproducibility improves with training scale (0.26→0.47 going 30×) but is far from the "same feature" bar even at 30K tokens. Whether it reaches 0.9 at production scale (1M+ tokens) is untested — that's the open question. Not a "crisis," a measured data point with a scale control.

Full paper draft + reproduce.sh: link below.

**6/10** Headline finding 2: I audited 4 Qwen instruct models from 0.5B to 3B params. The abliteration recipe's ability to suppress refusal **degrades monotonically with scale**:

- Qwen2-0.5B:   coeff −3 → refusal 0.33 → 0.00  **WORKS fully**
- Qwen2.5-0.5B: coeff −1..−3 graded → 0.67 → 0.33  **WORKS partially**
- Qwen2.5-1.5B: coeff −3 only → 0.33 → 0.67  **BACKFIRES**
- Qwen2.5-3B:   no negative-coefficient effect at all  **suppression gone**

**7/10** Same Qwen family. Both 0.5B models suppress refusal monotonically (canonical recipe-working pattern). The 1.5B model saturates and backfires. The 3B model loses suppression entirely — negative coefficients are dead, only positive coefficients (amplification) still respond. Most community abliteration targets sit at 3-9B, above the demonstrated working range.

Stage 4 scrubbing on Qwen2.5-1.5B formally rejects the L9+L10 attn circuit hypothesis: faithfulness **0.041**. Mechanistic evidence: refusal info lives in resid_post at L10-11 (recovery 0.5-1.0) but attention heads at those layers carry almost none (0.02-0.13). MLPs probably write it.

**8/10** Headline finding 3 (bonus): I ran our edge-level ACDC on the canonical Wang et al. 2022 IOI task on gpt2-small.

Recall: 3/12 canonical heads. Hits the late-layer name movers cleanly. Misses the entire upstream chain (s_inhibition, induction, duplicate_token).

Faithfulness 0.26 — flagged as partial. Honest validation > optimistic spin.

**8b/10** The platform also has:
- Interactive Gradio demo: `mech gradio`
- Pretrained SAE registry + sae_lens bridge
- Steering vector library: `mech apply-steering --vector refusal-qwen-2.5-1.5b-l10`
- Multi-model audit infra (8 candidate models prebuilt)
- Closed-loop iterate-from-run

**9/10** Reproducibility receipts on every run: environment.json with torch / numpy / transformer-lens versions + uv.lock SHA + seed + model weight hash. Months later anyone can verify why a result holds (or doesn't).

**10/10** Repo: https://github.com/ashlrai/mechanistic-interpretability
Release notes: /releases/tag/v0.1.0-preview
Docs: https://ashlrai.github.io/mechanistic-interpretability/
Full SAE seed-reproducibility writeup: /docs/publications/lesswrong_post.md

MIT licensed. Issues + PRs open.

---

## LessWrong / AlignmentForum post

Use the SAE seed-reproducibility writeup at `docs/publications/lesswrong_post.md` as the primary post. Cross-reference the abliteration audit as a shorter follow-up post a few days later if the first lands well.

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
