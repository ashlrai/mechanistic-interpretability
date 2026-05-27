# SAE Replication Crisis — artifact bundle

Files in this directory:

- **`paper.md`** — short paper draft (~1500 words). Preliminary report; 1 model, 1 corpus.
- **`thread.md`** — 12-tweet thread version for social distribution.
- **`metrics.json`** — frozen raw stability_report.json from the original 5-seed runs.
- **`reproduce.sh`** — exact commands a third party would run to reproduce the result from a fresh clone of the repo.

Headline numbers (full table in `paper.md`):

| Condition | Median best-match cosine | Stability ≥ 0.9 |
|---|---:|---:|
| Layer 0, 128f, full matrix | 0.095 | 0.16% |
| Layer 0, 128f, live-only | **0.500** | 0.48% |
| Layer 6, 128f, live-only | 0.323 | 0.00% |
| Layer 6, 512f, live-only | 0.257 | 0.00% |

The live-only layer-0 result (0.500) is the cleanest single-number summary. The 0.9 threshold of "same feature" is not crossed in any condition tested.
