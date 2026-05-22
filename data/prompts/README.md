# Prompt Datasets

Prompt datasets are small, local files used by experiment specs and loaders. The loader supports:

- JSONL files with one object per line. Each object must include `prompt`; `id` and `metadata` are optional. Extra JSON fields are folded into `metadata`.
- Plain text files with one prompt per non-empty line. Lines beginning with `#` are comments.

Each loaded `PromptRecord` and `PromptDataset` has a stable SHA-256 digest computed from normalized
record content. Hashes ignore file paths and JSON field ordering, but include prompt text, record
IDs, and metadata.

Curated starter files:

- `factual.jsonl`: simple factual completion prompts with expected answers in metadata.
- `clean_corrupted.jsonl`: clean and corrupted prompt pairs for activation patching.
- `ambiguous_polysemantic.txt`: short ambiguous prompts for polysemanticity probes.
