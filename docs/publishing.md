# Publishing to HuggingFace Hub

This platform can publish SAE weights, steering vectors, and investigation
writeups to a HuggingFace Hub repository so collaborators can load them
without cloning this repo.

## Quick start

```bash
# Preview everything that would be published (no upload, no credentials needed)
mech publish-all --user myorg --dry-run

# Authenticate once, then publish for real
huggingface-cli login
mech publish-all --user myorg
```

## Bundle formats

### SAE bundle (`kind: sae`)

Produced by `mech publish-sae --run-id N --repo <user>/<name>`.

| File | Description |
|---|---|
| `README.md` | Auto-generated with YAML frontmatter, model card, and usage snippet |
| `bundle_metadata.json` | Machine-readable manifest (spec + result + environment) |
| `sae_weights.safetensors` | Encoder/decoder weights (TopKSAE layout) |
| `sae_weights.safetensors.json` | Hyperparameters sidecar (n_features, k, input_dim, training stats) |
| `feature_analysis.json` | Per-feature top-activating prompts |
| `environment.json` | Full environment provenance (Python version, package versions, seed, torch runtime) |

Loading from Hub (once registered in `SAE_REGISTRY`):

```python
from mech_interp.sae.registry import load_pretrained_sae
sae, config = load_pretrained_sae("myorg/sae-run51-gpt2-medium")
```

### Steering vector bundle (`kind: steering`)

Produced by `mech publish-steering --vector <name> --repo <user>/<name>`.

| File | Description |
|---|---|
| `README.md` | Auto-generated with YAML frontmatter, vector card, and usage snippet |
| `bundle_metadata.json` | Machine-readable manifest |
| `direction.safetensors` | Unit-norm direction tensor under key `"direction"` |
| `direction.safetensors.json` | Extraction metadata sidecar (norm, quality, prompt counts) |

Loading from Hub — add `hf_repo` to `STEERING_REGISTRY` and the loader
falls back automatically when the local file is absent:

```python
from mech_interp.steering.registry import load_steering_vector
direction, metadata = load_steering_vector("sentiment-gpt2-medium-l8")
```

### Investigation bundle (`kind: investigation`)

Produced by `mech publish-investigation --slug <slug> --repo <user>/<name>`.

| File | Description |
|---|---|
| `README.md` | Auto-generated with YAML frontmatter and description |
| `bundle_metadata.json` | Machine-readable manifest |
| `investigation.md` | Full investigation writeup from `docs/investigations/<slug>.md` |
| *(optional)* `metrics.json`, `paper.md`, … | Any files from `docs/publications/<slug>_artifacts/` |

## Command reference

### `mech publish-sae`

```
mech publish-sae --run-id N --repo <user>/<name> [--dry-run] [--license MIT]
```

- `--run-id`: local run ID (e.g. `51`)
- `--repo`: target HF repo (`user/repo-name`)
- `--artifact-root`: override default `artifacts/` directory
- `--license`: SPDX license string for the README frontmatter
- `--dry-run`: print the bundle table; do not upload

### `mech publish-steering`

```
mech publish-steering --vector <name> --repo <user>/<name> [--dry-run]
```

- `--vector`: registry key from `mech list-steering`
- `--repo`: target HF repo

### `mech publish-investigation`

```
mech publish-investigation --slug <slug> --repo <user>/<name> [--dry-run]
```

- `--slug`: investigation filename without `.md` (e.g. `sae_replication_crisis`)

### `mech publish-all`

```
mech publish-all --user <user> [--sae-run-ids 51,52] [--dry-run]
```

Convenience command that bundles and uploads every SAE run (auto-detected or
explicit), every steering vector, and every investigation under one namespace.

Repo naming convention:

| Artifact | Repo name |
|---|---|
| SAE run 51 on gpt2-medium | `<user>/sae-run51-gpt2-medium` |
| Steering vector `sentiment-gpt2-medium-l8` | `<user>/sv-sentiment-gpt2-medium-l8` |
| Investigation `sae_replication_crisis` | `<user>/inv-sae-replication-crisis` |

## HuggingFace Hub fallback in registries

`load_steering_vector` now attempts a Hub download when `local_path` is
missing but `hf_repo` is set on the descriptor:

```python
SteeringVectorDescriptor(
    name="sentiment-gpt2-medium-l8",
    ...
    local_path=None,          # not cloned locally
    hf_repo="myorg/sv-sentiment-gpt2-medium-l8",
)
```

The download is cached by `huggingface_hub` in `~/.cache/huggingface/`.

## Authenticating

Publishing requires a HuggingFace account and write token:

```bash
huggingface-cli login
# paste your token from https://huggingface.co/settings/tokens
```

After login the token is cached in `~/.cache/huggingface/token`.
All subsequent `mech publish-*` calls use it automatically.

Org repos require the token to have write access to the target org.

## Dry-run output (example)

```
╭─────────── mech publish ───────────╮
│ DRY RUN — no files uploaded        │
│ Bundle: sae-run51-gpt2-medium      │
│ Kind:   sae                        │
│ Target: https://huggingface.co/... │
╰────────────────────────────────────╯

Files that would be uploaded
┌──────────────────────────┬──────────────────┬──────────┐
│ Staged name              │ Source path      │ Size     │
├──────────────────────────┼──────────────────┼──────────┤
│ README.md                │ (generated)      │  3.2 KB  │
│ bundle_metadata.json     │ (generated)      │  1.1 KB  │
│ sae_weights.safetensors  │ artifacts/run-…  │ 16.0 MB  │
│ sae_weights.safetensors… │ artifacts/run-…  │   407 B  │
│ environment.json         │ artifacts/run-…  │   578 B  │
│ feature_analysis.json    │ artifacts/run-…  │ 699.0 KB │
└──────────────────────────┴──────────────────┴──────────┘
```
