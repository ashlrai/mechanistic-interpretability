---
name: New experiment family
about: Propose a new mech-interp experiment family
title: "[FAMILY] "
labels: enhancement, family-request
---

## Paper reference

Cite the paper or technique this family would implement. Link to arXiv / blog / repo.

## Family name + enum addition

Proposed `ExperimentFamily` enum value (snake_case, e.g. `transcoder`, `causal_tracing`):

## YAML schema

What parameters the spec would accept. Use pydantic-style sketch:

```yaml
name: example-spec-name
family: <proposed_family>
backend: transformerlens
parameters:
  model: gpt2-small
  hook_site: blocks.0.hook_resid_pre
  # ... other params
```

## Artifacts produced

What `result.artifacts` would contain (per-run files + their JSON schema):

- `XYZ.json` — schema sketch
- `XYZ.safetensors` — what it stores

## Reference family

Closest existing family to model after (e.g. `polysemanticity_sae` for SAE-like work, `circuit_patching` for intervention-based work). Path: `src/mech_interp/experiments/<family>.py`.

## Estimated work

- New module: ~? LOC
- New YAML: 1
- New tests: ~?
- Estimated runtime on gpt2-small: ?

## Related families / proposal generator

Does this family have a natural follow-up family that a `ProposalGenerator` could emit? If so, sketch it.
