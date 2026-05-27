## Summary

(What this PR does + why)

## Test plan

- [ ] `bash scripts/check.sh` passes locally
- [ ] New tests added for new code paths
- [ ] `mech validate` passes (if YAMLs changed)
- [ ] `RUN_INTEGRATION_TESTS=1 bash scripts/check.sh` passes (if backend / family code changed)
- [ ] Docs updated (if behavior or CLI changed)

## Related issue

Fixes #

## Breaking change?

- [ ] Yes — explain migration path below
- [ ] No

## Checklist

- [ ] Followed commit convention (`<type>: <summary>`)
- [ ] No model weights committed
- [ ] No secrets in code or CI
- [ ] If new experiment family: spec YAML + integration test + `families.py` enum + `runner.py::experiment_for_spec` lazy-import block
