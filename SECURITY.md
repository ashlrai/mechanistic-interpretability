# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email the maintainer privately at `<security-contact-please-update>`. Include a description of the vulnerability, reproduction steps, and any proof-of-concept code. You will receive an acknowledgment within **24 hours**, an initial severity assessment within **7 days**, and a fix or coordinated public disclosure within **30 days**.

## Scope

**In scope:**

- Platform code in `src/mech_interp/` — orchestration, runner, CLI.
- The HuggingFace model adapter (`src/mech_interp/models/hf_adapter.py`), particularly path-traversal surfaces in `_get_module` and related helpers.
- The `trust_remote_code` configuration knob and any downstream escalation it enables.

**Out of scope:**

- Security of model weights or datasets we do not ship.
- Vulnerabilities in HuggingFace Hub infrastructure itself.
- Issues in third-party dependencies (report those upstream; we will patch promptly when upstream fixes land).

## Prior Security Fixes

This project takes security seriously. Two issues were already identified and resolved:

- **commit 903c994** — Silently escalated `trust_remote_code` flag and dunder-traversal path injection in `_get_module`. Both were fixed by adding explicit opt-in gating and input sanitization.

## Preferred Languages

Reports in English are preferred; Spanish is also fine.
