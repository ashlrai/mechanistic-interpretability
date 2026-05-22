# Roadmap

This project should grow from a reliable local experiment spine into a serious mechanistic
interpretability lab. Keep each stage runnable before moving to the next.

## Stage 1: Research Spine

- Validate experiment YAML before execution.
- Persist resolved specs, configs, run status, metrics, and artifact manifests.
- Add provider health checks for Ollama and LM Studio.
- Add a TransformerLens smoke path that does not require heavyweight model downloads in tests.
- Keep CI green on every push.

## Stage 2: TransformerLens Execution

- Load small supported models through the instrumented backend.
- Capture activations for selected hook names.
- Save tensor summaries by default and full tensors only when explicitly requested.
- Add smoke experiments for residual stream, MLP output, and attention pattern capture.

## Stage 3: First Real Experiment Families

- Polysemanticity: activation selectivity sweeps across curated prompt groups.
- Superposition: sparse feature sweeps and residual-stream feature geometry.
- Circuit patching: clean/corrupted prompt pairs and causal activation replacement.

## Stage 4: Local Scale

- Add resource planning for batch size, activation retention, dtype, and model size.
- Add resumable run queues and failure recovery.
- Add report generation that summarizes thousands of runs into research notes.

## Stage 5: Agentic Research Loop

- Let agents propose experiment matrices from prior results.
- Require every generated experiment to write a spec before execution.
- Add automated result triage, anomaly detection, and follow-up recommendations.
