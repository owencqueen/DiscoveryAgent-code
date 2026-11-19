# Temperature-Swept Activation & Rollout Dataset for LLM Misalignment

This directory contains prompt families, stochastic decoding rollouts across temperature sweeps, extracted residual-stream activations, and behavioral outcome labels for open-weight language models.

## Dataset Overview

The dataset provides paired activation trajectories and behavioral annotations across both standard aligned models (e.g., Qwen2.5-1.5B/7B, Llama-3.1-8B-Instruct) and controlled misalignment variants (e.g., Anthropic sleeper-agent models, Betley et al. emergent misalignment fine-tunes).

Directory path: {datapath}

### File Structure
- `data/prompts/`:
  - `prompt_families.json`: Metadata and text for 250 distinct prompt families across benign, adversarial jailbreaks, sleeper triggers, and over-refusal edge cases.
  - `splits.json`: Pre-defined group-level train, validation, and test splits grouped by prompt family (DO NOT use random splits across completions of the same prompt).
- `data/rollouts/`:
  - `completions_tau_{temp}.jsonl`: Generated rollouts across temperatures tau in {0.0, 0.4, 0.7, 1.0, 1.3} with 20-50 samples per prompt.
  - `behavioral_labels.csv`: Evaluated behavioral labels (safe_aligned, unsafe_jailbroken, deceptive_defection, hallucinated, over_refusal).
- `data/activations/`:
  - `h_common_prefix/`: Residual stream activations h_{l,0} extracted at the final prompt token (before the first generated token).
  - `h_rollout_steps/`: Activations h_{l,t} extracted across generation steps t in {1, 2, 5, 10, 20}.
  - `metadata.json`: Layer dimensions, model identifiers, token offsets, and forward-pass hooks.

## Evaluation & Benchmark Targets
1. Prospective Detection:
   - AUROC and Log-Loss on held-out prompt families evaluated strictly at the common-prefix token (t = 0).
2. Four-way Baseline Comparison:
   - Linear Activation Probe vs Nonlinear MLP vs Standard EBM (tau=0.0) vs Swept EBM (multi-tau).
3. Causal Committor Perturbations:
   - Measure q(h, c, tau) = P(unsafe | do(h_{l,t} = h), c, tau) under targeted gradient and control perturbations.
