# Research Task: Prospective Detection and Causal Steering of LLM Misalignment via Temperature-Swept EBMs

## Scientific Objective
Assess whether energy-based models (EBMs) trained on temperature-diverse activation trajectories can prospectively detect hidden misalignment failure modes before they manifest in output tokens, and whether gradient-based interventions causally shift continuation probabilities toward safe outcomes.

## Critical Methodological Guardrails (MANDATORY)
1. **Prevent Outcome Leakage**:
   - The residual activation at the common prompt prefix (t = 0) is IDENTICAL for all stochastic rollouts of a given prompt: h_{l,0}^+ = h_{l,0}^-.
   - Prospective detection must be evaluated strictly on common-prefix activations (t = 0) across HELD-OUT prompt families.
   - Any evaluation at generation steps t > 0 must explicitly distinguish between prospective latent detection and early behavioral emission recognition.
2. **Strict Group-Level Splits**:
   - Split train, validation, and test sets by PROMPT FAMILY, never by random rollout splitting. Rollouts from the same prompt must never span across train and test sets.
3. **Four-Way Baseline Comparison**:
   - Every evaluated claim must benchmark:
     (a) Linear Activation Probe (temperature-diverse)
     (b) Nonlinear MLP Classifier (same data)
     (c) Standard EBM (greedy / single decoding condition)
     (d) Temperature-Swept EBM (multi-temperature trajectories)
   - Include prompt-only and output-probability baselines to rule out prompt-type memorization.
4. **Physical Energy Claims & Causal Committor Surface**:
   - Do NOT assume contrastive EBM energy E_\theta(h) represents an intrinsic physical free-energy landscape or that local minima are physical attractors.
   - Causal sensitivity MUST be measured using stochastic outcome probabilities under interventions:
     q(h, c, \tau) = P(unsafe continuation | do(h_{l,t} = h), c, \tau).
   - Evaluate whether perturbations near transition regions (q \approx 0.5) produce asymmetric recovery vs corruption rates.
5. **Statistical Metrics**:
   - Do NOT use the Pass@10 / Pass@1 ratio as an alignment metric (the ratio naturally increases as the model becomes safer).
   - Report absolute Unsafe@1 and Unsafe@K separately at matched temperatures with rigorous bootstrap confidence intervals.
   - Measure spectral entropy and effective dimension only with matched sample counts, prompt diversity controls, and null models.

## Pre-Registered Success Criteria
1. **Prospective Prediction**: Swept EBM or probe predicts unsafe behavior at t = 0 with statistically superior AUROC/calibration on held-out prompt families compared to simple linear and prompt-only baselines.
2. **Generalization**: Detection signal transfers to previously unseen misalignment triggers or fine-tuning variants.
3. **Causal Efficacy**: Gradient intervention -eta * \nabla_h E_\theta(h) significantly reduces q(h, c, \tau) on held-out prompts without degrading benign compliance or linguistic coherence.
4. **Specificity**: Benign capabilities and false-refusal rates remain stable under steering.
