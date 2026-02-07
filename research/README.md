# Research: Novel ML/AI Connections

Ten potentially novel findings discovered by analyzing cross-domain mathematical
structures in the ML-ToolBox corpus. Each has a proof-of-concept implementation
with reproducible experiments.

## Findings

### 1. L^-2 Regularization (Heisenberg Regularizer)
**File:** `heisenberg_regularizer.py`

A new regularization penalty lambda/Var(w) that prevents weight collapse -- the exact
mathematical dual of standard L2 regularization. Where L2 penalizes weights
that are too spread out, L^-2 penalizes weights that are too concentrated.

**Key equation:** Loss = CrossEntropy + lambda_2*||w||^2 + lambda_H / (Var(w) + epsilon)

**Target problem:** Representational collapse in self-supervised learning,
attention head diversity, ensemble diversity.

### 2. Dissipative Grokking Predictor
**File:** `dissipative_grokking.py`

Models neural network training dynamics as a Prigogine dissipative system to
predict when grokking (sudden generalization after memorization) will occur.
Derives a critical learning-rate-to-weight-decay ratio that triggers the
phase transition from memorization to generalization.

**Key prediction:** Grokking occurs when eta/lambda exceeds a critical threshold
derived from the dissipative stability condition.

### 3. Gray Code Hyperparameter Search
**File:** `gray_code_hpo.py`

Uses reflected Gray codes for hyperparameter optimization. Each evaluation
changes exactly one hyperparameter, giving perfect single-factor attribution
as a free byproduct of the search -- something no existing HPO method provides.

**Key property:** Every step in the search is simultaneously an optimization
step AND a controlled experiment.

### 4. PID-Controlled Learning Rate
**File:** `pid_learning_rate.py`

Applies a PID (Proportional-Integral-Derivative) feedback controller to the
learning rate, treating loss as process variable and desired loss as setpoint.
Unlike open-loop schedulers (cosine, step), this is a closed-loop controller
that reacts to actual training dynamics.

**Key equation:** LR(t) = LR(t-1) * (1 + Kp*e + Ki*int(e) + Kd*de/dt)

**Target problem:** Pathological loss landscapes (noisy, plateau-cliff),
where fixed schedules fail.

### 5. Satisficing Optimization with Adaptive Aspiration
**File:** `satisficing_optimization.py`

Applies Herbert Simon's bounded rationality to hyperparameter optimization.
Instead of finding the global optimum, finds "good enough" solutions fast
using adaptive aspiration levels that learn from search history.

**Key equation:** A(t+1) = A(t) + alpha * (P(t) - A(t))

**Target problem:** HPO when compute is limited and "good enough" beats
"optimal but slow".

### 6. Helmholtz Free Energy Training
**File:** `helmholtz_training.py`

Uses the thermodynamic free energy F = E - T*S as a unified training
objective. Temperature scheduling provides principled explore-to-exploit
annealing, unifying label smoothing, entropy bonuses, and simulated
annealing under one framework.

**Key equation:** F = E - T*S (energy = loss, entropy = parameter spread)

**Target problem:** Generalization via principled regularization scheduling.

### 7. Complementarity-Guided Feature Engineering
**File:** `complementarity_features.py`

Adapts Bohr's wave-particle complementarity principle as a zero-cost
diagnostic for whether frequency-domain (FFT) features will improve a
model. High complementarity means spatial and frequency views give
different information.

**Key equation:** Complementarity = 1 - |rho(wave_magnitude, particle_magnitude)|

**Target problem:** "Should I add FFT features?" -- answered before
engineering them.

### 8. Channel Capacity Limits of Knowledge Distillation
**File:** `channel_capacity_distillation.py`

Models a student network as a noisy communication channel with Shannon
capacity C = B*log2(1+S/N). The teacher's information content is an
upper bound on what the student can learn, regardless of distillation
method. Also maps ensembles to repetition codes.

**Key equation:** C = bandwidth * log2(1 + signal/noise)

**Target problem:** "Is my student model too small to learn from this
teacher?" -- diagnosed before training.

### 9. Bell Inequality Tests for Feature Dependencies
**File:** `bell_feature_entanglement.py`

Adapts the CHSH Bell inequality from quantum mechanics to detect
non-linear feature dependencies that Pearson correlation misses.
Feature pairs that violate the classical bound S <= 2 need interaction
terms or non-linear processing.

**Key equation:** S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|, classical limit S <= 2

**Target problem:** Detecting which feature pairs need interaction
terms before adding them.

### 10. Boltzmann Brain AI
**File:** `boltzmann_brain.py`

A neural network that doesn't exist until needed, then spontaneously
self-assembles from random thermal fluctuations into a minimal working
configuration, solves the task, and dissolves back to vacuum state.
Zero storage cost between invocations.

**Key equation:** P(brain) = exp(-E(brain) / kT), where E = task_loss + disorder

**Target problem:** Edge AI with zero model storage, ephemeral intelligence,
privacy (no persistent model to steal).

## Running

```bash
# Each file is self-contained with experiments
python research/heisenberg_regularizer.py
python research/dissipative_grokking.py
python research/gray_code_hpo.py
python research/pid_learning_rate.py
python research/satisficing_optimization.py
python research/helmholtz_training.py
python research/complementarity_features.py
python research/channel_capacity_distillation.py
python research/bell_feature_entanglement.py
python research/boltzmann_brain.py
```

## Status

These are proof-of-concept implementations. None have been peer-reviewed
or validated at scale. Treat as exploratory research.