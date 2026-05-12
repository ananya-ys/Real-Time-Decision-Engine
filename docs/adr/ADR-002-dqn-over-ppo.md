# ADR-002: DQN over PPO for RTDE RL Policy

**Status:** Accepted  
**Date:** 2025-01-01  
**Authors:** RTDE Engineering  

---

## Context

Phase 5 of the RTDE build requires selecting a Reinforcement Learning algorithm
for the third policy tier. The primary candidates are:

- **DQN** (Deep Q-Network) — value-based, off-policy, discrete actions
- **PPO** (Proximal Policy Optimization) — policy gradient, on-policy, continuous/discrete

---

## Decision

We selected **DQN** for the RTDE RL policy tier.

---

## Rationale

### 1. Action Space is Discrete and Small (5 actions)

RTDE has exactly 5 discrete scaling actions:  
`SCALE_UP_3 | SCALE_UP_1 | HOLD | SCALE_DOWN_1 | SCALE_DOWN_3`

DQN maps directly to this structure: the Q-network outputs one value per action,
and we select `argmax Q(s, a)`.

PPO maintains a policy distribution over the action space. For 5 discrete actions,
this is more complex than needed and provides no architectural advantage.

**Verdict: DQN simpler and more natural for this action space.**

### 2. Off-Policy Learning = Sample Efficiency

DQN stores transitions in an **experience replay buffer** and reuses them for multiple 
gradient updates. Each transition `(s, a, r, s')` contributes to many training steps.

PPO is **on-policy**: it requires fresh trajectories for each update and discards them
after. This means more environment interactions are needed for the same learning progress.

In a production system where environment interactions are expensive (each decision 
has real cloud cost), sample efficiency matters.

**Verdict: DQN learns more from fewer interactions.**

### 3. Training Isolation Architecture

DQN separates **inference** (online, fast) from **training** (offline, Celery worker).

The `decide()` method only runs a forward pass — no gradient computation.  
The `train_step()` method runs batch gradient descent — called only by the Celery worker.

PPO requires computing policy gradients on the same trajectory used for decisions,
creating tighter coupling between inference and training. This would require more
complex worker coordination to maintain the P99 < 300ms SLO.

**Verdict: DQN training isolation is architecturally cleaner for this system.**

### 4. Target Network Prevents Oscillation

DQN's **target network** (a frozen copy updated every N steps) prevents the
"chasing your own tail" problem where Q-values oscillate and fail to converge.

Without a target network, the TD target changes every step, causing instability.
PPO avoids this differently (clipped surrogate objective), but for our discrete
action space, target network is a simpler, well-understood solution.

**Verdict: DQN's stability properties are better understood for this domain.**

---

## Consequences

### Positive
- Simpler implementation (no policy distribution, no entropy regularization)
- Natural fit for 5-action discrete space
- Strong sample efficiency via replay buffer
- Clean training isolation from inference path

### Negative
- Off-policy: Q-values may overestimate (double DQN mitigates, not implemented)
- No natural uncertainty quantification (vs. distributional RL)
- Exploration is handled externally (ExplorationGuard + Bandit warm-start) rather than natively

### Mitigations
- DQN overestimation: training on the same environment consistently mitigates in practice
- Exploration: three-tier architecture (Baseline → Bandit → RL) ensures warm-start experience

---

## Alternatives Considered

| Algorithm | Reason Rejected |
|-----------|-----------------|
| PPO       | On-policy (sample inefficient), more complex for discrete 5-action space |
| A3C/A2C   | Requires parallel environments; inference isolation harder to maintain |
| SAC       | Designed for continuous action spaces; poor fit for discrete scaling |
| Rainbow DQN | Overcomplicated for initial deployment; can upgrade incrementally |

---

## Implementation Reference

- `app/policies/rl_policy.py` — `QNetwork`, `ReplayBuffer`, `RLPolicy`
- `app/ml/state_normalizer.py` — normalized input to Q-network
- `app/worker/tasks.py` — `train_rl_policy` Celery task (training isolation)
- `tests/unit/test_rl_policy.py` — test coverage
