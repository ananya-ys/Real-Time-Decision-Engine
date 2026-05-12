# ADR-001: PolicyInterface Abstract Base Class

**Status:** Accepted  
**Date:** 2025-01-01  
**Authors:** RTDE Engineering  

---

## Context

RTDE requires three policy tiers: Baseline, Bandit, and RL. The system must be able to:
1. Swap the active policy at runtime without restarting.
2. Use any policy wherever another is used (Liskov Substitution).
3. Write `DecisionService` once, with zero changes needed when adding a new policy.
4. Enforce at typecheck time that all policies implement the required interface.

---

## Decision

We use an **Abstract Base Class (`PolicyInterface`)** defined in `app/policies/base_policy.py`.

Every policy (Baseline, Bandit, RL) inherits from `PolicyInterface` and implements:
- `decide(state, explore)` — required, returns `ScalingDecision`
- `update(state, action, reward)` — required, online learning update
- `get_checkpoint()` — required, serializes policy state
- `load_checkpoint(checkpoint)` — required, restores policy state
- `clip_instances(target, min, max)` — provided, not overridable

---

## Rationale

### Dependency Inversion

`DecisionService` depends on `PolicyInterface`, NOT on `BaselinePolicy` or `RLPolicy`:

```python
# WRONG — couples service to implementation
from app.policies.rl_policy import RLPolicy

class DecisionService:
    def __init__(self):
        self._policy = RLPolicy()   # can't swap without changing service

# CORRECT — depends on abstraction
from app.policies.base_policy import PolicyInterface

class DecisionService:
    def __init__(self):
        self._policy: PolicyInterface = BaselinePolicy()  # swap at runtime

    def set_active_policy(self, policy: PolicyInterface) -> None:
        self._policy = policy  # swap without any service change
```

### Typecheck Enforcement

Python ABCs enforce interface compliance at class definition time:

```python
class BrokenPolicy(PolicyInterface):
    # Missing decide() implementation
    pass

# Raises TypeError at import time — not at runtime in production
```

Without ABC: missing method discovered when called in production → 500 error.  
With ABC: discovered at `import` time during development.

### `clip_instances` as Non-Override Method

`clip_instances` is defined on `PolicyInterface` (not abstract) because:
- Every policy MUST apply safety clipping.
- The clipping logic must be identical across all policies.
- Making it overridable would allow a subclass to remove safety bounds.

---

## Consequences

### Positive
- `DecisionService` tests don't need to import any concrete policy.
- Adding a 4th policy tier (e.g., PPO) requires zero changes in `DecisionService`.
- `mypy --strict` enforces interface at lint time in CI.

### Negative
- Python ABCs are less strict than, e.g., Rust traits — `isinstance()` checks still needed at runtime.
- Async ABC methods require `@abstractmethod` + `async def` — not checked by standard ABC machinery.

### Mitigations
- Type annotations + mypy strict mode cover the async case.
- Integration tests verify concrete policies satisfy the contract end-to-end.

---

## Implementation Reference

- `app/policies/base_policy.py` — PolicyInterface ABC
- `app/policies/baseline_policy.py` — Baseline implementation
- `app/policies/bandit_policy.py` — Bandit implementation
- `app/policies/rl_policy.py` — RL implementation
- `app/services/decision_service.py` — Consumer of PolicyInterface
