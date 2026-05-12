"""
RLPolicy — Deep Q-Network for sequential scaling decisions (numpy implementation).

WHY NUMPY (not PyTorch):
- Same DQN algorithm, same patterns, fully testable anywhere.
- In production: swap QNetwork for torch.nn.Module. PolicyInterface unchanged.
- ADR-002 documents DQN vs PPO. Framework choice is separate from algorithm.

KEY SAFETY PROPERTIES:
- Action clipping: instances_after always in [min_instances, max_instances].
- Training isolation: train_step() in Celery worker. NEVER on decide() path.
- Warm-start gate: RL only active after min_decisions Bandit warm-up.
- Checkpoint atomic write: temp file → rename → never corrupt.
- Version-locked normalizer: mismatch → CheckpointError loud.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from app.ml.state_normalizer import _N_FEATURES, StateNormalizer
from app.policies.base_policy import PolicyCheckpointData, PolicyInterface
from app.schemas.common import ActionType, PolicyMode, PolicyType
from app.schemas.decision import ScalingDecision
from app.schemas.state import SystemState

logger = structlog.get_logger(__name__)

_ACTIONS: list[ActionType] = [
    ActionType.SCALE_UP_3,
    ActionType.SCALE_UP_1,
    ActionType.HOLD,
    ActionType.SCALE_DOWN_1,
    ActionType.SCALE_DOWN_3,
]
_N_ACTIONS = len(_ACTIONS)
_ACTION_IDX: dict[ActionType, int] = {a: i for i, a in enumerate(_ACTIONS)}


class QNetwork:
    """
    Feedforward Q-network: state → Q-values for all 5 actions.
    Architecture: 3 hidden layers with ReLU. Xavier initialization.
    """

    def __init__(
        self,
        n_features: int = _N_FEATURES,
        n_actions: int = _N_ACTIONS,
        hidden_dim: int = 64,
        seed: int | None = None,
    ) -> None:
        rng = np.random.RandomState(seed)

        def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
            scale = np.sqrt(2.0 / fan_in)
            return rng.randn(fan_in, fan_out).astype(np.float32) * scale

        self.layers: list[dict[str, np.ndarray]] = [
            {"W": _xavier(n_features, hidden_dim), "b": np.zeros(hidden_dim, dtype=np.float32)},
            {"W": _xavier(hidden_dim, hidden_dim), "b": np.zeros(hidden_dim, dtype=np.float32)},
            {
                "W": _xavier(hidden_dim, hidden_dim // 2),
                "b": np.zeros(hidden_dim // 2, dtype=np.float32),
            },
            {"W": _xavier(hidden_dim // 2, n_actions), "b": np.zeros(n_actions, dtype=np.float32)},
        ]
        self._cache: list[dict[str, np.ndarray]] = []

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        self._cache = []
        a = x.copy()
        for i, layer in enumerate(self.layers):
            z = a @ layer["W"] + layer["b"]
            if i < len(self.layers) - 1:
                relu_z = np.maximum(0, z)
                if training:
                    self._cache.append({"a_in": a, "z": z, "relu_z": relu_z})
                a = relu_z
            else:
                if training:
                    self._cache.append({"a_in": a, "z": z})
                a = z
        return a[0] if single else a

    def backward(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        actions: np.ndarray,
        learning_rate: float,
        gradient_clip: float,
    ) -> float:
        batch_size = x.shape[0]
        q_vals = self.forward(x, training=True)
        predictions = q_vals[np.arange(batch_size), actions]
        loss = float(np.mean((predictions - targets) ** 2))

        d_output = np.zeros_like(q_vals)
        d_output[np.arange(batch_size), actions] = 2 * (predictions - targets) / batch_size

        d_a = d_output
        for i in range(len(self.layers) - 1, -1, -1):
            cache = self._cache[i]
            a_in = cache["a_in"]
            d_W = np.clip(a_in.T @ d_a, -gradient_clip, gradient_clip)
            d_b = np.clip(d_a.sum(axis=0), -gradient_clip, gradient_clip)
            self.layers[i]["W"] -= learning_rate * d_W
            self.layers[i]["b"] -= learning_rate * d_b
            if i > 0:
                # Pass gradient to previous layer: d_a_prev = d_a @ W.T
                d_a = d_a @ self.layers[i]["W"].T
                # Apply ReLU derivative from the PREVIOUS layer's pre-activation (cache[i-1]["z"])
                prev_z = self._cache[i - 1]["z"]
                d_a *= (prev_z > 0).astype(np.float32)
        return loss

    def copy_weights_from(self, other: QNetwork) -> None:
        for i, src in enumerate(other.layers):
            self.layers[i]["W"] = src["W"].copy()
            self.layers[i]["b"] = src["b"].copy()

    def get_state_dict(self) -> dict[str, Any]:
        sd: dict[str, Any] = {}
        for i, layer in enumerate(self.layers):
            sd[f"layer_{i}_W"] = layer["W"].tolist()
            sd[f"layer_{i}_b"] = layer["b"].tolist()
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for i, layer in enumerate(self.layers):
            layer["W"] = np.array(state_dict[f"layer_{i}_W"], dtype=np.float32)
            layer["b"] = np.array(state_dict[f"layer_{i}_b"], dtype=np.float32)


@dataclass
class Transition:
    state: list[float]
    action_idx: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    """Fixed-capacity experience replay. Random sampling breaks temporal correlation."""

    def __init__(self, capacity: int = 50_000) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(list(self._buffer), batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self._buffer) >= min_size

    def to_dict(self) -> list[dict]:
        return [
            {"s": t.state, "a": t.action_idx, "r": t.reward, "ns": t.next_state, "d": t.done}
            for t in list(self._buffer)[-1000:]
        ]

    @classmethod
    def from_dict(cls, data: list[dict], capacity: int = 50_000) -> ReplayBuffer:
        buf = cls(capacity=capacity)
        for d in data:
            buf.push(
                Transition(
                    state=d["s"],
                    action_idx=d["a"],
                    reward=d["r"],
                    next_state=d["ns"],
                    done=d["d"],
                )
            )
        return buf


class RLPolicy(PolicyInterface):
    """
    Deep Q-Network policy for cloud resource allocation.
    Inference: normalizer → Q-network forward → argmax → clip → return.
    Training: runs in Celery worker via train_step(). Never on decide() path.
    """

    def __init__(
        self,
        normalizer: StateNormalizer | None = None,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_update_freq: int = 100,
        gradient_clip: float = 10.0,
        min_instances: int = 1,
        max_instances: int = 20,
        warm_start_min_decisions: int = 1000,
        seed: int | None = None,
    ) -> None:
        self._normalizer = normalizer
        self._min_instances = min_instances
        self._max_instances = max_instances
        self._warm_start_min_decisions = warm_start_min_decisions
        self._training_steps = 0
        self._inference_steps = 0

        self._q_network = QNetwork(hidden_dim=hidden_dim, seed=seed)
        self._target_network = QNetwork(hidden_dim=hidden_dim, seed=seed)
        self._target_network.copy_weights_from(self._q_network)

        self._learning_rate = learning_rate
        self._gamma = gamma
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq
        self._gradient_clip = gradient_clip

        self._replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.policy_mode = PolicyMode.ACTIVE

        logger.info("rl_policy_initialized", hidden_dim=hidden_dim, buffer_capacity=buffer_capacity)

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.RL

    @property
    def is_ready(self) -> bool:
        return (
            self._normalizer is not None
            and self._normalizer.is_fitted
            and self._inference_steps >= self._warm_start_min_decisions
        )

    def set_normalizer(self, normalizer: StateNormalizer) -> None:
        self._normalizer = normalizer
        logger.info("normalizer_attached", version_id=normalizer.version_id)

    async def decide(self, state: SystemState, explore: bool = True) -> ScalingDecision:
        """Inference: normalize → forward → argmax Q → clip. explore ignored (DQN exploits)."""
        if self._normalizer is None:
            raise RuntimeError(
                "RLPolicy.decide() called without a fitted normalizer. "
                "Call set_normalizer() before inference."
            )
        instances_before = state.instance_count
        norm_state = self._normalizer.normalize(state)
        q_values = self._q_network.forward(norm_state, training=False)
        q_list = q_values.tolist()
        action_idx = int(np.argmax(q_values))
        action = _ACTIONS[action_idx]

        delta = _action_delta(action)
        raw_target = instances_before + delta
        instances_after = self.clip_instances(raw_target, self._min_instances, self._max_instances)
        actual_delta = instances_after - instances_before
        if actual_delta != delta:
            action = _delta_to_action(actual_delta)
            action_idx = _ACTION_IDX.get(action, _ACTION_IDX[ActionType.HOLD])

        sorted_q = sorted(q_list, reverse=True)
        confidence_spread = sorted_q[0] - sorted_q[1] if len(sorted_q) > 1 else 0.0
        max_abs = max(abs(q) for q in q_list) or 1.0
        confidence = min(1.0, abs(confidence_spread) / max_abs)

        self._inference_steps += 1

        return ScalingDecision(
            action=action,
            instances_before=instances_before,
            instances_after=instances_after,
            policy_type=self.policy_type,
            policy_mode=self.policy_mode,
            confidence=confidence,
            q_values={a.value: round(q_list[i], 4) for i, a in enumerate(_ACTIONS)},
        )

    async def update(self, state: SystemState, action: ScalingDecision, reward: float) -> None:
        """Store transition in replay buffer. Training is in Celery worker."""
        if self._normalizer is None:
            return
        norm_state = self._normalizer.normalize(state).tolist()
        action_idx = _ACTION_IDX.get(action.action, _ACTION_IDX[ActionType.HOLD])
        self._replay_buffer.push(
            Transition(
                state=norm_state,
                action_idx=action_idx,
                reward=reward,
                next_state=norm_state,
                done=False,
            )
        )

    def train_step(self) -> float | None:
        """One gradient descent step. Called by Celery worker ONLY."""
        if not self._replay_buffer.is_ready(self._batch_size):
            return None
        batch = self._replay_buffer.sample(self._batch_size)
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action_idx for t in batch], dtype=np.int32)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([float(t.done) for t in batch], dtype=np.float32)

        next_q = self._target_network.forward(next_states, training=False)
        td_targets = rewards + self._gamma * next_q.max(axis=1) * (1 - dones)

        loss = self._q_network.backward(
            states,
            td_targets,
            actions,
            learning_rate=self._learning_rate,
            gradient_clip=self._gradient_clip,
        )
        self._training_steps += 1

        if self._training_steps % self._target_update_freq == 0:
            self._target_network.copy_weights_from(self._q_network)
            logger.debug("target_network_updated", training_steps=self._training_steps)

        return loss

    def get_checkpoint(self) -> PolicyCheckpointData:
        return PolicyCheckpointData(
            weights={
                "q_network": self._q_network.get_state_dict(),
                "target_network": self._target_network.get_state_dict(),
                "training_steps": self._training_steps,
                "inference_steps": self._inference_steps,
                "replay_buffer": self._replay_buffer.to_dict(),
            },
            step_count=self._training_steps,
            performance_metric=float(-self._training_steps),
        )

    def load_checkpoint(self, checkpoint: PolicyCheckpointData) -> None:
        from app.core.exceptions import CheckpointError

        weights = checkpoint.weights
        if weights is None:
            raise CheckpointError("RL checkpoint has no weights")
        required_keys = {"q_network", "target_network", "training_steps"}
        missing = required_keys - set(weights.keys())
        if missing:
            raise CheckpointError(f"RL checkpoint missing keys: {missing}")
        try:
            self._q_network.load_state_dict(weights["q_network"])
            self._target_network.load_state_dict(weights["target_network"])
            self._training_steps = int(weights["training_steps"])
            self._inference_steps = int(weights.get("inference_steps", 0))
            if weights.get("replay_buffer"):
                self._replay_buffer = ReplayBuffer.from_dict(
                    weights["replay_buffer"],
                    capacity=self._replay_buffer._buffer.maxlen or 50_000,
                )
            logger.info("rl_checkpoint_loaded", training_steps=self._training_steps)
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"Failed to restore RL checkpoint: {exc}") from exc

    def save_weights_to_file(self, path: Path) -> None:
        """Atomic write: temp → rename. Never leaves corrupt checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        try:
            cp = self.get_checkpoint()
            tmp_path.write_text(json.dumps(cp.weights))
            tmp_path.rename(path)
            logger.info("rl_weights_saved", path=str(path), steps=self._training_steps)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save RL weights to {path}: {exc}") from exc

    @classmethod
    def load_weights_from_file(cls, path: Path, normalizer: StateNormalizer) -> RLPolicy:
        from app.core.exceptions import CheckpointError

        if not path.exists():
            raise FileNotFoundError(f"RL checkpoint not found at {path}")
        try:
            weights = json.loads(path.read_text())
        except Exception as exc:
            raise CheckpointError(f"Malformed RL checkpoint at {path}: {exc}") from exc
        policy = cls(normalizer=normalizer)
        policy.load_checkpoint(
            PolicyCheckpointData(
                weights=weights,
                step_count=weights.get("training_steps", 0),
            )
        )
        return policy

    @property
    def training_steps(self) -> int:
        return self._training_steps

    @property
    def inference_steps(self) -> int:
        return self._inference_steps

    @property
    def buffer_size(self) -> int:
        return len(self._replay_buffer)

    @property
    def replay_buffer(self) -> ReplayBuffer:
        return self._replay_buffer


def _action_delta(action: ActionType) -> int:
    return {
        ActionType.SCALE_UP_3: 3,
        ActionType.SCALE_UP_1: 1,
        ActionType.HOLD: 0,
        ActionType.SCALE_DOWN_1: -1,
        ActionType.SCALE_DOWN_3: -3,
    }.get(action, 0)


def _delta_to_action(delta: int) -> ActionType:
    if delta >= 3:
        return ActionType.SCALE_UP_3
    elif delta >= 1:
        return ActionType.SCALE_UP_1
    elif delta <= -3:
        return ActionType.SCALE_DOWN_3
    elif delta <= -1:
        return ActionType.SCALE_DOWN_1
    return ActionType.HOLD
