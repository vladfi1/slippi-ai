"""Evaluates a policy."""

from concurrent import futures
import threading

import numpy as np
import ray
import tensorflow as tf
import tree
import typing as tp

from slippi_ai.policies import Policy
from slippi_ai import envs as env_lib
from slippi_ai import embed, saving, utils

Port = int
Trajectory = tp.Sequence[embed.StateActionReward]


class RolloutTiming(tp.NamedTuple):
  """Mean timings during a rollout."""
  inference: tp.Mapping[Port, float]
  env: float

class RolloutWorker:

  def __init__(
    self,
    policies: tp.Mapping[Port, Policy],
    env: env_lib.Environment,  # TODO: support multiple envs
    num_steps_per_rollout: int,
    compile: bool = True,
  ) -> None:
    assert set(policies) == set(env._opponents)
    self._policies = policies
    self._env = env
    self._num_steps_per_rollout = num_steps_per_rollout
    self._recurrent_states = {
        port: policy.initial_state(1)
        for port, policy in policies.items()
    }
    self._embed_action = {
        port: policy.controller_head.action_embedding()
        for port, policy in policies.items()
    }

    maybe_compile = tf.function if compile else lambda x: x
    self._sample_fns = {
        port: maybe_compile(policy.sample)
        for port, policy in policies.items()
    }

    self._last_state_action: tp.Dict[Port, embed.StateActionReward] = {}
    for port, game in self._env.current_state().items():
      action = self._embed_action[port].from_state(
          embed.ActionWithRepeat(game.p0.controller, 0))
      self._last_state_action[port] = embed.StateActionReward(
          state=game,
          action=action,
          reward=0.,
      )

  def _sample(
      self,
      port: Port,
      state_action: embed.StateActionReward,
  ) -> tp.Tuple[embed.ActionWithRepeat, embed.Controller]:
    policy = self._policies[port]
    sample_fn = self._sample_fns[port]

    # policies operate on batched inputs
    batched_state_action = tf.nest.map_structure(
        lambda x: np.expand_dims(x, 0),
        state_action)

    # CPU is faster for inference with small batch sizes
    with tf.device('/CPU:0'):
      # consider doing this in one TF call for all policies
      action_with_repeat, self._recurrent_states[port] = sample_fn(
          batched_state_action, self._recurrent_states[port])
    
    action_with_repeat: embed.ActionWithRepeat = tf.nest.map_structure(
        lambda x: np.squeeze(x.numpy(), 0),
        action_with_repeat)

    # we don't support action repeat and maybe should remove it
    assert action_with_repeat.repeat == 0

    # un-discretize the action
    embed_action = policy.controller_head.action_embedding()
    controller = embed_action.decode(action_with_repeat).action

    return action_with_repeat, controller

  def rollout(self) -> tp.Tuple[tp.Mapping[Port, Trajectory], RolloutTiming]:
    trajectories = {
        port: [state] for port, state in self._last_state_action.items()
    }

    env_profiler = utils.Profiler()
    policy_profilers = {port: utils.Profiler() for port in self._policies}

    for _ in range(self._num_steps_per_rollout):
      actions_with_repeat = {}
      controllers = {}

      for port, trajectory in trajectories.items():
        with policy_profilers[port]:
          actions_with_repeat[port], controllers[port] = self._sample(
              port, trajectory[-1])

      with env_profiler:
        observations = self._env.step(controllers)

      for port, (game, reward) in observations.items():
        state_action = embed.StateActionReward(
            state=game,
            action=actions_with_repeat[port],
            reward=reward,
        )
        trajectories[port].append(state_action)

    self._last_state_action = {p: t[-1] for p, t in trajectories.items()}

    inference_times = {
        port: profiler.mean_time()
        for port, profiler in policy_profilers.items()
    }

    timings = RolloutTiming(
        inference=inference_times,
        env=env_profiler.mean_time(),
    )

    return trajectories, timings

  def update_variables(
      self, updates: tp.Mapping[Port, tp.Sequence[np.ndarray]],
  ):
    for port, values in updates.items():
      for var, val in zip(self._policies[port].variables, values):
        var.assign(val)

  def stop(self):
    self._env.stop()

class RolloutMetrics(tp.NamedTuple):
  rewards: tp.Mapping[Port, float]
  timings: RolloutTiming


class SerializableRolloutWorker:
  """Takes in only serializable arguments."""

  def __init__(
      self,
      policy_configs: tp.Mapping[Port, dict],
      env_kwargs: dict,
      num_steps_per_rollout: int,
  ):
    policies = {
        port: saving.policy_from_config(config)
        for port, config in policy_configs.items()
    }
    for policy in policies.values():
      saving.init_variables(policy)

    env = env_lib.Environment(**env_kwargs)

    self._rollout_worker = RolloutWorker(
        policies, env, num_steps_per_rollout)
    
    self._lock = threading.Lock()

  def rollout(
      self,
      policy_vars: tp.Mapping[Port, tp.Sequence[np.ndarray]],
  ) -> RolloutMetrics:
    # We lock to protect against multiple ray remote calls.
    # I'm not sure if Ray Actors already do this.
    with self._lock:
      self._rollout_worker.update_variables(policy_vars)
      trajectories, timings = self._rollout_worker.rollout()

      rewards = {
          port: sum(x.reward for x in trajectory)
          for port, trajectory in trajectories.items()
      }

      return RolloutMetrics(rewards, timings)

  def stop(self):
    with self._lock:
      self._rollout_worker.stop()

RayRolloutWorker = ray.remote(SerializableRolloutWorker)


class Logger(tp.Protocol):
  def __call__(data: tree.Structure[tp.Any], step: tp.Optional[int]) -> None:
    """Log some data at a given step."""


class RemoteEvaluator:
  """Allows only one evaluation at a time."""

  def __init__(
      self,
      logger: Logger,
      **worker_kwargs,
  ):
    self._logger = logger
    self._worker = RayRolloutWorker.remote(**worker_kwargs)
    self._ready = threading.Event()
    self._ready.set()
    self._executor = futures.ThreadPoolExecutor(1)

  def _log_metrics(self, metrics_future, step: int):
    metrics: RolloutMetrics = ray.get(metrics_future)
    self._logger(metrics, step)
    self._ready.set()
    return metrics

  def rollout(
      self,
      step: int,
      policy_vars: tp.Mapping[Port, tp.Sequence[np.ndarray]],
  ) -> tp.Optional[futures.Future[RolloutMetrics]]:
    if not self._ready.is_set():
      return None
    self._ready.clear()
    metrics_future = self._worker.rollout.remote(policy_vars)
    # return self._log_metrics(metrics_future, step)
    return self._executor.submit(self._log_metrics, metrics_future, step)

  def stop(self):
    self._ready.wait()
    ray.wait([self._worker.stop.remote()])
    assert self._ready.is_set()
