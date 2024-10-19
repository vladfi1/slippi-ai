
"""Train (and test) a network via imitation learning."""

import dataclasses
import json
import os
import pickle
import time
import typing as tp

from absl import logging

import numpy as np
import tensorflow as tf

import wandb

from slippi_ai import (
    controller_heads,
    evaluators,
    flag_utils,
    nametags,
    networks,
    policies,
    saving,
    s3_lib,
    tf_utils,
    train_lib,
    utils,
)
from slippi_ai import q_learner as learner_lib
from slippi_ai import data as data_lib
from slippi_ai import q_function as q_lib
from slippi_ai import embed as embed_lib
from slippi_ai import dolphin as dolphin_lib

_field = utils.field

@dataclasses.dataclass
class RuntimeConfig:
  max_runtime: int = 1 * 60 * 60  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

  eval_every_n: int = 100  # number of training steps between evaluations
  num_eval_steps: int = 10  # number of batches per evaluation

@dataclasses.dataclass
class AgentConfig:
  batch_steps: int = 0
  compile: bool = True
  name: str = nametags.DEFAULT_NAME
  async_inference: bool = False

@dataclasses.dataclass
class RLEvaluatorConfig:
  use: bool = False
  # Seconds between evaluations. Note that the evaluator runs at around
  # half real-time, so this should be ~20x the rollout length if you want
  # to spend 10% of the time evaluating.
  # TODO: try running in parallel with training (so evaluator must be on CPU)
  interval_seconds: float = 15 * 60
  runtime_seconds: float = 60

  dolphin: dolphin_lib.DolphinConfig = _field(dolphin_lib.DolphinConfig)

  # env
  rollout_length: int = 600  # rollout chunk size
  num_envs: int = 1
  async_envs: bool = True
  num_env_steps: int = 0
  inner_batch_size: int = 1
  use_fake_envs: bool = False

  agent: AgentConfig = _field(AgentConfig)
  opponent: tp.Optional[str] = None
  opponent_name: str = nametags.DEFAULT_NAME
  gpu_inference: bool = True

@dataclasses.dataclass
class QFunctionConfig:
  network: dict = _field(lambda: networks.DEFAULT_CONFIG)

@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = _field(RuntimeConfig)
  rl_evaluator: RLEvaluatorConfig = _field(RLEvaluatorConfig)

  dataset: data_lib.DatasetConfig = _field(data_lib.DatasetConfig)
  data: data_lib.DataConfig = _field(data_lib.DataConfig)
  max_names: int = 16

  learner: learner_lib.LearnerConfig = _field(learner_lib.LearnerConfig)

  # These apply to both sample and q policies.
  # TODO: can we support distinct configurations here?
  policy: policies.PolicyConfig = _field(
      lambda: policies.PolicyConfig(train_value_head=False))
  network: dict = _field(lambda: networks.DEFAULT_CONFIG)
  controller_head: dict = _field(lambda: controller_heads.DEFAULT_CONFIG)

  embed: embed_lib.EmbedConfig = _field(embed_lib.EmbedConfig)

  q_function: QFunctionConfig = _field(QFunctionConfig)

  expt_root: str = 'experiments/q_learning'
  expt_dir: tp.Optional[str] = None
  tag: tp.Optional[str] = None

  # TODO: group these into their own subconfig
  save_to_s3: bool = False
  restore_tag: tp.Optional[str] = None
  restore_pickle: tp.Optional[str] = None
  initialize_policies_from: tp.Optional[str] = None

  is_test: bool = False  # for db management
  version: int = saving.VERSION

def _get_loss(stats: dict):
  return stats['total_loss'].numpy().mean()

def train(config: Config):
  tag = config.tag or train_lib.get_experiment_tag()
  # Might want to use wandb.run.dir instead, but it doesn't seem
  # to be set properly even when we try to override it.
  expt_dir = config.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.expt_root, tag)
    os.makedirs(expt_dir, exist_ok=True)
  config.expt_dir = expt_dir  # for wandb logging
  logging.info('experiment directory: %s', expt_dir)

  runtime = config.runtime

  if config.initialize_policies_from:
    logging.info(f'Initializing policies from {config.initialize_policies_from}')
    state = saving.load_state_from_disk(config.initialize_policies_from)
    sample_policy = saving.load_policy_from_state(state)
    q_policy = saving.load_policy_from_state(state)

    # Overwrite policy config
    imitation_config = state['config']
    config.policy = flag_utils.dataclass_from_dict(
        policies.PolicyConfig, imitation_config['policy'])
    config.network = imitation_config['network']
    config.controller_head = imitation_config['controller_head']
  else:
    config_dict = dataclasses.asdict(config)
    sample_policy = saving.policy_from_config(config_dict)
    q_policy = saving.policy_from_config(config_dict)

  q_function = q_lib.QFunction(
      network_config=config.q_function.network,
      embed_state_action=q_policy.embed_state_action,
      embed_action=q_policy.controller_embedding,
  )

  learner_kwargs = dataclasses.asdict(config.learner)
  learning_rate = tf.Variable(
      learner_kwargs['learning_rate'], name='learning_rate', trainable=False)
  learner_kwargs.update(learning_rate=learning_rate)
  learner = learner_lib.Learner(
      q_function=q_function,
      sample_policy=sample_policy,
      q_policy=q_policy,
      **learner_kwargs,
  )
  # Initialize variables without calling tf.gradient to avoid strange tf bug.
  # See https://github.com/vladfi1/slippi-ai/blob/q-learning-dbg/tests/bug.py
  learner.initialize_variables()

  logging.info("Network configuration")
  for comp in ['network', 'controller_head']:
    logging.info(f'Using {comp}: {getattr(config, comp)["name"]}')

  ### Dataset Creation ###
  dataset_config = config.dataset

  # Parse csv chars into list of enum values.
  char_filters = {}
  for key in ['allowed_characters', 'allowed_opponents']:
    chars_string = getattr(dataset_config, key)
    char_filters[key] = data_lib.chars_from_string(chars_string)

  train_replays, test_replays = data_lib.train_test_split(dataset_config)
  logging.info(f'Training on {len(train_replays)} replays, testing on {len(test_replays)}')

  pickle_path = os.path.join(expt_dir, 'latest.pkl')

  save_to_s3 = config.save_to_s3
  if save_to_s3 or config.restore_tag:
    if 'S3_CREDS' not in os.environ:
      raise ValueError('must set the S3_CREDS environment variable')

    s3_store = s3_lib.get_store()
    s3_keys = s3_lib.get_keys(tag)

  if config.restore_tag:
    restore_s3_keys = s3_lib.get_keys(config.restore_tag)
  elif save_to_s3:
    restore_s3_keys = s3_keys
  else:
    restore_s3_keys = None

  # attempt to restore parameters
  restored = False
  if restore_s3_keys is not None:
    try:
      restore_key = restore_s3_keys.combined
      obj = s3_store.get(restore_key)
      logging.info('restoring from %s', restore_key)
      combined_state = pickle.loads(obj)
      restored = True
      # TODO: do some config compatibility validation
    except KeyError:
      # TODO: re-raise if user specified restore_tag
      logging.info('no params found at %s', restore_key)
  elif config.restore_pickle:
    logging.info('restoring from %s', config.restore_pickle)
    with open(config.restore_pickle, 'rb') as f:
      combined_state = pickle.load(f)
    restored = True
  elif os.path.exists(pickle_path):
    logging.info('restoring from %s', pickle_path)
    with open(pickle_path, 'rb') as f:
      combined_state = pickle.load(f)
    restored = True
  else:
    logging.info('not restoring any params')

  if restored:
    name_map: dict[str, int] = combined_state['name_map']
  else:
    name_map = train_lib.create_name_map(train_replays, config.max_names)

  name_map_path = os.path.join(expt_dir, 'name_map.json')
  # Record name map
  print(name_map)
  with open(name_map_path, 'w') as f:
    json.dump(name_map, f)
  wandb.save(name_map_path, policy='now')

  num_codes = nametags.max_name_code(name_map) + 1
  encode_name = nametags.name_encoder(name_map)
  encode_name_uint8 = lambda name: np.uint8(encode_name(name))
  batch_encode_name = np.vectorize(encode_name_uint8)

  # Create data sources for train and test.
  data_config = dict(
      dataclasses.asdict(config.data),
      # TODO: call from_state in the learner instead
      embed_game=q_policy.embed_game,
      embed_controller=q_policy.controller_embedding,
      extra_frames=1 + q_policy.delay,
      name_map=name_map,
      **char_filters,
  )
  train_data = data_lib.make_source(replays=train_replays, **data_config)
  test_data = data_lib.make_source(replays=test_replays, **data_config)

  train_manager = train_lib.TrainManager(learner, train_data, dict(train=True))
  test_manager = train_lib.TrainManager(learner, test_data, dict(train=False))

  stats, _ = train_manager.step()
  logging.info('loss initial: %f', _get_loss(stats))
  del stats

  step = tf.Variable(0, trainable=False, name="step")

  # saving and restoring
  # TODO: move this into the Learner?
  tf_state = dict(
      step=step,
      sample_policy=sample_policy.variables,
      policy=q_policy.variables,  # other code expects a "policy" key
      q_function=q_function.variables,
      optimizers=dict(
          sample_policy=learner.sample_policy_optimizer.variables,
          q_policy=learner.q_policy_optimizer.variables,
          q_function=learner.q_function_optimizer.variables,
      ),
  )

  def get_tf_state():
    return tf.nest.map_structure(lambda v: v.numpy(), tf_state)

  def set_tf_state(state):
    tf.nest.map_structure(
      lambda var, val: var.assign(val),
      tf_state, state)

  def save():
    # Local Save
    tf_state = get_tf_state()

    # easier to always bundle the config with the state
    combined_state = dict(
        state=tf_state,
        config=dataclasses.asdict(config),
        name_map=name_map,
    )
    pickled_state = pickle.dumps(combined_state)

    logging.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)

    if save_to_s3:
      logging.info('saving state to S3: %s', s3_keys.combined)
      s3_store.put(s3_keys.combined, pickled_state)

  maybe_save = utils.Periodically(save, runtime.save_interval)

  if restored:
    set_tf_state(combined_state['state'])
    logging.info('loss post-restore: %f', _get_loss(test_manager.step()[0]))

  FRAMES_PER_MINUTE = 60 * 60

  step_tracker = utils.Tracker(step.numpy())
  epoch_tracker = utils.Tracker(0)
  log_tracker = utils.Tracker(time.time())

  @utils.periodically(runtime.log_interval)
  def maybe_log(train_stats: dict):
    """Do a test step, then log both train and test stats."""
    test_stats, _ = test_manager.step()

    elapsed_time = log_tracker.update(time.time())
    total_steps = step.numpy()
    steps = step_tracker.update(total_steps)
    # assume num_frames is constant per step
    num_frames = steps * train_stats['num_frames']

    epoch = train_stats['epoch']
    delta_epoch = epoch_tracker.update(epoch)

    sps = steps / elapsed_time
    mps = num_frames / FRAMES_PER_MINUTE / elapsed_time
    eph = delta_epoch / elapsed_time * 60 * 60
    data_time = train_manager.data_profiler.mean_time()
    step_time = train_manager.step_profiler.mean_time()

    timings = dict(
        sps=sps,
        mps=mps,
        eph=eph,
        data=data_time,
        step=step_time,
    )

    all_stats = dict(
        train=train_stats,
        test=test_stats,
        timings=timings,
    )
    train_lib.log_stats(all_stats, total_steps)

    train_loss = _get_loss(train_stats)
    test_loss = _get_loss(test_stats)

    print(f'step={total_steps} epoch={epoch:.3f}')
    print(f'sps={sps:.2f} mps={mps:.2f} eph={eph:.2e}')
    print(f'losses: train={train_loss:.4f} test={test_loss:.4f}')
    print(f'timing:'
          f' data={data_time:.3f}'
          f' step={step_time:.3f}')
    print()

  def maybe_eval(force: bool = False):
    total_steps = step.numpy()
    if (not force) and total_steps % runtime.eval_every_n != 0:
      return

    eval_results = [test_manager.step() for _ in range(runtime.num_eval_steps)]

    eval_stats, batches = zip(*eval_results)
    eval_stats = tf.nest.map_structure(tf_utils.to_numpy, eval_stats)
    eval_stats = tf.nest.map_structure(utils.stack, *eval_stats)

    to_log = dict(eval=eval_stats)
    train_lib.log_stats(to_log, total_steps)

    # Log losses aggregated by name.

    # Stats have shape [num_eval_steps, unroll_length, batch_size]
    time_mean = lambda x: np.mean(x, axis=1)
    loss = time_mean(eval_stats['q_function']['q']['loss'])
    assert loss.shape == (runtime.num_eval_steps, config.data.batch_size)

    # Metadata only has a batch dimension.
    metas: list[data_lib.ChunkMeta] = [batch.meta for batch in batches]
    meta: data_lib.ChunkMeta = tf.nest.map_structure(utils.stack, *metas)

    # Name of the player we're imitating.
    name = np.where(
        meta.info.swap, meta.info.meta.p1.name, meta.info.meta.p0.name)
    encoded_name = batch_encode_name(name)
    assert encoded_name.dtype == np.uint8
    assert encoded_name.shape == loss.shape

    loss_sums_and_counts = []
    for i in range(num_codes):
      mask = encoded_name == i
      loss_sums_and_counts.append((np.sum(loss * mask), np.sum(mask)))

    losses, counts = zip(*loss_sums_and_counts)
    to_log = dict(
        losses=np.array(losses, dtype=np.float32),
        counts=np.array(counts, dtype=np.uint32),
    )

    to_log = dict(eval_names=to_log)
    train_lib.log_stats(to_log, total_steps, take_mean=False)

  if config.rl_evaluator.use:
    if config.rl_evaluator.opponent:
      opponent_state = saving.load_state_from_disk(
          path=config.rl_evaluator.opponent)
    else:
      # TODO: default to sample policy?
      raise NotImplementedError('opponent not specified')

    AGENT_PORT = 1
    OPPONENT_PORT = 2

    dolphin_kwargs = config.rl_evaluator.dolphin.to_kwargs()
    # Characters are set in the evaluator from the data config.
    players = {
        AGENT_PORT: dolphin_lib.AI(),
        # TODO: allow playing against the in-game CPU like run_evaluator.py
        OPPONENT_PORT: dolphin_lib.AI(),
    }
    dolphin_kwargs.update(players=players)

    common_agent_kwargs = dataclasses.asdict(config.rl_evaluator.agent)

    agent_kwargs: dict[int, dict] = {}
    agent_state = dict(
        state=get_tf_state(),  # could drop everything but 'policy' key
        config=dataclasses.asdict(config),
        name_map=name_map,
    )
    agent_kwargs[AGENT_PORT] = dict(
        state=agent_state,
        **common_agent_kwargs,
    )
    agent_kwargs[OPPONENT_PORT] = dict(
        state=opponent_state,
        **common_agent_kwargs,
    )
    agent_kwargs[OPPONENT_PORT].update(
        name=config.rl_evaluator.opponent_name,
    )

    env_kwargs = {}
    if config.rl_evaluator.async_envs:
      env_kwargs.update(
          num_steps=config.rl_evaluator.num_env_steps,
          inner_batch_size=config.rl_evaluator.inner_batch_size,
      )

    rl_evaluator = evaluators.Evaluator(
        agent_kwargs=agent_kwargs,
        dolphin_kwargs=dolphin_kwargs,
        num_envs=config.rl_evaluator.num_envs,
        async_envs=config.rl_evaluator.async_envs,
        env_kwargs=dict(
            num_steps=config.rl_evaluator.num_env_steps,
            inner_batch_size=config.rl_evaluator.inner_batch_size,
        ),
        use_gpu=config.rl_evaluator.gpu_inference,
        use_fake_envs=config.rl_evaluator.use_fake_envs,
    )

    rl_evaluator.start()

    def rl_evaluate():
      total_steps = step.numpy()

      # TODO: might want to reset the environment here?
      rl_evaluator.update_variables({AGENT_PORT: q_policy.variables})
      start_time = time.perf_counter()
      run_time = 0
      rewards = []
      while run_time < config.rl_evaluator.runtime_seconds:
        metrics, timings = rl_evaluator.rollout(
            num_steps=config.rl_evaluator.rollout_length)
        rewards.append(metrics[AGENT_PORT].reward)

        run_time = time.perf_counter() - start_time

      total_reward = sum(rewards)
      num_frames = (
          config.rl_evaluator.num_envs
          * config.rl_evaluator.rollout_length
          * len(rewards))
      num_minutes = num_frames / (60 * 60)
      kdpm = total_reward / num_minutes
      fps = num_frames / run_time

      to_log = dict(
          ko_diff=kdpm,
          num_rollouts=len(rewards),
          num_minutes=num_minutes,
          fps=fps,
          timings=timings,
      )

      print(f'EVAL: kdpm={kdpm}, minutes={num_minutes:.2f}, fps={fps:.1f}\n')
      to_log['time'] = run_time

      train_lib.log_stats(to_log, total_steps)

    stop_rl_evaluator = rl_evaluator.stop
  else:
    rl_evaluate = lambda: None
    stop_rl_evaluator = lambda: None

  maybe_rl_eval = utils.Periodically(rl_evaluate, config.rl_evaluator.interval_seconds)

  try:

    # For burnin.
    maybe_eval(force=True)
    maybe_rl_eval()  # guaranteed to run the first time
    train_manager.step()
    step.assign_add(1)

    start_time = time.time()

    while time.time() - start_time < runtime.max_runtime:
      train_stats, _ = train_manager.step()
      step.assign_add(1)
      maybe_log(train_stats)
      maybe_eval()
      maybe_rl_eval()

      maybe_save()

    maybe_eval(force=True)
    rl_evaluate()
    save()

  finally:
    stop_rl_evaluator()
