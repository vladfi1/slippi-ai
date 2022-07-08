import time

from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import (
  embed,
  policies,
  saving,
  train_lib,
  controller_heads,
  networks,
  utils,
)

flags.DEFINE_string('tag', None, 'Experiment tag to pull from S3.')
flags.DEFINE_string('saved_model', None, 'Path to local saved model.')
flags.DEFINE_multi_integer('bs', None, 'batch size', required=True)
flags.DEFINE_integer('runtime', 5, 'How long to run, in seconds.')

_CONTROLLER_HEAD = ff.DEFINE_dict(
    'controller_head',
    **utils.get_flags_from_default(controller_heads.DEFAULT_CONFIG),
)

_NETWORK = ff.DEFINE_dict(
    'network',
    **utils.get_flags_from_default(networks.DEFAULT_CONFIG),
)

FLAGS = flags.FLAGS

def build_policy() -> policies.Policy:
  if FLAGS.saved_model:
    return saving.load_policy_from_disk(FLAGS.saved_model)
  elif FLAGS.tag:
    return saving.load_policy_from_s3(FLAGS.tag)
  else:
    embed_controller = embed.embed_controller_discrete  # TODO: configure

    return train_lib.build_policy(
        controller_head_config=_CONTROLLER_HEAD.value,
        max_action_repeat=0,
        network_config=_NETWORK.value,
        embed_controller=embed_controller,
    )

def run(policy: policies.Policy, batch_size: int):
  state_action = policy.embed_state_action.dummy([batch_size])
  hidden_state = policy.initial_state(batch_size)

  # warmup
  policy.sample(state_action, hidden_state)

  start_time = time.perf_counter()
  runtime = 0
  num_iters = 0

  while runtime < FLAGS.runtime:
    _, hidden_state = policy.sample(state_action, hidden_state)
    num_iters += 1
    runtime = time.perf_counter() - start_time

  sps = num_iters / runtime
  total_sps = sps * batch_size

  print(f'sps: {sps:.0f}, total_sps: {total_sps:.0f}')

  return sps, total_sps


def main(_):
  policy = build_policy()
  bss = FLAGS.bs
  stats = [run(policy, bs) for bs in bss]

  for n, (fps, sps) in zip(bss, stats):
    print(f'{n:03d}: fps: {fps:.1f}, sps: {sps:.1f}')

  for n, (fps, sps) in zip(bss, stats):
    print(f'{n} {fps:.1f} {sps:.1f}')


if __name__ == '__main__':
  app.run(main)
