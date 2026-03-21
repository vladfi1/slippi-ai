import dataclasses
import os
import sys
import typing as tp

from absl import flags
import sky

import fancyflags as ff
from slippi_ai import flag_utils

@dataclasses.dataclass
class Config:
  launch: bool = False
  gpu: str = 'RTX4090:1'
  cluster: tp.Optional[str] = None
  memory: int = 24

SKY = ff.DEFINE_dict('sky', **flag_utils.get_flags_from_dataclass(Config))

def serialize_non_default_flags() -> list[str]:
  """Serializes all non-default flags into a list of command line arguments."""
  args = []

  for module_name, flag_list in flags.FLAGS.flags_by_module_dict().items():
    for flag in flag_list:
      if flag.using_default_value:
        continue
      if flag.value is None:
        raise ValueError(f'Flag {module_name}:{flag.name} has a non-default value of None, which cannot be serialized.')

      if module_name == __name__:
        continue

      serialized = flag.serialize()
      print(f'Flag {module_name}.{flag.name} has non-default value {flag.value}, serialized as {serialized}')
      args.append(serialized)

  return args

ENV_VARS = [
    'WANDB_API_KEY',
    'FSSPEC_S3_KEY', 'FSSPEC_S3_SECRET', 'FSSPEC_S3_ENDPOINT_URL',
]

def launch(config: Config):
  vast_kwargs = dict(
      args='--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864',
      order='dlperf_usd-',
      query=['cuda_vers>=13.0'],
  )

  resources = sky.Resources(
      infra='vast',
      accelerators=config.gpu,
      memory=f'{config.memory}+',
      image_id='docker:vladfi/slippi-ai:jax',
      disk_size=16,
      _cluster_config_overrides=dict(
          vast=dict(create_instance_kwargs=vast_kwargs,)
      ),
  )

  run_commands = [
      'cd ~/sky_workdir/',
  ]

  main_file_path = sys.modules['__main__'].__file__
  assert main_file_path is not None
  main_file_relpath = os.path.relpath(main_file_path)

  run_command = [
      'python', main_file_relpath,
  ]

  non_default_args = serialize_non_default_flags()
  print('Extra args:', non_default_args)
  run_command.extend(non_default_args)
  print('Final run command:', '\n'.join(run_command))

  run_commands.append(' '.join(run_command))

  secrets = {}
  for key in ENV_VARS:
    if key not in os.environ:
      raise ValueError(f'Environment variable {key} is not set, but is required for launching on the cluster.')
    secrets[key] = os.environ[key]

  task = sky.Task(
      num_nodes=1,
      workdir='.',
      setup='''\
  cd ~/sky_workdir/
  pip install -r jax-requirements.txt
  pip install .[jax]
  ''',
      run='\n'.join(run_commands),
      resources=resources,
      secrets=secrets,
      file_mounts={'/root/.s3cfg': '~/.s3cfg'},
  )

  request_id = sky.launch(
      task,
      cluster_name=config.cluster,
      idle_minutes_to_autostop=10,
      down=True,
      fast=True,
  )
  print(f'Provisioning: sky api logs {request_id[:8]}')

  try:
      job_id, handle = sky.get(request_id)
      print(f'Launched job with ID: {job_id}')

      if handle is None:
          print('Failed to launch the job.')
          return

      cluster = handle.get_cluster_name()
      sky.tail_logs(cluster, job_id, follow=True)
  except KeyboardInterrupt:
      pass

def wrap(main: tp.Callable[[list[str]], None]):
  def wrapped_main(argv):
    config = flag_utils.dataclass_from_dict(Config, SKY.value)
    if config.launch:
      launch(config)
    else:
      main(argv)

  return wrapped_main
