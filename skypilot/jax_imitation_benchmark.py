"""Python equivalent of jax_imitation_benchmark.yaml.
"""

import subprocess

from absl import app, flags

import sky

MAX_STEP = flags.DEFINE_integer('max_step', 200, 'Maximum number of steps to train for.')
GPU = flags.DEFINE_string('gpu', 'RTX4090:1', 'GPU to use for training.')
MODEL_SIZE = flags.DEFINE_integer('model_size', 512, 'Hidden size of the model.')
CLUSTER = flags.DEFINE_string('cluster', 'jax-imitation-benchmark', 'Cluster to run on.')
PACK = flags.DEFINE_bool('pack', False, 'Whether to pack data for training.')
PREFETCH = flags.DEFINE_integer('prefetch', 1, 'Number of batches to prefetch.')
PROFILE = flags.DEFINE_bool('profile', False, 'Whether to profile the training run.')

def main(_):
    vast_kwargs = dict(
        args='--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864',
        order='dlperf_usd-',
        query=['cuda_vers>=13.0'],
    )

    resources = sky.Resources(
        infra='vast',
        accelerators=GPU.value,
        memory='16+',
        image_id='docker:vladfi/slippi-ai:jax',
        disk_size=16,
        _cluster_config_overrides=dict(
            vast=dict(create_instance_kwargs=vast_kwargs,)
        ),
    )

    run_commands = [
        'cd ~/sky_workdir/',
    ]

    run_command = [
        'python', 'slippi_ai/jax/launch_scripts/imitation/tx_like.py',
        '--toy_data',
        f'--config.runtime.max_step={MAX_STEP.value}',
        f'--net.hidden_size={MODEL_SIZE.value}',
        f'--config.learner.pack_data={PACK.value}',
        f'--config.runtime.prefetch={PREFETCH.value}',
    ]

    profile_trace_dir = '/tmp/profile_trace'

    if PROFILE.value:
        run_command.append(f'--config.runtime.profile_trace_dir={profile_trace_dir}')

    run_commands.append(' '.join(run_command))

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
    )

    request_id = sky.launch(
        task,
        cluster_name=CLUSTER.value,
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

        if PROFILE.value:
            subprocess.check_call(['rsync', '-Pavz', f'{cluster}:{profile_trace_dir}/', f'./untracked/profile_traces/'])
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
