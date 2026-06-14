if __name__ == '__main__':
    __spec__ = None  # https://github.com/python/cpython/issues/87115

    from absl import app
    import fancyflags as ff
    import wandb

    from slippi_ai import flag_utils
    from slippi_ai.jax.q import train_q_rl

    CONFIG = ff.DEFINE_dict(
        'config',
        **flag_utils.get_flags_from_default(train_q_rl.DEFAULT_CONFIG))

    WANDB = ff.DEFINE_dict(
        'wandb',
        project=ff.String('slippi-ai'),
        mode=ff.Enum('disabled', ['online', 'offline', 'disabled']),
        group=ff.String('jax-q-rl'),
        name=ff.String(None),
        notes=ff.String(None),
        dir=ff.String(None, 'directory to save logs'),
        tags=ff.StringList([]),
    )

    def main(_):
        config = flag_utils.dataclass_from_dict(
            train_q_rl.Config, CONFIG.value)

        wandb_kwargs = dict(WANDB.value)
        wandb.init(config=CONFIG.value, **wandb_kwargs)

        train_q_rl.run(config)

    app.run(main)
