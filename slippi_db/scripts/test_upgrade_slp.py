from absl import app, flags

from slippi_db import upgrade_slp

DOLPHIN = flags.DEFINE_string('dolphin', None, 'Path to Dolphin executable.', required=True)
SSBM_ISO = flags.DEFINE_string('iso', None, 'Path to SSBM ISO file.', required=True)
INPUT = flags.DEFINE_string('input', None, 'Input file or directory to convert.', required=True)

def main(_):
  dolphin_config = upgrade_slp.DolphinConfig(
      dolphin_path=DOLPHIN.value,
      ssbm_iso_path=SSBM_ISO.value,
  )

  input_path = INPUT.value

  upgrade_slp.test_upgrade_slp(
      input_path=input_path,
      dolphin_config=dolphin_config,
      in_memory=True)

if __name__ == '__main__':
  app.run(main)
