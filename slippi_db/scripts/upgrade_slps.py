import json

from absl import app, flags

from slippi_db import upgrade_slp

DOLPHIN = flags.DEFINE_string('dolphin', None, 'Path to Dolphin executable.', required=True)
SSBM_ISO = flags.DEFINE_string('iso', None, 'Path to SSBM ISO file.', required=True)
INPUT = flags.DEFINE_string('input', None, 'Input archive to convert.', required=True)
OUTPUT = flags.DEFINE_string('output', None, 'Output archive to write.', required=True)
NUM_THREADS = flags.DEFINE_integer('threads', 1, 'Number of threads to use for conversion.')
CHECK_SAME_PARSE = flags.DEFINE_bool('check_same_parse', True, 'Check if the replay has the same parse as the original.')
GZIP_OUTPUT = flags.DEFINE_bool('gzip_output', True, 'Compress the output archive with gzip.')
WORK_DIR = flags.DEFINE_string('work_dir', None, 'Optional working directory for temporary files.')
IN_MEMORY = flags.DEFINE_bool('in_memory', True, 'Use in-memory temporary files for conversion.')

def main(_):
  dolphin_config = upgrade_slp.DolphinConfig(
      dolphin_path=DOLPHIN.value,
      ssbm_iso_path=SSBM_ISO.value,
  )

  results = upgrade_slp.upgrade_archive(
      input_path=INPUT.value,
      output_path=OUTPUT.value,
      dolphin_config=dolphin_config,
      in_memory=IN_MEMORY.value,
      num_threads=NUM_THREADS.value,
      check_same_parse=CHECK_SAME_PARSE.value,
      gzip_output=GZIP_OUTPUT.value,
      work_dir=WORK_DIR.value,
      log_interval=30,
  )

  with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

if __name__ == '__main__':
  app.run(main)
