import json

from absl import app, flags

from slippi_db import upgrade_slp

DOLPHIN = flags.DEFINE_string('dolphin', None, 'Path to Dolphin executable.', required=True)
SSBM_ISO = flags.DEFINE_string('iso', None, 'Path to SSBM ISO file.', required=True)
INPUT = flags.DEFINE_string('input', None, 'Input archive to convert.', required=True)
OUTPUT = flags.DEFINE_string('output', None, 'Output archive to write.', required=True)
NUM_THREADS = flags.DEFINE_integer('threads', 1, 'Number of threads to use for conversion.')
CHECK_SAME_PARSE = flags.DEFINE_bool('check_same_parse', True, 'Check if the replay has the same parse as the original.')
WORK_DIR = flags.DEFINE_string('work_dir', None, 'Optional working directory for temporary files.')
IN_MEMORY = flags.DEFINE_bool('in_memory', True, 'Use in-memory temporary files for conversion.')
LOG_INTERVAL = flags.DEFINE_integer('log_interval', 30, 'Interval in seconds to log progress during conversion.')
CHECK_IF_NEEDED = flags.DEFINE_bool('check_if_needed', False, 'Check if the file needs conversion before processing.')

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
      work_dir=WORK_DIR.value,
      log_interval=LOG_INTERVAL.value,
      check_if_needed=CHECK_IF_NEEDED.value,
  )

  json_results = [
      (result.local_file.name, result.error, result.skipped)
      for result in results
  ]

  with open('upgrade_results.json', 'w') as f:
    json.dump(json_results, f, indent=2)

if __name__ == '__main__':
  app.run(main)
