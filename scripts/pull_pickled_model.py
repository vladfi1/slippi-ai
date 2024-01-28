import os

from absl import app, flags

from slippi_ai import s3_lib

FLAGS = flags.FLAGS
flags.DEFINE_string('tag', None, 'experiment tag', required=True)
flags.DEFINE_string('path', 'pickled_models/', 'path to put model')

def main(_):
  store = s3_lib.get_store()
  keys = s3_lib.get_keys(FLAGS.tag)
  combined_bytes = store.get(keys.combined)

  path = os.path.join(FLAGS.path, FLAGS.tag)

  with open(path, 'wb') as f:
    f.write(combined_bytes)

if __name__ == '__main__':
  app.run(main)
