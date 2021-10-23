import os
import io, tarfile

from absl import app, flags

from slippi_ai import s3_lib

FLAGS = flags.FLAGS
flags.DEFINE_string('tag', None, 'experiment tag', required=True)
flags.DEFINE_string('s3_creds', None, 's3 credentials')
flags.DEFINE_string('path', 'saved_models/', 'path to put model')

def main(_):
  store = s3_lib.get_store()
  keys = s3_lib.get_keys(FLAGS.tag)
  saved_model_bytes = store.get(keys.saved_model)
  saved_model_file = io.BytesIO(saved_model_bytes)
  
  path = os.path.join(FLAGS.path, FLAGS.tag)

  with tarfile.open(mode='r', fileobj=saved_model_file) as f:
    f.extractall(path)

if __name__ == '__main__':
  app.run(main)
