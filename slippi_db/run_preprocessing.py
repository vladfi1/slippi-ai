from absl import app
from absl import flags

from slippi_db import preprocessing

flags.DEFINE_string('path', None, 'path')

def main(_):
  preprocessing.assert_same_parse(flags.FLAGS.path)

if __name__ == '__main__':
  app.run(main)
