"""Convert parsed.sqlite back to parsed.pkl.

Reverses the flattening done by convert_parsed_to_sqlite.py.

Usage: python slippi_db/scripts/convert_sqlite_to_parsed.py --input=parsed.sqlite --output=parsed.pkl
"""

import pickle
import sqlite3

from absl import app, flags

from slippi_db.parse_local import convert_sqlite_to_pickle


ROOT = flags.DEFINE_string('root', None, 'Root directory containing parsed.sqlite')
INPUT = flags.DEFINE_string('input', None, 'Input SQLite database path')
OUTPUT = flags.DEFINE_string('output', None, 'Output parsed.pkl file')


def main(_):
  if ROOT.value is not None:
    input_path = f"{ROOT.value}/parsed.sqlite"
    output_path = f"{ROOT.value}/parsed.pkl"
  else:
    input_path = INPUT.value
    if input_path is None:
      raise ValueError("Either --root or --input must be specified")
    output_path = OUTPUT.value
    if output_path is None:
      raise ValueError("Either --root or --output must be specified")

  print(f"Converting {input_path} to {output_path}...")
  convert_sqlite_to_pickle(input_path, output_path)

if __name__ == '__main__':
  app.run(main)
