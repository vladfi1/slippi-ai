#!/usr/bin/env python
import subprocess
import sys

cluster_file = sys.argv[1]
script_file = sys.argv[2]

if not cluster_file.endswith('.yaml'):
  raise ValueError(f'Expected yaml file: {cluster_file}')

if not script_file.endswith('.py'):
  raise ValueError(f'Expected python file: {script_file}')

def quote(arg: str) -> str:
  return f'"{arg}"'

try:
  # Would be nicer to use the ray python API, but I'm not sure if it is supported.
  subprocess.check_call([
    'ray', 'submit', '--start', cluster_file,
    script_file, *map(quote, sys.argv[3:]),
  ])
finally:
  # Make sure the cluster is torn down even if we Ctrl-C
  subprocess.check_call(['ray', 'down', '-y', cluster_file])
