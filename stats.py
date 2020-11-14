import collections
import os
import pandas
import sqlite3
import melee

DB_PATH = os.path.expanduser('~/Scratch/melee_public_slp_dataset.sqlite3')

conn = sqlite3.connect(DB_PATH)
table = pandas.read_sql_query("SELECT * from replays", conn)

def to_name(c):
  return melee.Character(c).name

def print_matchups():
  p0 = map(to_name, table['css_character_0'])
  p1 = map(to_name, table['css_character_1'])

  matchups = map(tuple, map(sorted, zip(p0, p1)))
  matchups = collections.Counter(matchups)

  sorted_matchups = sorted(matchups.items(), key=lambda kv: kv[1])
  total_games = len(table)

  for (c1, c2), v in sorted_matchups:
    print(c1.rjust(13), c2.rjust(13), f'{100*v/total_games:.2f}')

if __name__ == '__main__':
  print_matchups()
