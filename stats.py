import collections
import os
import pandas
import sqlite3
import melee
import paths

conn = sqlite3.connect(paths.DB_PATH)
TABLE = pandas.read_sql_query("SELECT * from replays", conn)

def get_all_names():
  return TABLE.filename

def get_fox_ditto_names():
  table = TABLE
  table = table[table.in_game_character_0 == melee.Character.FOX.value]
  table = table[table.in_game_character_1 == melee.Character.FOX.value]
  return table.filename

SUBSETS = {
  "fox_dittos": get_fox_ditto_names,
  "all": get_all_names,
}

def to_name(c):
  return melee.Character(c).name

def print_matchups():
  p0 = map(to_name, table['in_game_character_0'])
  p1 = map(to_name, table['in_game_character_1'])

  matchups = map(tuple, map(sorted, zip(p0, p1)))
  matchups = collections.Counter(matchups)

  sorted_matchups = sorted(matchups.items(), key=lambda kv: kv[1])
  total_games = len(table)

  for (c1, c2), v in sorted_matchups:
    print(c1.rjust(13), c2.rjust(13), f'{100*v/total_games:.2f}')

if __name__ == '__main__':
  print_matchups()
