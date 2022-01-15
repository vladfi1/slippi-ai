"""
   Filter by a large amount of Slippi files, especially from Netplay, and train upon that dataset. This script produces a text document containing
   a list of Slippi files preprocessed and found to be usable for training based on the applied filter. Finally using this text document as the 
   input to make_dataset.py will completely render the Slippi file as a pickle.

   For help with input id's consult:
   https://github.com/hohav/py-slippi/blob/master/slippi/id.py
   
"""

import json
import multiprocessing

from absl import app, flags
from multiprocessing import Process
from pathlib import Path
from slippi import Game, id as sid
from slippi.parse import ParseError
from typing import List


FLAGS = flags.FLAGS

# integers
flags.DEFINE_integer('cores', multiprocessing.cpu_count(), 'Number of cores/processes to spawn.')

# strings
flags.DEFINE_string('src_dir', 'Slippi/', 'Source directory of dataset.')
flags.DEFINE_string('export_file', 'filtered_dataset.json' ,'Filtered slippi files exported to json.')

# game.end
flags.DEFINE_bool('ignore_lras', False, 'Ignore games that contain LRA Start input.')

# game.metadata
flags.DEFINE_integer('required_frames', 2200, 'Required number of frames for a game to have.')
flags.DEFINE_list('allowed_characters', [], 'Name of the characters allowed in game, example: "fox,ganondorf,sheik".')
flags.DEFINE_list('allowed_codes', [], 'Netplay codes, example "ABSZ#717,BLAH#420".')
flags.DEFINE_list('allowed_stages', [], 'Name of stage, space separated by underscores.')


def compare_end(game : Game) -> bool:
   """Compare end function.
   
   Args:
      game: Slippi recording.
      
   Returns:
      bool: True on successful comparison, able candidate.
   """
   successful = True
   if not FLAGS.ignore_lras and game.end is not None:
      successful &= game.end.lras_initiator is None
   return successful


def compare_metadata(game : Game) -> bool:
   """Compare metadata function, favors netplay related slippi files.

   Args: 
      game: Slippi recording.
   
   Returns:
      bool: True on successful comparison, able candidate.
   """
   successful = False
   if hasattr(game, 'metadata') and hasattr(game.metadata, 'players') and game.metadata.platform.value == 'dolphin': 
      players = game.metadata.players
      # Must have (2) players
      successful = (len([player for player in players if player is not None]) == 2)
      
      # Compare allowed_characters
      if FLAGS.allowed_characters:
         if all([player.characters for player in players if player is not None]) and all(len(player.characters) == 1 for player in players if player is not None):
            for character in FLAGS.allowed_characters:
               successful &= any([list(player.characters.keys())[0].name.lower() == str(character).lower() for player in players if player is not None])
         else:
            successful = False

      # Compare allowed_codes
      if FLAGS.allowed_codes:
         if all([hasattr(player, 'netplay') and player.netplay is not None for player in players if player is not None]):
            was_successful = False
            for code in FLAGS.allowed_codes:
               was_successful |= any([player.netplay.code.lower() == code.lower() for player in players if player is not None])
            successful &= was_successful
         else:
            successful = False
      
      # Compare required_frames
      if FLAGS.required_frames > 0:
         successful &= game.metadata.duration >= FLAGS.required_frames
   
   return successful


def compare_start(game : Game) -> bool:
   """Compare start function, favors netplay related slippi files.

   Args: 
      game: Slippi recording.
   
   Returns:
      bool: True on successful comparison, able candidate.
   """
   successful = False
   if hasattr(game, 'start'): 
      successful = True
      start = game.start
      if FLAGS.allowed_stages:
         successful &= str(start.stage.name).lower() in [str(x).lower() for x in FLAGS.allowed_stages]
      return successful


def processor(files : List[Path]):
   """Processing function for multiprocess.Process.
   
   Args:
      pypid: Integer of process.
      files: Files to be processed.
      
   Returns:
      Files that succeeded.
   """
   processed_list = []
   for file in files:
      if file.suffix == '.slp':
         try:
            game = Game(file)
            success = compare_metadata(game)
            success &= compare_start(game)
            success &= compare_end(game)

            if success:
               processed_list.append(file)
         except ParseError as pe:
            print(pe)
   return processed_list


def main(_):
   """TODO: Main is a giant cascading ruleset, this would be nice to
   apply a series of validation functions into a list and run all(list)
   to only grab what returns True."""
   processed_files = []
   source_directory = Path(FLAGS.src_dir)
   if source_directory.exists():
      files = [file for file in source_directory.iterdir()]
      partition_size = len(files) // max(1, FLAGS.cores)
      partitioned_set = [files[i:i + partition_size] for i in range(0, len(files), partition_size)]
      
      pool = multiprocessing.Pool()
      return_pool = list(pool.map(processor, [partition for partition in partitioned_set]))
      for ret in return_pool:
         for path in ret:
            processed_files.append(str(path))
   else:
      raise FileNotFoundError(f'Source directory {source_directory} does not exist.')

   if processed_files:
      with open(FLAGS.export_file, 'w') as fd:
         json.dump({'processed_slippi_files': processed_files}, fd)
      print(f'Final processed size {len(processed_files)}.')

if __name__ == '__main__':
   app.run(main)
