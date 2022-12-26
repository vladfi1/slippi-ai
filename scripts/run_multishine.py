#!/usr/bin/env python3
import argparse
import signal
import sys

import melee
from slippi_ai import dolphin, techskill

# This example program demonstrates how to use the Melee API to run a console,
#   setup controllers, and send button presses over to a console

def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
        raise argparse.ArgumentTypeError("%s is an invalid controller port. \
                                         Must be 1, 2, 3, or 4." % value)
    return ivalue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example of libmelee in action')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Debug mode. Creates a CSV of all game states')
    parser.add_argument('--address', '-a', default="127.0.0.1",
                        help='IP address of Slippi/Wii')
    parser.add_argument('--dolphin_executable_path', '-e', default=None,
                        help='The directory where dolphin is')
    parser.add_argument('--connect_code', '-t', default="",
                        help='Direct connect code to connect to in Slippi Online')
    parser.add_argument('--iso', default=None, type=str,
                        help='Path to melee iso.')
    parser.add_argument('--runtime', default=10, type=int,
                        help='Runtime in seconds.')

    args = parser.parse_args()

    # This logger object is useful for retroactively debugging issues in your bot
    #   You can write things to it each frame, and it will create a CSV file describing the match
    log = None
    if args.debug:
        log = melee.Logger()

    players = {
        port: dolphin.AI(melee.Character.FOX)
        for port in (1, 2)
    }

    console = dolphin.Dolphin(
        path=args.dolphin_executable_path,
        iso=args.iso,
        players=players,
    )

    # Create our Controller object
    #   The controller is the second primary object your bot will interact with
    #   Your controller is your way of sending button presses to the game, whether
    #   virtual or physical.
    agents = []

    for port in [1, 2]:
        agents.append(techskill.MultiShine(
            port, console.controllers[port]))

    costume = 0
    framedata = melee.framedata.FrameData()
    num_frames = 0

    # Main loop
    while num_frames < args.runtime * 60:
        # "step" to the next frame
        gamestate = console.step()
        for agent in agents:
            agent.step(gamestate)

    console.stop()
