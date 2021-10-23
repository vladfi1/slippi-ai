import itertools
from melee import enums

down_b = lambda c: c.simple_press(0.5, 0, enums.Button.BUTTON_B)
empty = lambda c: c.release_all()
jump = lambda c: c.simple_press(0.5, 0.5, enums.Button.BUTTON_Y)

multishine_inputs = [
    jump, empty, empty,
    # down_b, down_b, down_b,  # shine on first airborne frame
    down_b, empty, empty,  # shine on first airborne frame
    empty, empty,   # fall to ground
]

names = [
    "jump 1", "jump 2", "jump 3",
    "shine 1", "shine 2", "shine 3",
    "fall 1", "fall 2",
]

class MultiShine:
    def __init__(self, port: int, controller):
        self._port = port
        self._controller = controller
        self._input_sequence = itertools.cycle(zip(names, multishine_inputs))

    def step(self, gamestate):
        if gamestate.frame < 0:
            return

        # time.sleep((1 / SPEED - 1) / FPS)
        player = gamestate.players[self._port]
        name, action = next(self._input_sequence)

        # print(gamestate.frame, player.action, name)
        action(self._controller)

        self._controller.flush()
