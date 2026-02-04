"""Action space module for Slippi AI.

Provides discrete action spaces derived from clustering real controller inputs.
"""

from slippi_ai.action_space.spaces import (
    BUTTON_NAMES,
    DiscreteAction,
    CompactActionSpace,
    compact_action_space,
)

__all__ = [
    'BUTTON_NAMES',
    'DiscreteAction',
    'CompactActionSpace',
    'compact_action_space',
]
