#!/usr/bin/env python3
import sys
sys.path.insert(0, 'scripts')
import launcher

# Verify constants exist and are correct
assert hasattr(launcher, 'CHARACTERS'), "CHARACTERS not defined"
assert len(launcher.CHARACTERS) == 27, f"CHARACTERS should have 27 items, got {len(launcher.CHARACTERS)}"

assert launcher.CHARACTERS[0] == 'FOX', f"First character should be FOX, got {launcher.CHARACTERS[0]}"

assert hasattr(launcher, 'PLAYER_TYPES'), "PLAYER_TYPES not defined"
assert launcher.PLAYER_TYPES == ['ai', 'human', 'cpu'], f"PLAYER_TYPES mismatch: {launcher.PLAYER_TYPES}"

assert hasattr(launcher, 'MODELS_DIR'), "MODELS_DIR not defined"
assert hasattr(launcher, 'list_models'), "list_models function not defined"

# Verify classes exist
assert hasattr(launcher, 'GlobalPathsFrame'), "GlobalPathsFrame not defined"
assert hasattr(launcher, 'PlayerFrame'), "PlayerFrame not defined"

# Test that list_models works without error
models = launcher.list_models()
print(f"list_models() returned: {models} (type: {type(models).__name__})")

# Verify methods
gp_methods = ['values']
for method in gp_methods:
    assert hasattr(launcher.GlobalPathsFrame, method), f"GlobalPathsFrame.{method} not found"

pf_methods = ['values']
for method in pf_methods:
    assert hasattr(launcher.PlayerFrame, method), f"PlayerFrame.{method} not found"

print("All structure verification checks passed!")
