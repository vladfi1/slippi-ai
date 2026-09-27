"""Systematic evaluation of deployed agents.

The package keeps a sqlite database of every agent that has appeared in
`deployed_models/` (keyed on the md5 of the underlying `stripped_models/` file,
since deployed names come and go), every evaluation run between two agents, the
individual games those runs produced, and Bradley-Terry ratings fit to them.

Modules:
  db        schema and connection helpers
  agents    scanning deployed_models and registering agents
  matchups  which pairs still need evaluating
  runner    running scripts/run_evaluator.py and ingesting its results
  ratings   Bradley-Terry ratings via choix
"""
