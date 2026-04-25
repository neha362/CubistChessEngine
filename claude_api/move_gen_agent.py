"""
Compatibility shim. claude_api's search_engine.py and eval_engine.py import
`from move_gen_agent import ...`, but the actual module is named
`move_engine.py`. Re-export the relevant symbols here so both names resolve.
"""

from move_engine import *  # noqa: F401,F403
from move_engine import BoardState, Move, MoveGenerator, parse_fen  # noqa: F401
