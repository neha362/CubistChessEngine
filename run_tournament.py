import sys
sys.path.insert(0, 'adapter_code')
from engine_adapter import build_engine, run_tournament

engines = [
    build_engine('classical',       max_depth=3),
    build_engine('berserker_siege', max_depth=3),
    build_engine('berserker_chaos', max_depth=3),
    build_engine('mcts',            time_limit=3.0),
    build_engine('random'),
]
run_tournament(engines, max_moves=150)
