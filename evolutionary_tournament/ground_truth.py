"""Reference scores: built-in classical evaluator (Stockfish optional via env var)."""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path

import chess
import chess.engine

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "classical_minimax") not in sys.path:
    sys.path.insert(0, str(_ROOT / "classical_minimax"))

from chess_engine.eval import EvalAgent  # type: ignore
from chess_engine.move_gen import MoveGenAgent  # type: ignore
from chess_engine.search import SearchAgent  # type: ignore


def _try_open_stockfish() -> chess.engine.SimpleEngine | None:
    """
    Try to open Stockfish.  Returns None (silently) in every failure case,
    including Windows/Jupyter asyncio subprocess restrictions.

    To enable Stockfish, set the STOCKFISH_EXECUTABLE environment variable
    to the absolute path of your stockfish binary before importing this module.
    """
    # On Windows with a running asyncio event loop (e.g. inside a Jupyter
    # kernel), python-chess spawns a background thread whose new event loop
    # may use SelectorEventLoop, which cannot create subprocess transports.
    # Detect this early and skip the subprocess attempt entirely.
    if sys.platform == "win32":
        try:
            import asyncio as _asyncio
            _asyncio.get_running_loop()
            return None  # running inside Jupyter / async context — skip
        except RuntimeError:
            pass  # no running loop, safe to try

    envp = os.environ.get("STOCKFISH_EXECUTABLE", "").strip()
    cands = [envp] if envp else []
    cands += ["stockfish", "stockfish.exe"]

    for name in cands:
        if not name:
            continue
        # Resolve to an absolute path via PATH if needed
        if os.path.isabs(name) and os.path.isfile(name):
            path = name
        else:
            found = shutil.which(name)
            if not found:
                continue
            path = found
        try:
            return chess.engine.SimpleEngine.popen_uci(path)  # type: ignore[call-overload]
        except Exception:
            pass

    return None


@contextlib.contextmanager
def reference_engine():
    """Context manager yielding a Stockfish SimpleEngine or None."""
    e = _try_open_stockfish()
    try:
        yield e
    finally:
        if e is not None:
            with contextlib.suppress(chess.engine.EngineError, OSError, BrokenPipeError):
                e.quit()


def position_value_white(
    board: chess.Board,
    depth: int,
    engine: chess.engine.SimpleEngine | None,
) -> int:
    """Centipawn score for White. Uses Stockfish when available, built-in eval otherwise."""
    if engine is not None:
        with contextlib.suppress(chess.engine.EngineError, OSError, BrokenPipeError):
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=0.2))
            w = info["score"].white()
            if w.is_mate() and w.mate() is not None:
                m = w.mate()
                return 30000 - 10 * abs(m) if (m and m > 0) else -30000 + 10 * abs(m or 0)
            c = w.score()
            return int(c) if c is not None else 0
    return EvalAgent().evaluate(board)


def root_move_scores_sm(
    board: chess.Board,
    max_depth: int,
    engine: chess.engine.SimpleEngine | None,
) -> dict[str, int]:
    """
    UCI move → centipawn score for the side to move.
    Uses Stockfish multiPV when available, built-in SearchAgent otherwise.
    """
    legal = list(board.legal_moves)
    if not legal:
        return {}
    if len(legal) == 1:
        return {legal[0].uci(): 0}

    if engine is not None:
        n = min(len(legal), 64)
        try:
            infos = engine.analyse(
                board,
                chess.engine.Limit(depth=max(4, min(max_depth, 12)), time=0.4),
                multipv=n,
            )
        except (chess.engine.EngineError, OSError, BrokenPipeError, chess.engine.EngineTerminatedError):
            infos = None
        else:
            uci: dict[str, int] = {}
            for inf in infos:
                pvs = inf.get("pv", [])
                if not pvs:
                    continue
                sc = inf["score"].relative
                m0 = pvs[0]
                if sc.is_mate() and sc.mate() is not None:
                    mt = sc.mate()
                    cp = 30_000 - 10 * abs(mt) if (mt and mt > 0) else -30_000
                else:
                    cp = sc.score() if sc.score() is not None else 0
                uci[m0.uci()] = int(cp)
            for mv in legal:
                uci.setdefault(mv.uci(), 0)
            if uci:
                return uci

    s = SearchAgent(EvalAgent().evaluate, MoveGenAgent())
    per, _bm, _ = s.root_all_scores(board.copy(), max(2, min(max_depth, 5)))
    return {m.uci(): int(c) for m, c in per}
