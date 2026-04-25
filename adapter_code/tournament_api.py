from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

# Compatibility path: engine_adapter's classical imports expect `chess_engine` package.
REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSICAL_ROOT = REPO_ROOT / "classical_minimax"
if CLASSICAL_ROOT.exists():
    classical_str = str(CLASSICAL_ROOT)
    if classical_str not in sys.path:
        sys.path.insert(0, classical_str)
SCENARIOS_ROOT = REPO_ROOT / "scenarios"
if SCENARIOS_ROOT.exists():
    scenarios_str = str(SCENARIOS_ROOT)
    if scenarios_str not in sys.path:
        sys.path.insert(0, scenarios_str)

from engine_adapter import GameRunner, build_combo, build_engine
from tournament_trust_bridge import DEFAULT_TRUST_PATH, FAST_REFERENCE_DEPTH, TournamentTrustBridge


TRUST_BRIDGE = TournamentTrustBridge(persistence_path=DEFAULT_TRUST_PATH)
TRUST_LOCK = threading.Lock()
TRUST_STATUS: Dict[str, Any] = {
    "running": False,
    "queued": False,
    "last_started_at": None,
    "last_finished_at": None,
    "last_error": None,
    "last_summary": None,
    "current_game": None,
}


def _engine_kwargs(engine_id: str, depth: int, time_limit: float, mcts_iter: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if engine_id == "mcts":
        kwargs["iterations"] = mcts_iter
        kwargs["time_limit"] = time_limit
    elif engine_id == "neural_nn":
        kwargs["depth"] = depth
        kwargs["time_limit"] = time_limit
    elif engine_id in {"oracle", "oracle_direct", "random"}:
        # Keep defaults for these engines unless caller specifies globals.
        if engine_id in {"oracle", "oracle_direct"}:
            kwargs["time_limit"] = time_limit
    else:
        kwargs["max_depth"] = depth
        kwargs["time_limit"] = time_limit
    return kwargs


def _build_adapter(entry: Dict[str, Any], mode: str, depth: int, time_limit: float, mcts_iter: int):
    if mode == "engine":
        engine_id = entry.get("engineId")
        if not engine_id:
            raise ValueError("engine entry missing engineId")
        return build_engine(engine_id, **_engine_kwargs(engine_id, depth, time_limit, mcts_iter))

    if mode == "combo":
        search_id = entry.get("searchId")
        eval_id = entry.get("evalId")
        if not search_id or not eval_id:
            raise ValueError("combo entry must include searchId and evalId")
        combo_kwargs: Dict[str, Any] = {"max_depth": depth, "time_limit": time_limit}
        if search_id == "mcts_search":
            combo_kwargs = {"time_limit": time_limit, "iterations": mcts_iter}
        return build_combo(search_id, eval_id, **combo_kwargs)

    raise ValueError(f"Unsupported mode: {mode!r}")


def _trust_payload() -> Dict[str, Any]:
    with TRUST_LOCK:
        snapshot = TRUST_BRIDGE.snapshot()
        snapshot["status"] = dict(TRUST_STATUS)
        return snapshot


def _start_trust_training(
    *,
    start_fen: str,
    uci_moves: list[str],
    white_engine_id: str,
    black_engine_id: str,
    reference_depth: int,
    max_plies: int,
) -> Dict[str, Any]:
    with TRUST_LOCK:
        if TRUST_STATUS["running"] or TRUST_STATUS["queued"]:
            return {
                "queued": False,
                "started": False,
                "reason": "busy",
                "status": dict(TRUST_STATUS),
            }
        TRUST_STATUS.update(
            {
                "running": False,
                "queued": True,
                "last_started_at": None,
                "last_finished_at": None,
                "last_error": None,
                "last_summary": None,
                "current_game": {
                    "white": white_engine_id,
                    "black": black_engine_id,
                    "moves": len(uci_moves),
                    "reference_depth": reference_depth,
                    "max_plies": max_plies,
                },
            }
        )

    def _worker() -> None:
        with TRUST_LOCK:
            TRUST_STATUS["queued"] = False
            TRUST_STATUS["running"] = True
            TRUST_STATUS["last_started_at"] = time.time()
        try:
            summary = TRUST_BRIDGE.train_from_game(
                start_fen=start_fen,
                uci_moves=uci_moves,
                white_engine_id=white_engine_id,
                black_engine_id=black_engine_id,
                reference_depth=reference_depth,
                autosave=True,
                max_plies=max_plies,
            )
            with TRUST_LOCK:
                TRUST_STATUS["last_summary"] = summary
        except Exception as exc:  # noqa: BLE001
            with TRUST_LOCK:
                TRUST_STATUS["last_error"] = {
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
        finally:
            with TRUST_LOCK:
                TRUST_STATUS["running"] = False
                TRUST_STATUS["queued"] = False
                TRUST_STATUS["last_finished_at"] = time.time()

    thread = threading.Thread(target=_worker, name="trust-training", daemon=True)
    thread.start()
    return {
        "queued": True,
        "started": True,
        "reason": "background",
        "status": dict(TRUST_STATUS),
    }


class TournamentApiHandler(BaseHTTPRequestHandler):
    server_version = "TournamentAPI/1.0"

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "tournament_api",
                    "cwd": str(Path.cwd()),
                },
            )
            return
        if self.path.rstrip("/") == "/trust":
            self._send_json(
                200,
                {
                    "ok": True,
                    "trust": _trust_payload(),
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.rstrip("/")
        if route == "/trust/reset":
            try:
                with TRUST_LOCK:
                    if TRUST_STATUS["running"] or TRUST_STATUS["queued"]:
                        self._send_json(409, {"ok": False, "error": "Trust training is running; try again in a moment."})
                        return
                    TRUST_BRIDGE.reset()
                    TRUST_STATUS.update(
                        {
                            "running": False,
                            "queued": False,
                            "last_started_at": None,
                            "last_finished_at": None,
                            "last_error": None,
                            "last_summary": None,
                            "current_game": None,
                        }
                    )
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "trust": _trust_payload(),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=6)})
            return

        if route != "/play":
            self._send_json(404, {"ok": False, "error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))

            mode = str(payload.get("mode", "engine"))
            white_entry = payload.get("white")
            black_entry = payload.get("black")
            settings = payload.get("settings", {})
            if not isinstance(white_entry, dict) or not isinstance(black_entry, dict):
                raise ValueError("white and black entrants must be objects")

            depth = int(settings.get("depth", 2))
            time_limit = float(settings.get("timeLimit", 1.0))
            mcts_iter = int(settings.get("mctsIter", 100))
            max_moves = int(settings.get("maxMoves", 40))
            trust_reference_depth = int(settings.get("trustReferenceDepth", min(max(depth, FAST_REFERENCE_DEPTH), 3)))
            trust_max_plies = int(settings.get("trustMaxPlies", min(max_moves, 24)))
            fen = str(settings.get("fen") or "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

            white_adapter = _build_adapter(white_entry, mode, depth, time_limit, mcts_iter)
            black_adapter = _build_adapter(black_entry, mode, depth, time_limit, mcts_iter)

            runner = GameRunner()
            game = runner.play(white_adapter, black_adapter, fen=fen, max_moves=max_moves, verbose=False)
            trust_training = _start_trust_training(
                start_fen=fen,
                uci_moves=list(game.moves),
                white_engine_id=white_entry.get("name", white_adapter.name),
                black_engine_id=black_entry.get("name", black_adapter.name),
                reference_depth=trust_reference_depth,
                max_plies=trust_max_plies,
            )

            self._send_json(
                200,
                {
                    "ok": True,
                    "game": {
                        "white": white_entry.get("name", white_adapter.name),
                        "black": black_entry.get("name", black_adapter.name),
                        "winner": game.winner,
                        "reason": game.reason,
                        "moves": len(game.moves),
                        "uci_moves": list(game.moves),
                        "final_fen": game.final_fen,
                    },
                    "trust_training": trust_training,
                    "trust": _trust_payload(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                500,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=6),
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local API for running real Python tournament games")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), TournamentApiHandler)
    print(f"Tournament API listening on http://{args.host}:{args.port}")
    print("Endpoints: GET /health, GET /trust, POST /play, POST /trust/reset")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
