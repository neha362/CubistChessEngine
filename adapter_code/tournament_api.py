from __future__ import annotations

import argparse
import json
import sys
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
from tournament_trust_bridge import DEFAULT_TRUST_PATH, TournamentTrustBridge


TRUST_BRIDGE = TournamentTrustBridge(persistence_path=DEFAULT_TRUST_PATH)


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
                    "trust": TRUST_BRIDGE.snapshot(),
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.rstrip("/")
        if route == "/trust/reset":
            try:
                TRUST_BRIDGE.reset()
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "trust": TRUST_BRIDGE.snapshot(),
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

            depth = int(settings.get("depth", 4))
            time_limit = float(settings.get("timeLimit", 5.0))
            mcts_iter = int(settings.get("mctsIter", 400))
            max_moves = int(settings.get("maxMoves", 120))
            fen = str(settings.get("fen") or "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

            white_adapter = _build_adapter(white_entry, mode, depth, time_limit, mcts_iter)
            black_adapter = _build_adapter(black_entry, mode, depth, time_limit, mcts_iter)

            runner = GameRunner()
            game = runner.play(white_adapter, black_adapter, fen=fen, max_moves=max_moves, verbose=False)
            trust_training = TRUST_BRIDGE.train_from_game(
                start_fen=fen,
                uci_moves=list(game.moves),
                white_engine_id=white_entry.get("name", white_adapter.name),
                black_engine_id=black_entry.get("name", black_adapter.name),
                autosave=True,
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
