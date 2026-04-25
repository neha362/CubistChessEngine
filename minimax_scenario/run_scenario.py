#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import chess.pgn


REQUIRED_DECISION_KEYS = (
    "stated_evaluation_function",
    "candidate_rewrites",
    "scored_candidates",
    "chosen_rewrite",
    "reason_for_rewrite",
)

DEFAULT_ROUND_PROMPT = """You are improving your own chess engine.

Before writing code, state your evaluation function. Include:
1. What makes a rewrite good?
2. What risks are you penalizing?
3. How will you choose among candidate rewrites?

Then propose 3 possible rewrites.
Score each from 1-10.
Choose the best rewrite.
Implement only that rewrite.
Explain why it should improve play.
"""


@dataclass
class Model:
    name: str
    engine_type: str
    source_dir: Path
    improve_command: str | None
    test_command: str | None
    move_time_ms: int


@dataclass
class GameOutcome:
    result: float  # 1.0 = white win, 0.5 draw, 0.0 black win
    illegal_moves: int
    avg_move_time_ms: float
    pgn: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def copy_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def run_command(command: str, cwd: Path, env: dict[str, str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env={**os.environ, **env},
        shell=True,
        text=True,
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


class Adapter:
    def choose_move(self, model_dir: Path, fen: str, move_time_ms: int) -> str:
        raise NotImplementedError


def purge_modules(prefixes: tuple[str, ...]) -> None:
    to_del = [k for k in sys.modules.keys() if any(k == p or k.startswith(f"{p}.") for p in prefixes)]
    for k in to_del:
        del sys.modules[k]


class ClassicalAdapter(Adapter):
    def choose_move(self, model_dir: Path, fen: str, move_time_ms: int) -> str:
        depth = max(1, min(4, move_time_ms // 300))
        purge_modules(("chess_engine",))
        sys.path.insert(0, str(model_dir))
        try:
            from chess_engine.eval import EvalAgent
            from chess_engine.move_gen import MoveGenAgent
            from chess_engine.search import SearchAgent
        finally:
            sys.path.pop(0)
        board = chess.Board(fen)
        engine = SearchAgent(EvalAgent().evaluate, MoveGenAgent())
        return engine.best_move(board, depth).uci()


class BerserkerAdapter(Adapter):
    def choose_move(self, model_dir: Path, fen: str, move_time_ms: int) -> str:
        purge_modules(("movegen_agent", "berserker_search_agent", "berserker_eval_agent"))
        sys.path.insert(0, str(model_dir))
        try:
            from movegen_agent import from_fen
            from berserker_search_agent import search
        finally:
            sys.path.pop(0)
        state = from_fen(fen)
        result = search(state, max_depth=4, movetime_ms=move_time_ms, verbose=False)
        return result.uci


class MonteCarloAdapter(Adapter):
    def choose_move(self, model_dir: Path, fen: str, move_time_ms: int) -> str:
        purge_modules(("movegen_agent", "rollout_agent", "mcts_agent"))
        sys.path.insert(0, str(model_dir))
        try:
            from movegen_agent import from_fen, sq_name
            from mcts_agent import mcts_search
        finally:
            sys.path.pop(0)
        state = from_fen(fen)
        res = mcts_search(state, movetime_ms=move_time_ms, max_iter=1000, verbose=False)
        if res.best_move is None:
            raise ValueError("MCTS returned no move")
        frm, to, promo = res.best_move
        return f"{sq_name(frm)}{sq_name(to)}{promo}"


class ClaudeApiAdapter(Adapter):
    def choose_move(self, model_dir: Path, fen: str, move_time_ms: int) -> str:
        import importlib.util
        purge_modules(("move_gen_agent", "search_engine", "eval_engine"))

        def load(mod_name: str, file_path: Path):
            spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load {file_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            return module

        # Some files import move_gen_agent; alias move_engine.py for compatibility.
        move_mod = load("move_gen_agent", model_dir / "move_engine.py")
        eval_mod = load("eval_engine", model_dir / "eval_engine.py")
        search_mod = load("search_engine", model_dir / "search_engine.py")
        state = move_mod.parse_fen(fen)
        depth = max(2, min(5, move_time_ms // 300))
        engine = search_mod.SearchAgent(eval_fn=eval_mod.Evaluator().evaluate, max_depth=depth, time_limit=max(0.2, move_time_ms / 1000))
        move, _, _ = engine.search(state)
        if move is None:
            raise ValueError("Claude API search returned no move")
        return move.uci()


class MockNNAdapter(Adapter):
    def __init__(self, helper_js: Path) -> None:
        self._helper = helper_js

    def choose_move(self, model_dir: Path, fen: str, move_time_ms: int) -> str:
        proc = subprocess.run(
            ["node", str(self._helper), str(model_dir), fen, str(move_time_ms)],
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"mock_nn helper failed: {proc.stderr.strip()}")
        return proc.stdout.strip()


def make_adapter(engine_type: str, scenario_dir: Path) -> Adapter:
    if engine_type == "classical_minimax":
        return ClassicalAdapter()
    if engine_type == "aggressive_berserker":
        return BerserkerAdapter()
    if engine_type == "mcts":
        return MonteCarloAdapter()
    if engine_type == "claude_api":
        return ClaudeApiAdapter()
    if engine_type == "mock_nn":
        return MockNNAdapter(scenario_dir / "mock_nn_choose_move.js")
    raise ValueError(f"Unknown engine_type: {engine_type}")


def ensure_decision_file(path: Path, model: str, round_idx: int) -> dict[str, Any]:
    if not path.exists():
        payload = {
            "stated_evaluation_function": f"{model} default placeholder (no improver command provided).",
            "candidate_rewrites": [
                "Improve move ordering",
                "Tune evaluation coefficients",
                "Improve search depth/time management",
            ],
            "scored_candidates": [
                {"rewrite": "Improve move ordering", "score": 7},
                {"rewrite": "Tune evaluation coefficients", "score": 7},
                {"rewrite": "Improve search depth/time management", "score": 7},
            ],
            "chosen_rewrite": "No automatic rewrite executed",
            "reason_for_rewrite": "Improver command not configured for this model/round.",
            "round": round_idx,
            "model": model,
        }
        write_json(path, payload)
    payload = read_json(path)
    missing = [k for k in REQUIRED_DECISION_KEYS if k not in payload]
    if missing:
        raise ValueError(f"Decision JSON missing keys {missing}: {path}")
    return payload


def parse_test_payload(path: Path, rc: int) -> tuple[int, int]:
    if path.exists():
        p = read_json(path)
        return int(p.get("passed", 0)), int(p.get("failed", 0))
    return (1, 0) if rc == 0 else (0, 1)


def play_single_game(
    white_adapter: Adapter,
    white_dir: Path,
    white_name: str,
    black_adapter: Adapter,
    black_dir: Path,
    black_name: str,
    move_time_ms: int,
    max_plies: int,
    start_fen: str,
) -> GameOutcome:
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    game.setup(chess.Board(start_fen))
    node = game
    timings: list[float] = []
    illegal = 0

    for _ in range(max_plies):
        if board.is_game_over():
            break
        adapter = white_adapter if board.turn == chess.WHITE else black_adapter
        engine_dir = white_dir if board.turn == chess.WHITE else black_dir
        t0 = time.perf_counter()
        try:
            uci = adapter.choose_move(engine_dir, board.fen(), move_time_ms)
            move = chess.Move.from_uci(uci)
        except Exception:
            illegal += 1
            result = 0.0 if board.turn == chess.WHITE else 1.0
            game.headers["Result"] = "0-1" if result == 0.0 else "1-0"
            return GameOutcome(result=result, illegal_moves=illegal, avg_move_time_ms=statistics.mean(timings) if timings else 0.0, pgn=str(game))
        dt = (time.perf_counter() - t0) * 1000.0
        timings.append(dt)
        if move not in board.legal_moves:
            illegal += 1
            result = 0.0 if board.turn == chess.WHITE else 1.0
            game.headers["Result"] = "0-1" if result == 0.0 else "1-0"
            return GameOutcome(result=result, illegal_moves=illegal, avg_move_time_ms=statistics.mean(timings) if timings else 0.0, pgn=str(game))
        board.push(move)
        node = node.add_variation(move)

    if board.is_game_over():
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            result = 0.5
            game.headers["Result"] = "1/2-1/2"
        elif outcome.winner == chess.WHITE:
            result = 1.0
            game.headers["Result"] = "1-0"
        else:
            result = 0.0
            game.headers["Result"] = "0-1"
    else:
        result = 0.5
        game.headers["Result"] = "1/2-1/2"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    return GameOutcome(
        result=result,
        illegal_moves=illegal,
        avg_move_time_ms=statistics.mean(timings) if timings else 0.0,
        pgn=str(game),
    )


def play_match(
    model_a: tuple[str, Adapter, Path],
    model_b: tuple[str, Adapter, Path],
    games: int,
    move_time_ms: int,
    max_plies: int,
    opening_fens: list[str],
) -> dict[str, Any]:
    a_name, a_adapter, a_dir = model_a
    b_name, b_adapter, b_dir = model_b
    a_points = 0.0
    a_wins = 0
    b_wins = 0
    draws = 0
    illegal = 0
    avg_times: list[float] = []
    pgns: list[str] = []

    for i in range(games):
        start_fen = opening_fens[i % len(opening_fens)]
        if i % 2 == 0:
            g = play_single_game(a_adapter, a_dir, a_name, b_adapter, b_dir, b_name, move_time_ms, max_plies, start_fen)
            a_points += g.result
            if g.result == 1.0:
                a_wins += 1
            elif g.result == 0.0:
                b_wins += 1
            else:
                draws += 1
        else:
            g = play_single_game(b_adapter, b_dir, b_name, a_adapter, a_dir, a_name, move_time_ms, max_plies, start_fen)
            a_points += 1.0 - g.result
            if g.result == 0.0:
                a_wins += 1
            elif g.result == 1.0:
                b_wins += 1
            else:
                draws += 1
        illegal += g.illegal_moves
        avg_times.append(g.avg_move_time_ms)
        pgns.append(g.pgn)
    return {
        "wins": a_wins,
        "draws": draws,
        "losses": b_wins,
        "points": a_points,
        "games": games,
        "win_rate": a_points / games,
        "illegal_moves": illegal,
        "average_move_time_ms": statistics.mean(avg_times) if avg_times else 0.0,
        "pgns": pgns,
    }


def run_tactics(adapter: Adapter, model_dir: Path, model_name: str, move_time_ms: int, tactics: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    illegal = 0
    details: list[dict[str, Any]] = []
    for item in tactics:
        fen = item["fen"]
        expected = set(item.get("best_moves", []))
        try:
            uci = adapter.choose_move(model_dir, fen, move_time_ms)
            move = chess.Move.from_uci(uci)
            legal = move in chess.Board(fen).legal_moves
        except Exception:
            uci = "error"
            legal = False
        if not legal:
            illegal += 1
        ok = (uci in expected) if expected else legal
        if ok:
            correct += 1
        details.append({"fen": fen, "expected": sorted(expected), "chosen": uci, "pass": ok, "legal": legal})
    return {
        "model": model_name,
        "positions": len(tactics),
        "correct": correct,
        "accuracy": (correct / len(tactics)) if tactics else 0.0,
        "illegal_moves": illegal,
        "details": details,
    }


def load_models(cfg: dict[str, Any], repo_root: Path) -> list[Model]:
    out: list[Model] = []
    for m in cfg["models"]:
        out.append(
            Model(
                name=m["name"],
                engine_type=m["engine_type"],
                source_dir=(repo_root / m["source_dir"]).resolve(),
                improve_command=m.get("improve_command"),
                test_command=m.get("test_command"),
                move_time_ms=int(m.get("move_time_ms", cfg.get("move_time_ms", 700))),
            )
        )
    return out


def run(config_path: Path) -> int:
    repo_root = config_path.resolve().parent.parent
    cfg = read_json(config_path)
    rounds = int(cfg.get("rounds", 5))
    games_per_match = int(cfg.get("games_per_match", 4))
    max_plies = int(cfg.get("max_plies", 120))
    out_root = (repo_root / cfg.get("output_dir", "scenario_outputs")).resolve()
    run_dir = out_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scenario_dir = Path(__file__).resolve().parent
    models = load_models(cfg, repo_root)
    model_by_name = {m.name: m for m in models}
    baseline_name = cfg["baseline_model"]
    if baseline_name not in model_by_name:
        raise ValueError(f"baseline_model '{baseline_name}' is not in models list")

    opening_fens = cfg.get("opening_fens", [chess.STARTING_FEN])
    tactics = read_json((repo_root / cfg.get("tactics_file", "scenario/tactics_default.json")).resolve())["positions"]

    write_json(
        run_dir / "manifest.json",
        {
            "started_at_utc": utc_now(),
            "rounds": rounds,
            "games_per_match": games_per_match,
            "max_plies": max_plies,
            "baseline_model": baseline_name,
            "models": [m.__dict__ | {"source_dir": str(m.source_dir)} for m in models],
        },
    )

    # Round 0 snapshots (shared baseline/original code)
    snapshots: dict[str, dict[int, Path]] = {m.name: {} for m in models}
    for m in models:
        dst = run_dir / "snapshots" / m.name / "round_0"
        copy_dir(m.source_dir, dst)
        snapshots[m.name][0] = dst

    summary_csv = run_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "round",
                "stated_evaluation_function",
                "chosen_rewrite",
                "reason_for_rewrite",
                "tests_passed",
                "tests_failed",
                "win_rate_vs_baseline",
                "win_rate_vs_previous_self",
                "win_rate_vs_other_models",
                "illegal_move_count",
                "average_move_time_ms",
            ],
        )
        writer.writeheader()

    for r in range(1, rounds + 1):
        # Improvement + tests first for all models
        round_decisions: dict[str, dict[str, Any]] = {}
        round_tests: dict[str, dict[str, Any]] = {}
        for m in models:
            prev = snapshots[m.name][r - 1]
            cur = run_dir / "snapshots" / m.name / f"round_{r}"
            copy_dir(prev, cur)
            snapshots[m.name][r] = cur

            prompt_path = run_dir / "prompts" / m.name / f"round_{r}.txt"
            decision_path = run_dir / "decisions" / m.name / f"round_{r}.json"
            test_json_path = run_dir / "tests" / m.name / f"round_{r}.json"
            write_text(prompt_path, DEFAULT_ROUND_PROMPT)

            if m.improve_command:
                env = {
                    "SCENARIO_MODEL": m.name,
                    "SCENARIO_ROUND": str(r),
                    "SCENARIO_PROMPT_PATH": str(prompt_path),
                    "SCENARIO_DECISION_JSON_PATH": str(decision_path),
                }
                rc, out, err = run_command(m.improve_command, cur, env)
                write_text(run_dir / "logs" / m.name / f"improve_round_{r}.log", f"EXIT={rc}\n\nSTDOUT:\n{out}\n\nSTDERR:\n{err}\n")
                if rc != 0:
                    raise RuntimeError(f"Improve command failed for {m.name} round {r}")
            decision = ensure_decision_file(decision_path, m.name, r)
            round_decisions[m.name] = decision

            if m.test_command:
                env_t = {
                    "SCENARIO_MODEL": m.name,
                    "SCENARIO_ROUND": str(r),
                    "SCENARIO_TEST_JSON_PATH": str(test_json_path),
                }
                trc, tout, terr = run_command(m.test_command, cur, env_t)
                write_text(run_dir / "logs" / m.name / f"test_round_{r}.log", f"EXIT={trc}\n\nSTDOUT:\n{tout}\n\nSTDERR:\n{terr}\n")
                passed, failed = parse_test_payload(test_json_path, trc)
            else:
                passed, failed = (0, 0)
            round_tests[m.name] = {"passed": passed, "failed": failed}

        # Benchmarks
        for m in models:
            adapter = make_adapter(m.engine_type, scenario_dir)
            cur = snapshots[m.name][r]

            baseline = model_by_name[baseline_name]
            baseline_adapter = make_adapter(baseline.engine_type, scenario_dir)
            vs_baseline = play_match(
                (m.name, adapter, cur),
                (baseline.name, baseline_adapter, snapshots[baseline.name][0]),
                games_per_match,
                m.move_time_ms,
                max_plies,
                opening_fens,
            )
            write_json(run_dir / "matches" / m.name / f"round_{r}_vs_baseline.json", vs_baseline)

            prev_adapter = make_adapter(m.engine_type, scenario_dir)
            vs_prev = play_match(
                (m.name, adapter, cur),
                (f"{m.name}_prev", prev_adapter, snapshots[m.name][r - 1]),
                games_per_match,
                m.move_time_ms,
                max_plies,
                opening_fens,
            )
            write_json(run_dir / "matches" / m.name / f"round_{r}_vs_previous_self.json", vs_prev)

            other_rates: list[float] = []
            illegal_total = vs_baseline["illegal_moves"] + vs_prev["illegal_moves"]
            times = [vs_baseline["average_move_time_ms"], vs_prev["average_move_time_ms"]]
            for other in models:
                if other.name == m.name:
                    continue
                other_adapter = make_adapter(other.engine_type, scenario_dir)
                other_match = play_match(
                    (m.name, adapter, cur),
                    (other.name, other_adapter, snapshots[other.name][r]),
                    games_per_match,
                    m.move_time_ms,
                    max_plies,
                    opening_fens,
                )
                write_json(run_dir / "matches" / m.name / f"round_{r}_vs_{other.name}.json", other_match)
                other_rates.append(other_match["win_rate"])
                illegal_total += other_match["illegal_moves"]
                times.append(other_match["average_move_time_ms"])

            tactics_result = run_tactics(adapter, cur, m.name, m.move_time_ms, tactics)
            write_json(run_dir / "tactics" / m.name / f"round_{r}.json", tactics_result)
            illegal_total += tactics_result["illegal_moves"]

            row = {
                "model": m.name,
                "round": r,
                "stated_evaluation_function": round_decisions[m.name]["stated_evaluation_function"],
                "chosen_rewrite": round_decisions[m.name]["chosen_rewrite"],
                "reason_for_rewrite": round_decisions[m.name]["reason_for_rewrite"],
                "tests_passed": round_tests[m.name]["passed"],
                "tests_failed": round_tests[m.name]["failed"],
                "win_rate_vs_baseline": vs_baseline["win_rate"],
                "win_rate_vs_previous_self": vs_prev["win_rate"],
                "win_rate_vs_other_models": statistics.mean(other_rates) if other_rates else 0.0,
                "illegal_move_count": illegal_total,
                "average_move_time_ms": statistics.mean(times) if times else 0.0,
            }
            with summary_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)
            write_json(run_dir / "round_reports" / m.name / f"round_{r}.json", row)
            print(f"[OK] {m.name} round {r} complete")

    write_json(run_dir / "finished.json", {"finished_at_utc": utc_now(), "summary_csv": str(summary_csv)})
    print(f"Scenario complete. Results in: {run_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run self-improvement scenario for multiple chess engines.")
    parser.add_argument("--config", required=True, help="Path to scenario config JSON.")
    args = parser.parse_args()
    return run(Path(args.config).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
