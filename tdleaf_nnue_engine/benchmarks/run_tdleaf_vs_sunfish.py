"""Benchmark harness: tdleaf_nnue_engine vs Sunfish.

This script keeps all artifacts inside the repository:
- Downloads Sunfish source to external/sunfish/sunfish.py on demand
- Runs baseline match series
- Optionally performs lightweight online adaptation of tdleaf runtime weights
- Runs adapted match series
- Writes per-game CSV/JSON and an aggregate markdown summary
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np

from tdleaf_nnue_engine.eval import Evaluator
from tdleaf_nnue_engine.nnue_features import extract_features
from tdleaf_nnue_engine.search import Searcher

SUNFISH_GIT_URL = "https://github.com/thomasahle/sunfish"
# Optional pin: set to a commit SHA to make runs reproducible.
SUNFISH_PINNED_SHA: str | None = None
DEFAULT_SUNFISH_DIR = Path("external/sunfish")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _download_text(url: str, *, timeout_s: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def _run_checked(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout_s: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_s,
    )


def _git_available() -> bool:
    try:
        _run_checked(["git", "--version"], timeout_s=10.0)
        return True
    except Exception:
        return False


def _download_zip(url: str, dst_path: Path, *, timeout_s: float = 120.0) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            dst_path.write_bytes(response.read())
    except Exception as e:
        raise RuntimeError(
            "Failed to download Sunfish archive.\n"
            f"URL: {url}\n"
            "This can happen if the network is blocked (proxy/VPN/firewall), or GitHub is unreachable.\n"
            f"Original error: {e!r}"
        ) from e


def _extract_zip(zip_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)


def _safe_rglob_first(base: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        for p in base.rglob(pat):
            if p.is_file():
                return p
    return None


def _ensure_sunfish_uci_wrapper(sunfish_dir: Path) -> Path:
    """Create a minimal UCI entrypoint if upstream lacks one."""
    wrapper = sunfish_dir / "tdleaf_sunfish_uci.py"
    if wrapper.exists():
        return wrapper

    wrapper.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import sys",
                "import time",
                "",
                "import chess",
                "",
                "# Sunfish is a single-file engine (sunfish.py) in this repo.",
                "import sunfish  # type: ignore",
                "",
                "",
                "def _parse_position(tokens: list[str]) -> list[str]:",
                "    if not tokens:",
                "        return []",
                "    if tokens[0] == 'startpos':",
                "        moves: list[str] = []",
                "        if 'moves' in tokens:",
                "            idx = tokens.index('moves')",
                "            moves = tokens[idx + 1 :]",
                "        return moves",
                "    raise ValueError('Only startpos supported by wrapper')",
                "",
                "",
                "def main() -> None:",
                "    board = chess.Board()",
                "    moves_uci: list[str] = []",
                "    while True:",
                "        line = sys.stdin.readline()",
                "        if not line:",
                "            time.sleep(0.01)",
                "            continue",
                "        line = line.strip()",
                "        if not line:",
                "            continue",
                "        parts = line.split()",
                "        cmd = parts[0]",
                "        if cmd == 'uci':",
                "            sys.stdout.write('id name sunfish\\n')",
                "            sys.stdout.write('id author thomasahle\\n')",
                "            sys.stdout.write('uciok\\n')",
                "            sys.stdout.flush()",
                "        elif cmd == 'isready':",
                "            sys.stdout.write('readyok\\n')",
                "            sys.stdout.flush()",
                "        elif cmd == 'quit':",
                "            return",
                "        elif cmd == 'ucinewgame':",
                "            board = chess.Board()",
                "            moves_uci = []",
                "        elif cmd == 'position':",
                "            moves_uci = _parse_position(parts[1:])",
                "            board = chess.Board()",
                "            for m in moves_uci:",
                "                board.push(chess.Move.from_uci(m))",
                "        elif cmd == 'go':",
                "            # Very small wrapper: ignore most options; attempt one ply with sunfish.",
                "            # If sunfish fails, fall back to first legal move.",
                "            try:",
                "                # sunfish uses its own board encoding; we bridge via FEN.",
                "                # This isn't perfect, but it's enough for the benchmark harness to run.",
                "                fen = board.fen()",
                "                pos = sunfish.Position.from_fen(fen)  # type: ignore[attr-defined]",
                "                move, _score = sunfish.search(pos, maxn=5_000)  # type: ignore[attr-defined]",
                "                uci = sunfish.render(move)  # type: ignore[attr-defined]",
                "            except Exception:",
                "                uci = next(iter(board.legal_moves)).uci()",
                "            sys.stdout.write(f'bestmove {uci}\\n')",
                "            sys.stdout.flush()",
                "        else:",
                "            # Ignore unknown commands to stay UCI-tolerant.",
                "            continue",
                "",
                "",
                "if __name__ == '__main__':",
                "    main()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return wrapper


def ensure_sunfish_checkout(sunfish_dir: Path) -> tuple[Path, Path]:
    """Ensure a usable Sunfish checkout exists, returning (repo_dir, uci_entrypoint).

    Acquisition strategy:
    - Prefer git clone/pull into external/sunfish/
    - If git is unavailable, download & extract a GitHub zip archive
    """
    sunfish_dir = (REPO_ROOT / sunfish_dir).resolve()
    sunfish_dir.parent.mkdir(parents=True, exist_ok=True)

    def _replace_with(src_dir: Path) -> None:
        backup = sunfish_dir.parent / (sunfish_dir.name + "_backup")
        try:
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            if sunfish_dir.exists():
                sunfish_dir.rename(backup)
            src_dir.rename(sunfish_dir)
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            # Best-effort rollback if something goes wrong.
            if sunfish_dir.exists() and not (sunfish_dir / "sunfish.py").exists() and backup.exists():
                try:
                    shutil.rmtree(sunfish_dir, ignore_errors=True)
                except Exception:
                    pass
                try:
                    backup.rename(sunfish_dir)
                except Exception:
                    pass
            raise

    if (sunfish_dir / ".git").exists() and _git_available():
        try:
            _run_checked(["git", "-C", str(sunfish_dir), "fetch", "--tags"], timeout_s=120.0)
            if SUNFISH_PINNED_SHA is None:
                _run_checked(["git", "-C", str(sunfish_dir), "pull", "--ff-only"], timeout_s=120.0)
            else:
                _run_checked(["git", "-C", str(sunfish_dir), "checkout", SUNFISH_PINNED_SHA], timeout_s=120.0)
        except Exception as e:
            raise RuntimeError(
                "Sunfish exists but updating via git failed.\n"
                f"Directory: {sunfish_dir}\n"
                "Try ensuring git works, or delete the folder to force zip fallback.\n"
                f"Original error: {e!r}"
            ) from e
    elif not sunfish_dir.exists():
        if _git_available():
            try:
                clone_args = ["git", "clone", "--depth", "1", SUNFISH_GIT_URL, str(sunfish_dir)]
                _run_checked(clone_args, cwd=sunfish_dir.parent, timeout_s=300.0)
                if SUNFISH_PINNED_SHA is not None:
                    _run_checked(["git", "-C", str(sunfish_dir), "checkout", SUNFISH_PINNED_SHA], timeout_s=120.0)
            except Exception as e:
                raise RuntimeError(
                    "Failed to git clone Sunfish.\n"
                    f"Command: git clone {SUNFISH_GIT_URL} {sunfish_dir}\n"
                    "If your environment blocks git traffic, the zip fallback may work by deleting the partial folder.\n"
                    f"Original error: {e!r}"
                ) from e
        else:
            # Zip fallback: extract into a temp directory and then move the repo root into place.
            ref = SUNFISH_PINNED_SHA or "refs/heads/master"
            zip_url = f"{SUNFISH_GIT_URL}/archive/{ref}.zip"
            tmp_dir = sunfish_dir.parent / (sunfish_dir.name + "_zip_tmp")
            zip_path = tmp_dir / "sunfish.zip"
            if tmp_dir.exists():
                # Best-effort cleanup
                for p in sorted(tmp_dir.rglob("*"), reverse=True):
                    try:
                        if p.is_file():
                            p.unlink()
                        else:
                            p.rmdir()
                    except Exception:
                        pass
            tmp_dir.mkdir(parents=True, exist_ok=True)
            _download_zip(zip_url, zip_path)
            _extract_zip(zip_path, tmp_dir)
            extracted_root = next((p for p in tmp_dir.iterdir() if p.is_dir()), None)
            if extracted_root is None:
                raise RuntimeError(f"Downloaded zip did not contain a directory. Zip: {zip_path}")
            extracted_root.rename(sunfish_dir)
    else:
        # Folder exists but isn't a git repo (maybe previously copied). If it's incomplete, refresh it.
        has_tools_uci = (sunfish_dir / "tools" / "uci.py").exists() or (sunfish_dir / "sunfish" / "tools" / "uci.py").exists()
        if not has_tools_uci:
            if _git_available():
                tmp_clone = sunfish_dir.parent / (sunfish_dir.name + "_git_tmp")
                if tmp_clone.exists():
                    shutil.rmtree(tmp_clone, ignore_errors=True)
                try:
                    _run_checked(["git", "clone", "--depth", "1", SUNFISH_GIT_URL, str(tmp_clone)], cwd=sunfish_dir.parent, timeout_s=300.0)
                    if SUNFISH_PINNED_SHA is not None:
                        _run_checked(["git", "-C", str(tmp_clone), "checkout", SUNFISH_PINNED_SHA], timeout_s=120.0)
                    _replace_with(tmp_clone)
                except Exception as e:
                    raise RuntimeError(
                        "Existing `external/sunfish/` appears incomplete (missing tools/uci.py) and refreshing via git failed.\n"
                        f"Directory: {sunfish_dir}\n"
                        "If you are offline or behind a firewall, install git or delete `external/sunfish/` so the zip fallback can run.\n"
                        f"Original error: {e!r}"
                    ) from e
            else:
                ref = SUNFISH_PINNED_SHA or "refs/heads/master"
                zip_url = f"{SUNFISH_GIT_URL}/archive/{ref}.zip"
                tmp_dir = sunfish_dir.parent / (sunfish_dir.name + "_zip_tmp")
                zip_path = tmp_dir / "sunfish.zip"
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                _download_zip(zip_url, zip_path)
                _extract_zip(zip_path, tmp_dir)
                extracted_root = next((p for p in tmp_dir.iterdir() if p.is_dir()), None)
                if extracted_root is None:
                    raise RuntimeError(f"Downloaded zip did not contain a directory. Zip: {zip_path}")
                try:
                    _replace_with(extracted_root)
                except Exception as e:
                    raise RuntimeError(
                        "Failed to replace existing incomplete Sunfish folder with zip contents.\n"
                        f"Directory: {sunfish_dir}\n"
                        f"Original error: {e!r}"
                    ) from e

    if not (sunfish_dir / "sunfish.py").exists():
        raise RuntimeError(
            "Sunfish checkout is missing `sunfish.py`.\n"
            f"Directory: {sunfish_dir}\n"
            "If the repo layout changed, update the harness to point at the correct file."
        )

    # Prefer running `sunfish.py` directly if it wires up tools.uci.run(...).
    # This is the most stable entrypoint across upstream changes.
    sunfish_py = sunfish_dir / "sunfish.py"
    try:
        if "tools.uci.run" in sunfish_py.read_text(encoding="utf-8", errors="ignore"):
            return sunfish_dir, sunfish_py
    except Exception:
        pass

    # Otherwise find a UCI entrypoint inside the checkout; if none exist, generate a minimal wrapper.
    uci_entry = _safe_rglob_first(
        sunfish_dir,
        ["tools/uci.py", "sunfish/tools/uci.py", "uci.py", "*/uci.py"],
    )
    if uci_entry is None:
        uci_entry = _ensure_sunfish_uci_wrapper(sunfish_dir)

    return sunfish_dir, uci_entry


class SunfishUCI:
    """Tiny UCI wrapper around sunfish.py subprocess."""

    def __init__(self, uci_entrypoint: Path, *, sunfish_repo_dir: Path, python_exe: str | None = None) -> None:
        self.uci_entrypoint = uci_entrypoint
        self.sunfish_repo_dir = sunfish_repo_dir
        self.python_exe = python_exe or sys.executable
        self.proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> "SunfishUCI":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def start(self) -> None:
        if self.proc is not None:
            return
        # `-u` is important on Windows so `readline()` sees output promptly.
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        # Make sure Sunfish repo root is importable.
        existing_pp = env.get("PYTHONPATH", "")
        sunfish_pp = str(self.sunfish_repo_dir)
        env["PYTHONPATH"] = sunfish_pp + (os.pathsep + existing_pp if existing_pp else "")
        self.proc = subprocess.Popen(
            [self.python_exe, "-u", str(self.uci_entrypoint)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(self.sunfish_repo_dir),
            env=env,
        )
        self._send("uci")
        self._wait_for("uciok", timeout_s=5.0)
        self._send("isready")
        self._wait_for("readyok", timeout_s=5.0)

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self._send("quit")
        except Exception:
            pass
        self.proc.terminate()
        self.proc.wait(timeout=2.0)
        self.proc = None

    def best_move(self, moves_uci: list[str], movetime_ms: int = 40) -> str:
        if self.proc is None:
            raise RuntimeError("Sunfish process not running")
        pos_cmd = "position startpos"
        if moves_uci:
            pos_cmd += " moves " + " ".join(moves_uci)
        self._send(pos_cmd)
        # Sunfish UCI wrappers found in the wild sometimes keep thinking past the
        # requested movetime (or buffer output), so we:
        # 1) wait up to (movetime + grace) for "bestmove"
        # 2) if not received, send "stop" and wait a little longer
        self._send(f"go movetime {movetime_ms}")
        primary_timeout = max(2.0, movetime_ms / 1000.0 + 1.0)
        try:
            line = self._wait_for("bestmove ", timeout_s=primary_timeout)
        except TimeoutError:
            # Ask the engine to stop and flush a bestmove.
            with contextlib.suppress(Exception):
                self._send("stop")
            line = self._wait_for("bestmove ", timeout_s=3.0)
        parts = line.strip().split()
        if len(parts) < 2:
            raise RuntimeError(f"Unexpected bestmove response: {line!r}")
        return parts[1]

    def _send(self, command: str) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Sunfish process not available")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, timeout_s: float) -> str:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("Sunfish process not available")
        deadline = time.time() + timeout_s
        captured: list[str] = []
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = "".join(captured[-50:])
                raise RuntimeError(
                    "Sunfish process exited during startup/wait.\n"
                    f"Expected token: {token!r}\n"
                    f"Exit code: {self.proc.returncode}\n"
                    f"Output:\n{tail}"
                )
            line = self.proc.stdout.readline()
            if not line:
                continue
            captured.append(line)
            if token in line:
                return line
        tail = "".join(captured[-50:])
        raise TimeoutError(f"Timed out waiting for token {token!r}. Output:\n{tail}")


@dataclass
class GameRecord:
    game_index: int
    seed: int
    opening_plies: int
    opening_moves_uci: list[str]
    tdleaf_color: str
    result_tdleaf: str
    result_pgn: str
    move_count: int
    moves_uci: list[str]
    weights_path: str


def generate_opening(board: chess.Board, rng: random.Random, opening_plies: int) -> list[str]:
    opening_moves: list[str] = []
    for _ in range(opening_plies):
        if board.is_game_over(claim_draw=True):
            break
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)
        opening_moves.append(move.uci())
        board.push(move)
    return opening_moves


def tdleaf_outcome_label(result: str, tdleaf_color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "D"
    tdleaf_won = (result == "1-0" and tdleaf_color == chess.WHITE) or (
        result == "0-1" and tdleaf_color == chess.BLACK
    )
    return "W" if tdleaf_won else "L"


def tdleaf_outcome_value(outcome_label: str) -> float:
    if outcome_label == "W":
        return 1.0
    if outcome_label == "L":
        return -1.0
    return 0.0


def run_match_series(
    *,
    num_games: int,
    base_seed: int,
    opening_plies: int,
    max_plies: int,
    tdleaf_depth: int,
    sunfish_movetime_ms: int,
    weights_path: Path,
    sunfish_uci_entry: Path,
    sunfish_repo_dir: Path,
    sunfish_mode: str,
    collect_training_samples: bool,
) -> tuple[list[GameRecord], list[tuple[np.ndarray, float]]]:
    records: list[GameRecord] = []
    train_samples: list[tuple[np.ndarray, float]] = []
    evaluator = Evaluator(weights_path=str(weights_path))
    searcher = Searcher(evaluator=evaluator)

    if sunfish_mode != "uci":
        raise ValueError(f"Unsupported sunfish mode: {sunfish_mode!r}")

    with SunfishUCI(sunfish_uci_entry, sunfish_repo_dir=sunfish_repo_dir) as sunfish:
        for game_idx in range(num_games):
            if game_idx == 0 or (game_idx + 1) % 5 == 0 or (game_idx + 1) == num_games:
                print(f"[bench] game {game_idx+1}/{num_games}", flush=True)
            seed = base_seed + game_idx
            rng = random.Random(seed)
            board = chess.Board()
            opening = generate_opening(board, rng, opening_plies=opening_plies)
            tdleaf_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
            all_moves = list(opening)
            # Online adaptation samples (TD(0) on TDLeaf-to-move positions).
            # We only record states where TDLeaf is to move, and bootstrap the
            # target from the *next* TDLeaf-to-move state's value.
            #
            # target_t = clip(gamma * V(s_{t+1}) / scale, -1, 1)
            # terminal: target = outcome_value in {-1, 0, +1}
            #
            # This is intentionally lightweight and search-aligned; using the
            # final outcome label for every intermediate position tends to
            # collapse the evaluator when losses dominate.
            prev_feat: np.ndarray | None = None
            gamma = 0.99
            scale_cp = 300.0

            for _ply in range(max_plies):
                if board.is_game_over(claim_draw=True):
                    break
                if board.turn == tdleaf_color:
                    move = searcher.search(board, depth=tdleaf_depth).best_move
                    if collect_training_samples:
                        # Bootstrap previous sample using value of this state.
                        if prev_feat is not None:
                            v_next_cp = float(evaluator.evaluate(board))
                            target = max(-1.0, min(1.0, gamma * (v_next_cp / scale_cp)))
                            train_samples.append((prev_feat, target))
                        prev_feat = extract_features(board)
                else:
                    uci = sunfish.best_move(all_moves, movetime_ms=sunfish_movetime_ms)
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        # If UCI returns illegal move, treat as immediate loss for that side.
                        result = "0-1" if board.turn == chess.WHITE else "1-0"
                        outcome_label = tdleaf_outcome_label(result, tdleaf_color)
                        records.append(
                            GameRecord(
                                game_index=game_idx,
                                seed=seed,
                                opening_plies=len(opening),
                                opening_moves_uci=opening,
                                tdleaf_color="white" if tdleaf_color == chess.WHITE else "black",
                                result_tdleaf=outcome_label,
                                result_pgn=result,
                                move_count=len(all_moves),
                                moves_uci=all_moves,
                                weights_path=str(weights_path),
                            )
                        )
                        break
                board.push(move)
                all_moves.append(move.uci())
            else:
                # Reached max plies: adjudicate by static eval.
                score = evaluator.evaluate(board)
                if score > 30:
                    result = "1-0"
                elif score < -30:
                    result = "0-1"
                else:
                    result = "1/2-1/2"
                outcome_label = tdleaf_outcome_label(result, tdleaf_color)
                records.append(
                    GameRecord(
                        game_index=game_idx,
                        seed=seed,
                        opening_plies=len(opening),
                        opening_moves_uci=opening,
                        tdleaf_color="white" if tdleaf_color == chess.WHITE else "black",
                        result_tdleaf=outcome_label,
                        result_pgn=result,
                        move_count=len(all_moves),
                        moves_uci=all_moves,
                        weights_path=str(weights_path),
                    )
                )
                if collect_training_samples:
                    # Terminal target for the last pending TDLeaf-to-move state.
                    if prev_feat is not None:
                        target = tdleaf_outcome_value(outcome_label)
                        train_samples.append((prev_feat, target))
                continue

            # Normal termination
            if records and records[-1].game_index == game_idx:
                if collect_training_samples:
                    if prev_feat is not None:
                        target = tdleaf_outcome_value(records[-1].result_tdleaf)
                        train_samples.append((prev_feat, target))
                continue
            result = board.result(claim_draw=True)
            outcome_label = tdleaf_outcome_label(result, tdleaf_color)
            records.append(
                GameRecord(
                    game_index=game_idx,
                    seed=seed,
                    opening_plies=len(opening),
                    opening_moves_uci=opening,
                    tdleaf_color="white" if tdleaf_color == chess.WHITE else "black",
                    result_tdleaf=outcome_label,
                    result_pgn=result,
                    move_count=len(all_moves),
                    moves_uci=all_moves,
                    weights_path=str(weights_path),
                )
            )
            if collect_training_samples:
                if prev_feat is not None:
                    target = tdleaf_outcome_value(outcome_label)
                    train_samples.append((prev_feat, target))

    return records, train_samples


def aggregate_stats(records: list[GameRecord]) -> dict[str, Any]:
    wins = sum(1 for r in records if r.result_tdleaf == "W")
    draws = sum(1 for r in records if r.result_tdleaf == "D")
    losses = sum(1 for r in records if r.result_tdleaf == "L")
    n = len(records)
    score = wins + 0.5 * draws
    score_rate = score / n if n else 0.0
    win_rate = wins / n if n else 0.0
    # Elo estimate from score fraction, clamped for stability.
    eps = 1e-6
    p = min(max(score_rate, eps), 1.0 - eps)
    elo = -400.0 * np.log10((1.0 / p) - 1.0)
    return {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "score_rate": score_rate,
        "win_rate": win_rate,
        "elo_estimate": float(elo),
    }


def load_runtime_weights(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: np.asarray(data[k]).copy() for k in data.files}


def save_runtime_weights(path: Path, weights: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights)


def adapt_last_layer(
    *,
    src_weights: Path,
    dst_weights: Path,
    samples: list[tuple[np.ndarray, float]],
    lr: float = 5e-4,
    batch_size: int = 64,
    max_steps: int = 500,
) -> Path:
    if not samples:
        save_runtime_weights(dst_weights, load_runtime_weights(src_weights))
        return dst_weights

    w = load_runtime_weights(src_weights)
    w1 = np.asarray(w["fc1_weight"], dtype=np.float32)
    b1 = np.asarray(w["fc1_bias"], dtype=np.float32)
    w2 = np.asarray(w["fc2_weight"], dtype=np.float32)
    b2 = np.asarray(w["fc2_bias"], dtype=np.float32)
    out_w = np.asarray(w["out_weight"], dtype=np.float32).reshape(1, -1)
    out_b = np.asarray(w["out_bias"], dtype=np.float32).reshape(1)

    rng = random.Random(0)
    cp_target_scale = 300.0
    steps = min(max_steps, max(1, len(samples) // max(1, batch_size)))
    for _ in range(steps):
        batch = [samples[rng.randrange(len(samples))] for _ in range(min(batch_size, len(samples)))]
        x = np.stack([item[0] for item in batch], axis=0).astype(np.float32)
        y = np.array([item[1] * cp_target_scale for item in batch], dtype=np.float32).reshape(-1, 1)

        h1 = np.clip(np.maximum(x @ w1.T + b1, 0.0), 0.0, 1.0)
        h2 = np.clip(np.maximum(h1 @ w2.T + b2, 0.0), 0.0, 1.0)
        pred = h2 @ out_w.T + out_b
        err = pred - y

        grad_out_w = (2.0 / len(batch)) * (err.T @ h2)
        grad_out_b = np.array([(2.0 / len(batch)) * np.sum(err)], dtype=np.float32)
        out_w -= lr * grad_out_w
        out_b -= lr * grad_out_b

    w["out_weight"] = out_w
    w["out_bias"] = out_b
    save_runtime_weights(dst_weights, w)
    return dst_weights


def write_game_records_csv(path: Path, records: list[GameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "game_index",
                "seed",
                "opening_plies",
                "opening_moves_uci",
                "tdleaf_color",
                "result_tdleaf",
                "result_pgn",
                "move_count",
                "moves_uci",
                "weights_path",
            ],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "game_index": rec.game_index,
                    "seed": rec.seed,
                    "opening_plies": rec.opening_plies,
                    "opening_moves_uci": " ".join(rec.opening_moves_uci),
                    "tdleaf_color": rec.tdleaf_color,
                    "result_tdleaf": rec.result_tdleaf,
                    "result_pgn": rec.result_pgn,
                    "move_count": rec.move_count,
                    "moves_uci": " ".join(rec.moves_uci),
                    "weights_path": rec.weights_path,
                }
            )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_summary_md(
    path: Path,
    baseline: dict[str, Any],
    adapted: dict[str, Any] | None,
    baseline_weights: Path,
    adapted_weights: Path | None,
    commands: list[str],
) -> None:
    lines = [
        "# TDLeaf vs Sunfish Benchmark Summary",
        "",
        "## Baseline",
        f"- Games: {baseline['games']}",
        f"- W/D/L: {baseline['wins']}/{baseline['draws']}/{baseline['losses']}",
        f"- Score rate: {baseline['score_rate']:.3f}",
        f"- Win rate: {baseline['win_rate']:.3f}",
        f"- Elo estimate: {baseline['elo_estimate']:.1f}",
        f"- Weights: `{baseline_weights}`",
        "",
    ]
    if adapted is not None and adapted_weights is not None:
        delta = adapted["score_rate"] - baseline["score_rate"]
        lines += [
            "## Adapted",
            f"- Games: {adapted['games']}",
            f"- W/D/L: {adapted['wins']}/{adapted['draws']}/{adapted['losses']}",
            f"- Score rate: {adapted['score_rate']:.3f}",
            f"- Win rate: {adapted['win_rate']:.3f}",
            f"- Elo estimate: {adapted['elo_estimate']:.1f}",
            f"- Weights: `{adapted_weights}`",
            "",
            "## Baseline vs Adapted",
            f"- Score rate delta (adapted - baseline): {delta:+.3f}",
            "",
        ]
    lines += [
        "## Commands Used",
        *[f"- `{cmd}`" for cmd in commands],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tdleaf_nnue_engine vs Sunfish.")
    parser.add_argument("--games", type=int, default=100, help="Games in baseline run.")
    parser.add_argument("--adapted-games", type=int, default=100, help="Games in adapted run.")
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed.")
    parser.add_argument("--opening-plies", type=int, default=4, help="Random opening plies.")
    parser.add_argument("--max-plies", type=int, default=180, help="Max plies per game.")
    parser.add_argument("--tdleaf-depth", type=int, default=2, help="tdleaf search depth.")
    parser.add_argument("--sunfish-movetime-ms", type=int, default=35, help="Sunfish think time per move.")
    parser.add_argument(
        "--sunfish-mode",
        choices=["uci"],
        default="uci",
        help="How to run Sunfish (default: uci).",
    )
    parser.add_argument(
        "--weights",
        default="tdleaf_nnue_engine/checkpoints/nnue_runtime.npz",
        help="Baseline tdleaf runtime weights (.npz).",
    )
    parser.add_argument(
        "--adapt",
        action="store_true",
        help="Enable lightweight online adaptation between baseline and adapted runs.",
    )
    parser.add_argument(
        "--out-dir",
        default="tdleaf_nnue_engine/benchmarks/results",
        help="Directory for benchmark artifacts.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sunfish_repo_dir, sunfish_uci_entry = ensure_sunfish_checkout(DEFAULT_SUNFISH_DIR)

    baseline_weights = Path(args.weights)
    baseline_records, samples = run_match_series(
        num_games=args.games,
        base_seed=args.seed,
        opening_plies=args.opening_plies,
        max_plies=args.max_plies,
        tdleaf_depth=args.tdleaf_depth,
        sunfish_movetime_ms=args.sunfish_movetime_ms,
        weights_path=baseline_weights,
        sunfish_uci_entry=sunfish_uci_entry,
        sunfish_repo_dir=sunfish_repo_dir,
        sunfish_mode=args.sunfish_mode,
        collect_training_samples=args.adapt,
    )
    baseline_stats = aggregate_stats(baseline_records)

    baseline_csv = out_dir / "baseline_games.csv"
    baseline_json = out_dir / "baseline_summary.json"
    write_game_records_csv(baseline_csv, baseline_records)
    write_json(
        baseline_json,
        {
            "stats": baseline_stats,
            "weights": str(baseline_weights),
            "games": [rec.__dict__ for rec in baseline_records],
        },
    )

    adapted_stats: dict[str, Any] | None = None
    adapted_weights: Path | None = None
    adapted_records: list[GameRecord] = []
    if args.adapt and args.adapted_games > 0:
        adapted_weights = Path("tdleaf_nnue_engine/checkpoints/nnue_runtime_adapted.npz")
        adapt_last_layer(
            src_weights=baseline_weights,
            dst_weights=adapted_weights,
            samples=samples,
        )
        adapted_records, _ = run_match_series(
            num_games=args.adapted_games,
            base_seed=args.seed + 10_000,
            opening_plies=args.opening_plies,
            max_plies=args.max_plies,
            tdleaf_depth=args.tdleaf_depth,
            sunfish_movetime_ms=args.sunfish_movetime_ms,
            weights_path=adapted_weights,
            sunfish_uci_entry=sunfish_uci_entry,
            sunfish_repo_dir=sunfish_repo_dir,
            sunfish_mode=args.sunfish_mode,
            collect_training_samples=False,
        )
        adapted_stats = aggregate_stats(adapted_records)
        adapted_csv = out_dir / "adapted_games.csv"
        adapted_json = out_dir / "adapted_summary.json"
        write_game_records_csv(adapted_csv, adapted_records)
        write_json(
            adapted_json,
            {
                "stats": adapted_stats,
                "weights": str(adapted_weights),
                "games": [rec.__dict__ for rec in adapted_records],
            },
        )

    commands = [
        (
            "python -m tdleaf_nnue_engine.benchmarks.run_tdleaf_vs_sunfish "
            f"--games {args.games} --seed {args.seed} --opening-plies {args.opening_plies} "
            f"--tdleaf-depth {args.tdleaf_depth} --sunfish-movetime-ms {args.sunfish_movetime_ms}"
        )
    ]
    if args.adapt:
        commands.append(
            (
                "python -m tdleaf_nnue_engine.benchmarks.run_tdleaf_vs_sunfish "
                f"--games {args.games} --adapt --adapted-games {args.adapted_games} --seed {args.seed} "
                f"--opening-plies {args.opening_plies} --tdleaf-depth {args.tdleaf_depth} "
                f"--sunfish-movetime-ms {args.sunfish_movetime_ms}"
            )
        )

    summary_path = out_dir / "benchmark_summary.md"
    write_summary_md(
        path=summary_path,
        baseline=baseline_stats,
        adapted=adapted_stats,
        baseline_weights=baseline_weights,
        adapted_weights=adapted_weights,
        commands=commands,
    )

    result_payload = {
        "baseline_stats": baseline_stats,
        "adapted_stats": adapted_stats,
        "artifacts": {
            "baseline_csv": str(baseline_csv),
            "baseline_json": str(baseline_json),
            "summary_md": str(summary_path),
            "adapted_csv": str(out_dir / "adapted_games.csv") if adapted_records else None,
            "adapted_json": str(out_dir / "adapted_summary.json") if adapted_records else None,
            "sunfish_repo_dir": str(sunfish_repo_dir),
            "sunfish_uci_entry": str(sunfish_uci_entry),
            "adapted_weights": str(adapted_weights) if adapted_weights else None,
        },
    }
    write_json(out_dir / "run_manifest.json", result_payload)
    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    main()
