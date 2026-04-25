"""
Iterative online adaptation loop vs Sunfish.

Runs N iterations of:
  1) play K games vs Sunfish (collecting training samples)
  2) write per-iteration artifacts (CSV + JSON summary)
  3) adapt runtime weights (last-layer) from those samples
  4) feed adapted weights into the next iteration

Example:
  python -m tdleaf_nnue_engine.benchmarks.iterative_adapt_loop --iters 10 --games-per-iter 50
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from .run_tdleaf_vs_sunfish import (
    DEFAULT_SUNFISH_DIR,
    adapt_last_layer,
    aggregate_stats,
    ensure_sunfish_checkout,
    run_match_series,
    write_game_records_csv,
    write_json,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Iteratively adapt tdleaf runtime weights vs Sunfish.")
    p.add_argument("--iters", type=int, default=10, help="Number of iterations (loops).")
    p.add_argument("--games-per-iter", type=int, default=50, help="Games per iteration.")
    p.add_argument("--seed", type=int, default=1234, help="Base RNG seed (iteration i uses seed + i*10000).")
    p.add_argument("--opening-plies", type=int, default=4, help="Random opening plies.")
    p.add_argument("--max-plies", type=int, default=180, help="Max plies per game.")
    p.add_argument("--tdleaf-depth", type=int, default=2, help="tdleaf search depth.")
    p.add_argument("--sunfish-movetime-ms", type=int, default=35, help="Sunfish think time per move.")
    p.add_argument(
        "--weights",
        default="tdleaf_nnue_engine/checkpoints/nnue_runtime.npz",
        help="Initial tdleaf runtime weights (.npz).",
    )
    p.add_argument(
        "--out-dir",
        default="tdleaf_nnue_engine/benchmarks/iterative_results",
        help="Base directory for per-iteration artifacts.",
    )
    p.add_argument(
        "--final-weights",
        default="tdleaf_nnue_engine/checkpoints/nnue_runtime_final.npz",
        help="Where to write the final weights after all iterations.",
    )
    args = p.parse_args()

    if args.iters < 1:
        raise SystemExit("--iters must be >= 1")
    if args.games_per_iter < 1:
        raise SystemExit("--games-per-iter must be >= 1")

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    sunfish_repo_dir, sunfish_uci_entry = ensure_sunfish_checkout(DEFAULT_SUNFISH_DIR)

    current_weights = Path(args.weights)
    if not current_weights.exists():
        raise SystemExit(f"weights file not found: {current_weights}")

    iter_weights_dir = Path("tdleaf_nnue_engine/checkpoints/iterative")
    iter_weights_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "iters": args.iters,
        "games_per_iter": args.games_per_iter,
        "opening_plies": args.opening_plies,
        "max_plies": args.max_plies,
        "tdleaf_depth": args.tdleaf_depth,
        "sunfish_movetime_ms": args.sunfish_movetime_ms,
        "sunfish_repo_dir": str(sunfish_repo_dir),
        "sunfish_uci_entry": str(sunfish_uci_entry),
        "start_weights": str(current_weights),
        "iterations": [],
    }

    t0_all = time.time()
    print(
        f"[tdleaf] iterative adapt loop: iters={args.iters} games_per_iter={args.games_per_iter} "
        f"opening_plies={args.opening_plies} tdleaf_depth={args.tdleaf_depth} "
        f"sunfish_movetime_ms={args.sunfish_movetime_ms}",
        flush=True,
    )
    print(f"[tdleaf] start weights: {current_weights}", flush=True)

    for i in range(args.iters):
        iter_start = time.time()
        iter_dir = out_base / f"iter_{i+1:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        base_seed = int(args.seed) + i * 10_000
        print(
            f"\n[tdleaf] iter {i+1}/{args.iters}  seed={base_seed}  weights={current_weights}",
            flush=True,
        )
        print(f"[tdleaf] playing {args.games_per_iter} games vs sunfish…", flush=True)
        records, samples = run_match_series(
            num_games=int(args.games_per_iter),
            base_seed=base_seed,
            opening_plies=int(args.opening_plies),
            max_plies=int(args.max_plies),
            tdleaf_depth=int(args.tdleaf_depth),
            sunfish_movetime_ms=int(args.sunfish_movetime_ms),
            weights_path=current_weights,
            sunfish_uci_entry=sunfish_uci_entry,
            sunfish_repo_dir=sunfish_repo_dir,
            sunfish_mode="uci",
            collect_training_samples=True,
        )

        stats = aggregate_stats(records)
        games_csv = iter_dir / "games.csv"
        summary_json = iter_dir / "summary.json"
        write_game_records_csv(games_csv, records)
        write_json(
            summary_json,
            {"stats": stats, "weights": str(current_weights), "games": [r.__dict__ for r in records]},
        )
        print(
            f"[tdleaf] iter {i+1} results: W/D/L={stats['wins']}/{stats['draws']}/{stats['losses']} "
            f"score_rate={stats['score_rate']:.3f} elo_est={stats['elo_estimate']:.1f}",
            flush=True,
        )
        print(f"[tdleaf] wrote: {games_csv}", flush=True)
        print(f"[tdleaf] wrote: {summary_json}", flush=True)
        print(f"[tdleaf] adapting last layer from {len(samples)} samples…", flush=True)

        next_weights = iter_weights_dir / f"nnue_runtime_iter_{i+1:02d}.npz"
        adapt_last_layer(src_weights=current_weights, dst_weights=next_weights, samples=samples)
        print(f"[tdleaf] wrote: {next_weights}", flush=True)

        manifest["iterations"].append(
            {
                "iter": i + 1,
                "seed": base_seed,
                "weights_in": str(current_weights),
                "weights_out": str(next_weights),
                "stats": stats,
                "games_csv": str(games_csv),
                "summary_json": str(summary_json),
            }
        )

        current_weights = next_weights
        print(f"[tdleaf] iter {i+1} done in {(time.time() - iter_start):.1f}s", flush=True)

    final_path = Path(args.final_weights)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(current_weights, final_path)
    manifest["final_weights"] = str(final_path)

    write_json(out_base / "manifest.json", manifest)  # single index of all runs
    print(f"\n[tdleaf] wrote manifest: {out_base / 'manifest.json'}", flush=True)
    print(f"[tdleaf] final weights: {final_path}", flush=True)
    print(f"[tdleaf] total elapsed: {(time.time() - t0_all)/60.0:.2f} min", flush=True)


if __name__ == "__main__":
    main()

