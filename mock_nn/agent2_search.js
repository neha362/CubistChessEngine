/**
 * ============================================================
 * AGENT 2 — SEARCH AGENT
 * Neural-Inspired Chess Engine · Hackathon Edition
 * ============================================================
 *
 * Responsibilities:
 *  - Iterative deepening alpha-beta search
 *  - Transposition table (Zobrist-keyed)
 *  - Move ordering (MVV-LVA, killer moves, history heuristic)
 *  - Quiescence search to avoid horizon effect
 *  - Stub interface to Eval Agent (swap in real eval freely)
 *
 * Interface:
 *  search(state, opts) → { move, score, depth, nodes, pv }
 *
 * opts:
 *  {
 *    maxDepth:  number   (default 4)
 *    timeLimitMs: number (default 2000)
 *    evalFn:    fn(state) → number   (centipawns, from side-to-move POV)
 *    onDepth:   fn({ depth, score, move, nodes, pv }) — called each IID step
 *  }
 *
 * The evalFn stub defaults to a simple material count so this agent
 * can be tested completely independently of Agent 3.
 * ============================================================
 */

'use strict';

const { getMoves, isInCheck, applyMove } = require('./agent1_movegen');

// ─── Piece values (for stubs and move ordering) ──────────────

const PIECE_VAL = { P: 100, N: 320, B: 330, R: 500, Q: 900, K: 20000 };

// ─── Default eval stub (material only) ───────────────────────
// Replace this with Agent 3's evalFn in production.

function stubEval(state) {
  let score = 0;
  for (const p of state.board) {
    if (!p) continue;
    const val = PIECE_VAL[p[1]] ?? 0;
    score += p[0] === state.side ? val : -val;
  }
  return score;
}

// ─── Zobrist hashing for TT ──────────────────────────────────
// Pre-generate random 32-bit keys for each (piece, square) pair + side.

const ZOBRIST = (() => {
  const rand = () => Math.floor(Math.random() * 2 ** 32);
  const pieces = ['wP','wN','wB','wR','wQ','wK','bP','bN','bB','bR','bQ','bK'];
  const table  = {};
  for (const p of pieces) {
    table[p] = Array.from({ length: 64 }, rand);
  }
  return {
    table,
    sideToMove: rand(),
  };
})();

function zobristHash(board, side) {
  let h = 0;
  for (let s = 0; s < 64; s++) {
    if (board[s]) h ^= ZOBRIST.table[board[s]][s];
  }
  if (side === 'b') h ^= ZOBRIST.sideToMove;
  return h;
}

// ─── Transposition Table ─────────────────────────────────────

const TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2;
const TT_SIZE  = 1 << 20; // ~1M entries

class TranspositionTable {
  constructor() { this.table = new Array(TT_SIZE); }

  index(hash) { return hash % TT_SIZE; }

  store(hash, depth, score, flag, move) {
    const idx = this.index(hash);
    const entry = this.table[idx];
    // Replace if empty, deeper, or same depth (age replacement)
    if (!entry || entry.depth <= depth) {
      this.table[idx] = { hash, depth, score, flag, move };
    }
  }

  probe(hash, depth, alpha, beta) {
    const entry = this.table[this.index(hash)];
    if (!entry || entry.hash !== hash || entry.depth < depth) return null;
    if (entry.flag === TT_EXACT) return entry;
    if (entry.flag === TT_LOWER && entry.score >= beta)  return entry;
    if (entry.flag === TT_UPPER && entry.score <= alpha) return entry;
    return null; // bounds don't cut but we can still use the move
  }

  getBestMove(hash) {
    const entry = this.table[this.index(hash)];
    return (entry && entry.hash === hash) ? entry.move : null;
  }

  clear() { this.table = new Array(TT_SIZE); }
}

// ─── Killer move table ───────────────────────────────────────
// Stores 2 non-capture moves that caused a beta cutoff at each depth.

class KillerTable {
  constructor(maxDepth) {
    this.killers = Array.from({ length: maxDepth + 1 }, () => [null, null]);
  }
  store(depth, move) {
    if (!move.promo && !this.board?.[move.to]) {
      const [k0] = this.killers[depth];
      if (!movesEqual(k0, move)) {
        this.killers[depth][1] = k0;
        this.killers[depth][0] = move;
      }
    }
  }
  isKiller(depth, move) {
    return this.killers[depth].some(k => k && movesEqual(k, move));
  }
}

function movesEqual(a, b) {
  return a && b && a.from === b.from && a.to === b.to && a.promo === b.promo;
}

// ─── History heuristic ───────────────────────────────────────
// Tracks how often a quiet move caused a cutoff.

class HistoryTable {
  constructor() { this.table = {}; }
  key(move) { return `${move.from}-${move.to}`; }
  update(move, depth) {
    const k = this.key(move);
    this.table[k] = (this.table[k] ?? 0) + depth * depth;
  }
  score(move) { return this.table[this.key(move)] ?? 0; }
}

// ─── Move ordering ────────────────────────────────────────────

/**
 * Score each move for ordering. Higher score → searched first.
 *
 * Priority (descending):
 *  1. TT best move
 *  2. Winning captures (MVV-LVA)
 *  3. Promotions
 *  4. Killer moves
 *  5. History heuristic
 *  6. Quiet moves
 */
function scoreMoves(board, moves, ttMove, depth, killers, history) {
  return moves.map(move => {
    let score = 0;

    if (movesEqual(move, ttMove)) {
      score = 2_000_000;
    } else if (board[move.to]) {
      // MVV-LVA: Most Valuable Victim, Least Valuable Attacker
      const victim   = PIECE_VAL[board[move.to][1]] ?? 0;
      const attacker = PIECE_VAL[board[move.from][1]] ?? 0;
      score = 1_000_000 + victim * 10 - attacker;
    } else if (move.promo) {
      score = 900_000 + (PIECE_VAL[move.promo[1]] ?? 0);
    } else if (move.castle) {
      score = 500_000;
    } else if (killers?.isKiller(depth, move)) {
      score = 800_000;
    } else {
      score = history?.score(move) ?? 0;
    }

    return { ...move, _score: score };
  }).sort((a, b) => b._score - a._score);
}

// ─── Quiescence search ────────────────────────────────────────
// Extend search on captures/checks to avoid horizon effect.

function quiesce(state, alpha, beta, evalFn, stats) {
  stats.nodes++;

  const standPat = evalFn(state);
  if (standPat >= beta) return beta;
  if (standPat > alpha) alpha = standPat;

  const moves  = getMoves(state);
  const inCheck = isInCheck(state.board, state.side);

  // In check: search all moves to escape
  // Not in check: only look at captures + promotions
  const candidates = inCheck
    ? moves
    : moves.filter(m => state.board[m.to] || m.promo || m.epCapture);

  for (const move of candidates) {
    const child = applyMove(state, move);
    const score = -quiesce(child, -beta, -alpha, evalFn, stats);
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }

  return alpha;
}

// ─── Alpha-beta negamax ───────────────────────────────────────

function alphaBeta(state, depth, alpha, beta, tt, killers, history, evalFn, stats) {
  stats.nodes++;

  const hash   = zobristHash(state.board, state.side);
  const ttHit  = tt.probe(hash, depth, alpha, beta);
  if (ttHit?.flag === TT_EXACT) return { score: ttHit.score, move: ttHit.move };

  const moves   = getMoves(state);
  const inCheck = isInCheck(state.board, state.side);

  // Terminal nodes
  if (moves.length === 0) {
    return { score: inCheck ? -20000 + stats.nodes : 0, move: null };
  }
  if (depth === 0) {
    const score = quiesce(state, alpha, beta, evalFn, stats);
    return { score, move: null };
  }

  const ttMove  = tt.getBestMove(hash);
  const ordered = scoreMoves(state.board, moves, ttMove, depth, killers, history);

  let   best     = { score: -Infinity, move: null };
  let   origAlpha = alpha;

  for (const move of ordered) {
    const child = applyMove(state, move);
    const { score: raw } = alphaBeta(
      child, depth - 1, -beta, -alpha,
      tt, killers, history, evalFn, stats
    );
    const score = -raw;

    if (score > best.score) { best = { score, move }; }
    if (score > alpha)       { alpha = score; }

    if (alpha >= beta) {
      // Beta cutoff — update killers and history for quiet moves
      if (!state.board[move.to] && !move.promo) {
        killers.store(depth, move);
        history.update(move, depth);
      }
      break;
    }
  }

  // Store in TT
  const flag = best.score <= origAlpha ? TT_UPPER
             : best.score >= beta      ? TT_LOWER
             : TT_EXACT;
  tt.store(hash, depth, best.score, flag, best.move);

  return best;
}

// ─── Iterative deepening ──────────────────────────────────────

/**
 * Main search entry point.
 *
 * @param {object} state      Chess state from Agent 1
 * @param {object} opts
 * @param {number}   opts.maxDepth      Max search depth (default 4)
 * @param {number}   opts.timeLimitMs   Time limit in ms (default 2000)
 * @param {Function} opts.evalFn        fn(state) → centipawns (default: stub)
 * @param {Function} opts.onDepth       Callback per IID depth
 * @returns {{ move, score, depth, nodes, pv }}
 */
function search(state, opts = {}) {
  const {
    maxDepth    = 4,
    timeLimitMs = 2000,
    evalFn      = stubEval,
    onDepth     = null,
  } = opts;

  const tt      = new TranspositionTable();
  const killers = new KillerTable(maxDepth + 2);
  const history = new HistoryTable();
  const stats   = { nodes: 0 };
  const start   = Date.now();

  let bestResult = { move: null, score: 0, depth: 0, nodes: 0, pv: [] };

  for (let d = 1; d <= maxDepth; d++) {
    if (Date.now() - start > timeLimitMs) break;

    const result = alphaBeta(
      state, d, -Infinity, Infinity,
      tt, killers, history, evalFn, stats
    );

    if (result.move) {
      bestResult = {
        move:  result.move,
        score: result.score,
        depth: d,
        nodes: stats.nodes,
        pv:    extractPV(state, tt, d),
      };
    }

    if (onDepth) onDepth({ ...bestResult, elapsedMs: Date.now() - start });

    // Mate found — no point searching deeper
    if (Math.abs(result.score) > 15000) break;
  }

  return bestResult;
}

// ─── Principal Variation extraction ─────────────────────────

function extractPV(state, tt, maxDepth) {
  const pv   = [];
  let   cur  = state;
  const seen = new Set();

  for (let d = 0; d < maxDepth; d++) {
    const hash = zobristHash(cur.board, cur.side);
    if (seen.has(hash)) break; // repetition guard
    seen.add(hash);

    const move = tt.getBestMove(hash);
    if (!move) break;

    pv.push(move);
    cur = applyMove(cur, move);
  }
  return pv;
}

// ─── Move → algebraic notation ───────────────────────────────

function moveToAlg(move) {
  const files = 'abcdefgh';
  const from  = files[move.from % 8] + (8 - Math.floor(move.from / 8));
  const to    = files[move.to   % 8] + (8 - Math.floor(move.to   / 8));
  return from + to + (move.promo ? move.promo[1].toLowerCase() : '');
}

// ─── Exports ─────────────────────────────────────────────────

module.exports = {
  search,
  stubEval,
  moveToAlg,
  zobristHash,
};

// ─── Self-test (node agent2_search.js) ───────────────────────

if (require.main === module) {
  const { makeStartState } = require('./agent1_movegen');

  console.log('=== Agent 2: Search self-test (stub eval) ===');
  const state = makeStartState();

  const result = search(state, {
    maxDepth: 4,
    timeLimitMs: 5000,
    evalFn: stubEval,
    onDepth: ({ depth, score, move, nodes, elapsedMs }) => {
      console.log(
        `  depth ${depth}: ${move ? moveToAlg(move) : '(none)'}` +
        `  score=${score > 0 ? '+' : ''}${score}cp` +
        `  nodes=${nodes}  time=${elapsedMs}ms`
      );
    },
  });

  console.log('\nBest move:', moveToAlg(result.move));
  console.log('Score:    ', result.score, 'cp');
  console.log('Depth:    ', result.depth);
  console.log('Nodes:    ', result.nodes);
  console.log('PV:       ', result.pv.map(moveToAlg).join(' '));
}
