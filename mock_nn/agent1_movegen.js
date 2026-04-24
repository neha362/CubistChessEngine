/**
 * ============================================================
 * AGENT 1 — MOVE GENERATOR
 * Neural-Inspired Chess Engine · Hackathon Edition
 * ============================================================
 *
 * Responsibilities:
 *  - Generate pseudo-legal moves for all piece types
 *  - Filter to strictly legal moves (no leaving king in check)
 *  - Handle all edge cases: en passant, castling, promotions, pins
 *
 * Interface:
 *  getMoves(state)  → Move[]          (legal moves for side to move)
 *  isInCheck(state) → boolean
 *  applyMove(state, move) → State     (used internally + by search)
 *
 * Move shape:
 *  { from: 0-63, to: 0-63, promo?: 'wQ'|'wR'|...,
 *    castle?: 'wK'|'wQ'|'bK'|'bQ',
 *    epCapture?: true, ep2?: true }
 *
 * State shape:
 *  { board: string[64], side: 'w'|'b',
 *    castling: { wK, wQ, bK, bQ },
 *    ep: number|null }
 * ============================================================
 */

'use strict';

// ─── Helpers ────────────────────────────────────────────────

const rank = sq => Math.floor(sq / 8);
const file = sq => sq % 8;
const sq   = (r, f) => r * 8 + f;
const opp  = side => side === 'w' ? 'b' : 'w';

const inBounds = (r, f) => r >= 0 && r < 8 && f >= 0 && f < 8;

// ─── Knight / King delta tables ─────────────────────────────

const KNIGHT_DELTAS = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
const KING_DELTAS   = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
const BISHOP_RAYS   = [[-1,-1],[-1,1],[1,-1],[1,1]];
const ROOK_RAYS     = [[-1,0],[1,0],[0,-1],[0,1]];
const QUEEN_RAYS    = [...BISHOP_RAYS, ...ROOK_RAYS];

const SLIDING = new Set(['B','R','Q']);

// ─── Core: pseudo-legal move generation ─────────────────────

/**
 * Generate all pseudo-legal moves for `side`.
 * Pseudo-legal = correct piece movement, ignores own-king safety.
 *
 * @param {string[]} board  64-element array, e.g. 'wP', 'bK', null
 * @param {string}   side   'w' or 'b'
 * @param {number|null} ep  en-passant target square (or null)
 * @param {{ wK,wQ,bK,bQ }} castling  castling rights
 * @returns {object[]} array of move objects
 */
function generatePseudoLegal(board, side, ep, castling) {
  const moves = [];

  for (let s = 0; s < 64; s++) {
    const piece = board[s];
    if (!piece || piece[0] !== side) continue;

    const pt = piece[1];
    const r  = rank(s);
    const f  = file(s);

    switch (pt) {
      case 'P': generatePawnMoves(board, s, r, f, side, ep, moves); break;
      case 'N': generateLeaperMoves(board, s, r, f, side, KNIGHT_DELTAS, moves); break;
      case 'K': generateKingMoves(board, s, r, f, side, castling, moves); break;
      default:
        if (SLIDING.has(pt)) {
          const rays = pt === 'B' ? BISHOP_RAYS
                     : pt === 'R' ? ROOK_RAYS
                     : QUEEN_RAYS;
          generateSlidingMoves(board, s, r, f, side, rays, moves);
        }
    }
  }

  return moves;
}

// ─── Pawn moves ─────────────────────────────────────────────

function generatePawnMoves(board, s, r, f, side, ep, moves) {
  const dir       = side === 'w' ? -1 : 1;
  const startRank = side === 'w' ? 6 : 1;
  const promRank  = side === 'w' ? 0 : 7;

  // Single push
  const one = sq(r + dir, f);
  if (r + dir >= 0 && r + dir < 8 && !board[one]) {
    if (r + dir === promRank) {
      pushPromotions(s, one, side, moves);
    } else {
      moves.push({ from: s, to: one });

      // Double push from starting rank
      if (r === startRank) {
        const two = sq(r + 2 * dir, f);
        if (!board[two]) {
          // ep2 flag: this move creates an en-passant target
          moves.push({ from: s, to: two, ep2: true });
        }
      }
    }
  }

  // Captures (diagonal)
  for (const df of [-1, 1]) {
    const nf = f + df;
    const nr = r + dir;
    if (!inBounds(nr, nf)) continue;

    const ts = sq(nr, nf);

    // Normal capture
    if (board[ts] && board[ts][0] === opp(side)) {
      if (nr === promRank) {
        pushPromotions(s, ts, side, moves);
      } else {
        moves.push({ from: s, to: ts });
      }
    }

    // En passant capture
    if (ep === ts) {
      moves.push({ from: s, to: ts, epCapture: true });
    }
  }
}

function pushPromotions(from, to, side, moves) {
  for (const pp of ['Q', 'R', 'B', 'N']) {
    moves.push({ from, to, promo: side + pp });
  }
}

// ─── Leaper (knight / king base) ────────────────────────────

function generateLeaperMoves(board, s, r, f, side, deltas, moves) {
  for (const [dr, df] of deltas) {
    const nr = r + dr, nf = f + df;
    if (!inBounds(nr, nf)) continue;
    const ts = sq(nr, nf);
    if (!board[ts] || board[ts][0] !== side) {
      moves.push({ from: s, to: ts });
    }
  }
}

// ─── Sliding pieces ─────────────────────────────────────────

function generateSlidingMoves(board, s, r, f, side, rays, moves) {
  for (const [dr, df] of rays) {
    let nr = r + dr, nf = f + df;
    while (inBounds(nr, nf)) {
      const ts = sq(nr, nf);
      if (board[ts]) {
        if (board[ts][0] !== side) moves.push({ from: s, to: ts }); // capture
        break; // ray blocked
      }
      moves.push({ from: s, to: ts });
      nr += dr; nf += df;
    }
  }
}

// ─── King moves + castling ───────────────────────────────────

function generateKingMoves(board, s, r, f, side, castling, moves) {
  generateLeaperMoves(board, s, r, f, side, KING_DELTAS, moves);

  // Castling — only generate if rights exist and path is clear.
  // Legality of not passing through check is enforced in filterLegal.
  if (side === 'w' && r === 7 && f === 4) {
    if (castling.wK && !board[61] && !board[62] && board[63] === 'wR')
      moves.push({ from: s, to: 62, castle: 'wK' });
    if (castling.wQ && !board[59] && !board[58] && !board[57] && board[56] === 'wR')
      moves.push({ from: s, to: 58, castle: 'wQ' });
  }
  if (side === 'b' && r === 0 && f === 4) {
    if (castling.bK && !board[5] && !board[6] && board[7] === 'bR')
      moves.push({ from: s, to: 6, castle: 'bK' });
    if (castling.bQ && !board[3] && !board[2] && !board[1] && board[0] === 'bR')
      moves.push({ from: s, to: 2, castle: 'bQ' });
  }
}

// ─── Apply move → new state ──────────────────────────────────

/**
 * Apply a move to a state, returning a brand-new state.
 * Pure function — does not mutate input.
 *
 * @param {object} state  { board, side, castling, ep }
 * @param {object} move
 * @returns {object} new state
 */
function applyMove(state, move) {
  const board    = [...state.board];
  const castling = { ...state.castling };
  let   ep       = null;

  const piece = board[move.from];

  // Place piece (or promotion piece) on destination
  board[move.to]   = move.promo ?? piece;
  board[move.from] = null;

  // En passant capture: remove the pawn that was captured
  if (move.epCapture) {
    const capturedRank = rank(move.to) + (piece[0] === 'w' ? 1 : -1);
    board[sq(capturedRank, file(move.to))] = null;
  }

  // Double pawn push: set en-passant target square
  if (move.ep2) {
    ep = sq(
      (rank(move.from) + rank(move.to)) / 2,
      file(move.to)
    );
  }

  // Castling: also move the rook
  if (move.castle) {
    const rookMoves = {
      wK: [63, 61], wQ: [56, 59],
      bK: [ 7,  5], bQ: [ 0,  3],
    };
    const [rookFrom, rookTo] = rookMoves[move.castle];
    board[rookTo]   = board[rookFrom];
    board[rookFrom] = null;
  }

  // Update castling rights
  if (piece === 'wK') { castling.wK = false; castling.wQ = false; }
  if (piece === 'bK') { castling.bK = false; castling.bQ = false; }
  if (move.from === 56 || move.to === 56) castling.wQ = false;
  if (move.from === 63 || move.to === 63) castling.wK = false;
  if (move.from ===  0 || move.to ===  0) castling.bQ = false;
  if (move.from ===  7 || move.to ===  7) castling.bK = false;

  return {
    board,
    side:     opp(state.side),
    castling,
    ep,
  };
}

// ─── Check detection ─────────────────────────────────────────

/**
 * Is `side`'s king currently in check?
 *
 * @param {string[]} board
 * @param {string}   side
 * @returns {boolean}
 */
function isInCheck(board, side) {
  const kingSq = board.findIndex(p => p === side + 'K');
  if (kingSq === -1) return true; // king captured — treat as check

  // Generate all opponent pseudo-legal moves and see if any attack the king
  const oppMoves = generatePseudoLegal(board, opp(side), null, {
    wK: false, wQ: false, bK: false, bQ: false,
  });
  return oppMoves.some(m => m.to === kingSq);
}

// ─── Filter pseudo-legal → legal ────────────────────────────

/**
 * Return only moves that don't leave the moving side's king in check.
 * Also validates castling doesn't pass through attacked squares.
 *
 * @param {object} state  { board, side, castling, ep }
 * @returns {object[]} legal moves
 */
function getMoves(state) {
  const { board, side, ep, castling } = state;
  const pseudo = generatePseudoLegal(board, side, ep, castling);

  return pseudo.filter(move => {
    // For castling, also check that king doesn't pass through check
    if (move.castle) {
      const passThroughSq = {
        wK: 61, wQ: 59, bK: 5, bQ: 3,
      }[move.castle];

      // King's starting square must not be in check
      if (isInCheck(board, side)) return false;

      // Passing square must not be attacked
      const passBoard = [...board];
      passBoard[passThroughSq] = side + 'K';
      passBoard[move.from]     = null;
      if (isInCheck(passBoard, side)) return false;
    }

    const newState = applyMove({ board, side, castling, ep }, move);
    return !isInCheck(newState.board, side);
  });
}

// ─── Perft (move-count correctness test) ────────────────────

/**
 * Count leaf nodes at given depth. Use to verify correctness:
 *   perft(startState, 1) === 20
 *   perft(startState, 2) === 400
 *   perft(startState, 3) === 8902
 *   perft(startState, 4) === 197281
 *   perft(startState, 5) === 4865609
 *
 * @param {object} state
 * @param {number} depth
 * @returns {number}
 */
function perft(state, depth) {
  if (depth === 0) return 1;
  const moves = getMoves(state);
  if (depth === 1) return moves.length;

  let count = 0;
  for (const move of moves) {
    count += perft(applyMove(state, move), depth - 1);
  }
  return count;
}

// ─── Starting position factory ───────────────────────────────

function makeStartState() {
  const board = Array(64).fill(null);
  const backRank = ['R','N','B','Q','K','B','N','R'];
  for (let f = 0; f < 8; f++) {
    board[f]      = 'b' + backRank[f];
    board[8  + f] = 'bP';
    board[48 + f] = 'wP';
    board[56 + f] = 'w' + backRank[f];
  }
  return {
    board,
    side:     'w',
    castling: { wK: true, wQ: true, bK: true, bQ: true },
    ep:       null,
  };
}

// ─── Exports ────────────────────────────────────────────────

module.exports = {
  getMoves,
  isInCheck,
  applyMove,
  perft,
  makeStartState,
  // Internals exposed for testing
  generatePseudoLegal,
};

// ─── Quick self-test (run directly: node agent1_movegen.js) ──

if (require.main === module) {
  const state = makeStartState();
  console.log('=== Agent 1: Move Generator self-test ===');
  console.log('Start position legal moves:', getMoves(state).length, '(expect 20)');
  console.log('White in check at start:',    isInCheck(state.board, 'w'), '(expect false)');
  console.log('Perft(1):', perft(state, 1), '(expect 20)');
  console.log('Perft(2):', perft(state, 2), '(expect 400)');
  console.log('Perft(3):', perft(state, 3), '(expect 8902)');
  // perft(4) takes ~1s, perft(5) ~30s in pure JS
  console.log('Perft(4):', perft(state, 4), '(expect 197281)');
}
