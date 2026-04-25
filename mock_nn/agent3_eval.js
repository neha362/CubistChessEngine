/**
 * ============================================================
 * AGENT 3 — EVAL AGENT (Neural-Inspired Pattern Matcher)
 * Neural-Inspired Chess Engine · Hackathon Edition
 * ============================================================
 *
 * Responsibilities:
 *  - Layer 1: Material counting (raw piece values)
 *  - Layer 2: Piece-square tables (positional bonuses)
 *  - Layer 3: Neural heuristic pattern table (the "AI" layer)
 *             Each rule is an "if position looks like X → adjust score by Y"
 *             distilling chess knowledge into a rule table without training.
 *  - Layer 4: Game-phase blending (interpolate midgame ↔ endgame PSTs)
 *
 * Interface:
 *  evaluate(state)         → number (centipawns, from side-to-move POV)
 *  scorePosition(state)    → EvalBreakdown (full detail for debugging)
 *  addPattern(pattern)     → void (extend the rule table at runtime)
 *  testPosition(fen, expected) → { pass, actual, diff }
 *
 * Standalone testable — no dependency on Agent 1 or Agent 2.
 * ============================================================
 */

'use strict';

// ─── Piece values (centipawns) ───────────────────────────────

const PIECE_VAL = {
  P: 100,
  N: 320,
  B: 330,
  R: 500,
  Q: 900,
  K: 20000,
};

// ─── Piece-square tables ──────────────────────────────────────
// Index 0 = a8 (rank 8, file a), index 63 = h1 (rank 1, file h).
// All tables are from White's perspective.
// Black's PST is obtained by mirroring vertically (see pstIndex below).

const PST_MID = {
  P: [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
  ],
  N: [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
  ],
  B: [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
  ],
  R: [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
  ],
  Q: [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
  ],
  // Midgame king: stay behind pawn shield, prefer castled position
  K: [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
  ],
};

// Endgame king PST: wants to be active and central
const PST_END_KING = [
  -50,-40,-30,-20,-20,-30,-40,-50,
  -30,-20,-10,  0,  0,-10,-20,-30,
  -30,-10, 20, 30, 30, 20,-10,-30,
  -30,-10, 30, 40, 40, 30,-10,-30,
  -30,-10, 30, 40, 40, 30,-10,-30,
  -30,-10, 20, 30, 30, 20,-10,-30,
  -30,-30,  0,  0,  0,  0,-30,-30,
  -50,-30,-30,-30,-30,-30,-30,-50,
];

// PST index for Black: mirror vertically
function pstIndex(sq, isWhite) {
  if (isWhite) return sq;
  const r = Math.floor(sq / 8);
  const f = sq % 8;
  return (7 - r) * 8 + f;
}

// ─── Game phase ───────────────────────────────────────────────
// Based on remaining non-pawn material. Full = 24, endgame = 0.

const PHASE_WEIGHTS = { N: 1, B: 1, R: 2, Q: 4 };
const MAX_PHASE     = 24;

function getPhase(board) {
  let phase = 0;
  for (const p of board) {
    if (p && PHASE_WEIGHTS[p[1]]) phase += PHASE_WEIGHTS[p[1]];
  }
  return Math.min(phase, MAX_PHASE); // 24=opening/mid, 0=endgame
}

// ─── Neural heuristic pattern table ───────────────────────────
//
// Each pattern is:
// {
//   id:      string    — unique identifier
//   label:   string    — human-readable name
//   cond:    string    — English description of what triggers it
//   bonus:   number    — centipawns added when condition is met (negative = penalty)
//   match:   fn(board, side, phase) → number   — how many times the pattern fires
//                                                 (return float 0..1 for partial match)
// }
//
// Extending: call addPattern({ id, label, cond, bonus, match }) at runtime
// to inject new rules — e.g. Claude-generated rules from the hackathon.

const PATTERN_TABLE = [

  // ── Pawn structure ──────────────────────────────────────────

  {
    id:    'doubled_pawn',
    label: 'Doubled pawn penalty',
    cond:  'Two or more own pawns on the same file',
    bonus: -30,
    match(board, side) {
      let count = 0;
      for (let f = 0; f < 8; f++) {
        let pawns = 0;
        for (let r = 0; r < 8; r++) {
          if (board[r * 8 + f] === side + 'P') pawns++;
        }
        if (pawns > 1) count += pawns - 1;
      }
      return count;
    },
  },

  {
    id:    'isolated_pawn',
    label: 'Isolated pawn penalty',
    cond:  'Own pawn with no friendly pawns on adjacent files',
    bonus: -20,
    match(board, side) {
      let count = 0;
      for (let f = 0; f < 8; f++) {
        let hasPawn = false;
        for (let r = 0; r < 8; r++) {
          if (board[r * 8 + f] === side + 'P') { hasPawn = true; break; }
        }
        if (!hasPawn) continue;
        let hasNeighbour = false;
        for (const nf of [f - 1, f + 1]) {
          if (nf < 0 || nf > 7) continue;
          for (let r = 0; r < 8; r++) {
            if (board[r * 8 + nf] === side + 'P') { hasNeighbour = true; break; }
          }
          if (hasNeighbour) break;
        }
        if (!hasNeighbour) count++;
      }
      return count;
    },
  },

  {
    id:    'passed_pawn',
    label: 'Passed pawn bonus',
    cond:  'Own pawn with no enemy pawns in front on same or adjacent files',
    bonus: 40,
    match(board, side) {
      const opp   = side === 'w' ? 'b' : 'w';
      const dir   = side === 'w' ? -1 : 1; // direction toward enemy
      let count   = 0;

      for (let s = 0; s < 64; s++) {
        if (board[s] !== side + 'P') continue;
        const r = Math.floor(s / 8);
        const f = s % 8;
        let blocked = false;

        // Check all squares in front on same + adjacent files
        let nr = r + dir;
        while (nr >= 0 && nr < 8) {
          for (const nf of [f - 1, f, f + 1]) {
            if (nf < 0 || nf > 7) continue;
            if (board[nr * 8 + nf] === opp + 'P') { blocked = true; break; }
          }
          if (blocked) break;
          nr += dir;
        }
        if (!blocked) count++;
      }
      return count;
    },
  },

  // ── Center control ──────────────────────────────────────────

  {
    id:    'center_control',
    label: 'Center control',
    cond:  'Own piece occupies e4/d4/e5/d5 (squares 27,28,35,36)',
    bonus: 15,
    match(board, side) {
      const CENTER = [27, 28, 35, 36];
      return CENTER.filter(s => board[s] && board[s][0] === side).length;
    },
  },

  {
    id:    'extended_center',
    label: 'Extended center influence',
    cond:  'Pawn or piece on c3-f3 / c6-f6 ring squares',
    bonus: 8,
    match(board, side) {
      const EXT = [18,19,20,21,26,29,34,37,42,43,44,45];
      return EXT.filter(s => board[s] && board[s][0] === side).length;
    },
  },

  // ── Piece coordination ───────────────────────────────────────

  {
    id:    'bishop_pair',
    label: 'Bishop pair bonus',
    cond:  'Both bishops still on board',
    bonus: 30,
    match(board, side) {
      const bishops = board.filter(p => p === side + 'B');
      return bishops.length >= 2 ? 1 : 0;
    },
  },

  {
    id:    'rook_open_file',
    label: 'Rook on open file',
    cond:  'Own rook on a file with no pawns of either colour',
    bonus: 25,
    match(board, side) {
      let count = 0;
      for (let f = 0; f < 8; f++) {
        let hasPawn = false;
        for (let r = 0; r < 8; r++) {
          const p = board[r * 8 + f];
          if (p && p[1] === 'P') { hasPawn = true; break; }
        }
        if (hasPawn) continue;
        for (let r = 0; r < 8; r++) {
          if (board[r * 8 + f] === side + 'R') count++;
        }
      }
      return count;
    },
  },

  {
    id:    'rook_semi_open_file',
    label: 'Rook on semi-open file',
    cond:  'Own rook on a file with no own pawns (but may have enemy pawn)',
    bonus: 12,
    match(board, side) {
      const opp = side === 'w' ? 'b' : 'w';
      let count = 0;
      for (let f = 0; f < 8; f++) {
        let hasOwnPawn = false, hasOppPawn = false;
        for (let r = 0; r < 8; r++) {
          const p = board[r * 8 + f];
          if (p === side + 'P') hasOwnPawn = true;
          if (p === opp + 'P')  hasOppPawn = true;
        }
        if (!hasOwnPawn && hasOppPawn) {
          for (let r = 0; r < 8; r++) {
            if (board[r * 8 + f] === side + 'R') count++;
          }
        }
      }
      return count;
    },
  },

  {
    id:    'rook_seventh_rank',
    label: 'Rook on seventh rank',
    cond:  'Own rook on rank 7 (rank 2 for Black) — trapping enemy king',
    bonus: 40,
    match(board, side) {
      const targetRank = side === 'w' ? 1 : 6; // rank 7 from side's view
      let count = 0;
      for (let f = 0; f < 8; f++) {
        if (board[targetRank * 8 + f] === side + 'R') count++;
      }
      return count;
    },
  },

  {
    id:    'knight_outpost',
    label: 'Knight outpost',
    cond:  'Knight on a square that cannot be attacked by enemy pawns and is supported by own pawn',
    bonus: 30,
    match(board, side) {
      const opp     = side === 'w' ? 'b' : 'w';
      const oppDir  = side === 'w' ? 1 : -1;
      let count     = 0;

      for (let s = 0; s < 64; s++) {
        if (board[s] !== side + 'N') continue;
        const r = Math.floor(s / 8);
        const f = s % 8;

        // Not attackable by enemy pawns
        let safe = true;
        for (const df of [-1, 1]) {
          const nf = f + df;
          const nr = r + oppDir;
          if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
            if (board[nr * 8 + nf] === opp + 'P') { safe = false; break; }
          }
        }
        if (!safe) continue;

        // Supported by own pawn
        const ownDir = side === 'w' ? 1 : -1;
        let supported = false;
        for (const df of [-1, 1]) {
          const nf = f + df;
          const nr = r + ownDir;
          if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
            if (board[nr * 8 + nf] === side + 'P') { supported = true; break; }
          }
        }
        if (supported) count++;
      }
      return count;
    },
  },

  // ── King safety ─────────────────────────────────────────────

  {
    id:    'king_pawn_shield',
    label: 'King pawn shield',
    cond:  'Pawns directly in front of castled king (3 squares)',
    bonus: 18,
    match(board, side, phase) {
      if (phase < 8) return 0; // endgame — king wants to be active
      const kingSq = board.findIndex(p => p === side + 'K');
      if (kingSq === -1) return 0;
      const r   = Math.floor(kingSq / 8);
      const f   = kingSq % 8;
      const dir = side === 'w' ? -1 : 1;
      let shield = 0;
      for (const df of [-1, 0, 1]) {
        const nf = f + df;
        if (nf < 0 || nf > 7) continue;
        const shieldSq = (r + dir) * 8 + nf;
        if (shieldSq >= 0 && shieldSq < 64 && board[shieldSq] === side + 'P') {
          shield++;
        }
      }
      return shield;
    },
  },

  {
    id:    'open_king_file',
    label: 'Open file near king penalty',
    cond:  'No own pawns on king\'s file — king is exposed',
    bonus: -35,
    match(board, side, phase) {
      if (phase < 6) return 0;
      const kingSq = board.findIndex(p => p === side + 'K');
      if (kingSq === -1) return 0;
      const kf = kingSq % 8;
      for (let r = 0; r < 8; r++) {
        if (board[r * 8 + kf] === side + 'P') return 0;
      }
      return 1;
    },
  },

  // ── Endgame patterns ─────────────────────────────────────────

  {
    id:    'king_activity_endgame',
    label: 'King activity in endgame',
    cond:  'King distance from center (closer is better in endgame)',
    bonus: 10,
    match(board, side, phase) {
      if (phase > 8) return 0; // only in endgame
      const kingSq = board.findIndex(p => p === side + 'K');
      if (kingSq === -1) return 0;
      const r = Math.floor(kingSq / 8);
      const f = kingSq % 8;
      // Manhatten distance from d4 (sq 35), max 7
      const dist = Math.abs(r - 3) + Math.abs(f - 3);
      return (7 - dist) * 0.15; // fractional match value
    },
  },

  {
    id:    'connected_rooks',
    label: 'Connected rooks',
    cond:  'Two own rooks on the same rank or file with no pieces between them',
    bonus: 20,
    match(board, side) {
      const rooks = [];
      for (let s = 0; s < 64; s++) {
        if (board[s] === side + 'R') rooks.push(s);
      }
      if (rooks.length < 2) return 0;
      let count = 0;
      for (let i = 0; i < rooks.length; i++) {
        for (let j = i + 1; j < rooks.length; j++) {
          const [a, b] = [rooks[i], rooks[j]];
          const ra = Math.floor(a / 8), fa = a % 8;
          const rb = Math.floor(b / 8), fb = b % 8;
          if (ra === rb) { // same rank
            let clear = true;
            for (let f = Math.min(fa,fb)+1; f < Math.max(fa,fb); f++) {
              if (board[ra * 8 + f]) { clear = false; break; }
            }
            if (clear) count++;
          } else if (fa === fb) { // same file
            let clear = true;
            for (let r = Math.min(ra,rb)+1; r < Math.max(ra,rb); r++) {
              if (board[r * 8 + fa]) { clear = false; break; }
            }
            if (clear) count++;
          }
        }
      }
      return count;
    },
  },

  {
    id:    'bad_bishop',
    label: 'Bad bishop penalty',
    cond:  'Bishop blocked by own pawns on same colour squares',
    bonus: -15,
    match(board, side) {
      let penalty = 0;
      for (let s = 0; s < 64; s++) {
        if (board[s] !== side + 'B') continue;
        const bishopColor = (Math.floor(s / 8) + s % 8) % 2;
        let pawnsOnColor = 0;
        for (let ps = 0; ps < 64; ps++) {
          if (board[ps] !== side + 'P') continue;
          if ((Math.floor(ps / 8) + ps % 8) % 2 === bishopColor) pawnsOnColor++;
        }
        // More pawns on same colour = worse bishop
        penalty += Math.max(0, pawnsOnColor - 2);
      }
      return penalty;
    },
  },
];

// ─── Evaluate a single pattern ───────────────────────────────

function evalPattern(board, side, pattern, phase) {
  const fires = pattern.match(board, side, phase);
  return fires * pattern.bonus;
}

// ─── Main evaluation function ────────────────────────────────

/**
 * Full position evaluation.
 * Returns centipawns from the side-to-move's perspective.
 *
 * @param {object} state  { board, side, ... }
 * @returns {number}
 */
function evaluate(state) {
  return scorePosition(state).total;
}

/**
 * Detailed breakdown of the evaluation (for debugging / agent logs).
 *
 * @param {object} state
 * @returns {EvalBreakdown}
 */
function scorePosition(state) {
  const { board, side } = state;
  const phase = getPhase(board);
  const phaseRatio = phase / MAX_PHASE; // 1.0 = full middlegame, 0.0 = endgame

  let material = 0;
  let pst      = 0;

  for (let s = 0; s < 64; s++) {
    const p = board[s];
    if (!p) continue;

    const pt      = p[1];
    const isWhite = p[0] === 'w';
    const sign    = isWhite ? 1 : -1;

    // Material
    material += sign * (PIECE_VAL[pt] ?? 0);

    // PST (phase-blended for King)
    const idx = pstIndex(s, isWhite);
    if (pt === 'K') {
      const midScore = PST_MID.K[idx] ?? 0;
      const endScore = PST_END_KING[idx] ?? 0;
      pst += sign * (phaseRatio * midScore + (1 - phaseRatio) * endScore);
    } else {
      pst += sign * ((PST_MID[pt]?.[idx]) ?? 0);
    }
  }

  // Pattern table
  const patternScores = {};
  let   patternTotal  = 0;
  let   firedCount    = 0;

  for (const pattern of PATTERN_TABLE) {
    const own = evalPattern(board, side, pattern, phase);
    const opp = evalPattern(board, side === 'w' ? 'b' : 'w', pattern, phase);
    const net = own - opp;
    patternScores[pattern.id] = net;
    if (net !== 0) { patternTotal += net; firedCount++; }
  }

  // From side-to-move's perspective
  const sign  = side === 'w' ? 1 : -1;
  const total = sign * Math.round(material + pst) + patternTotal;

  return {
    total,
    material:     sign * Math.round(material),
    pst:          sign * Math.round(pst),
    patternTotal,
    patternScores,
    phase,
    firedCount,
  };
}

// ─── Add custom patterns at runtime ──────────────────────────

/**
 * Inject a new heuristic rule into the pattern table.
 * Use this to extend the "neural" layer with Claude-generated rules.
 *
 * @param {{ id, label, cond, bonus, match }} pattern
 */
function addPattern(pattern) {
  if (!pattern.id || !pattern.match) {
    throw new Error('Pattern must have id and match function');
  }
  const existing = PATTERN_TABLE.findIndex(p => p.id === pattern.id);
  if (existing >= 0) {
    PATTERN_TABLE[existing] = pattern; // replace
  } else {
    PATTERN_TABLE.push(pattern);
  }
}

// ─── Standalone position tester ──────────────────────────────

/**
 * Test the evaluator against a known position.
 * Useful for unit testing eval correctness without Agent 1 or 2.
 *
 * @param {object} state   partial state with board[] and side
 * @param {number} expectedSign  +1 if side should be winning, -1 if losing, 0 if equal
 * @param {string} description
 */
function testPosition(state, expectedSign, description) {
  const result = scorePosition(state);
  const actualSign = result.total > 10 ? 1 : result.total < -10 ? -1 : 0;
  const pass = actualSign === expectedSign;

  console.log(`${pass ? 'PASS' : 'FAIL'} — ${description}`);
  console.log(`  Score: ${result.total > 0 ? '+' : ''}${result.total}cp`);
  console.log(`  Material: ${result.material}cp  PST: ${result.pst}cp  Patterns: ${result.patternTotal}cp`);
  if (!pass) {
    console.log(`  Expected: ${expectedSign > 0 ? 'winning' : expectedSign < 0 ? 'losing' : 'equal'}`);
    console.log(`  Actual:   ${actualSign > 0 ? 'winning' : actualSign < 0 ? 'losing' : 'equal'}`);
  }
  const fired = Object.entries(result.patternScores)
    .filter(([,v]) => v !== 0)
    .map(([k,v]) => `${k}: ${v > 0 ? '+' : ''}${v}`);
  if (fired.length) console.log('  Fired patterns:', fired.join(', '));
  console.log('');
  return { pass, score: result.total, breakdown: result };
}

// ─── Exports ─────────────────────────────────────────────────

module.exports = {
  evaluate,
  scorePosition,
  addPattern,
  testPosition,
  PATTERN_TABLE,
  PIECE_VAL,
  PST_MID,
};

// ─── Self-test (node agent3_eval.js) ─────────────────────────

if (require.main === module) {
  console.log('=== Agent 3: Eval Agent self-test ===\n');

  // Test 1: Starting position should be ~equal
  const startBoard = Array(64).fill(null);
  const backRank = ['R','N','B','Q','K','B','N','R'];
  for (let f = 0; f < 8; f++) {
    startBoard[f]      = 'b' + backRank[f];
    startBoard[8  + f] = 'bP';
    startBoard[48 + f] = 'wP';
    startBoard[56 + f] = 'w' + backRank[f];
  }
  testPosition({ board: startBoard, side: 'w' }, 0, 'Starting position — should be near equal');

  // Test 2: White up a queen — should be clearly winning
  const queenUpBoard = [...startBoard];
  queenUpBoard[3] = null; // remove black queen
  testPosition({ board: queenUpBoard, side: 'w' }, 1, 'White up a queen — should be winning');

  // Test 3: Black up a rook — should be losing for White
  const rookUpBoard = [...startBoard];
  rookUpBoard[63] = null; // remove white h-rook
  testPosition({ board: rookUpBoard, side: 'w' }, -1, 'White down a rook — should be losing');

  // Test 4: Custom pattern injection
  console.log('--- Testing custom pattern injection ---');
  addPattern({
    id:    'fianchetto',
    label: 'Fianchetto bonus',
    cond:  'Bishop on g2/b2 (or g7/b7) with pawn structure intact',
    bonus: 20,
    match(board, side) {
      const targets = side === 'w' ? [54, 49] : [14, 9]; // g2,b2 / g7,b7
      return targets.filter(s => board[s] === side + 'B').length;
    },
  });

  const fianchettoBoard = [...startBoard];
  fianchettoBoard[49] = null; // b2 pawn removed
  fianchettoBoard[49] = 'wB'; // bishop on b2
  const result = scorePosition({ board: fianchettoBoard, side: 'w' });
  console.log('Fianchetto pattern score:', result.patternScores['fianchetto'], 'cp');
  console.log('Pattern table size:', PATTERN_TABLE.length, 'rules\n');

  // Summary
  console.log('Pattern table:');
  for (const p of PATTERN_TABLE) {
    console.log(`  [${p.bonus > 0 ? '+' : ''}${p.bonus}cp] ${p.id} — ${p.cond}`);
  }
}
