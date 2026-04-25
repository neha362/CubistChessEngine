'use strict';
const {getMoves, isInCheck, applyMove, makeStartState} = require('./nn_move_gen');
const {search, moveToAlg} = require('./nn_search');
const {evaluate} = require('./nn_eval');

function fenToState(fen) {
  const parts = fen.split(' ');
  const rankStrs = parts[0].split('/');
  const board = Array(64).fill(null);
  const pieceMap = {
    P:'wP',N:'wN',B:'wB',R:'wR',Q:'wQ',K:'wK',
    p:'bP',n:'bN',b:'bB',r:'bR',q:'bQ',k:'bK'
  };
  for (let r = 0; r < 8; r++) {
    let c = 0;
    for (const ch of rankStrs[r]) {
      if (/\d/.test(ch)) { c += parseInt(ch); }
      else { board[r * 8 + c] = pieceMap[ch]; c++; }
    }
  }
  const side = parts[1] === 'w' ? 'w' : 'b';
  const cas = parts[2] || '-';
  const castling = {
    wK: cas.includes('K'), wQ: cas.includes('Q'),
    bK: cas.includes('k'), bQ: cas.includes('q')
  };
  const ep = parts[3] === '-' ? null : (() => {
    const f = 'abcdefgh'.indexOf(parts[3][0]);
    const rank = 8 - parseInt(parts[3][1]);
    return rank * 8 + f;
  })();
  return { board, side, castling, ep };
}

process.stdin.setEncoding('utf8');
let buf = '';
process.stdin.on('data', chunk => {
  buf += chunk;
  const lines = buf.split('\n'); buf = lines.pop();
  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      const req = JSON.parse(line);
      if (req.cmd === 'move') {
        const state = fenToState(req.fen);
        const moves = getMoves(state);
        if (!moves.length) {
          process.stdout.write(JSON.stringify({ uci: 'none' }) + '\n');
          continue;
        }
        const result = search(state, {
          maxDepth: req.depth || 3,
          timeLimitMs: req.timeLimitMs || 3000
        });
        const uci = result.move ? moveToAlg(result.move) : 'none';
        process.stdout.write(JSON.stringify({ uci, score: result.score }) + '\n');
      } else if (req.cmd === 'eval') {
        const state = fenToState(req.fen);
        const score = evaluate(state);
        process.stdout.write(JSON.stringify({ score }) + '\n');
      } else if (req.cmd === 'ping') {
        process.stdout.write(JSON.stringify({ pong: true }) + '\n');
      }
    } catch (e) {
      process.stdout.write(JSON.stringify({ error: String(e) }) + '\n');
    }
  }
});
