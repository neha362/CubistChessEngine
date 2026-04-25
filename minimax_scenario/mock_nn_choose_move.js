#!/usr/bin/env node
'use strict';

const path = require('path');

function parseFenToState(fen) {
  const [placement, side, castlingRaw, epRaw] = fen.trim().split(/\s+/);
  const rows = placement.split('/');
  const board = Array(64).fill(null);

  for (let r = 0; r < 8; r++) {
    let f = 0;
    for (const ch of rows[r]) {
      if (/\d/.test(ch)) {
        f += Number(ch);
        continue;
      }
      const color = ch === ch.toUpperCase() ? 'w' : 'b';
      const piece = ch.toUpperCase();
      board[r * 8 + f] = color + piece;
      f += 1;
    }
  }

  let ep = null;
  if (epRaw && epRaw !== '-') {
    const file = epRaw.charCodeAt(0) - 'a'.charCodeAt(0);
    const rank = Number(epRaw[1]);
    const row = 8 - rank;
    ep = row * 8 + file;
  }

  const castling = {
    wK: castlingRaw && castlingRaw.includes('K'),
    wQ: castlingRaw && castlingRaw.includes('Q'),
    bK: castlingRaw && castlingRaw.includes('k'),
    bQ: castlingRaw && castlingRaw.includes('q'),
  };

  return { board, side, castling, ep };
}

function moveToUci(move) {
  const files = 'abcdefgh';
  const from = files[move.from % 8] + (8 - Math.floor(move.from / 8));
  const to = files[move.to % 8] + (8 - Math.floor(move.to / 8));
  let promo = '';
  if (move.promo) {
    promo = move.promo[1].toLowerCase();
  }
  return from + to + promo;
}

function main() {
  const [, , engineDir, fen, moveTimeMs] = process.argv;
  if (!engineDir || !fen) {
    console.error('Usage: node mock_nn_choose_move.js <engineDir> <fen> <moveTimeMs>');
    process.exit(2);
  }
  const searchMod = require(path.join(engineDir, 'agent2_search.js'));
  const evalMod = require(path.join(engineDir, 'agent3_eval.js'));
  const state = parseFenToState(fen);
  const budget = Number(moveTimeMs || '700');
  const result = searchMod.search(state, {
    maxDepth: Math.max(2, Math.min(5, Math.floor(budget / 250))),
    timeLimitMs: budget,
    evalFn: evalMod.evaluate,
  });
  if (!result || !result.move) {
    console.error('No move produced by mock_nn search.');
    process.exit(1);
  }
  process.stdout.write(moveToUci(result.move));
}

main();
