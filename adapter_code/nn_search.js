'use strict';
const{getMoves,isInCheck,applyMove}=require('./nn_move_gen');
const{evaluate:defaultEval}=require('./nn_eval');
const PIECE_VAL={P:100,N:320,B:330,R:500,Q:900,K:20000};

function search(state,opts={}){
  const{maxDepth=4,timeLimitMs=2000,evalFn=defaultEval}=opts;
  const start=Date.now();let bestResult={move:null,score:0};
  for(let d=1;d<=maxDepth;d++){
    if(Date.now()-start>timeLimitMs)break;
    const result=negamax(state,d,-Infinity,Infinity,evalFn);
    if(result.move)bestResult=result;
    if(Math.abs(result.score)>15000)break;
  }
  return bestResult;
}
function negamax(state,depth,alpha,beta,evalFn){
  const moves=getMoves(state);
  if(!moves.length){const inCheck=isInCheck(state.board,state.side);return{score:inCheck?-20000:0,move:null};}
  if(depth===0)return{score:evalFn(state),move:null};
  const ordered=moves.slice().sort((a,b)=>movePriority(state.board,b)-movePriority(state.board,a));
  let best={score:-Infinity,move:null};
  for(const move of ordered){
    const child=applyMove(state,move);
    const{score:raw}=negamax(child,depth-1,-beta,-alpha,evalFn);
    const score=-raw;
    if(score>best.score){best={score,move};}
    if(score>alpha)alpha=score;
    if(alpha>=beta)break;
  }
  return best;
}
function movePriority(board,move){
  if(board[move.to]){const vv=PIECE_VAL[board[move.to][1]]??0,av=PIECE_VAL[board[move.from][1]]??0;return 100000+10*vv-av;}
  if(move.promo)return 90000;
  return 0;
}
function moveToAlg(move){
  const files='abcdefgh';
  const from=files[move.from%8]+(8-Math.floor(move.from/8));
  const to=files[move.to%8]+(8-Math.floor(move.to/8));
  return from+to+(move.promo?move.promo[1].toLowerCase():'');
}
module.exports={search,moveToAlg};
