'use strict';
const rank=sq=>Math.floor(sq/8),file=sq=>sq%8,sq=(r,f)=>r*8+f,opp=side=>side==='w'?'b':'w';
const inBounds=(r,f)=>r>=0&&r<8&&f>=0&&f<8;
const KNIGHT_DELTAS=[[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
const KING_DELTAS=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
const BISHOP_RAYS=[[-1,-1],[-1,1],[1,-1],[1,1]];
const ROOK_RAYS=[[-1,0],[1,0],[0,-1],[0,1]];
const QUEEN_RAYS=[...BISHOP_RAYS,...ROOK_RAYS];
const SLIDING=new Set(['B','R','Q']);

function generatePseudoLegal(board,side,ep,castling){
  const moves=[];
  for(let s=0;s<64;s++){
    const piece=board[s];if(!piece||piece[0]!==side)continue;
    const pt=piece[1],r=rank(s),f=file(s);
    if(pt==='P')genPawn(board,s,r,f,side,ep,moves);
    else if(pt==='N')genLeaper(board,s,r,f,side,KNIGHT_DELTAS,moves);
    else if(pt==='K')genKing(board,s,r,f,side,castling,moves);
    else if(SLIDING.has(pt))genSlider(board,s,r,f,side,pt==='B'?BISHOP_RAYS:pt==='R'?ROOK_RAYS:QUEEN_RAYS,moves);
  }
  return moves;
}
function genPawn(board,s,r,f,side,ep,moves){
  const dir=side==='w'?-1:1,startRank=side==='w'?6:1,promRank=side==='w'?0:7;
  const one=sq(r+dir,f);
  if(r+dir>=0&&r+dir<8&&!board[one]){
    if(r+dir===promRank)for(const pp of'QRBN')moves.push({from:s,to:one,promo:side+pp});
    else{moves.push({from:s,to:one});if(r===startRank&&!board[sq(r+2*dir,f)])moves.push({from:s,to:sq(r+2*dir,f),ep2:true});}
  }
  for(const df of[-1,1]){const nf=f+df,nr=r+dir;if(!inBounds(nr,nf))continue;const ts=sq(nr,nf);
    if(board[ts]&&board[ts][0]===opp(side)){if(nr===promRank)for(const pp of'QRBN')moves.push({from:s,to:ts,promo:side+pp});else moves.push({from:s,to:ts});}
    if(ep===ts)moves.push({from:s,to:ts,epCapture:true});}
}
function genLeaper(board,s,r,f,side,deltas,moves){
  for(const[dr,df]of deltas){const nr=r+dr,nf=f+df;if(!inBounds(nr,nf))continue;const ts=sq(nr,nf);if(!board[ts]||board[ts][0]!==side)moves.push({from:s,to:ts});}
}
function genSlider(board,s,r,f,side,rays,moves){
  for(const[dr,df]of rays){let nr=r+dr,nf=f+df;while(inBounds(nr,nf)){const ts=sq(nr,nf);if(board[ts]){if(board[ts][0]!==side)moves.push({from:s,to:ts});break;}moves.push({from:s,to:ts});nr+=dr;nf+=df;}}
}
function genKing(board,s,r,f,side,castling,moves){
  genLeaper(board,s,r,f,side,KING_DELTAS,moves);
  if(side==='w'&&r===7&&f===4){
    if(castling.wK&&!board[61]&&!board[62]&&board[63]==='wR')moves.push({from:s,to:62,castle:'wK'});
    if(castling.wQ&&!board[59]&&!board[58]&&!board[57]&&board[56]==='wR')moves.push({from:s,to:58,castle:'wQ'});
  }
  if(side==='b'&&r===0&&f===4){
    if(castling.bK&&!board[5]&&!board[6]&&board[7]==='bR')moves.push({from:s,to:6,castle:'bK'});
    if(castling.bQ&&!board[3]&&!board[2]&&!board[1]&&board[0]==='bR')moves.push({from:s,to:2,castle:'bQ'});
  }
}
function applyMove(state,move){
  const board=[...state.board],castling={...state.castling};let ep=null;
  const piece=board[move.from];board[move.to]=move.promo??piece;board[move.from]=null;
  if(move.epCapture){const capturedRank=rank(move.to)+(piece[0]==='w'?1:-1);board[sq(capturedRank,file(move.to))]=null;}
  if(move.ep2)ep=sq((rank(move.from)+rank(move.to))/2,file(move.to));
  if(move.castle){const rookMoves={wK:[63,61],wQ:[56,59],bK:[7,5],bQ:[0,3]};const[rf,rt]=rookMoves[move.castle];board[rt]=board[rf];board[rf]=null;}
  if(piece==='wK'){castling.wK=false;castling.wQ=false;}if(piece==='bK'){castling.bK=false;castling.bQ=false;}
  if(move.from===56||move.to===56)castling.wQ=false;if(move.from===63||move.to===63)castling.wK=false;
  if(move.from===0||move.to===0)castling.bQ=false;if(move.from===7||move.to===7)castling.bK=false;
  return{board,side:opp(state.side),castling,ep};
}
function isInCheck(board,side){
  const kingSq=board.findIndex(p=>p===side+'K');if(kingSq===-1)return true;
  const oppMoves=generatePseudoLegal(board,opp(side),null,{wK:false,wQ:false,bK:false,bQ:false});
  return oppMoves.some(m=>m.to===kingSq);
}
function getMoves(state){
  const{board,side,ep,castling}=state;
  return generatePseudoLegal(board,side,ep,castling).filter(move=>{
    if(move.castle){
      const passThroughSq={wK:61,wQ:59,bK:5,bQ:3}[move.castle];
      if(isInCheck(board,side))return false;
      const pb=[...board];pb[passThroughSq]=side+'K';pb[move.from]=null;
      if(isInCheck(pb,side))return false;
    }
    return!isInCheck(applyMove({board,side,castling,ep},move).board,side);
  });
}
function makeStartState(){
  const board=Array(64).fill(null);
  const br=['R','N','B','Q','K','B','N','R'];
  for(let f=0;f<8;f++){board[f]='b'+br[f];board[8+f]='bP';board[48+f]='wP';board[56+f]='w'+br[f];}
  return{board,side:'w',castling:{wK:true,wQ:true,bK:true,bQ:true},ep:null};
}
module.exports={getMoves,isInCheck,applyMove,makeStartState,generatePseudoLegal};
