from typing import Iterable, Optional
import chess

_PIECE_VALUES={chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900,chess.KING:0}

class MoveGen:
    name="BerserkerMoveGen"
    def legal_moves(self,board): return list(board.legal_moves)
    def ordered_moves(self,board,hint_move=None):
        moves=list(board.legal_moves)
        moves.sort(key=lambda m:self._move_priority(board,m,hint_move),reverse=True)
        return moves
    def _move_priority(self,board,move,hint_move):
        if hint_move is not None and move==hint_move: return 10_000_000
        score=0
        if board.gives_check(move): score+=1_000_000
        if board.is_capture(move):
            if board.is_en_passant(move): victim_val=_PIECE_VALUES[chess.PAWN]
            else:
                victim=board.piece_at(move.to_square)
                victim_val=_PIECE_VALUES[victim.piece_type] if victim else 0
            attacker=board.piece_at(move.from_square)
            attacker_val=_PIECE_VALUES[attacker.piece_type] if attacker else 0
            score+=100_000+10*victim_val-attacker_val
        if move.promotion is not None: score+=50_000+_PIECE_VALUES.get(move.promotion,0)
        file=chess.square_file(move.to_square); rank=chess.square_rank(move.to_square)
        center_distance=abs(3.5-file)+abs(3.5-rank)
        score+=int(20-4*center_distance)
        return score

default=MoveGen()
