import chess
PAWN=100;KNIGHT=320;BISHOP=330;ROOK=500;QUEEN=900
_MATERIAL={chess.PAWN:PAWN,chess.KNIGHT:KNIGHT,chess.BISHOP:BISHOP,chess.ROOK:ROOK,chess.QUEEN:QUEEN,chess.KING:0}
def _pst_table_index(sq,color):
    f=chess.square_file(sq); r=chess.square_rank(sq)
    r_adj=r if color==chess.WHITE else 7-r
    return(7-r_adj)*8+f
_PST_P=[0,0,0,0,0,0,0,0,5,5,5,5,5,5,5,5,10,10,15,20,20,15,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0]
_PST_N=[-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50]
_PST_B=[-20,-10,-10,-10,-10,-10,-10,-20,-10,5,0,0,0,0,5,-10,-10,10,10,10,10,10,10,-10,-10,0,10,15,15,10,0,-10,-10,5,5,10,10,5,5,-10,-10,0,5,10,10,5,0,-10,-10,0,0,0,0,0,0,-10,-20,-10,-10,-10,-10,-10,-10,-20]
_PST_R=[0,0,0,5,5,0,0,0,0,0,0,5,5,0,0,0,0,0,0,5,5,0,0,0,0,0,0,5,5,0,0,0,0,0,0,5,5,0,0,0,0,0,0,5,5,0,0,0,0,0,0,5,5,0,0,0,0,0,0,5,5,0,0,0]
def _open_file_bonus(board,sq,color):
    f=chess.square_file(sq)
    for r in range(8):
        s=chess.square(f,r); p=board.piece_at(s)
        if p and p.piece_type==chess.PAWN and p.color==color: return 0
    return 15
class EvalAgent:
    def evaluate(self,board):
        score=0
        for sq in chess.SQUARES:
            p=board.piece_at(sq)
            if p is None: continue
            idx=_pst_table_index(sq,p.color)
            mat=_MATERIAL[p.piece_type]; pst=0
            if p.piece_type==chess.PAWN: pst=_PST_P[idx]
            elif p.piece_type==chess.KNIGHT: pst=_PST_N[idx]
            elif p.piece_type==chess.BISHOP: pst=_PST_B[idx]
            elif p.piece_type==chess.ROOK: pst=_PST_R[idx]+_open_file_bonus(board,sq,p.color)
            contrib=mat+pst
            score+=contrib if p.color==chess.WHITE else -contrib
        return score
