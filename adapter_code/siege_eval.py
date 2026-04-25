import chess

PIECE_VALUES={chess.PAWN:60,chess.KNIGHT:280,chess.BISHOP:300,chess.ROOK:450,chess.QUEEN:700,chess.KING:0}

PAWN_PST=[0,0,0,0,0,0,0,0,0,0,0,-5,-5,0,0,0,5,10,10,5,5,10,10,5,10,15,15,20,20,15,15,10,20,25,30,35,35,30,25,20,40,50,55,60,60,55,50,40,70,80,85,90,90,85,80,70,0,0,0,0,0,0,0,0]
KNIGHT_PST=[-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,5,5,0,-20,-40,-30,5,10,15,15,10,5,-30,-30,0,20,25,25,20,0,-30,-20,10,30,35,35,30,10,-20,-10,20,35,40,40,35,20,-10,0,15,25,30,30,25,15,0,-50,-40,-30,-30,-30,-30,-40,-50]
BISHOP_PST=[-20,-10,-10,-10,-10,-10,-10,-20,-10,10,5,5,5,5,10,-10,-10,5,15,15,15,15,5,-10,-10,5,15,20,20,15,5,-10,-10,10,15,20,20,15,10,-10,-10,15,20,25,25,20,15,-10,-10,10,5,5,5,5,10,-10,-20,-10,-10,-10,-10,-10,-10,-20]
ROOK_PST=[0,0,5,10,10,5,0,0,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,5,5,5,5,5,5,0,5,10,10,10,10,10,10,5,10,15,15,15,15,15,15,10,20,25,25,25,25,25,25,20,0,0,5,10,10,5,0,0]
QUEEN_PST=[-20,-10,-10,-5,-5,-10,-10,-20,-10,0,5,0,0,0,0,-10,-10,5,10,10,10,10,5,-10,-5,0,10,15,15,10,0,-5,0,10,15,20,20,15,10,0,5,15,20,25,25,20,15,5,10,15,20,25,25,20,15,10,-10,0,0,0,0,0,0,-10]
KING_PST=[10,20,5,0,0,5,20,10,10,10,-5,-5,-5,-5,10,10,-5,-10,-10,-10,-10,-10,-10,-5,-10,-15,-15,-20,-20,-15,-15,-10,-15,-20,-20,-25,-25,-20,-20,-15,-15,-20,-20,-25,-25,-20,-20,-15,-15,-20,-20,-25,-25,-20,-20,-15,-15,-20,-20,-25,-25,-20,-20,-15]
PST={chess.PAWN:PAWN_PST,chess.KNIGHT:KNIGHT_PST,chess.BISHOP:BISHOP_PST,chess.ROOK:ROOK_PST,chess.QUEEN:QUEEN_PST,chess.KING:KING_PST}

def _king_zone(king_square):
    zone=chess.SquareSet()
    kf,kr=chess.square_file(king_square),chess.square_rank(king_square)
    for df in(-1,0,1):
        for dr in(-1,0,1):
            f,r=kf+df,kr+dr
            if 0<=f<8 and 0<=r<8: zone.add(chess.square(f,r))
    return zone

class Evaluator:
    name="Berserker2"
    KING_ZONE_ATTACK_BONUS={chess.PAWN:15,chess.KNIGHT:25,chess.BISHOP:25,chess.ROOK:40,chess.QUEEN:80,chess.KING:0}
    ATTACKER_COUNT_BONUS=[0,0,30,80,150,250,400,600,800]
    CHECK_BONUS=50; OPEN_FILE_VS_KING=35; PAWN_STORM_BONUS=25

    def evaluate(self,board):
        if board.is_checkmate(): return -99999 if board.turn==chess.WHITE else 99999
        if board.is_stalemate() or board.is_insufficient_material(): return 0
        score=0
        for square,piece in board.piece_map().items():
            value=PIECE_VALUES[piece.piece_type]
            pst_sq=square if piece.color==chess.WHITE else chess.square_mirror(square)
            positional=PST[piece.piece_type][pst_sq]
            sign=1 if piece.color==chess.WHITE else -1
            score+=sign*(value+positional)
        score+=self._king_zone_score(board,chess.WHITE)
        score-=self._king_zone_score(board,chess.BLACK)
        if board.is_check(): score+=-self.CHECK_BONUS if board.turn==chess.WHITE else self.CHECK_BONUS
        score+=self._pawn_storm_score(board)
        return score

    def _king_zone_score(self,board,attacker_color):
        enemy_king_sq=board.king(not attacker_color)
        if enemy_king_sq is None: return 0
        zone=_king_zone(enemy_king_sq); total=0; attacker_count=0
        for square,piece in board.piece_map().items():
            if piece.color!=attacker_color or piece.piece_type==chess.KING: continue
            attacks=board.attacks(square); zone_hits=len(attacks&zone)
            if zone_hits>0:
                total+=zone_hits*self.KING_ZONE_ATTACK_BONUS[piece.piece_type]; attacker_count+=1
        idx=min(attacker_count,len(self.ATTACKER_COUNT_BONUS)-1)
        total+=self.ATTACKER_COUNT_BONUS[idx]
        return total

    def _pawn_storm_score(self,board):
        score=0
        for color in(chess.WHITE,chess.BLACK):
            enemy_king=board.king(not color)
            if enemy_king is None: continue
            ek_file=chess.square_file(enemy_king); ek_rank=chess.square_rank(enemy_king)
            sign=1 if color==chess.WHITE else -1
            for square in board.pieces(chess.PAWN,color):
                pf=chess.square_file(square); pr=chess.square_rank(square)
                if abs(pf-ek_file)>1: continue
                advancement=pr if color==chess.WHITE else(7-pr)
                rank_distance=abs(pr-ek_rank)
                if advancement>=4 and rank_distance<=4:
                    score+=sign*self.PAWN_STORM_BONUS*(advancement-3)
        return score

default=Evaluator()
