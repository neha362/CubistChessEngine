"""
Evaluation Agent — the Berserker personality.

Game-theoretic framing: we are choosing a utility function U(s) over states s.
The search will optimize U via minimax. Different choices of U define
different equilibria — i.e. different "personalities."

Berserker's utility is biased toward STATES THAT THREATEN THE ENEMY KING. The
bias terms, in priority order:

  1. King-zone attacks: pieces attacking squares adjacent to the enemy king.
  2. Tempo-on-king: bonus for delivering checks.
  3. Open files / diagonals pointing at the enemy king.
  4. Sacrifice tolerance: material is discounted vs. the standard valuation,
     so the engine will give up a pawn (or even a piece) for sufficient
     attacking compensation.
  5. Pawn storm: advanced pawns near the enemy king are heavily rewarded.
  6. King exposure: enemy king on an open file or with few defenders is great;
     own king's exposure is barely penalized (Berserker doesn't fear the wind).

This is NOT a "rational" evaluator. It will make objectively bad trades. That's
the entire point — the experiment is whether the resulting strategy beats
rational engines through chaos and initiative.
"""

import chess

# Note: material values are LOWER than a rational engine. A standard engine
# values a queen at 900cp; Berserker values it at 700cp because it's willing
# to trade queen for attack. Pawns are worth almost nothing to Berserker —
# 60cp instead of 100 — because pawns are fuel for the storm.
PIECE_VALUES = {
    chess.PAWN:   60,
    chess.KNIGHT: 280,
    chess.BISHOP: 300,
    chess.ROOK:   450,
    chess.QUEEN:  700,
    chess.KING:   0,
}

# Aggressive PSTs: knights and bishops that point at the enemy camp get bonuses.
# Pawns that march forward get rewarded heavily — Berserker pushes pawns
# at the king instead of building solid structures.
PAWN_PST = [
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,  -5,  -5,   0,   0,   0,
     5,  10,  10,   5,   5,  10,  10,   5,
    10,  15,  15,  20,  20,  15,  15,  10,
    20,  25,  30,  35,  35,  30,  25,  20,   # advancing pawns
    40,  50,  55,  60,  60,  55,  50,  40,   # storming pawns
    70,  80,  85,  90,  90,  85,  80,  70,   # pawns about to promote
     0,   0,   0,   0,   0,   0,   0,   0,
]

# Knights forward of the 4th rank get aggressive bonuses.
KNIGHT_PST = [
   -50, -40, -30, -30, -30, -30, -40, -50,
   -40, -20,   0,   5,   5,   0, -20, -40,
   -30,   5,  10,  15,  15,  10,   5, -30,
   -30,   0,  20,  25,  25,  20,   0, -30,
   -20,  10,  30,  35,  35,  30,  10, -20,   # outposts
   -10,  20,  35,  40,  40,  35,  20, -10,   # forward outposts
     0,  15,  25,  30,  30,  25,  15,   0,
   -50, -40, -30, -30, -30, -30, -40, -50,
]

# Bishops on long diagonals pointing at the enemy king area.
BISHOP_PST = [
   -20, -10, -10, -10, -10, -10, -10, -20,
   -10,  10,   5,   5,   5,   5,  10, -10,
   -10,   5,  15,  15,  15,  15,   5, -10,
   -10,   5,  15,  20,  20,  15,   5, -10,
   -10,  10,  15,  20,  20,  15,  10, -10,
   -10,  15,  20,  25,  25,  20,  15, -10,   # active bishops
   -10,  10,   5,   5,   5,   5,  10, -10,
   -20, -10, -10, -10, -10, -10, -10, -20,
]

# Rooks on open files toward the enemy.
ROOK_PST = [
     0,   0,   5,  10,  10,   5,   0,   0,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
     0,   5,   5,   5,   5,   5,   5,   0,
     5,  10,  10,  10,  10,  10,  10,   5,
    10,  15,  15,  15,  15,  15,  15,  10,   # rook lifts
    20,  25,  25,  25,  25,  25,  25,  20,   # 7th rank — Berserker loves this
     0,   0,   5,  10,  10,   5,   0,   0,
]

# Queen wants to be near the enemy king. Aggressive queen sorties early on are
# usually a bad idea in real chess — Berserker does them anyway.
QUEEN_PST = [
   -20, -10, -10,  -5,  -5, -10, -10, -20,
   -10,   0,   5,   0,   0,   0,   0, -10,
   -10,   5,  10,  10,  10,  10,   5, -10,
    -5,   0,  10,  15,  15,  10,   0,  -5,
     0,  10,  15,  20,  20,  15,  10,   0,
     5,  15,  20,  25,  25,  20,  15,   5,   # invasion squares
    10,  15,  20,  25,  25,  20,  15,  10,
   -10,   0,   0,   0,   0,   0,   0, -10,
]

# King PST: Berserker barely cares about its own king safety. The penalties for
# walking the king up the board are MUCH smaller than a rational engine would
# use. This is the "never plays safely" property.
KING_PST = [
    10,  20,   5,   0,   0,   5,  20,  10,
    10,  10,  -5,  -5,  -5,  -5,  10,  10,
    -5, -10, -10, -10, -10, -10, -10,  -5,
   -10, -15, -15, -20, -20, -15, -15, -10,
   -15, -20, -20, -25, -25, -20, -20, -15,
   -15, -20, -20, -25, -25, -20, -20, -15,
   -15, -20, -20, -25, -25, -20, -20, -15,
   -15, -20, -20, -25, -25, -20, -20, -15,
]

PST = {
    chess.PAWN: PAWN_PST, chess.KNIGHT: KNIGHT_PST, chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST, chess.QUEEN: QUEEN_PST, chess.KING: KING_PST,
}


def _king_zone(king_square: int) -> chess.SquareSet:
    """The 3x3 region around a king. Attacks on these squares = pressure on king."""
    zone = chess.SquareSet()
    kf, kr = chess.square_file(king_square), chess.square_rank(king_square)
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            f, r = kf + df, kr + dr
            if 0 <= f < 8 and 0 <= r < 8:
                zone.add(chess.square(f, r))
    return zone


class Evaluator:
    """
    Berserker evaluation. Implements the EvalAgent protocol.

    To swap personalities, write another class with the same evaluate() signature
    and different weights. That's the experiment your team is running.
    """

    name = "Berserker"

    # Bonuses for attacking the squares around the enemy king. These add up
    # FAST when multiple pieces converge on the king zone — which is exactly
    # the Berserker pattern.
    KING_ZONE_ATTACK_BONUS = {
        chess.PAWN: 15, chess.KNIGHT: 25, chess.BISHOP: 25,
        chess.ROOK: 40, chess.QUEEN: 80, chess.KING: 0,
    }

    # Bonus per attacker on the enemy king zone, scaling super-linearly.
    # Two attackers > 2x one attacker — coordination is rewarded.
    ATTACKER_COUNT_BONUS = [0, 0, 30, 80, 150, 250, 400, 600, 800]

    CHECK_BONUS         = 50      # the side to move is in check
    OPEN_FILE_VS_KING   = 35      # rook/queen on open file pointing at enemy king
    PAWN_STORM_BONUS    = 25      # per pawn close to enemy king

    def evaluate(self, board: chess.Board) -> int:
        # Terminal states: mate is the ultimate goal for Berserker, so we don't
        # discount it. Stalemate is 0 (we'd rather lose attacking than draw).
        if board.is_checkmate():
            return -99999 if board.turn == chess.WHITE else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0

        # --- Material + PST (with Berserker's discounted material weights) ---
        for square, piece in board.piece_map().items():
            value = PIECE_VALUES[piece.piece_type]
            pst_sq = square if piece.color == chess.WHITE else chess.square_mirror(square)
            positional = PST[piece.piece_type][pst_sq]
            sign = 1 if piece.color == chess.WHITE else -1
            score += sign * (value + positional)

        # --- King-zone attacks: the personality core ---
        # For each side, count attacks on the enemy king's zone, weighted by
        # the type of attacker. This is the term that rewards "winding up" an
        # attack even before any tactics resolve.
        score += self._king_zone_score(board, chess.WHITE)
        score -= self._king_zone_score(board, chess.BLACK)

        # --- Tempo on king ---
        if board.is_check():
            # The side to move is in check — bad for them.
            score += -self.CHECK_BONUS if board.turn == chess.WHITE else self.CHECK_BONUS

        # --- Pawn storm: pawns advanced near the enemy king are gold ---
        score += self._pawn_storm_score(board)

        return score

    def _king_zone_score(self, board: chess.Board, attacker_color: chess.Color) -> int:
        """How much pressure does attacker_color have on the OPPOSING king?"""
        enemy_king_sq = board.king(not attacker_color)
        if enemy_king_sq is None:
            return 0  # Shouldn't happen mid-game, but defensive.

        zone = _king_zone(enemy_king_sq)
        total = 0
        attacker_count = 0

        for square, piece in board.piece_map().items():
            if piece.color != attacker_color or piece.piece_type == chess.KING:
                continue
            # board.attacks(sq) returns squares this piece attacks from sq.
            attacks = board.attacks(square)
            zone_hits = len(attacks & zone)
            if zone_hits > 0:
                total += zone_hits * self.KING_ZONE_ATTACK_BONUS[piece.piece_type]
                attacker_count += 1

        # Coordination bonus: many attackers on one king is super-linearly good.
        idx = min(attacker_count, len(self.ATTACKER_COUNT_BONUS) - 1)
        total += self.ATTACKER_COUNT_BONUS[idx]
        return total

    def _pawn_storm_score(self, board: chess.Board) -> int:
        """Pawns advanced on or near the enemy king's file get bonuses."""
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            enemy_king = board.king(not color)
            if enemy_king is None:
                continue
            ek_file = chess.square_file(enemy_king)
            ek_rank = chess.square_rank(enemy_king)
            sign = 1 if color == chess.WHITE else -1

            for square in board.pieces(chess.PAWN, color):
                pf = chess.square_file(square)
                pr = chess.square_rank(square)
                # Pawn must be on or adjacent to the king's file.
                if abs(pf - ek_file) > 1:
                    continue
                # How far has it advanced toward the enemy?
                advancement = pr if color == chess.WHITE else (7 - pr)
                # How close to the enemy king's rank?
                rank_distance = abs(pr - ek_rank)
                if advancement >= 4 and rank_distance <= 4:
                    score += sign * self.PAWN_STORM_BONUS * (advancement - 3)
        return score


default = Evaluator()
