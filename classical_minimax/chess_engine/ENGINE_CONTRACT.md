# Engine interface contract

## Move generation
from chess_engine.move_gen import MoveGenAgent
import chess

agent = MoveGenAgent()
board = chess.Board()                        # or chess.Board(fen="...")
moves = agent.generate_moves(board)          # returns list[chess.Move]
# identical to list(board.legal_moves)

## Terminal detection (shared, import this — do not reimplement)
from chess_engine.game_state import is_terminal, terminal_score

## Search interface (swap your own SearchAgent behind the UCI facade)
class YourSearchAgent:
    def best_move(self, board: chess.Board, max_depth: int) -> chess.Move:
        ...  # must return a legal chess.Move

## UCI process contract
stdin:  position fen <fen> / position startpos moves <moves>
        go depth <n>
stdout: bestmove <uci_move>   # e.g. e2e4, e1g1, e7e8q

## Note on monte_carlo/movegen_agent.py
If your stack uses a custom GameState/squares representation instead of 
chess.Board, add an adapter:
    chess_board = chess.Board(fen=your_state.to_fen())
Do not re-implement legal move generation — use MoveGenAgent or board.legal_moves.
