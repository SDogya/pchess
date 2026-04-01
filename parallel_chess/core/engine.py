import numpy as np
from .board import EMPTY, KING
from .rules import get_pseudo_legal_moves, decode_move


class ParallelResolutionEngine:
    """
    Resolves simultaneously submitted moves from both players.
    All collision rules are applied in strict priority order.
    """

    def resolve_step(
        self,
        board: np.ndarray,
        move_w: tuple[int, int] | None,
        move_b: tuple[int, int] | None,
    ) -> tuple[np.ndarray, dict]:
        """
        move_w, move_b: (from_sq, to_sq) encoded moves, or None for null-move.
        Returns updated board and info dict.
        """
        info = {
            "white_illegal": False,
            "black_illegal": False,
            "mutual_destruction": False,
            "swap_collision": False,
            "white_captured": None,
            "black_captured": None,
            "white_king_dead": False,
            "black_king_dead": False,
        }

        legal_w = get_pseudo_legal_moves(board, 1)
        legal_b = get_pseudo_legal_moves(board, -1)

        move_w = self._validate_move(move_w, legal_w, info, "white")
        move_b = self._validate_move(move_b, legal_b, info, "black")

        new_board = board.copy()

        if move_w is None and move_b is None:
            return new_board, info

        w_fr, w_ff, w_tr, w_tf = decode_move(*move_w) if move_w else (None,) * 4
        b_fr, b_ff, b_tr, b_tf = decode_move(*move_b) if move_b else (None,) * 4

        # --- Mutual destruction: both target same square ---
        if move_w and move_b and (w_tr, w_tf) == (b_tr, b_tf):
            info["mutual_destruction"] = True
            new_board[w_fr, w_ff] = EMPTY
            new_board[b_fr, b_ff] = EMPTY
            new_board[w_tr, w_tf] = EMPTY
            self._check_kings(board, new_board, info)
            return new_board, info

        # --- Swap collision: pieces cross paths ---
        if move_w and move_b and (w_tr, w_tf) == (b_fr, b_ff) and (b_tr, b_tf) == (w_fr, w_ff):
            info["swap_collision"] = True
            new_board[w_fr, w_ff] = EMPTY
            new_board[b_fr, b_ff] = EMPTY
            self._check_kings(board, new_board, info)
            return new_board, info

        # --- Standard resolution: apply moves independently ---
        if move_w:
            captured = int(board[w_tr, w_tf])
            new_board[w_tr, w_tf] = board[w_fr, w_ff]
            new_board[w_fr, w_ff] = EMPTY
            if captured != EMPTY:
                info["white_captured"] = abs(captured)

        if move_b:
            # Re-read from new_board to handle the case where white already moved
            captured = int(new_board[b_tr, b_tf])
            new_board[b_tr, b_tf] = board[b_fr, b_ff]
            new_board[b_fr, b_ff] = EMPTY
            if captured != EMPTY and np.sign(captured) != -1:
                info["black_captured"] = abs(captured)

        self._check_kings(board, new_board, info)
        return new_board, info

    def _validate_move(
        self,
        move: tuple[int, int] | None,
        legal_mask: np.ndarray,
        info: dict,
        color: str,
    ) -> tuple[int, int] | None:
        if move is None:
            return None
        from_sq, to_sq = move
        fr, ff, tr, tf = decode_move(from_sq, to_sq)
        if not (0 <= fr < 8 and 0 <= ff < 8 and 0 <= tr < 8 and 0 <= tf < 8):
            info[f"{color}_illegal"] = True
            return None
        if not legal_mask[fr, ff, tr, tf]:
            info[f"{color}_illegal"] = True
            return None
        return move

    def _check_kings(self, old_board: np.ndarray, new_board: np.ndarray, info: dict):
        white_king_was = np.any(old_board == KING)
        black_king_was = np.any(old_board == -KING)
        info["white_king_dead"] = white_king_was and not np.any(new_board == KING)
        info["black_king_dead"] = black_king_was and not np.any(new_board == -KING)
