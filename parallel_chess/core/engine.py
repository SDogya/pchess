import numpy as np
from .board import EMPTY, KING
from .rules import get_pseudo_legal_moves, decode_move

class ParallelResolutionEngine:
    """
    Разрешает одновременно поданные ходы обоих игроков.
    Применяет правила столкновений и превращений.
    """

    def resolve_step(
        self,
        board: np.ndarray,
        move_w: tuple[int, int] | None,
        move_b: tuple[int, int] | None,
    ) -> tuple[np.ndarray, dict]:
        """
        Применяет ходы и возвращает обновленную доску и словарь info для начисления наград.
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
            "white_promoted_count": 0,  # Используем явное имя для связи с наградами
            "black_promoted_count": 0,
        }

        legal_w = get_pseudo_legal_moves(board, 1)
        legal_b = get_pseudo_legal_moves(board, -1)

        move_w = self._validate_move(move_w, legal_w, info, "white")
        move_b = self._validate_move(move_b, legal_b, info, "black")

        new_board = board.copy()

        if move_w is None and move_b is None:
            return new_board, info

        w_fr, w_ff, w_tr, w_tf = decode_move(move_w[0], move_w[1]) if move_w else (None,) * 4
        b_fr, b_ff, b_tr, b_tf = decode_move(move_b[0], move_b[1]) if move_b else (None,) * 4

        # --- 1. Взаимное уничтожение (целят в одну клетку) ---
        if move_w and move_b and (w_tr, w_tf) == (b_tr, b_tf):
            info["mutual_destruction"] = True
            new_board[w_fr, w_ff] = EMPTY
            new_board[b_fr, b_ff] = EMPTY
            new_board[w_tr, w_tf] = EMPTY
            self._check_kings(board, new_board, info)
            return new_board, info

        # --- 2. Столкновение при пересечении (Swap) ---
        if (move_w and move_b and (w_tr, w_tf) == (b_fr, b_ff)
                and (b_tr, b_tf) == (w_fr, w_ff)):
            info["swap_collision"] = True
            new_board[w_fr, w_ff] = EMPTY
            new_board[b_fr, b_ff] = EMPTY
            self._check_kings(board, new_board, info)
            return new_board, info

        # --- 3. Обычное выполнение ходов ---
        if move_w:
            captured = int(board[w_tr, w_tf])
            new_board[w_tr, w_tf] = board[w_fr, w_ff]
            new_board[w_fr, w_ff] = EMPTY
            if captured != EMPTY:
                info["white_captured"] = abs(captured)

        if move_b:
            # Проверяем, не съела ли белая фигура ту, что черные хотели защитить
            captured = int(new_board[b_tr, b_tf])
            new_board[b_tr, b_tf] = board[b_fr, b_ff]
            new_board[b_fr, b_ff] = EMPTY
            if captured != EMPTY and np.sign(captured) != -1:
                info["black_captured"] = abs(captured)

        # Проверка состояния королей
        self._check_kings(board, new_board, info)
        
        # --- ОБРАБОТКА ПРЕВРАЩЕНИЙ ---
        # Вызываем после всех перемещений, но до возврата результата
        self._handle_promotions(new_board, info)

        return new_board, info

    def _handle_promotions(self, board: np.ndarray, info: dict):
        """
        Находит пешки на краях и превращает их в ферзей, записывая счетчик в info.
        """
        # Белые пешки (1) превращаются в ферзей (5) на 0-й строке
        white_mask = (board[0, :] == 1)
        white_count = int(np.sum(white_mask))
        if white_count > 0:
            board[0, white_mask] = 5
            info["white_promoted_count"] = white_count

        # Черные пешки (-1) превращаются в ферзей (-5) на 7-й строке
        black_mask = (board[7, :] == -1)
        black_count = int(np.sum(black_mask))
        if black_count > 0:
            board[7, black_mask] = -5
            info["black_promoted_count"] = black_count

    def _validate_move(self, move, legal_mask, info, color):
        if move is None: return None
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