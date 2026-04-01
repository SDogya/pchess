import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..core.board import BoardState, KING
from ..core.engine import ParallelResolutionEngine
from ..core.rules import get_pseudo_legal_moves, moves_mask_to_list, encode_move

PIECE_REWARDS = {
    1: 1,
    2: 3,
    3: 3,
    4: 5,
    5: 9,
    6: 4,
}  # king reward handled via termination


class SimultaneousChessEnv(gym.Env):
    """
    Simultaneous-move chess: both players submit actions at the same time.
    observation_space: (8, 8) int8 grid, positive=white, negative=black.
    action_space: Dict with MultiDiscrete([64, 64]) per player.
                  action[0] = from_sq (rank*8+file), action[1] = to_sq.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self, render_mode: str = None, max_steps: int = 200, starting_fen: str = None
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.starting_fen = starting_fen
        self.engine = ParallelResolutionEngine()

        move_space = spaces.Discrete(4096)
        self.observation_space = spaces.Box(low=-6, high=6, shape=(8, 8), dtype=np.int8)
        self.action_space = spaces.Dict({"white": move_space, "black": move_space})
        self._board: np.ndarray = None
        self._step_count: int = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = BoardState(self.starting_fen)
        self._board = state.grid.copy()
        self._step_count = 0
        return self._board.copy(), {}

    def step(self, action: dict) -> tuple[np.ndarray, float, bool, bool, dict]:
    # Декодируем скаляр (0-4095) в кортеж (from_sq, to_sq)
        def _decode(act):
            if act is None: return None
            if hasattr(act, "__iter__"): return tuple(act)
            return (int(act) // 64, int(act) % 64)

        move_w = _decode(action.get("white"))
        move_b = _decode(action.get("black"))

        # Движок обрабатывает столкновения и взятия
        new_board, info = self.engine.resolve_step(self._board, move_w, move_b)
        self._board = new_board
        self._step_count += 1

        # Обновляем награды с учетом новой ценности фигур
        info["rewards"] = self._compute_rewards(info)

        # МЯСНОЙ РЕЖИМ: Игра продолжается, пока у обеих сторон есть хотя бы одна фигура
        white_pieces = np.any(self._board > 0)
        black_pieces = np.any(self._board < 0)
        
        # Терминация наступает только при полном истреблении одного из цветов
        terminated = bool(not white_pieces or not black_pieces)
        
        # Дополнительная проверка на патовое состояние (если ходить некому, игра стопается)
        if not terminated:
            has_moves_w = np.any(get_pseudo_legal_moves(self._board, 1))
            has_moves_b = np.any(get_pseudo_legal_moves(self._board, -1))
            if not has_moves_w or not has_moves_b:
                terminated = True

        truncated = bool(self._step_count >= self.max_steps)
        obs = self._board.copy()
    
        return obs, 0.0, terminated, truncated, info

    def _compute_rewards(self, info: dict) -> dict:
        r_w, r_b = 0.0, 0.0

        # if info["white_illegal"]: r_w -= 10.0  # noqa: E701
        # if info["black_illegal"]: r_b -= 10.0  # noqa: E701

        # Награда за любые взятия, включая короля
        if info["white_captured"] is not None:
            r_w += PIECE_REWARDS.get(info["white_captured"], 0)
        if info["black_captured"] is not None:
            r_b += PIECE_REWARDS.get(info["black_captured"], 0)

        # Штраф за коллизии (встречное взятие или прыжок на одну клетку)
        if info["mutual_destruction"] or info["swap_collision"]:
            r_w -= 0.5
            r_b -= 0.5
        r_w += info.get("white_promoted_count", 0) * 8.0
        r_b += info.get("black_promoted_count", 0) * 8.0
    

        return {"white": r_w, "black": r_b}

    def get_legal_moves(self, color: int) -> list[tuple[int, int]]:
        """Returns list of (from_sq, to_sq) legal moves for given color (+1/-1)."""
        mask = get_pseudo_legal_moves(self._board, color)
        raw = moves_mask_to_list(mask)
        return [encode_move(*m) for m in raw]

    def render(self):
        if self.render_mode == "ansi":
            state = BoardState.__new__(BoardState)
            state.grid = self._board
            print(state)
        elif self.render_mode == "human":
            from ..render.visualizer import BoardRenderer

            BoardRenderer.render_frame(self._board, {})

    @property
    def board(self) -> np.ndarray:
        return self._board
