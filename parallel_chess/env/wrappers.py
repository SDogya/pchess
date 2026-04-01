import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable

from .chess_env import SimultaneousChessEnv
from ..core.rules import get_pseudo_legal_moves, moves_mask_to_list, encode_move


def random_opponent_policy(board: np.ndarray, color: int) -> tuple[int, int]:
    mask = get_pseudo_legal_moves(board, color)
    legal = moves_mask_to_list(mask)
    if not legal:
        return (0, 0)
    idx = np.random.randint(len(legal))
    return encode_move(*legal[idx])


class SingleAgentSelfPlayWrapper(gym.Wrapper):
    def __init__(
        self,
        env: SimultaneousChessEnv,
        opponent_policy: Callable = None,
        agent_color: int = 1,
    ):
        super().__init__(env)
        self.opponent_policy = opponent_policy or random_opponent_policy
        self.agent_color = agent_color
        self.opponent_color = -agent_color

        self.observation_space = env.observation_space
        # Плоское пространство для MaskablePPO
        self.action_space = spaces.Discrete(4096)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._agent_obs(obs), info

    def step(self, action: int | np.integer):
        action = int(action)  # Гарантируем чистый int для математики

        # 1. Декодируем действие агента из плоского в координаты (from_sq, to_sq)
        fr, ff = (action // 64) // 8, (action // 64) % 8
        tr, tf = (action % 64) // 8, (action % 64) % 8

        # 2. Если мы играем за черных, "разворачиваем" координаты обратно для движка
        if self.agent_color == -1:
            fr, tr = 7 - fr, 7 - tr

        agent_move = (fr * 8 + ff, tr * 8 + tf)

        # 3. Получаем ход оппонента
        board = self.env.board
        opp_obs = board * self.opponent_color
        if self.opponent_color == -1:
            opp_obs = np.flipud(opp_obs)

        # Политика оппонента работает с оригинальной геометрией доски
        opp_move = self.opponent_policy(board, self.opponent_color)

        # 4. Отправляем в среду
        if self.agent_color == 1:
            action_dict = {"white": agent_move, "black": opp_move}
        else:
            action_dict = {"white": opp_move, "black": agent_move}

        # Базовая среда теперь возвращает 0.0 вместо словаря
        obs, _base_reward, terminated, truncated, info = self.env.step(action_dict)

        # Достаем реальную награду конкретного агента из info
        agent_reward = (
            info["rewards"]["white"]
            if self.agent_color == 1
            else info["rewards"]["black"]
        )

        return self._agent_obs(obs), float(agent_reward), terminated, truncated, info

    def _agent_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs * self.agent_color
        if self.agent_color == -1:
            return np.flipud(obs)  # Черные всегда видят доску "снизу вверх"
        return obs

    def action_masks(self) -> np.ndarray:
        board = self.env.board
        mask_4d = get_pseudo_legal_moves(board, self.agent_color)

        if self.agent_color == -1:
            mask_4d = mask_4d[::-1, :, ::-1, :]

        # Использовать flatten() вместо reshape(4096) для гарантии contiguous памяти
        flat = mask_4d.flatten()
        
        if not np.any(flat):
            flat[0] = True 
            
        return flat