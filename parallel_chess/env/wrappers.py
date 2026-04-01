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
    """
    Wraps SimultaneousChessEnv into a single-agent environment.
    The wrapped agent always plays as white (from white's perspective).
    Observation is inverted (*-1) when feeding to the opponent so the net
    always sees itself as positive pieces moving "upward".

    opponent_policy: callable(board: np.ndarray, color: int) -> (from_sq, to_sq)
    """

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
        self.action_space = spaces.MultiDiscrete([64, 64])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._agent_obs(obs), info

    def step(self, action):
        board = self.env.board
        opp_move = self.opponent_policy(board * self.opponent_color, self.opponent_color)

        if self.agent_color == 1:
            action_dict = {"white": action, "black": opp_move}
        else:
            action_dict = {"white": opp_move, "black": action}

        obs, rewards, terminated, truncated, info = self.env.step(action_dict)
        agent_reward = rewards["white"] if self.agent_color == 1 else rewards["black"]

        return self._agent_obs(obs), agent_reward, terminated, truncated, info

    def _agent_obs(self, obs: np.ndarray) -> np.ndarray:
        # Always return board from agent's perspective (own pieces positive)
        return obs * self.agent_color

    def get_legal_action_mask(self) -> np.ndarray:
        """
        Returns flat boolean array of shape (64*64,) for action masking in PPO.
        Useful with MaskablePPO from sb3-contrib.
        """
        board = self.env.board
        mask_4d = get_pseudo_legal_moves(board, self.agent_color)
        flat = mask_4d.reshape(64, 64)
        # from_sq * 64 + to_sq indexing
        return flat.flatten()
