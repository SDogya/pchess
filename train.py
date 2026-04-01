"""
Quick demo + PPO training sketch.

Run demo:
    python train.py --demo

Train with SB3 PPO:
    python train.py --train --steps 100000
"""

import argparse
import numpy as np

from parallel_chess import (
    SimultaneousChessEnv,
    SingleAgentSelfPlayWrapper,
    random_opponent_policy,
)


def run_demo(n_steps: int = 20, render: bool = True):
    env = SimultaneousChessEnv(render_mode="human" if render else "ansi", max_steps=200)

    obs, _ = env.reset()

    total_rewards = {"white": 0.0, "black": 0.0}

    for step in range(n_steps):
        legal_w = env.get_legal_moves(1)
        legal_b = env.get_legal_moves(-1)

        move_w = legal_w[np.random.randint(len(legal_w))] if legal_w else (0, 0)
        move_b = legal_b[np.random.randint(len(legal_b))] if legal_b else (0, 0)

        obs, rewards, terminated, truncated, info = env.step({"white": move_w, "black": move_b})

        total_rewards["white"] += rewards["white"]
        total_rewards["black"] += rewards["black"]

        print(f"Step {step+1:3d} | W:{rewards['white']:+.1f}  B:{rewards['black']:+.1f} | {_fmt_info(info)}")

        if render:
            env.render()

        if terminated or truncated:
            print("Game over!", "White wins" if info["black_king_dead"] else "Black wins" if info["white_king_dead"] else "Draw/Truncated")
            break

    print(f"\nTotal rewards — White: {total_rewards['white']:.1f}  Black: {total_rewards['black']:.1f}")
    env.close()


def train_ppo(total_timesteps: int = 100_000):
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("Install RL deps: uv pip install -e '.[rl]'")
        return

    base_env = SimultaneousChessEnv(max_steps=200)
    env = SingleAgentSelfPlayWrapper(base_env, opponent_policy=random_opponent_policy)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_parallel_chess")
    print("Model saved to ppo_parallel_chess.zip")


def _fmt_info(info: dict) -> str:
    parts = []
    if info["white_illegal"]: parts.append("W_illegal")
    if info["black_illegal"]: parts.append("B_illegal")
    if info["mutual_destruction"]: parts.append("MUTUAL_DEST")
    if info["swap_collision"]: parts.append("SWAP")
    if info["white_captured"]: parts.append(f"W_cap={info['white_captured']}")
    if info["black_captured"]: parts.append(f"B_cap={info['black_captured']}")
    return " ".join(parts) if parts else "-"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",  action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--demo-steps", type=int, default=300)
    args = parser.parse_args()

    if args.demo:
        run_demo(n_steps=args.demo_steps, render=not args.no_render)
    elif args.train:
        train_ppo(args.steps)
    else:
        print("Use --demo or --train. See --help.")
