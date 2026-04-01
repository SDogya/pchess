import argparse
import numpy as np
import gymnasium as gym
import os
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from parallel_chess import (
    SimultaneousChessEnv,
    SingleAgentSelfPlayWrapper,
    random_opponent_policy,
)

# Отключаем валидацию аргументов распределения, чтобы избежать падения на Simplex() constraint
# Это критично для шахмат, где из-за fp32 сумма вероятностей может быть 0.999999 вместо 1.0
torch.distributions.Distribution.set_default_validate_args(False)

def train_ppo(total_timesteps: int, load_path: str, save_path: str, resume: bool):
    """
    Основной цикл обучения.
    """
    
    # Исправленная функция маскирования
    def mask_fn(env: gym.Env) -> np.ndarray:
        # Мы обращаемся к обертке SelfPlay, которая умеет отдавать плоскую маску 4096
        # Если метод в обертке называется action_masks, вызываем его
        return env.action_masks()

    # Сборка окружения
    base_env = SimultaneousChessEnv(max_steps=200)
    # Обертка берет на себя логику "один агент играет за обе стороны или против бота"
    # и преобразование координат для черных
    env = SingleAgentSelfPlayWrapper(base_env, opponent_policy=random_opponent_policy)
    # Стандартная обертка для MaskablePPO
    env = ActionMasker(env, action_mask_fn=mask_fn)

    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Используемое устройство: {device}")

    # Логика загрузки/создания модели
    if load_path and os.path.exists(load_path):
        print(f"[*] Загрузка модели из: {load_path}")
        model = MaskablePPO.load(load_path, env=env, device=device)
    elif resume and os.path.exists("ppo_parallel_chess.zip"):
        print("[*] Продолжение обучения модели ppo_parallel_chess.zip")
        model = MaskablePPO.load("ppo_parallel_chess.zip", env=env, device=device)
    else:
        print("[!] Создание новой модели MaskablePPO...")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=128, # Увеличил для стабильности на GPU
            n_epochs=10,
            learning_rate=3e-4,
            device=device,
            tensorboard_log="./ppo_chess_tensorboard/"
        )

    print(f"[*] Запуск обучения на {total_timesteps} шагов...")
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            reset_num_timesteps=not resume,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[!] Обучение прервано пользователем. Сохраняем текущее состояние...")

    # Сохранение
    final_name = save_path if save_path else "ppo_parallel_chess.zip"
    if not final_name.endswith(".zip"):
        final_name += ".zip"
    
    model.save(final_name)
    print(f"[*] Модель сохранена как: {final_name}")


def run_demo(n_steps: int = 300, render: bool = True):
    """
    Демонстрация случайной игры для проверки визуализации и наград.
    """
    env = SimultaneousChessEnv(render_mode="human" if render else "ansi", max_steps=200)
    obs, _ = env.reset()
    total_rewards = {"white": 0.0, "black": 0.0}

    for step in range(n_steps):
        # Берем случайные легальные ходы из окружения
        legal_w = env.get_legal_moves(1)
        legal_b = env.get_legal_moves(-1)

        move_w = legal_w[np.random.randint(len(legal_w))] if legal_w else (0, 0)
        move_b = legal_b[np.random.randint(len(legal_b))] if legal_b else (0, 0)

        obs, _, terminated, truncated, info = env.step({"white": move_w, "black": move_b})

        rewards = info["rewards"]
        total_rewards["white"] += rewards["white"]
        total_rewards["black"] += rewards["black"]

        if (step + 1) % 10 == 0 or terminated or truncated:
            print(f"Step {step+1:3d} | W_total:{total_rewards['white']:5.1f} B_total:{total_rewards['black']:5.1f} | {_fmt_info(info)}")

        if render:
            env.render()
        
        if terminated or truncated:
            print(f"--- GAME OVER at step {step+1} ---")
            break

    env.close()

def _fmt_info(info: dict) -> str:
    """Форматирование отладочной информации."""
    p = []
    if info.get("white_captured"): p.append(f"W_cap:{info['white_captured']}")
    if info.get("black_captured"): p.append(f"B_cap:{info['black_captured']}")
    if info.get("white_promoted_count"): p.append(f"W_PROM:{info['white_promoted_count']}")
    if info.get("black_promoted_count"): p.append(f"B_PROM:{info['black_promoted_count']}")
    if info.get("mutual_destruction"): p.append("MUTUAL_DEST")
    if info.get("swap_collision"): p.append("SWAP")
    return " | ".join(p) if p else "no events"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Chess Training")
    parser.add_argument("--demo", action="store_true", help="Запустить демо со случайными ходами")
    parser.add_argument("--train", action="store_true", help="Запустить обучение")
    parser.add_argument("--steps", type=int, default=100000, help="Количество шагов обучения")
    parser.add_argument("--load-path", type=str, help="Путь к загружаемой модели")
    parser.add_argument("--save-path", type=str, help="Путь для сохранения модели")
    parser.add_argument("--resume", action="store_true", help="Продолжить обучение существующей модели")
    parser.add_argument("--no-render", action="store_true", help="Отключить рендер в демо")
    
    args = parser.parse_args()

    if args.demo:
        run_demo(render=not args.no_render)
    elif args.train:
        train_ppo(args.steps, args.load_path, args.save_path, args.resume)
    else:
        print("Используйте флаг --train или --demo. Справка: --help")