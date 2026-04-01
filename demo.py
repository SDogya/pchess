import torch
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from parallel_chess import SimultaneousChessEnv
from parallel_chess.core.rules import get_pseudo_legal_moves

def get_agent_obs(board: np.ndarray, color: int) -> np.ndarray:
    obs = board * color
    return np.flipud(obs) if color == -1 else obs

def get_action_mask(board: np.ndarray, color: int) -> np.ndarray:
    mask_4d = get_pseudo_legal_moves(board, color)
    if color == -1:
        mask_4d = mask_4d[::-1, :, ::-1, :]
    
    flat = mask_4d.flatten().astype(bool)
    if not np.any(flat):
        flat[0] = True 
    return flat

def decode_action(action: int, color: int) -> tuple[int, int]:
    fr, ff = (action // 64) // 8, (action // 64) % 8
    tr, tf = (action % 64) // 8, (action % 64) % 8
    if color == -1:
        fr, tr = 7 - fr, 7 - tr
    return (fr * 8 + ff, tr * 8 + tf)

def get_deterministic_action(model: MaskablePPO, obs: np.ndarray, mask: np.ndarray) -> int:
    obs_batched = np.expand_dims(obs, axis=0)
    obs_tensor, _ = model.policy.obs_to_tensor(obs_batched)
    
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=model.device).unsqueeze(0)
    
    with torch.no_grad():
        features = model.policy.extract_features(obs_tensor)
        latent_pi = model.policy.mlp_extractor.forward_actor(features)
        logits = model.policy.action_net(latent_pi)
        
        logits[~mask_tensor] = -1e8
        
        if torch.isnan(logits).any():
            valid_idx = torch.where(mask_tensor[0])[0]
            return valid_idx[0].item() if len(valid_idx) > 0 else 0
            
        return torch.argmax(logits, dim=1).item()

def run_match(model_path_white: str, model_path_black: str, max_steps: int = 200) -> None:
    env = SimultaneousChessEnv(render_mode="human", max_steps=max_steps)
    
    print(f"Загрузка модели белых: {model_path_white}")
    model_w = MaskablePPO.load(model_path_white)
    
    print(f"Загрузка модели черных: {model_path_black}")
    model_b = MaskablePPO.load(model_path_black)
    
    env.reset()
    env.render()
    
    for step in range(max_steps):
        board = env.board
        
        # Инференс модели белых
        obs_w = get_agent_obs(board, 1)
        mask_w = get_action_mask(board, 1)
        action_w = get_deterministic_action(model_w, obs_w, mask_w)
        move_w = decode_action(action_w, 1)
        
        # Инференс модели черных
        obs_b = get_agent_obs(board, -1)
        mask_b = get_action_mask(board, -1)
        action_b = get_deterministic_action(model_b, obs_b, mask_b)
        move_b = decode_action(action_b, -1)
        
        _, _, terminated, truncated, info = env.step({
            "white": move_w,
            "black": move_b
        })
        
        env.render()
        
        if terminated or truncated:
            print("\nМатч завершен.")
            if info.get("white_king_dead") and info.get("black_king_dead"):
                print("Результат: Ничья (Взаимное уничтожение королей)")
            elif info.get("black_king_dead"):
                print("Результат: Победа белых")
            elif info.get("white_king_dead"):
                print("Результат: Победа черных")
            else:
                print("Результат: Лимит ходов исчерпан / Пат")
            break
            
    print("Закройте окно графика для выхода.")
    plt.show(block=True)
    env.close()

if __name__ == "__main__":
    # Укажи пути к двум разным чекпоинтам моделей
    run_match(
        model_path_white="ppo_parallel_chess3.zip", 
        model_path_black="lol.zip", 

        max_steps=200
    )