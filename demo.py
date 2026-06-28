import os
import argparse
import gymnasium as gym
import torch
import numpy as np
import time
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent


def add_noise(state, std=0.05):
    """Thêm nhiễu Gaussian vào quan sát."""
    return state + np.random.normal(0, std, size=state.shape).astype(np.float32)


def run_episode(agent, env, algo, state, use_noise=False):
    ep_reward = 0
    done = False
    step = 0

    while not done:
        obs = add_noise(state) if use_noise else state

        if algo == "dqn":
            action = agent.select_action(obs, epsilon=0.0)
        else:
            action, _, _ = agent.select_action(obs, deterministic=True)

        state, reward, terminated, truncated, _ = env.step(action)
        state = state.astype(np.float32)
        done = terminated or truncated
        ep_reward += reward
        step += 1
        time.sleep(0.02)

    return ep_reward, step


def run_demo(algo, episodes, mode, render_mode="human"):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if algo == "dqn":
        agent = DQNAgent(state_dim, action_dim)
        model_path = "models/dqn_best.pth"
    else:
        agent = PPOAgent(state_dim, action_dim)
        model_path = "models/ppo_best.pth"

    if os.path.exists(model_path):
        print(f"--- Đang tải mô hình {algo.upper()} từ {model_path} ---\n")
        agent.load(model_path)
    else:
        print(f"Lỗi: Không tìm thấy {model_path}. Hãy chạy train.py trước!")
        return

    use_noise = (mode == "noisy")
    if use_noise:
        print("[Chế độ Nhiễu] Thêm Gaussian noise (std=0.05) vào mỗi quan sát\n")

    for ep in range(episodes):
        state, _ = env.reset()
        state = state.astype(np.float32)
        print(f"Hiệp {ep + 1}...")
        reward, steps = run_episode(agent, env, algo, state, use_noise=use_noise)
        print(f"  → {steps} bước | Điểm: {reward:.0f}\n")
        time.sleep(0.8)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "noisy"])
    args = parser.parse_args()
    run_demo(args.algo, args.episodes, args.mode)