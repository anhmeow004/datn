import os
import argparse
import gymnasium as gym
import torch
import numpy as np
import socket

socket.gethostname = lambda: "CartPole-Agent"

from torch.utils.tensorboard import SummaryWriter
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent, PPOMemory

SOLVED_MEAN_REWARD = 450
SOLVED_WINDOW = 100

PPO_TRAJECTORY_SIZE = 2048
PPO_BATCH_SIZE = 64


def train_dqn(env, log_dir, save_path):
    writer = SummaryWriter(os.path.join(log_dir, "DQN"))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    epsilon = 1.0
    eps_decay = 0.995
    eps_min = 0.01
    best_reward = 0

    if os.path.exists(save_path):
        print(f"Dang kiem tra diem so cao nhat cua model cu ({save_path})...")
        agent.load(save_path)
        test_rewards = []
        for _ in range(5):
            s, _ = env.reset()
            r_sum, done = 0, False
            while not done:
                a = agent.select_action(s, 0.0)
                s, r, t, tr, _ = env.step(a)
                done = t or tr
                r_sum += r
            test_rewards.append(r_sum)
        best_reward = np.mean(test_rewards)
        print(f"Model cu dang co diem trung binh: {best_reward:.2f}")

    total_rewards = []
    ep = 0

    while True:
        state, _ = env.reset()
        ep_reward, ep_length, done = 0, 0, False
        ep_losses = []

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)
            state = next_state
            ep_reward += reward
            ep_length += 1

        epsilon = max(eps_min, epsilon * eps_decay)
        total_rewards.append(ep_reward)
        mean_rewards = float(np.mean(total_rewards[-SOLVED_WINDOW:]))

        writer.add_scalar("Reward", ep_reward, ep)
        writer.add_scalar("Reward_100", mean_rewards, ep)
        writer.add_scalar("EpisodeLength", ep_length, ep)
        writer.add_scalar("Epsilon", epsilon, ep)
        if ep_losses:
            mean_loss = float(np.mean(ep_losses))
            writer.add_scalar("Loss/DQN", mean_loss, ep)
        loss_str = f", Loss: {np.mean(ep_losses):.4f}" if ep_losses else ""

        print(
            f"Episode {ep}, Reward: {ep_reward:.2f}, "
            f"Mean_{SOLVED_WINDOW}: {mean_rewards:.2f}, "
            f"Epsilon: {epsilon:.2f}{loss_str}"
        )

        if mean_rewards > best_reward:
            best_reward = mean_rewards
            agent.save(save_path)

        if mean_rewards >= SOLVED_MEAN_REWARD:
            print(f"Solved in {ep} episodes! Mean reward (last {SOLVED_WINDOW}) = {mean_rewards:.2f}")
            break

        ep += 1

    writer.close()
    return agent


def train_ppo(env, log_dir, save_path):
    writer = SummaryWriter(os.path.join(log_dir, "PPO"))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)
    memory = PPOMemory()

    best_mean_reward = 0.0

    if os.path.exists(save_path):
        agent.load(save_path)
        print(f"Loaded PPO model: {save_path}")

    total_rewards = []
    ep = 0
    timestep = 0
    update_count = 0
    actor_loss = 0.0
    critic_loss = 0.0

    while True:
        state, _ = env.reset()
        ep_reward, ep_length, done = 0, 0, False

        while not done:
            action, logprob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(terminated)
            memory.values.append(value)

            state = next_state
            ep_reward += reward
            ep_length += 1
            timestep += 1

            if timestep % PPO_TRAJECTORY_SIZE == 0:

                if done:
                    last_value = 0.0
                else:
                    with torch.no_grad():
                        _, lv = agent.policy_old(
                            torch.FloatTensor(next_state).to(agent.device)
                        )
                    last_value = lv.item()

                actor_loss, critic_loss = agent.update(
                    memory,
                    last_value,
                    batch_size=PPO_BATCH_SIZE
                )

                memory.clear()

                writer.add_scalar(
                    "Loss/PPO_Actor",
                    actor_loss,
                    update_count
                )

                writer.add_scalar(
                    "Loss/PPO_Critic",
                    critic_loss,
                    update_count
                )

                update_count += 1

        total_rewards.append(ep_reward)
        mean_rewards = float(np.mean(total_rewards[-SOLVED_WINDOW:]))

        writer.add_scalar("Reward", ep_reward, ep)
        writer.add_scalar("Reward_100", mean_rewards, ep)
        writer.add_scalar("EpisodeLength", ep_length, ep)

        if update_count > 0:
            print(
                f"Episode {ep}, Reward: {ep_reward:.2f}, "
                f"Mean_{SOLVED_WINDOW}: {mean_rewards:.2f}, "
                f"ActorLoss: {actor_loss:.4f}, "
                f"CriticLoss: {critic_loss:.4f}"
            )
        else:
            print(
                f"Episode {ep}, Reward: {ep_reward:.2f}, "
                f"Mean_{SOLVED_WINDOW}: {mean_rewards:.2f}"
            )

        
        if mean_rewards > best_mean_reward:
            best_mean_reward = mean_rewards
            agent.save(save_path)

        if mean_rewards >= SOLVED_MEAN_REWARD:
            print(f"Solved in {ep} episodes! Mean reward ({SOLVED_WINDOW}) = {mean_rewards:.2f}")
            agent.save(save_path)
            break

        ep += 1

    writer.close()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    env_name = "CartPole-v1"
    render_mode = "human" if args.render else None
    env = gym.make(env_name, render_mode=render_mode)

    save_path = f"models/{args.algo}_best.pth"
    if args.algo == "dqn":
        train_dqn(env, "logs", save_path)
    else:
        train_ppo(env, "logs", save_path)