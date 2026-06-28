import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from src.models.networks import ActorCritic


class PPOAgent:
    def __init__(self, state_dim, action_dim,
                 lr_actor=1e-3, lr_critic=3e-3,
                 gamma=0.99, K_epochs=4, eps_clip=0.2, lam=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lam = lam

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        
        self.optimizer_actor = optim.Adam(self.policy.actor_net.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.policy.critic_net.parameters(), lr=lr_critic)

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            logits, value = self.policy_old(state)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    def _compute_gae(self, rewards, values, is_terminals, last_value):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(len(rewards))):
            mask = 0.0 if is_terminals[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def update(self, memory, last_value=0.0, batch_size=64):
        old_states   = torch.FloatTensor(np.array(memory.states)).to(self.device)
        old_actions  = torch.LongTensor(np.array(memory.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs)).to(self.device)
        old_values   = np.array(memory.values, dtype=np.float32)

        advantages_np, returns_np = self._compute_gae(
            memory.rewards, old_values, memory.is_terminals, last_value
        )

        advantages = torch.FloatTensor(advantages_np).to(self.device)
        returns    = torch.FloatTensor(returns_np).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        n = len(old_states)

        last_actor_loss = 0.0
        last_critic_loss = 0.0
            
        for _ in range(self.K_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = torch.LongTensor(indices[start:start + batch_size]).to(self.device)

                b_states   = old_states[idx]
                b_actions  = old_actions[idx]
                b_logprobs = old_logprobs[idx]
                b_adv      = advantages[idx]
                b_returns  = returns[idx]

                logits, state_values = self.policy(b_states)
                probs = torch.softmax(logits, dim=-1)
                dist  = Categorical(probs)
                logprobs     = dist.log_prob(b_actions)
                dist_entropy = dist.entropy()
                state_values = state_values.squeeze(-1)

                ratios = torch.exp(logprobs - b_logprobs.detach())
                surr1  = ratios * b_adv
                surr2  = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv

                actor_loss = (-torch.min(surr1, surr2) - 0.01 * dist_entropy).mean()
                last_actor_loss = actor_loss.item()
                self.optimizer_actor.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy.actor_net.parameters(), 0.5)
                self.optimizer_actor.step()

                critic_loss = self.MseLoss(state_values, b_returns)
                last_critic_loss = critic_loss.item()
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.critic_net.parameters(), 0.5)
                self.optimizer_critic.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        return last_actor_loss, last_critic_loss

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())


class PPOMemory:
    def __init__(self):
        self.actions     = []
        self.states      = []
        self.logprobs    = []
        self.rewards     = []
        self.is_terminals = []
        self.values      = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]