import os
import numpy as np
import torch

from experiment.training import Agent
from experiment.environment import Environment

def load_agent(arm, checkpoint_path, device="cpu"):
    agent = Agent(arm=arm, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.encoder.load_state_dict(checkpoint["encoder"])
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.encoder.eval()
    agent.actor.eval()
    return agent

def rollout_returns(agent, noise_mode, num_episodes=30, seed=1000):
    env = Environment(noise_mode=noise_mode, seed=seed)
    returns = []
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                z = agent.encoder(obs_t)
                action = agent.actor.deterministic(z)
                action = action.squeeze(0).numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
        returns.append(ep_return)
    return np.array(returns)

def main(dir="results", arms=("baseline", "prototype", "classifier"), seeds=(0, 1, 2), checkpoint_step=600_000, num_episodes=30):
    rows = []
    for arm in arms:
        for seed in seeds:
            run = f"{arm}_seed{seed}"
            checkpoint_path = os.path.join(dir, run, f"checkpoint_{checkpoint_step}.pt")
            agent = load_agent(arm, checkpoint_path)
            train_returns = rollout_returns(agent, noise_mode="walk", num_episodes=num_episodes)
            shift_returns = rollout_returns(agent, noise_mode="iid", num_episodes=num_episodes)

            row = {
                "arm": arm,
                "seed": seed,
                "train_mean": np.mean(train_returns),
                "shift_mean": np.mean(shift_returns),
                "degradation": train_returns.mean() - shift_returns.mean()
            }
            rows.append(row)

            print(f"{run:20s} train_mean={row['train_mean']:8.1f} shift_mean={row['shift_mean']:8.1f} degradation={row['degradation']:8.1f}")

    return rows

if __name__ == "__main__":
    main()