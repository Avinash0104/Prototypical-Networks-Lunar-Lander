import os
import numpy as np
import torch

from experiment.training import Agent
from experiment.environment import Environment, FrameStacker

def load_agent(arm, checkpoint_path, device="cpu"):
    agent = Agent(arm=arm, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.encoder.load_state_dict(checkpoint["encoder"])
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.encoder.eval()
    agent.actor.eval()
    return agent

def rollout_returns(agent, num_episodes=30, seed=1000):
    env = Environment(seed=seed)
    stacker = FrameStacker()
    returns = []
    for ep in range(num_episodes):
        frame, info = env.reset(seed=seed + ep)
        stack = stacker.reset(frame)
        done = False
        ep_return = 0.0
        while not done:
            stack_t = torch.from_numpy(stack).float().unsqueeze(0)
            with torch.no_grad():
                z = agent.encoder(stack_t)
                action = agent.actor.deterministic(z)
                action = action.squeeze(0).numpy()
            frame, reward, terminated, truncated, info = env.step(action)
            stack = stacker.step(frame)
            done = terminated or truncated
            ep_return += reward
        returns.append(ep_return)
    return np.array(returns)

def main(dir="results", arms=("baseline", "prototype", "classifier"), seeds=(0, 1, 2), checkpoint_step=500_000, num_episodes=30):
    rows = []
    for arm in arms:
        for seed in seeds:
            run = f"{arm}_seed{seed}"
            checkpoint_path = os.path.join(dir, run, f"checkpoint_{checkpoint_step}.pt")
            agent = load_agent(arm, checkpoint_path)
            returns = rollout_returns(agent, num_episodes=num_episodes)

            row = {
                "arm": arm,
                "seed": seed,
                "mean_return": np.mean(returns),
            }
            rows.append(row)

            print(f"{run:20s} mean_return={row['mean_return']:8.1f}")

    return rows

if __name__ == "__main__":
    main()
