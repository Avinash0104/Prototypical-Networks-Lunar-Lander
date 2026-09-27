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

def rollout_returns(agent, num_episodes=30, seed=1000, frame_stack=4):
    env = Environment(seed=seed)
    stacker = FrameStacker(frame_stack)
    returns = []
    for ep in range(num_episodes):
        frame, info = env.reset(seed=seed + ep)
        obs_stack = stacker.reset(frame)
        done = False
        ep_return = 0.0
        while not done:
            obs_t = torch.from_numpy(obs_stack).float().unsqueeze(0)
            with torch.no_grad():
                z = agent.encoder(obs_t)
                action = agent.actor.deterministic(z)
                action = action.squeeze(0).numpy()
            frame, reward, terminated, truncated, info = env.step(action)
            obs_stack = stacker.push(frame)
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
            returns = rollout_returns(agent, num_episodes=num_episodes, seed=1000)

            row = {
                "arm": arm,
                "seed": seed,
                "return_mean": np.mean(returns),
                "return_std": np.std(returns),
            }
            rows.append(row)

            print(f"{run:20s} return_mean={row['return_mean']:8.1f} return_std={row['return_std']:6.1f}")

    return rows

if __name__ == "__main__":
    main()