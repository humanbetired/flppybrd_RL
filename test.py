import numpy as np
import pygame
import torch
import gym
from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import os

MODEL_PATH = "model/best_flappy_model.pth"
MAX_STEPS = 1000

def test():
    env = FlappyBirdEnv(render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: File model {MODEL_PATH} tidak ditemukan. Jalankan train.py terlebih dahulu.")
        return

    agent.model.load_state_dict(torch.load(MODEL_PATH))
    agent.model.eval()
    agent.epsilon = 0.0

    print("Memulai visualisasi dengan model terbaik...")
    state, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    while not done and step_count < MAX_STEPS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        action = agent.act(state)
        q_values = agent.model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
        print(f"Langkah {step_count}: State: {state}, Action: {action}, Q-values: {q_values}")
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward
        env.render()
        step_count += 1

    print(f"Visualisasi selesai. Skor akhir: {env.score}, Total Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    test()