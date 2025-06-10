import numpy as np
import pygame
import torch
import gym
from flappy_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import os
import matplotlib.pyplot as plt

# Hyperparameters
EPISODES = 15000
MAX_STEPS = 2000
UPDATE_TARGET_EVERY = 200
MODEL_PATH = "model/best_flappy_model.pth"

def train():
    env = FlappyBirdEnv(render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    best_score = -float('inf')
    best_reward = -float('inf')
    scores = []
    rewards = []
    losses = []
    global_step = 0

    for e in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        episode_loss = 0
        loss_count = 0
        episode_score = 0
        q_values_log = []

        while not done and step_count < MAX_STEPS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            global_step += 1
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done or truncated)
            state = next_state
            total_reward += reward + 0.1  # Reward kecil per langkah
            episode_score = max(episode_score, env.score)

            q_values = agent.model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
            q_values_log.append(q_values[0])

            loss = agent.train()
            if loss is not None:
                episode_loss += loss
                loss_count += 1

            if global_step % UPDATE_TARGET_EVERY == 0:
                agent.update_target()

            agent.decay_epsilon()
            step_count += 1

        if loss_count > 0:
            episode_loss /= loss_count
        losses.append(episode_loss)
        scores.append(episode_score)
        rewards.append(total_reward)

        avg_q_values = np.mean(q_values_log, axis=0) if q_values_log else np.zeros(2)
        print(f"Episode: {e+1}/{EPISODES}, Skor: {episode_score}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Loss: {episode_loss:.4f}, Avg Q-values: {avg_q_values}")

        if episode_score > best_score or (episode_score == best_score and total_reward > best_reward):
            best_score = episode_score
            best_reward = total_reward
            torch.save(agent.model.state_dict(), MODEL_PATH)
            print(f"Model terbaik disimpan pada episode {e+1} dengan skor {best_score} dan reward {total_reward:.2f}")

        if e % 100 == 99:
            avg_score = np.mean(scores[-100:])
            avg_reward = np.mean(rewards[-100:])
            print(f"Rata-rata 100 episode terakhir - Skor: {avg_score:.2f}, Reward: {avg_reward:.2f}")

    env.close()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Skor')
    plt.title('Skor per Episode')
    
    plt.subplot(1, 3, 2)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    
    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss per Episode')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

if __name__ == "__main__":
    train()