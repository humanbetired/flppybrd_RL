import numpy as np
import pygame
import random
import gym
from gym import spaces
import os

pygame.init()

# Konstanta
WIDTH = 288
HEIGHT = 512
GRAVITY = 0.5
JUMP = -7.5
PIPE_WIDTH = 52
PIPE_GAP = 100
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
BASE_HEIGHT = 112
BASE_SPEED = 3
ANIMATION_SPEED = 0.2

WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
BLACK = (0, 0, 0)

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode="human"):
        super(FlappyBirdEnv, self).__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, -10, -PIPE_WIDTH, 0, -HEIGHT, -HEIGHT], dtype=np.float32),
            high=np.array([HEIGHT, 10, WIDTH, HEIGHT, HEIGHT, HEIGHT], dtype=np.float32),
            dtype=np.float32
        )

        self.assets = {}
        try:
            self.assets['bird'] = [
                pygame.transform.scale(pygame.image.load(os.path.join('assets', 'bird1.png')), (BIRD_WIDTH, BIRD_HEIGHT)),
                pygame.transform.scale(pygame.image.load(os.path.join('assets', 'bird2.png')), (BIRD_WIDTH, BIRD_HEIGHT)),
                pygame.transform.scale(pygame.image.load(os.path.join('assets', 'bird3.png')), (BIRD_WIDTH, BIRD_HEIGHT))
            ]
            self.assets['pipe'] = pygame.transform.scale(pygame.image.load(os.path.join('assets', 'pipe.png')), (PIPE_WIDTH, 320))
            self.assets['background'] = pygame.transform.scale(pygame.image.load(os.path.join('assets', 'background.png')), (WIDTH, HEIGHT))
            self.assets['base'] = pygame.transform.scale(pygame.image.load(os.path.join('assets', 'base.png')), (WIDTH, BASE_HEIGHT))
        except FileNotFoundError:
            print("Assets not found, using fallback rendering")
            self.assets['bird'] = [pygame.Surface((BIRD_WIDTH, BIRD_HEIGHT)) for _ in range(3)]
            for surf in self.assets['bird']:
                surf.fill(RED)
            self.assets['pipe'] = pygame.Surface((PIPE_WIDTH, 320))
            self.assets['pipe'].fill(GREEN)
            self.assets['background'] = pygame.Surface((WIDTH, HEIGHT))
            self.assets['background'].fill(BLUE)
            self.assets['base'] = pygame.Surface((WIDTH, BASE_HEIGHT))
            self.assets['base'].fill(BLACK)

        self.bird_frame = 0
        self.base_x = 0
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.bird_y = HEIGHT // 2
        self.bird_vel = 0
        self.pipe_x = WIDTH
        self.pipe_gap_y = random.randint(150, 400 - PIPE_GAP)
        self.score = 0
        self.done = False
        self.bird_frame = 0
        self.base_x = 0

        if self.render_mode == "human" and self.screen is None:
            pygame.display.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Flappy Bird RL")
            self.clock = pygame.time.Clock()

        return self.get_state(), {}

    def get_state(self):
        pipe_center = self.pipe_gap_y
        dist_to_top = self.pipe_gap_y - PIPE_GAP // 2 - self.bird_y
        dist_to_bottom = self.pipe_gap_y + PIPE_GAP // 2 - self.bird_y
        return np.array([
            self.bird_y / HEIGHT,
            self.bird_vel / 10,
            (self.pipe_x - 50) / WIDTH,
            self.pipe_gap_y / HEIGHT,
            dist_to_top / HEIGHT,
            dist_to_bottom / HEIGHT
        ], dtype=np.float32)

    def step(self, action):
        self.bird_vel += GRAVITY
        if action == 1:
            self.bird_vel = JUMP
        self.bird_y += self.bird_vel

        self.pipe_x -= 3
        self.base_x = (self.base_x - BASE_SPEED) % WIDTH
        reward = 0.1  # Reward dasar per langkah

        if self.pipe_x < -PIPE_WIDTH:
            self.pipe_x = WIDTH
            self.pipe_gap_y = random.randint(150, 400 - PIPE_GAP)
            self.score += 1
            reward = 50.0  # Reward melewati pipa

        # Check collisions
        bird_rect = pygame.Rect(50, int(self.bird_y), BIRD_WIDTH, BIRD_HEIGHT)
        pipe_top_rect = pygame.Rect(self.pipe_x, 0, PIPE_WIDTH, self.pipe_gap_y - PIPE_GAP // 2)
        pipe_bottom_rect = pygame.Rect(self.pipe_x, self.pipe_gap_y + PIPE_GAP // 2, PIPE_WIDTH, HEIGHT)
        if (
            self.bird_y < 0 or
            self.bird_y + BIRD_HEIGHT > HEIGHT - BASE_HEIGHT or
            bird_rect.colliderect(pipe_top_rect) or
            bird_rect.colliderect(pipe_bottom_rect)
        ):
            self.done = True
            reward = -20.0  # Penalti kolisi

        next_state = self.get_state()
        if self.render_mode == "human":
            self.render()

        return next_state, reward, self.done, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        self.screen.blit(self.assets['background'], (0, 0))
        self.screen.blit(self.assets['base'], (self.base_x, HEIGHT - BASE_HEIGHT))
        self.screen.blit(self.assets['base'], (self.base_x + WIDTH, HEIGHT - BASE_HEIGHT))
        pipe_top = pygame.transform.flip(self.assets['pipe'], False, True)
        self.screen.blit(pipe_top, (self.pipe_x, self.pipe_gap_y - PIPE_GAP // 2 - 320))
        self.screen.blit(self.assets['pipe'], (self.pipe_x, self.pipe_gap_y + PIPE_GAP // 2))
        bird = self.assets['bird'][int(self.bird_frame)]
        self.screen.blit(bird, (50, int(self.bird_y)))
        font = pygame.font.SysFont(None, 36)
        score_surface = font.render(f"Skor: {self.score}", True, WHITE)
        self.screen.blit(score_surface, (10, 10))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None