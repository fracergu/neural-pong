import pygame
import sys
import random
import argparse
import numpy as np
import tensorflow as tf
import os
from collections import deque



# Pygame initialization
pygame.init()

# CLI arguments
parser = argparse.ArgumentParser(description="Pong Game with Neural Network Training")
parser.add_argument("--mode", choices=["train", "play"], required=True, help="Mode to run the game in. 'train' for training the model, 'play' for playing against the trained model.")
args = parser.parse_args()

# Constants
WIDTH, HEIGHT = 640, 480
PADDLE_WIDTH, PADDLE_HEIGHT = 15, 75
BALL_WIDTH, BALL_HEIGHT = 15, 15
SPEED = 5
MEMORY_SIZE = 1000
BATCH_SIZE = 32
GAMMA = 0.95
REFRESH_RATE = 1000

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

font = pygame.font.SysFont(None, 55)

# Define neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(3, activation='softmax')
])


# Create the model if it doesn't exist
if not os.path.exists("pong_model.h5"):
    model.compile(optimizer='adam', loss='mse')
    model.save_weights("pong_model.h5")


if args.mode == "play":
    model.load_weights("pong_model.h5")

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()

memory = deque(maxlen=MEMORY_SIZE)

def get_action(state):
    action_probabilities = model.predict(np.array([state]))
    return np.argmax(action_probabilities[0])

def train_model():
    if len(memory) < BATCH_SIZE:
        return

    minibatch = random.sample(memory, BATCH_SIZE)
    
    states = np.array([i[0] for i in minibatch])
    actions = np.array([i[1] for i in minibatch])
    rewards = np.array([i[2] for i in minibatch])
    next_states = np.array([i[3] for i in minibatch])

    target = rewards + GAMMA * np.amax(model.predict_on_batch(next_states), axis=1)
    target_f = model.predict_on_batch(states)
    for i, action in enumerate(actions):
        target_f[i][action] = target[i]
    
    with tf.GradientTape() as tape:
        predictions = model(states)
        loss = loss_fn(target_f, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Classes
class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.first_hit = True
        self.reset()  

    def move(self, left_paddle, right_paddle):
        self.x += self.dx
        self.y += self.dy

        # Bounce off the top and bottom edges
        if self.y <= 0 or self.y >= HEIGHT - BALL_HEIGHT:
            self.dy = -self.dy

        # Bounce off the paddles
        if (self.dx < 0 and left_paddle.x < self.x < left_paddle.x + PADDLE_WIDTH and left_paddle.y < self.y < left_paddle.y + PADDLE_HEIGHT) or \
           (self.dx > 0 and right_paddle.x < self.x < right_paddle.x + PADDLE_WIDTH and right_paddle.y < self.y < right_paddle.y + PADDLE_HEIGHT):
            if self.first_hit:
                self.dx = -self.dx * 1.5 
                self.dy *= 1.5 
                self.first_hit = False
            else:
                self.dx = -self.dx

        # Reset the ball if it goes past the paddles
        if self.x < 0:
            self.reset()
            return 'right'
        elif self.x > WIDTH:
            self.reset()
            return 'left'
        return None

    def reset(self):
        self.x = WIDTH // 2 - BALL_WIDTH // 2
        self.y = HEIGHT // 2 - BALL_HEIGHT // 2
        self.dx = random.choice([-1, 1]) * (SPEED - 2)  
        self.dy = random.choice([-1, 1]) * random.uniform(0.5, 1.0) * (SPEED - 2)
        self.first_hit = True

    def draw(self):
        pygame.draw.rect(screen, WHITE, pygame.Rect(self.x, self.y, BALL_WIDTH, BALL_HEIGHT))

class Paddle:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, dy):
        self.y += dy
        self.y = max(0, min(HEIGHT - PADDLE_HEIGHT, self.y))

    def draw(self):
        pygame.draw.rect(screen, WHITE, pygame.Rect(self.x, self.y, PADDLE_WIDTH, PADDLE_HEIGHT))

ball = Ball(WIDTH // 2 - BALL_WIDTH // 2, HEIGHT // 2 - BALL_HEIGHT // 2)
left_paddle = Paddle(0, HEIGHT // 2 - PADDLE_HEIGHT // 2)
right_paddle = Paddle(WIDTH - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2)

left_score = 0
right_score = 0


try:
    iteration_count = 0  
    while True:
        iteration_count += 1

        if iteration_count % REFRESH_RATE == 0:
            model.save_weights("pong_model.h5")
            print(f"Iteration {iteration_count}, Score: {left_score} - {right_score}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Game state
        left_state = [ball.x - left_paddle.x, ball.y - left_paddle.y, ball.dx, ball.dy, SPEED]
        right_state = [right_paddle.x - ball.x, ball.y - right_paddle.y, -ball.dx, -ball.dy, SPEED]

        # Training mode: both paddles controlled by the NN
        if args.mode == "train":
            left_action = get_action(left_state)
            right_action = get_action(right_state)

            if left_action == 0:
                left_paddle.move(-SPEED)
            elif left_action == 1:
                left_paddle.move(SPEED)
            
            if right_action == 0:
                right_paddle.move(-SPEED)
            elif right_action == 1:
                right_paddle.move(SPEED)

        # Play mode: left paddle controlled by the player, right paddle controlled by the NN
        elif args.mode == "play":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                left_paddle.move(-SPEED)
            if keys[pygame.K_s]:
                left_paddle.move(SPEED)

            left_action = -1  # Dummy value
            right_action = get_action(right_state)
            if right_action == 0:
                right_paddle.move(-SPEED)
            elif right_action == 1:
                right_paddle.move(SPEED)

        result = ball.move(left_paddle, right_paddle)
        if result == 'left':
            left_score += 1
            reward = 1
            memory.append((left_state, left_action, reward, [ball.x - left_paddle.x, ball.y - left_paddle.y, ball.dx, ball.dy, SPEED]))
            memory.append((right_state, right_action, -reward, [right_paddle.x - ball.x, ball.y - right_paddle.y, -ball.dx, -ball.dy, SPEED]))

        elif result == 'right':
            right_score += 1
            reward = 1
            memory.append((right_state, right_action, reward, [right_paddle.x - ball.x, ball.y - right_paddle.y, -ball.dx, -ball.dy, SPEED]))
            memory.append((left_state, left_action, -reward, [ball.x - left_paddle.x, ball.y - left_paddle.y, ball.dx, ball.dy, SPEED]))

        if args.mode == "train":
            train_model()

        score_display = font.render(f"{left_score} - {right_score}", True, WHITE)

        screen.fill(BLACK)
        ball.draw()
        left_paddle.draw()
        right_paddle.draw()
        screen.blit(score_display, (WIDTH // 2 - score_display.get_width() // 2, 10))
        pygame.display.flip()
        pygame.time.Clock().tick(60)
      
except KeyboardInterrupt:
    model.save_weights("pong_model.h5")
    pygame.quit()
    sys.exit()
