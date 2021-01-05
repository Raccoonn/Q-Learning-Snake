

import pygame
import numpy as np




class snakeGame_v3:
    """
    Return state as 0, 1 array of obeservations

    See: https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36

    """
    def __init__(self, screen_Width=800, screen_Height=800, N_sqrs=25, difficulty=25):

        self.screen_Width = screen_Width
        self.screen_Height = screen_Height

        self.N_sqrs = N_sqrs
        self.difficulty = difficulty

        self.dx = screen_Width // N_sqrs
        self.dy = screen_Height // N_sqrs


    def state_observation(self):
        """
        Preform state observation, returning 0, 1 array as describe in article.
        """
        state = np.zeros(12)
        
        # Create state observation
        sx, sy = self.snake_head
        ax, ay = self.food_pos
        
        # Food locations
        if ay < sy:
            state[0] = 1
        if ax > sx:
            state[1] = 1
        if ay > sy:
            state[2] = 1
        if ax < sx:
            state[3] = 1
        
        # Obstacles
        # walls
        if sx == 0:
            state[4] = 1
        if sy == self.N_sqrs - 1:
            state[5] = 1
        if sx == self.N_sqrs - 1:
            state[6] = 1
        if sy == 0:
            state[7] = 1

        # body
        hx, hy = self.snake_head
        for seg in self.snake_body[3:]:
            if seg[0] == hx and seg[1] == hy-1:
                state[4] = 1
            if seg[1] == hy and seg[0] == hx+1:
                state[5] = 1
            if seg[0] == hx and seg[1] == hy+1:
                state[6] = 1
            if seg[1] == hy and seg[0] == hx-1:
                state[7] = 1

        # Direction
        if self.direction == 'UP':
            state[8] = 1
        if self.direction == 'RIGHT':
            state[9] = 1
        if self.direction == 'DOWN':
            state[10] = 1
        if self.direction == 'LEFT':
            state[11] = 1

        return state

    
    def reset(self):
        """
        Reset snake body and direction to starting location,
        also spawn a new food at a random location
        """
        self.direction = 'RIGHT'

        self.snake_head = [4, 8]
        self.snake_body = [[4, 8], [4, 7], [4, 6]]

        self.last_len = len(self.snake_body)

        self.food_pos = [np.random.randint(5, self.N_sqrs),
                         np.random.randint(5, self.N_sqrs)]

        self.score = 0
        
        done = False
        reward = 0

        state = self.state_observation()

        return done, reward, state




    def step(self, action, frame, buffer=500):
        """
            Reward: - +10 for eating
                    - +1 for going closer
                    - -1 for going further
                    - -100 for crashing
        """
        done = False
        reward = 0

        # Making sure the snake cannot move in the opposite self.direction instantaneously
        if action == 0 and self.direction != 'DOWN':
            self.direction = 'UP'
        if action == 1 and self.direction != 'UP':
            self.direction = 'DOWN'
        if action == 2 and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if action == 3 and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        # Track apple distance
        ax = abs(self.snake_head[0] - self.food_pos[0])
        ay = abs(self.snake_head[1] - self.food_pos[1])

        # Moving the snake
        if self.direction == 'UP':
            self.snake_head[1] -= 1
        if self.direction == 'DOWN':
            self.snake_head[1] += 1
        if self.direction == 'LEFT':
            self.snake_head[0] -= 1
        if self.direction == 'RIGHT':
            self.snake_head[0] += 1

        ax_ = abs(self.snake_head[0] - self.food_pos[0])
        ay_ = abs(self.snake_head[1] - self.food_pos[1]) 

        # Tracking apple distance before and after moving, assign
        # reward if closer to apple or further
        if ax_ < ax or ay_ < ay:
            reward += 1
        elif ax_ > ax or ay_ > ay:
            reward -= 1


        self.snake_body.insert(0, list(self.snake_head))

        # Check if snake eats food
        if self.snake_head[0] == self.food_pos[0] and self.snake_head[1] == self.food_pos[1]:
            reward += 10
            self.score += 1
            while True:
                self.food_pos = [np.random.randint(5, self.N_sqrs), 
                                 np.random.randint(5, self.N_sqrs)]
                if self.food_pos not in self.snake_body:
                    break
        else:
            self.snake_body.pop()


        # Flag to break snake looping
        if frame % buffer == 0 and self.last_len == len(self.snake_body):
            done = True
            reward -= 300
        else:
            self.last_len = len(self.snake_body)


        # Getting out of bounds
        if self.snake_head[0] < 0 or self.snake_head[0] == self.N_sqrs:
            done = True
            reward -= 100
        if self.snake_head[1] < 0 or self.snake_head[1] == self.N_sqrs:
            done = True
            reward -= 100
        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_head[0] == block[0] and self.snake_head[1] == block[1]:
                done = True
                reward -= 100


        state = self.state_observation()

        # Check is surrounded on 2 sides by obstacles and break
        # pseudo lose condition to stop looping
        if state[4] == 1 and state[6] == 1:
            done = True
            reward -= 100
        if state[5] == 1 and state[7] == 1:
            done = True
            reward -= 100


        return done, reward, state





    def setup_window(self):
        """
        Setup game window if rendering
        """
        pygame.init()
        pygame.display.set_caption('Snake_v3')
        self.screen = pygame.display.set_mode((self.screen_Width, self.screen_Height))

        self.black = pygame.Color(0, 0, 0)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)

        self.clock = pygame.time.Clock()




    def render(self):
        """
        Render the grid on screen
        """
        pygame.display.flip()
        self.screen.fill(self.black)

        # Update grid locations for snake and food
        grid = np.zeros((self.N_sqrs, self.N_sqrs))

        for idx in self.snake_body:
            grid[idx[0], idx[1]] = 1

        grid[self.food_pos[0], self.food_pos[1]] = 2

        # Draw squares according to grid
        for i in range(self.N_sqrs):
            for j in range(self.N_sqrs):
                # snake
                if grid[i, j] == 1:
                    pygame.draw.rect(self.screen, self.green, 
                                     pygame.Rect(i*self.dx, j*self.dy, self.dx, self.dy))
                # food
                elif grid[i, j] == 2:
                    pygame.draw.rect(self.screen, self.red,
                                     pygame.Rect(i*self.dx, j*self.dy, self.dx, self.dy))

        self.clock.tick(self.difficulty)





