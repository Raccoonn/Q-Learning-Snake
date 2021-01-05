


import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_dqn import Agent
from snakeEnvironment import snakeGame_v3



"""

    Using arrays to create grid,
        - observation is this grid
        - 3 observations make up a state
        - An observation buffer is created to return current states

    - Train with a buffer to prevent looping ~500
        - Remove or increase after training

"""




if __name__ == '__main__':

    strategy = tf.distribute.MirroredStrategy()

    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


    episode_limit = 5000

    show = True

    load = True

    train = False


    screen_Width = 600
    screen_Height = 600

    N_sqrs = 25

    input_dims = 12

    action_space = [0, 1, 2, 3]
    n_actions = len(action_space)

    batch_size = 512
    fc1_dims = 512
    fc2_dims = 256

    filename = 'snake_supertrained.h5'

    # Assign agent
    agent = Agent(alpha=0.00025, gamma=0.95, epsilon=1, epsilon_dec=0.995, epsilon_end=0.01,
                        batch_size=batch_size, input_dims=input_dims, n_actions=n_actions, mem_size=500000,
                        fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                        fname=filename)


    if load == True:
        agent.load_model(filename)
        agent.epsilon = 0
        print('\n\n... Model Loaded ...\n\n')


    env = snakeGame_v3(screen_Width, screen_Height, N_sqrs, difficulty=60)

    if show == True:
        env.setup_window()

    
    reward_store = []
    avg_reward = []
    epsilon_store = []


    p_i, p_syms = 0, ('\\', '|', '/', '-')

    for episode in range(1, episode_limit+1):

        total_reward = 0

        done, reward, state = env.reset()

        frame = 0
        start_time = time.time()
        print('\n')

        while not done:
            print('Playing a game...  ' + p_syms[p_i], end='\r')
            p_i = (p_i+1) % 4

            frame += 1

            if show == True:
                env.render()


            action = agent.choose_action(state)

            done, reward, state_ = env.step(action, frame, buffer=1000)

            total_reward += reward

            if train == True:
                agent.remember(state, action, reward, state_, int(done))
                agent.learn()


            state = state_





        eps = list(range(episode))

        reward_store.append(total_reward)
        avg_reward.append(np.mean(reward_store))
        epsilon_store.append(agent.epsilon)
        

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Reward')
        ax1.plot(eps, avg_reward, color='red')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon')
        ax2.plot(eps, epsilon_store, color='blue')

        plt.savefig('progress.png')
        plt.close()



        print('\n\nEpisode: ', episode)
        print('Total reward: ', total_reward)
        print('Score:  ', env.score)
        print('Time elapsed: ', time.time()-start_time)


        try:
            agent.save_model()
        except:
            pass


    if show == True:
        pygame.quit()












