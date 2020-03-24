import os
import gym
import numpy as np
from variables import *
import random
from matplotlib import pyplot as plt

class MultiArmBandits(object):
    def __init__(self):
        self.env = gym.make(environment)
        self.q_values = np.load(q_value_path) if os.path.exists(q_value_path) else np.random.uniform(0,1,self.env.action_space.n)

    def agent(self):
        global eps
        total_rewards = []
        for episode in range(num_episodes):
            observation = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                p = random.random()
                if p < eps:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_values)

                new_observation, reward, done, _ = self.env.step(action)

                if done:
                    reward = panelty

                episode_reward += reward
                eps = eps / np.sqrt(episode+1)

                old_q = self.q_values[action]
                new_q = old_q + learning_rate * (reward - old_q)
                self.q_values[action] = new_q

            total_rewards.append(episode_reward)
            if (episode + 1)% verbose == 0:
                print("episode :", episode+1," reward: ",episode_reward)
        MultiArmBandits.plot_cumulative_rewards(total_rewards,num_episodes)

    @staticmethod
    def plot_cumulative_rewards(total_rewards,num_episodes):
        cum_rewards = np.cumsum(total_rewards)
        cum_average_reward = cum_rewards / np.arange(1,num_episodes+1)

        fig = plt.figure()
        plt.plot(cum_average_reward)
        fig.suptitle('epsilon_greedy', fontsize=20)
        plt.xlabel('episode number')
        plt.ylabel('Cumulative Average Reward')
        fig.savefig('epsilon_greedy.png')

    def save_q_table(self): # save q values to use as initial value in next run
        np.save(q_value_path, self.q_values)


if __name__ == "__main__":
    bandit =  MultiArmBandits()
    bandit.agent()
    bandit.save_q_table()