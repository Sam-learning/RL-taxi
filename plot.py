import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.title('CartPole-v0')
    plt.xlabel('epoch')
    plt.ylabel('rewards')


def taxi():
    plt.figure(figsize=(10, 5))
    plt.title('Taxi-v3')
    plt.xlabel('epoch')
    plt.ylabel('rewards')
    rewards = np.load("./Rewards/taxi_rewards.npy").transpose()
    rewards_avg = np.mean(rewards, axis=1)
    plt.plot([i for i in range(3000)], rewards_avg[:3000],
             label='taxi', color='gray')
    plt.legend(loc="best")
    plt.savefig("./Plots/taxi.png")
    plt.show()
    plt.close()




if __name__ == "__main__":
    '''
    Plot the trend of Rewards
    '''  
    os.makedirs("./Plots", exist_ok=True)

    taxi()
