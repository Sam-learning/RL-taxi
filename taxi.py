import numpy as np
import os
import gym
from tqdm import tqdm

total_reward = []


class Agent():
    def __init__(self, env, epsilon=0.95, learning_rate=0.8, gamma=0.9):
        """
        Parameters:
            env: target enviornment.
            epsilon: Determinds the explore/expliot rate of the agent.
            learning_rate: Learning rate of the agent.
            gamma: discount rate of the agent.
        """
        self.env = env

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize qtable
        self.qtable = np.zeros((env.observation_space.n, env.action_space.n))

        self.qvalue_rec = []

    def choose_action(self, state):
        """
        Choose the best action with given state and epsilon.

        Parameters:
            state: A representation of the current state of the enviornment.
            epsilon: Determines the explore/expliot rate of the agent.

        Returns:
            action: The action to be evaluated.
        """
        # Begin your code
        # TODO

        self.epsilon -= 0.00005
        self.epsilon = max(self.epsilon, 0.10)

        if np.random.random() > self.epsilon:           # select greedy action with probability epsilon
          return np.argmax(self.qtable[state])
        else:                                          # otherwise, select an action randomly
          return env.action_space.sample()

        # End your code

    def learn(self, state, action, reward, next_state, done):
        """
        Calculate the new q-value base on the reward and state transformation observered after taking the action.

        Parameters:
            state: The state of the enviornment before taking the action.
            action: The exacuted action.
            reward: Obtained from the enviornment after taking the action.
            next_state: The state of the enviornment after taking the action.
            done: A boolean indicates whether the episode is done.

        Returns:
            None (Don't need to return anything)
        """
        # Begin your code
        # TODO
        next_action = self.choose_action(state)

        #Get the current q-value
        current = self.qtable[state ,action] 

        #calculate next state max q value if episode not done yet
        #episode:  If done is True means current state is a terminal state, and there are no more actions to be taken in the current episode. 
        Qsa_next = np.max(self.qtable[next_state]) if done is False else 0 

        #target q value = immediate reward + discounted max q_value
        #Calculate new Q-value by formula
        target = reward + (self.gamma * Qsa_next)           
        new_value = current + self.learning_rate * (target - current)

        #updates the Q-table
        self.qtable[state ,action] = new_value


        # End your code
        np.save("./Tables/taxi_table.npy", self.qtable)

    def check_max_Q(self, state):
        """
        - Implement the function calculating the max Q value of given state.
        - Check the max Q value of initial state

        Parameter:
            state: the state to be check.
        Return:
            max_q: the max Q value of given state
        """
        # Begin your code
        # TODO
        max_q = np.max(self.qtable[state])
        return max_q
        # End your code


def extract_state(ori_state):
        state = []
        if ori_state % 4 == 0:
            state.append('R')
        else:
            state.append('G')
        
        ori_state = ori_state // 4
        if ori_state % 5 == 2:
            state.append('Y')
        else:
            state.append('B')
        
        print(f"Initail state:\ntaxi at (2, 2), passenger at {state[1]}, destination at {state[0]}")
        

def train(env):
    """
    Train the agent on the given environment.

    Paramenter:
        env: the given environment.

    Return:
        None
    """
    training_agent = Agent(env)
    episode = 3000
    rewards = []
    episode_rewards = np.zeros(20)
    for ep in tqdm(range(episode)):
        state = env.reset()
        done = False

        count = 0
        
        #training_agent.epsilon -= 0.001
        #training_agent.epsilon = max(training_agent.epsilon, 0.1)
        while True:
            action = training_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            training_agent.learn(state, action, reward, next_state, done)
            count += reward

            if done:
                rewards.append(count)
                episode_rewards[int(ep%20)] = count
                break

            state = next_state
        
        if ep >= 20:
          if ep%200 == 0:
            print(np.mean(episode_rewards))
            print(training_agent.epsilon)
            

    total_reward.append(rewards)


def test(env):
    """
    Test the agent on the given environment.

    Paramenters:
        env: the given environment.

    Return:
        None
    """
    testing_agent = Agent(env)
    testing_agent.qtable = np.load("./Tables/taxi_table.npy")
    rewards = []

    for _ in range(100):
        state = testing_agent.env.reset()
        count = 0
        while True:
            action = np.argmax(testing_agent.qtable[state])
            next_state, reward, done, _ = testing_agent.env.step(action)
            count += reward
            if done == True:
                rewards.append(count)
                break

            state = next_state
    
    state = 248 # Do not change this value
    print(f"average reward: {np.mean(rewards)}")
    extract_state(state)
    print(f"max Q:{testing_agent.check_max_Q(state)}")


if __name__ == "__main__":

    env = gym.make("Taxi-v3")
    os.makedirs("./Tables", exist_ok=True)

    # training section:
    for i in range(1):
      print(f"#{i + 1} training progress")
      train(env)   
    # testing section:
    test(env)
        
    os.makedirs("./Rewards", exist_ok=True)

    np.save("./Rewards/taxi_rewards.npy", np.array(total_reward))

    env.close()
