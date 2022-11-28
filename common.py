import gym
import numpy as np
import time
import random

# references
# https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python/blob/master/Chapter03/3.12%20Value%20Iteration%20-%20Frozen%20Lake%20Problem.ipynb
# https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python/blob/master/Chapter03/3.13%20Policy%20Iteration%20-%20Frozen%20Lake%20Problem.ipynb
# https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python/blob/master/Chapter05/5.5%20Taxi%20Problem%20-%20Q%20Learning.ipynb


def value_iteration(env, gamma=1.0):
    # https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python/blob/master/Chapter03/3.12%20Value%20Iteration%20-%20Frozen%20Lake%20Problem.ipynb
    # initialize value table with zeros
    value_table = np.zeros(env.observation_space.n)

    # set number of iterations and threshold
    no_of_iterations = 100000
    threshold = 1e-5

    max_values = []
    average_values = []
    times = []
    start_time = time.time()
    iterations = []
    errors = []
    for i in range(no_of_iterations):
        # On each iteration, copy the value table to the updated_value_table
        updated_value_table = np.copy(value_table)
        max_values.append(np.max(updated_value_table))
        average_values.append(np.mean(updated_value_table))
        iterations.append(i)

        # Now we calculate Q Value for each actions in the state
        # and update the value of a state with maximum Q value

        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_states_rewards = []
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))

                Q_value.append(np.sum(next_states_rewards))

            value_table[state] = max(Q_value)

        # we will check whether we have reached the convergence i.e whether the difference
        # between our value table and updated value table is very small. But how do we know it is very
        # small? We set some threshold and then we will see if the difference is less
        # than our threshold, if it is less, we break the loop and return the value function as optimal
        # value function

        times.append(time.time() - start_time)
        errors.append(np.sum(np.fabs(updated_value_table - value_table)))
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            print('Value-iteration converged at iteration #%d.' % (i + 1))
            break

    return value_table, iterations, max_values, average_values, times, errors

def extract_policy(value_table, env, gamma=1.0):
    # https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python/blob/master/Chapter03/3.12%20Value%20Iteration%20-%20Frozen%20Lake%20Problem.ipynb
    # initialize the policy with zeros
    policy = np.zeros(env.observation_space.n)

    for state in range(env.observation_space.n):

        # initialize the Q table for a state
        Q_table = np.zeros(env.action_space.n)

        # compute Q value for all ations in the state
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

        # select the action which has maximum Q value as an optimal action of the state
        policy[state] = np.argmax(Q_table)

    return policy


def compute_value_function(env, policy, gamma=1.0):
    # https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python/blob/master/Chapter03/3.13%20Policy%20Iteration%20-%20Frozen%20Lake%20Problem.ipynb
    # initialize value table with zeros
    value_table = np.zeros(env.nS)

    # set the threshold
    threshold = 1e-5

    while True:

        # copy the value table to the updated_value_table
        updated_value_table = np.copy(value_table)

        # for each state in the environment, select the action according to the policy and compute the value table
        for state in range(env.nS):
            action = policy[state]

            # build the value table with the selected action
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                                      for trans_prob, next_state, reward_prob, _ in env.P[state][action]])

        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break

    return value_table


def policy_iteration(env, gamma=1.0):
    # https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python/blob/master/Chapter03/3.13%20Policy%20Iteration%20-%20Frozen%20Lake%20Problem.ipynb
    # Initialize policy with zeros
    old_policy = np.zeros(env.observation_space.n)
    no_of_iterations = 200000

    max_values = []
    average_values = []
    times = []
    start_time = time.time()
    iterations = []
    errors = []

    for i in range(no_of_iterations):

        # compute the value function
        new_value_function = compute_value_function(env, old_policy, gamma)

        max_values.append(np.max(new_value_function))
        average_values.append(np.mean(new_value_function))
        iterations.append(i)

        # Extract new policy from the computed value function
        new_policy = extract_policy(new_value_function, env, gamma)

        # Then we check whether we have reached convergence i.e whether we found the optimal
        # policy by comparing old_policy and new policy if it same we will break the iteration
        # else we update old_policy with new_policy

        times.append(time.time() - start_time)
        errors.append(np.sum(old_policy != new_policy))
        if (np.all(old_policy == new_policy)):
            print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        old_policy = new_policy

    return new_policy, iterations, max_values, average_values, times, errors


class Q_Learner:
    def __init__(self, env, alpha=0.4, gamma=0.9, epsilon=0.017, stats_freq=100, epsilon_decay=1.0):
        self.env = env
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.stats_freq = stats_freq
        for s in range(env.observation_space.n):
            for a in range(env.action_space.n):
                self.q[(s, a)] = 0.0

    def update_q_table(self, prev_state, action, reward, nextstate, alpha, gamma):

        qa = max([self.q[(nextstate, a)] for a in range(self.env.action_space.n)])
        self.q[(prev_state, action)] += alpha * (reward + gamma * qa - self.q[(prev_state, action)])

    def epsilon_greedy_policy(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return self.env.action_space.sample()
        else:
            return max(list(range(self.env.action_space.n)), key=lambda x: self.q[(state, x)])

    def run(self, num_iterations):
        average_values = []
        times = []
        start_time = time.time()
        iterations = []

        for i in range(num_iterations):
            r = 0

            prev_state = self.env.reset()

            while True:
                # if self.epsilon <= 0.0001:
                #     return average_values, times, iterations

                # In each state, we select the action by epsilon-greedy policy
                action = self.epsilon_greedy_policy(prev_state, self.epsilon)

                # then we perform the action and move to the next state, and receive the reward
                nextstate, reward, done, _ = self.env.step(action)

                # Next we update the Q value using our update_q_table function
                # which updates the Q value by Q learning update rule

                self.update_q_table(prev_state, action, reward, nextstate, self.alpha, self.gamma)

                # Finally we update the previous state as next state
                prev_state = nextstate

                # Store all the rewards obtained
                r += reward

                # we will break the loop, if we are at the terminal state of the episode
                if done:
                    break

            if i % self.stats_freq == 0:
                self.epsilon *= self.epsilon_decay
                _, value = self.extract_policy()

                average_values.append(value)
                times.append(time.time() - start_time)
                iterations.append(i)
        return average_values, times, iterations

    def extract_policy(self):
        # https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python/blob/master/Chapter03/3.12%20Value%20Iteration%20-%20Frozen%20Lake%20Problem.ipynb
        # initialize the policy with zeros
        policy = np.zeros(self.env.observation_space.n)
        value = 0
        for state in range(self.env.observation_space.n):
            # select the action which has maximum Q value as an optimal action of the state
            policy[state] = max(list(range(self.env.action_space.n)), key=lambda x: self.q[(state, x)])
            value += self.q[(state, policy[state])]

        return policy, value