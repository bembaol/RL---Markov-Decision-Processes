##Importation
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns

import hiive.mdptoolbox as mdptoolbox
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp


##Constants
title_fontsize = 24
fontsize = 24
legend_fontsize = 18
default_figure_size = (15, 8)

PROBLEM = "Forest"

SIZE = 10
R1=10
R2=20
FireProb=0.3
P, R = mdptoolbox.example.forest(S=SIZE, r1=R1, r2=R2, p=FireProb)
print(P)
print(R)

##Functions
def Average(lst):
    return sum(lst) / len(lst)

def average_value_of_policy(P, R, policy, iterations=100):
    np.random.seed(0)
    rewards = []
    # print("policy: ", policy)
    for i in range(iterations):
        for starting_position in range(P.shape[-1]):
            # print("Starting Position {}".format(starting_position))
            position = starting_position
            reward = 0
            while starting_position == 0 or position != 0:
                # print("Action {}".format(policy[position]))
                trans_probs = P[policy[position]][position]
                prob = np.random.rand()
                for state, p in enumerate(trans_probs):
                    if prob <= p:
                        reward += R[position, policy[position]]
                        position = state
                        if starting_position == 0:
                            starting_position = state
                        # print("Next State {}".format(position))
                        break
                    else:
                        prob -= p
            rewards.append(reward)
    return np.mean(rewards)

def plot_forest_problem_space(size=10):
    # https://stackoverflow.com/questions/43971138/python-plotting-colored-grid-based-on-values
    # create discrete colormap
    cmap = colors.ListedColormap(['lightgreen', 'lightblue'])

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    wait = mpatches.Patch(color='lightgreen', label='Wait')
    cut = mpatches.Patch(color='lightblue', label='Cut')
    fig.legend(handles=[wait, cut], fontsize=20, loc='center right')
    fig.suptitle("Forest Problem Space", fontsize=18)
    plt.xlabel('State', fontsize=18)
    fig.set_figheight(4)
    fig.set_figwidth(9)

    wait = np.zeros((1,size))
    cut = np.ones((1,size))
    print(wait.astype(np.int32))
    print(cut.astype(np.int32))

    cmap = colors.ListedColormap(['lightgreen', 'lightblue'])
    ax1.imshow(cut.astype(np.int32), cmap=cmap)
    cmap = colors.ListedColormap(['lightblue', 'lightblue'])
    ax2.imshow(cut.astype(np.int32), cmap=cmap)

    text = ax1.text(0, -0.1, "0", horizontalalignment='center', verticalalignment='center', color="black")
    text = ax2.text(0, -0.1, "0", horizontalalignment='center', verticalalignment='center', color="black")
    for i in range(1,size-1):
        text = ax1.text(i, -0.1, "0", horizontalalignment='center', verticalalignment='center', color="black")
        text = ax2.text(i, -0.1, "1", horizontalalignment='center', verticalalignment='center', color="black")
    text = ax1.text(size-1, -0.1, str(R1), horizontalalignment='center', verticalalignment='center', color="black")
    text = ax2.text(size-1, -0.1, str(R2), horizontalalignment='center', verticalalignment='center', color="black")

    # draw gridlines
    for ax in [ax1, ax2]:
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-.5, size, 1));
        ax.set_xticklabels(np.arange(0, size+1, 1), fontsize=20)

        ax.set_yticks(np.arange(-.5, 1, 1));
        ax.axes.yaxis.set_ticks([])
    ax1.set_ylabel("Wait Action", fontsize=16, rotation=0, labelpad=80)
    ax2.set_ylabel("Cut Action", fontsize=16, rotation=0, labelpad=80)


def plot_forest_policy(ax, policy, y_label=""):
    # https://stackoverflow.com/questions/43971138/python-plotting-colored-grid-based-on-values
    # create discrete colormap
    cmap = colors.ListedColormap(['lightgreen', 'lightblue'])

    policy = np.array(policy)
    policy = policy.reshape(1, len(policy))

    ax.imshow(policy.astype(np.int32), cmap=cmap)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, policy.shape[1], 1));
    ax.set_xticklabels(np.arange(0, policy.shape[1]+1, 1), fontsize=20)

    ax.set_yticks(np.arange(-.5, 1, 1));
    ax.axes.yaxis.set_ticks([])
    ax.set_ylabel(y_label, fontsize=16, rotation=0, labelpad=80)

def create_policy_figure(num_policies, title=""):
    fig, ax = plt.subplots(num_policies,1, sharex=True)
    wait = mpatches.Patch(color='lightgreen', label='Wait')
    cut = mpatches.Patch(color='lightblue', label='Cut')
    fig.legend(handles=[wait, cut], fontsize=20)
    fig.suptitle(title, fontsize=18)
    plt.xlabel('State', fontsize=18)
    fig.set_figheight(8)
    fig.set_figwidth(9)
    return fig, ax

def gamma_parameter_tuning(P, R, agent_type='Policy Iteration'):
    run_stats = []
    title = "Change In Policy With Discount Factor\n{}".format(agent_type)
    gammas = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    fig, ax = create_policy_figure(len(gammas), title)

    for gamma, ax in zip(gammas, ax):
        if agent_type == 'Policy Iteration':
            agent = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
        elif agent_type == 'Value Iteration':
            agent = mdptoolbox.mdp.ValueIteration(P, R, gamma)
        agent_dict = agent.run()
        plot_forest_policy(ax, agent.policy, "Discount\nFactor: {:.2f}".format(gamma))

        df = pd.DataFrame(agent_dict)
        run_stats.append(df)

    plt.show()
    fig = plt.figure(figsize=default_figure_size)
    plt.title(title, fontsize=fontsize)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=fontsize)
    for df, gamma in zip(run_stats, gammas):
            plt.plot(df['Iteration'], df['Mean V'], label='Discount Factor: {:.2f}'.format(gamma))
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel("Average Reward", fontsize=fontsize)
    plt.legend(fontsize=16)
    plt.grid()
    plt.show()

def pi_vi_convergence_plot(P, R, gamma, agent_type='Policy Iteration', x_tick_spacing=2):
    title = "{} Convergence Plot for {} Probelm Size = {}\n Discount Factor = {}".format(agent_type, PROBLEM, SIZE, gamma)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=default_figure_size, sharex=True)
    fig.suptitle(title, fontsize=18)

    ax1.set_xlabel("Iterations", fontsize=16)
    ax2.set_xlabel("Iterations", fontsize=16)
    ax3.set_xlabel("Iterations", fontsize=16)

    plt.setp(ax1.get_yticklabels(), fontsize=16)
    plt.setp(ax2.get_yticklabels(), fontsize=16)
    plt.setp(ax3.get_yticklabels(), fontsize=16)

    plt.setp(ax1.get_xticklabels(), fontsize=16)
    plt.setp(ax2.get_xticklabels(), fontsize=16)
    plt.setp(ax3.get_xticklabels(), fontsize=16)

    if agent_type == 'Policy Iteration':
        agent = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
    elif agent_type == 'Value Iteration':
        agent = mdptoolbox.mdp.ValueIteration(P, R, gamma)
    agent_dict = agent.run()

    df = pd.DataFrame(agent_dict)

    ax1.plot(df['Iteration'], df['Mean V'], label='Mean V', color="blue")
    ax2.plot(df['Iteration'], df['Error'], label="Error", color="orange")
    ax3.plot(df['Iteration'], df['Time'], label="Time", color="green")

    ax1.set_ylabel("Average Utility", fontsize=16)
    ax2.set_ylabel("Error", fontsize=16)
    ax3.set_ylabel("Time (s)", fontsize=16)

    ax1.set_xticks([i for i in range(0, max(df['Iteration'])+x_tick_spacing, x_tick_spacing)])
    ax2.set_xticks([i for i in range(0, max(df['Iteration'])+x_tick_spacing, x_tick_spacing)])
    ax3.set_xticks([i for i in range(0, max(df['Iteration'])+x_tick_spacing, x_tick_spacing)])

    ax1.grid()
    ax2.grid()
    ax3.grid()

    plt.tight_layout()
    plt.show()

def plot_problem_size(df):
    title = "{} Problem Size Comparison".format(PROBLEM)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=default_figure_size, sharex=True)
    fig.suptitle(title, fontsize=18)

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')

    fontsize=16
    plt.setp(ax1.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax2.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax3.get_yticklabels(), fontsize=fontsize)

    plt.setp(ax1.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax2.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax3.get_xticklabels(), fontsize=fontsize)

    sns.set_context("paper", rc={"font.size": fontsize, "axes.titlesize": fontsize, "axes.labelsize": fontsize})
    sns.barplot(x='problem size', y='iterations', hue='algorithm', data=df, ax=ax1)
    sns.barplot(x='problem size', y='time', hue='algorithm', data=df, ax=ax2)
    sns.barplot(x='problem size', y='value', hue='algorithm', data=df, ax=ax3)

    plt.tight_layout()
    plt.show()

def q_convergence_plot(P, R, agent_type='Q-Learning', x_tick_spacing=2):
    title = "{} Convergence Plot for {}".format(agent_type, PROBLEM)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=default_figure_size, sharex=True)
    fig.suptitle(title, fontsize=18)

    ax1.set_xlabel("Iterations", fontsize=16)
    ax2.set_xlabel("Iterations", fontsize=16)

    plt.setp(ax1.get_yticklabels(), fontsize=16)
    plt.setp(ax2.get_yticklabels(), fontsize=16)

    plt.setp(ax1.get_xticklabels(), fontsize=16)
    plt.setp(ax2.get_xticklabels(), fontsize=16)

    agent = mdptoolbox.mdp.QLearning(P, R, gamma=0.99,
                                 alpha=0.5, alpha_decay=0.25, alpha_min=0.01,
                                 epsilon=epsilon, epsilon_min=0.01, epsilon_decay=1.0, n_iter=1000000)
    agent.run()

    df = pd.DataFrame(agent.run_stats)

    ax1.plot(df['Iteration'], df['Mean V'], label='Mean V', color="blue")
    ax2.plot(df['Iteration'], df['Time'], label="Time", color="green")

    ax1.set_ylabel("Average Utility", fontsize=16)
    ax2.set_ylabel("Time (s)", fontsize=16)

#     ax1.set_xticks([i for i in range(0, max(df['Iteration'])+x_tick_spacing, x_tick_spacing)])
#     ax2.set_xticks([i for i in range(0, max(df['Iteration'])+x_tick_spacing, x_tick_spacing)])
#     ax3.set_xticks([i for i in range(0, max(df['Iteration'])+x_tick_spacing, x_tick_spacing)])

    ax1.grid()
    ax2.grid()

    plt.tight_layout()
    plt.show()

##Solving Forest MDP Using Value Iteration & Policy Iteration
DEMO = False
if DEMO:
    fig, ax = create_policy_figure(10, "Demo")
    for i in range(10):
        print(i)
        plot_forest_policy(ax[i], pi.policy, y_label="Gamma: X")

    plt.tight_layout()
    plt.show()

plot_forest_problem_space(10)

gamma_parameter_tuning(P, R, 'Value Iteration')

gamma_parameter_tuning(P, R, 'Policy Iteration')


#Display Graphs just for Gama = 0.99
title=""
fig, ax = create_policy_figure(1, title)
agent = mdptoolbox.mdp.ValueIteration(P, R, 0.99)
agent.run()
plot_forest_policy(ax, agent.policy)
plt.show()

vi = mdptoolbox.mdp.ValueIteration(P, R, 0.99)
vi_dict = vi.run()
print(vi.policy)
df = pd.DataFrame(vi_dict)
plt.plot(df['Iteration'], df['Reward'], label='Reward')
plt.plot(df['Iteration'], df['Mean V'], label='Mean V')
plt.plot(df['Iteration'], df['Error'], label="Error")
plt.legend()
plt.show()

pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.99)
pi_dict = pi.run()
print(pi.policy)
df = pd.DataFrame(pi_dict)
plt.plot(df['Iteration'], df['Reward'], label='Reward')
plt.plot(df['Iteration'], df['Mean V'], label='Mean V')
plt.plot(df['Iteration'], df['Error'], label="Error")
plt.legend()
plt.show()

pi_vi_convergence_plot(P, R, 0.99, "Value Iteration")

pi_vi_convergence_plot(P, R, 0.99, "Policy Iteration")

##Comparison with Q Learning
sizes = [10, 500, 5000]
q_iterations = {10: 500000, 500: 5000000, 5000: 20000000}
size_compare = []
for s in sizes:
    P_, R_ = mdptoolbox.example.forest(S=s, r1=R1, r2=R2, p=FireProb)
    vi_ = mdptoolbox.mdp.ValueIteration(P_, R_, 0.99)
    vi_.run()
    df = pd.DataFrame(vi_.run_stats)
    value = average_value_of_policy(P_, R_, vi_.policy)
    print("size = {} vi num_iterations = {} time = {} mean_value = {}".format(s, max(df['Iteration']), max(df['Time']), value))
    size_compare.append({"problem size": s, "algorithm": "Value Iteration", "time": max(df['Time']), "iterations" :max(df['Iteration']), 'value': value})
    pi_ = mdptoolbox.mdp.PolicyIteration(P_, R_, 0.99)
    pi_.run()
    df = pd.DataFrame(pi_.run_stats)
    value = average_value_of_policy(P_, R_, pi_.policy)
    print("size = {} pi num_iterations = {} time = {} mean_value = {}".format(s, max(df['Iteration']), max(df['Time']), value))
    print("Value Iteration policy equals Policy Iteration Policy ", np.all(vi_.policy == pi_.policy))
    size_compare.append({"problem size": s, "algorithm": "Policy Iteration", "time": max(df['Time']), "iterations" :max(df['Iteration']), "value": value})

    q_ = mdptoolbox.mdp.QLearning(P_, R_, gamma=0.99,
                             alpha=0.5, alpha_decay=0.25, alpha_min=0.01,
                             epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.9, n_iter=q_iterations[s])

    q_.run()
    df = pd.DataFrame(q_.run_stats)
    value = average_value_of_policy(P_, R_, q_.policy)
    print("size = {} q num_iterations = {} time = {} mean_value = {}".format(s, max(df['Iteration']), max(df['Time']), value))
    print("Q Learning Policy equals policy iteration ", np.all(q_.policy == pi_.policy))
    size_compare.append({"problem size": s, "algorithm": "Q Learning", "time": max(df['Time']), "iterations" :max(df['Iteration']), 'value': value})



df = pd.DataFrame(size_compare)
plot_problem_size(df)

## Just QLearning
P, R = mdptoolbox.example.forest(S=SIZE, r1=R1, r2=R2, p=0.3)

q = mdptoolbox.mdp.QLearning(P, R, gamma=0.99, n_iter = 1000000)
q_di = q.run()
df = pd.DataFrame(q_di)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=default_figure_size, sharex=True)
fig.suptitle(title, fontsize=18)

ax1.set_xlabel("Iterations", fontsize=16)
ax2.set_xlabel("Iterations", fontsize=16)

ax1.set_ylabel("Average Reward", fontsize=16)
ax2.set_ylabel("Max Reward", fontsize=16)

ax1.plot(df['Iteration'], df['Mean V'], label='Mean V', color="blue")
ax2.plot(df['Iteration'], df['Max V'], label="Error", color="orange")

plt.show()


##QLearning Tuning
P, R = mdptoolbox.example.forest(S=SIZE, r1=R1, r2=R2, p=FireProb)

#Epsilon Tuning
np.random.seed(0)
epsilon_list = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95]
epsilon_q_runstats = []
plt.figure(figsize=default_figure_size)
title = "{} Problem Q Learning Epsilon".format(PROBLEM)
plt.title(title, fontsize=18)
for epsilon in epsilon_list:
    print("Running epsilon = {}".format(epsilon))
    q = mdptoolbox.mdp.QLearning(P, R, gamma=0.99,
                                 alpha=0.5, alpha_decay=0.25, alpha_min=0.01,
                                 epsilon=epsilon, epsilon_min=0.01, epsilon_decay=0.9, n_iter=1000000)

    q.run()
    epsilon_q_runstats.append(q.run_stats)
    print(q.policy)
    df = pd.DataFrame(q.run_stats)

    plt.plot(df['Iteration'], df['Mean V'], label='epsilon = {:.2f}'.format(epsilon))

plt.xlabel('Iterations', fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('Average Utility', fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.show()

#Epsilon Decay Tuning
np.random.seed(0)
epsilon_decay_list = np.arange(0, 1, 0.1)
epsilon_decay_q_runstats = []
plt.figure(figsize=default_figure_size)
title = "{} Problem Q Learning Epsilon Decay".format(PROBLEM)
plt.title(title, fontsize=18)
for epsilon_decay in epsilon_decay_list:
    print("Running epsilon decay = {}".format(epsilon_decay))
    q = mdptoolbox.mdp.QLearning(P, R, gamma=0.99,
                                 alpha=0.5, alpha_decay=0.25, alpha_min=0.01,
                                 epsilon=0.5, epsilon_min=0.01, epsilon_decay=epsilon_decay, n_iter=1000000)

    q.run()
    epsilon_decay_q_runstats.append(q.run_stats)
    print(q.policy)
    df = pd.DataFrame(q.run_stats)

    plt.plot(df['Iteration'], df['Mean V'], label='Epsilon Decay = {:.2f}'.format(epsilon_decay))

plt.xlabel('Iterations', fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('Average Utility', fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.show()

#Alpha Tuning
np.random.seed(0)
alpha_list = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
alpha_q_run_stats = []
plt.figure(figsize=default_figure_size)
title = "{} Problem Q Learning Alpha".format(PROBLEM)
plt.title(title, fontsize=18)
for alpha in alpha_list:
    print("Running alpha = {}".format(alpha))
    q = mdptoolbox.mdp.QLearning(P, R, gamma=0.99,
                                 alpha=alpha, alpha_decay=0.25, alpha_min=.01,
                                 epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.9, n_iter=1000000)

    q.run()
    alpha_q_run_stats.append(q.run_stats)
    print(q.policy)
    df = pd.DataFrame(q.run_stats)

    plt.plot(df['Iteration'], df['Mean V'], label='Learning Rate = {:.2f}'.format(alpha))
    print(np.max(df["Time"]))

plt.xlabel('Iterations', fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('Average Utility', fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.legend()
plt.show()

#Alpha Decay Tuning
np.random.seed(0)
alpha_decay_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
alpha_decay_q_runstats = []
plt.figure(figsize=default_figure_size)
title = "{} Problem Q Learning Learning Rate Decay".format(PROBLEM)
plt.title(title, fontsize=18)
for alpha_decay in alpha_decay_list:
    print("Running alpha decay = {}".format(alpha_decay))
    q = mdptoolbox.mdp.QLearning(P, R, gamma=0.99,
                                 alpha=0.5, alpha_decay=alpha_decay, alpha_min=0.01,
                                 epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.9, n_iter=1000000)

    q.run()
    alpha_decay_q_runstats.append(q.run_stats)
    print(q.policy)
    df = pd.DataFrame(q.run_stats)

    plt.plot(df['Iteration'], df['Mean V'], label='Learning Rate Decay = {:.2f}'.format(alpha_decay))
    print(np.max(df["Time"]))

plt.xlabel('Iterations', fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('Average Utility', fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.legend()
plt.show()

# Policy
title="{} Problem Q-Learning Policy".format(PROBLEM)
fig, ax = create_policy_figure(1, title)
q = mdptoolbox.mdp.QLearning(P, R, gamma=0.99,
                             alpha=0.5, alpha_decay=0.25, alpha_min=0.01,
                             epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.9, n_iter=1000000)
q.run()
print(q.policy)
plot_forest_policy(ax, q.policy)
plt.show()

# Sum up
q_convergence_plot(P,R)
