import random
import pickle
from environment import CustomerSupportEnv, ACTIONS

env = CustomerSupportEnv()

Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.05

episodes = 30000


def get_q(state, action):
    return Q.get((state, action), 0)


for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:

        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            q_values = [get_q(state, a) for a in ACTIONS]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
            action = random.choice(best_actions)

        next_state, reward, done = env.step(action)

        old_q = get_q(state, action)
        next_max = max([get_q(next_state, a) for a in ACTIONS])

        new_q = old_q + alpha * (reward + gamma * next_max - old_q)
        Q[(state, action)] = new_q

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 500 == 0:
        print(f"Episode {episode} Reward: {round(total_reward, 2)}")

with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)

print("Training complete.....")