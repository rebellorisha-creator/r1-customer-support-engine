import pickle
import random
import matplotlib.pyplot as plt
from environment import CustomerSupportEnv, ACTIONS

env = CustomerSupportEnv()

Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995

episodes = 200
rewards_per_episode = []

def get_q(state, action):
    return Q.get((state, action), 0.0)

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    done = False

    while not done:
        # epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            qs = [get_q(state, a) for a in ACTIONS]
            action = ACTIONS[qs.index(max(qs))]

        next_state, reward, done = env.step(action)

        # Q update
        max_future = max([get_q(next_state, a) for a in ACTIONS])

        old_q = get_q(state, action)

        new_q = old_q + alpha * (reward + gamma * max_future - old_q)

        Q[(state, action)] = new_q

        state = next_state
        total_reward += reward

    epsilon *= epsilon_decay
    rewards_per_episode.append(total_reward)

    print(f"Episode {ep+1} Reward: {round(total_reward,2)}")

# 💾 Save model
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)

print("Training complete. Model saved.")

# 📈 Plot graph
plt.plot(rewards_per_episode)
plt.title("Training Reward Over Time")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()