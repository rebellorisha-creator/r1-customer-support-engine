import random
from env import CustomerSupportEnv

env = CustomerSupportEnv()

ACTIONS = [
    "auto_reply",
    "ask_clarification",
    "escalate",
    "route_to_sales",
    "route_to_tech"
]

# Q-table
Q = {}

# Hyperparameters
alpha = 0.1   # learning rate
gamma = 0.9   # future reward importance
epsilon = 0.2 # exploration

def get_state_key(state):
    return f"{state['sentiment']}_{state['urgency']}_{state['complexity']}_{state['category']}"

def choose_action(state_key):
    # explore
    if random.random() < epsilon:
        return random.choice(ACTIONS)

    # exploit
    if state_key not in Q:
        Q[state_key] = {a: 0 for a in ACTIONS}

    return max(Q[state_key], key=Q[state_key].get)


# TRAINING LOOP
episodes = 1000

for episode in range(episodes):

    state = env.reset()
    state_key = get_state_key(state)

    done = False

    while not done:

        action = choose_action(state_key)

        result = env.step(action)

        next_state = result["observation"]
        reward = result["reward"]
        done = result["done"]

        next_state_key = get_state_key(next_state)

        # initialize Q values
        if state_key not in Q:
            Q[state_key] = {a: 0 for a in ACTIONS}

        if next_state_key not in Q:
            Q[next_state_key] = {a: 0 for a in ACTIONS}

        # 🔥 Q-learning update
        Q[state_key][action] = (
            Q[state_key][action]
            + alpha * (
                reward + gamma * max(Q[next_state_key].values())
                - Q[state_key][action]
            )
        )

        state_key = next_state_key

print("Training finished!")

# Save Q-table
import pickle
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)