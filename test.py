import pickle
from environment import CustomerSupportEnv, ACTIONS

# Load Q-table
with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)

env = CustomerSupportEnv()


def get_q(state, action):
    return Q.get((state, action), 0)


def run_query(query):
    state = env.reset(query)
    done = False

    final_action = None
    final_reward = 0

    while not done:
        q_values = [get_q(state, a) for a in ACTIONS]
        action = ACTIONS[q_values.index(max(q_values))]

        next_state, reward, done = env.step(action)

        final_action = action
        final_reward = reward

        state = next_state

    return {
        "state": state,
        "action": final_action,
        "reward": final_reward
    }