from environment import CustomerSupportEnv, ACTIONS
import pickle

# Load trained Q-table
with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)

def get_q(state, action):
    return Q.get((state, action), 0)

def choose_best_action(state):
    q_values = [get_q(state, a) for a in ACTIONS]
    max_q = max(q_values)
    best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
    return best_actions[0]

def run_query(query):
    env = CustomerSupportEnv()
    state = env.reset(query)

    action = choose_best_action(state)
    next_state, reward,done, _= env.step(action)

    return {
        "state": state,
        "action": action,
        "reward": reward
    }
