import pickle
from env import CustomerSupportEnv

# load Q-table
with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)

env = CustomerSupportEnv()

def get_state_key(state):
    return f"{state['sentiment']}_{state['urgency']}_{state['complexity']}_{state['category']}"

state = env.reset()

done = False

while not done:
    key = get_state_key(state)

    if key in Q:
        action = max(Q[key], key=Q[key].get)
    else:
        action = "ask_clarification"

    result = env.step(action)

    print("STATE:", state)
    print("ACTION:", action)
    print("REWARD:", result["reward"])
    print("----")

    state = result["observation"]
    done = result["done"]