import pickle
from environment import CustomerSupportEnv, ACTIONS

env = CustomerSupportEnv()

with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)

def get_q(state, action):
    return Q.get((state, action), 0.0)

episodes = 5

for ep in range(episodes):
    state = env.reset()
    done = False

    while not done:
        qs = [get_q(state, a) for a in ACTIONS]
        action = ACTIONS[qs.index(max(qs))]

        state, reward, done = env.step(action)