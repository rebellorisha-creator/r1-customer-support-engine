import pickle
from environment import CustomerSupportEnv, ACTIONS

with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)

env = CustomerSupportEnv()

query = input("Enter query: ")

state = env.reset(query)
done = False


def get_q(state, action):
    return Q.get((state, action), 0)


while not done:
    q_values = [get_q(state, a) for a in ACTIONS]
    action = ACTIONS[q_values.index(max(q_values))]

    next_state, reward, done = env.step(action)

    print(f"Action: {action} | Reward: {reward}")

    state = next_state

print("\nDecision:", action)
print("Final State:", state)
