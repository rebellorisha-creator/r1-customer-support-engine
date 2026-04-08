from environment import CustomerSupportEnv, ACTIONS
import pickle

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

        # ----------------------------
        # 🔥 RULE-BASED FIX (IMPORTANT)
        # ----------------------------
        sentiment, urgency, complexity = state[0], state[1], state[2]

        if sentiment == "angry" and complexity == "complex":
            action = "ESCALATE"   # override RL decision

        # ----------------------------
        # STEP
        # ----------------------------
        next_state, reward, done, _ = env.step(action)

        final_action = action
        final_reward = reward

        state = next_state

    return {
        "state": state,
        "action": final_action,
        "reward": final_reward
    }