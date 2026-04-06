import random

SENTIMENTS = ["happy", "neutral", "angry"]
URGENCIES = ["low", "medium", "high"]
COMPLEXITIES = ["simple", "medium", "complex"]

ACTIONS = ["AI", "HUMAN", "ESCALATE", "COMPENSATE"]

# 💰 Cost for each action
ACTION_COST = {
    "AI": 0.1,
    "HUMAN": 0.5,
    "ESCALATE": 0.7,
    "COMPENSATE": 1.0
}

SAMPLE_QUERIES = [
    "My payment failed and I'm upset!",
    "How to reset password?",
    "Order not arrived, urgent!",
    "App keeps crashing!",
    "I want refund right now!",
]

def get_state(query):
    q = query.lower()

    sentiment = "angry" if any(word in q for word in ["upset", "refund", "angry"]) else "neutral"
    urgency = "high" if "urgent" in q or "now" in q else "medium"
    complexity = "complex" if any(word in q for word in ["failed", "crashing", "crashed"]) else "simple"

    return sentiment, urgency, complexity


class CustomerSupportEnv:
    def __init__(self):
        self.max_steps = 3
        self.step_count = 0
        self.prev_action = "NONE"
        self.done = False
        self.query = ""
        self.state = None

    def reset(self, query=None):
          if query:
              self.query = query
          else:
              self.query = random.choice(SAMPLE_QUERIES)

    # ✅ ALWAYS initialize
          self.step_count = 0
          self.prev_action = "NONE"
          self.done = False

          self.state = (*get_state(self.query), self.step_count, self.prev_action)

          return self.state

    def step(self, action):
        self.step_count += 1

        # 🎯 Success probability logic
        sentiment, urgency, complexity, _, _ = self.state

        success = False

        if action == "ESCALATE" and sentiment == "angry":
            success = True
        elif action == "AI" and complexity == "simple":
            success = True
        elif action == "HUMAN" and complexity == "complex":
            success = True
        elif action == "COMPENSATE" and sentiment == "angry":
            success = True
        else:
            success = random.random() < 0.3  # some randomness

        # 💰 Reward = success - cost
        reward = (1 if success else -1) - ACTION_COST[action]

        print(f"Action: {action} | Success: {success} | Reward: {round(reward,2)}")

        self.prev_action = action

        # next state
        self.state = (*get_state(self.query), self.step_count, self.prev_action)

        if success or self.step_count >= self.max_steps:
            self.done = True

        return self.state, round(reward, 2), self.done