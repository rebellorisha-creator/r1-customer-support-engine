import random

ACTIONS = [
    "auto_reply",
    "ask_clarification",
    "escalate",
    "route_to_sales",
    "route_to_tech"
]

class CustomerSupportEnv:

    def __init__(self):
        self.max_steps = 3        #  ADD HERE
        self.current_step = 0
        self.history = []        #  ADD HERE
        self.state = {}

    #  RESET FUNCTION
    def reset(self, task="easy"):
        self.task = task
        self.current_step = 0
        self.history = []        # reset history

        self.state = {
            "sentiment": random.choice(["positive", "neutral", "negative"]),
            "urgency": random.choice(["low", "medium", "high"]),
            "complexity": random.choice(["simple", "complex"]),
            "category": random.choice(["billing", "technical", "general"]),
            "step": 0
        }

        return self.state

    #  REWARD FUNCTION
    def calculate_reward(self, action):
        reward = 0

        if self.state["urgency"] == "high" and action == "escalate":
            reward += 1

        if self.state["complexity"] == "simple" and action == "auto_reply":
            reward += 1

        if action == "ask_clarification":
            reward += 0.3

        if self.state["urgency"] == "high" and action != "escalate":
            reward -= 0.5

        #  NEW RULE
        if self.state["category"] == "technical" and action == "route_to_tech":
            reward += 1

        return reward

    #  STEP FUNCTION
    def step(self, action):

        self.current_step += 1                     #  ADD HERE
        self.history.append(action)                #  ADD HERE

        reward = self.calculate_reward(action)

        done = self.current_step >= self.max_steps  #  ADD HERE

        # update state step count
        self.state["step"] = self.current_step

        #  DEBUG LOGS
        print(f"STATE: {self.state}")
        print(f"ACTION: {action}")
        print(f"REWARD: {reward}")

        return {
            "observation": self.state,
            "reward": reward,
            "done": done,
            "info": {"history": self.history}   # ✅ ADD HERE
        }