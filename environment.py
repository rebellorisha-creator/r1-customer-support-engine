import random
from textblob import TextBlob

ACTIONS = ["AI", "HUMAN", "ESCALATE", "COMPENSATE"]

class CustomerSupportEnv:

    def __init__(self):
        self.state = None
        self.done = False
        self.query = ""
        self.step_count = 0
        self.max_steps = 1

    # -------------------------
    # RESET
    # -------------------------
    def reset(self, query):
        self.query = query
        self.done = False
        self.step_count = 0

        self.state = (*self.get_state(query), self.step_count)
        return self.state

    # -------------------------
    # STATE EXTRACTION
    # -------------------------
    def get_state(self, query):
        query = query.lower()

        # Sentiment
        polarity = TextBlob(query).sentiment.polarity
        if polarity < -0.2 or any (word in query for word in ["angry", "upset", "frustrated", "terrible"]):
            sentiment = "angry"
        elif polarity > 0.2:
            sentiment = "happy"
        else:
            sentiment = "neutral"

        # Urgency
        if any(word in query for word in ["urgent", "immediately", "now"]):
            urgency = "high"
        elif any(word in query for word in ["soon", "fast"]):
            urgency = "medium"
        else:
            urgency = "low"

        # Complexity
        if any(word in query for word in ["refund", "payment", "error", "failed", "crash"]):
            complexity = "complex"
        else:
            complexity = "simple"

        # Issue type
        if "refund" in query:
            issue_type = "refund"
        elif "payment" in query:
            issue_type = "payment"
        elif "crash" in query or "error" in query:
            issue_type = "technical"
        elif "manager" in query:
            issue_type = "escalation"
        else:
            issue_type = "general"

        return (sentiment, urgency, complexity, issue_type)

    # -------------------------
    # STEP FUNCTION
    # -------------------------
    def step(self, action):

        sentiment, urgency, complexity, issue_type, step = self.state

        reward = 0
        success = False

        query = self.query.lower()
        if("terrible" in query or "bad" in query or "worst" in query or "fix this now" in query):
            if action=="ESCALATE":
                return self.state, 20, True, {}
            else:
                return self.state, -20, True, {}

        # -------------------------
        #  MANAGER / ESCALATION CASE
        # -------------------------
        if "manager" in query or issue_type == "escalation":
            if action in ["ESCALATE", "HUMAN"]:
                reward += 10
                success = True
            else:
                reward -= 10

        # -------------------------
        #  ANGRY USER
        # -------------------------
        elif sentiment == "angry":
            if action == "ESCALATE":
                reward += 8
                success = True
            elif action == "HUMAN":
                reward += 5
                success = True
            elif action == "COMPENSATE":
                reward += 6
                success = True
            else:
                reward -= 5

        # -------------------------
        #  COMPLEX ISSUE
        # -------------------------
        elif complexity == "complex":
            if action in ["HUMAN", "ESCALATE"]:
                reward += 6
                success = True
            elif action == "COMPENSATE":
                reward += 4
                success = True
            else:
                reward -= 3

        # -------------------------
        #  SIMPLE ISSUE
        # -------------------------
        else:
            if action == "AI":
                reward += 5
                success = True
            else:
                reward -= 2

        # -------------------------
        # ⏹ END EPISODE
        # -------------------------
        self.step_count += 1

        if success or self.step_count >= self.max_steps:
            self.done = True

        self.state = (*self.get_state(self.query), self.step_count)

        return self.state, reward, self.done, {}
