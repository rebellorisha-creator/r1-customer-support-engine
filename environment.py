import random
from textblob import TextBlob

# ----------------------------
# SENTIMENT DETECTION (NLP)
# ----------------------------
def detect_sentiment(query):
    analysis = TextBlob(query)
    polarity = analysis.sentiment.polarity

    if polarity < -0.3:
        return "angry"
    elif polarity > 0.3:
        return "happy"
    else:
        return "neutral"


# ----------------------------
# STATE SPACE
# ----------------------------
SENTIMENTS = ["happy", "neutral", "angry"]
URGENCIES = ["low", "medium", "high"]
COMPLEXITIES = ["simple", "medium", "complex"]

ACTIONS = ["AI", "HUMAN", "ESCALATE", "COMPENSATE"]

# ----------------------------
# ACTION COST (REALISM)
# ----------------------------
ACTION_COST = {
    "AI": 0.1,
    "HUMAN": 0.5,
    "ESCALATE": 0.7,
    "COMPENSATE": 1.0
}

# ----------------------------
# SAMPLE DATA
# ----------------------------
SAMPLE_QUERIES = [
    "My payment failed and I'm upset!",
    "How to reset password?",
    "Order not arrived, urgent!",
    "App keeps crashing!",
    "I want refund right now!",
    "You guys are useless and stupid",
    "Thanks for the quick help!"
]


# ----------------------------
# STATE GENERATION
# ----------------------------
def get_state(query):
    q = query.lower()

    # ✅ NLP sentiment
    sentiment = detect_sentiment(query)

    # 🔥 Boost strong negative words
    if any(word in q for word in ["stupid", "useless", "hate", "worst"]):
        sentiment = "angry"

    # urgency detection
    if any(word in q for word in ["urgent", "now", "immediately"]):
        urgency = "high"
    elif any(word in q for word in ["soon", "asap"]):
        urgency = "medium"
    else:
        urgency = "low"

    # complexity detection
    if any(word in q for word in ["failed", "crashing", "crashed", "error", "not working"]):
        complexity = "complex"
    elif any(word in q for word in ["how", "help", "guide"]):
        complexity = "medium"
    else:
        complexity = "simple"

    return sentiment, urgency, complexity


# ----------------------------
# RL ENVIRONMENT
# ----------------------------
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

        self.step_count = 0
        self.prev_action = "NONE"
        self.done = False

        self.state = (*get_state(self.query), self.step_count, self.prev_action)
        return self.state

    def step(self, action):
        self.step_count += 1

        sentiment, urgency, complexity, _, _ = self.state

        # ----------------------------
        # SUCCESS LOGIC
        # ----------------------------
        success = False

        if action == "ESCALATE" and sentiment == "angry":
            success = True
        elif action == "AI" and complexity == "simple":
            success = True
        elif action == "HUMAN" and complexity in ["medium", "complex"]:
            success = True
        elif action == "COMPENSATE" and sentiment == "angry":
            success = True
        else:
            success = random.random() < 0.3  # exploration noise

        # ----------------------------
        # REWARD FUNCTION
        # ----------------------------
        reward = (1 if success else -1) - ACTION_COST[action]

        # slight penalty for overusing humans
        if action == "HUMAN":
            reward -= 0.1

        print(f"Action: {action} | Success: {success} | Reward: {round(reward, 2)}")

        self.prev_action = action

        # next state
        self.state = (*get_state(self.query), self.step_count, self.prev_action)

        # done condition
        if success or self.step_count >= self.max_steps:
            self.done = True

        return self.state, round(reward, 2), self.done