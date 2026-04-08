import random
from textblob import TextBlob

ACTIONS = ["AI", "HUMAN", "ESCALATE", "COMPENSATE"]

SAMPLE_QUERIES = [
    "I want a refund",
    "My app crashed",
    "Where is my order",
    "This is the worst service",
    "Fix this now!!!",
    "I am happy with the service",
    "My delivery is delayed",
    "Payment failed",
    "I need to talk to manager"
]


#  FINAL SENTIMENT (FIXED)
def detect_sentiment(text):
    text = text.lower()

    # Force angry cases
    if any(word in text for word in [
        "manager", "complaint", "worst", "angry",
        "frustrated", "late", "delayed", "not working",
        "issue", "problem", "bad service"
    ]):
        return "angry"

    polarity = TextBlob(text).sentiment.polarity

    if polarity < -0.2:
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
# STATE GENERATION
# ----------------------------
def get_state(query):
    q = query.lower()

    #  NLP sentiment
    sentiment = detect_sentiment(query)

    #  Boost strong negative words
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


    # urgency
    if any(word in q for word in ["urgent", "now", "immediately", "right now", "manager"]):
        urgency = "high"
    elif any(word in q for word in ["soon", "waiting", "delay", "late", "delayed"]):
        urgency = "medium"
    else:
        urgency = "low"

    # complexity
    if any(word in q for word in ["crash", "error", "failed", "bug"]):
        complexity = "complex"
    else:
        complexity = "simple"

    # issue type
    if "refund" in q:
        issue_type = "refund"
    elif "payment" in q:
        issue_type = "payment"
    elif any(word in q for word in ["delivery", "order", "late", "delayed"]):
        issue_type = "delivery"
    else:
        issue_type = "general"

    return sentiment, urgency, complexity, issue_type


#  ENVIRONMENT
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

    def reset(self, query=None):
        self.step_count = 0
        self.done = False

        if query:
            self.query = query
        else:
            self.query = random.choice(SAMPLE_QUERIES)

        self.state = (*get_state(self.query), self.step_count)
        return self.state

    def step(self, action):
        self.step_count += 1

        sentiment, urgency, complexity, issue_type = get_state(self.query)

        reward = 0
        success = False

        #  ANGRY + HIGH → ESCALATE (VERY STRONG)
        if sentiment == "angry" and urgency == "high":
            if action == "ESCALATE":
                reward += 10
                success = True
            else:
                reward -= 10

        #  DELIVERY → ESCALATE / HUMAN
        elif issue_type == "delivery":
            if action == "ESCALATE":
                reward += 7
                success = True
            elif action == "HUMAN":
                reward += 5
                success = True
            else:
                reward -= 6

        #  PAYMENT → HUMAN / ESCALATE
        elif issue_type == "payment":
            if action in ["HUMAN", "ESCALATE"]:
                reward += 8
                success = True
            else:
                reward -= 7

        #  COMPLEX → HUMAN
        elif complexity == "complex":
            if action == "HUMAN":
                reward += 6
                success = True
            else:
                reward -= 5

        #  SIMPLE → AI
        else:
            if action == "AI":
                reward += 3
                success = True
            else:
                reward -= 2

        # stop episode
        if success or self.step_count >= self.max_steps:
            self.done = True

        self.state = (*get_state(self.query), self.step_count, action)

        return self.state, reward, self.done