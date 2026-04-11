import os
from env import CustomerSupportEnv
from openai import OpenAI

# Initialize environment
env = CustomerSupportEnv()

# Initialize LLM client (IMPORTANT)
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)


def get_action_from_llm(state):
    try:
        prompt = f"""
You are a customer support agent.

State:
sentiment: {state['sentiment']}
urgency: {state['urgency']}
complexity: {state['complexity']}
category: {state['category']}

Choose ONE action from:
auto_reply, ask_clarification, escalate, route_to_sales, route_to_tech

Only return action name.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        action = response.choices[0].message.content.strip().lower()

        valid_actions = [
            "auto_reply",
            "ask_clarification",
            "escalate",
            "route_to_sales",
            "route_to_tech"
        ]

        for a in valid_actions:
            if a in action:
                return a

        return "ask_clarification"

    except Exception as e:
        print(f"LLM error: {e}", flush=True)
        return "ask_clarification"

def run_task():
    print("[START] task=customer_support", flush=True)

    state = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        steps += 1

        # ✅ LLM decision
        action = get_action_from_llm(state)

        try:
            result = env.step(action)
        except Exception as e:
            print(f"Step error: {e}", flush=True)
            break

        reward = result["reward"]
        total_reward += reward

        print(f"[STEP] step={steps} reward={reward}", flush=True)

        state = result["observation"]
        done = result["done"]

    score = total_reward / steps if steps > 0 else 0

    print(f"[END] task=customer_support score={score} steps={steps}", flush=True)


if __name__ == "__main__":
    run_task()