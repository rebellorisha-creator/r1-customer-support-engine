
from env import CustomerSupportEnv
import random

# Initialize environment
env = CustomerSupportEnv()

def run_task(task_name="customer_support"):

    # START block
    print(f"[START] task={task_name}", flush=True)

    # Reset environment
    state = env.reset()

    total_reward = 0
    steps = 0

    done = False

    while not done:
        steps += 1

        # Choose random action
        action = random.choice([
            "auto_reply",
            "ask_clarification",
            "escalate",
            "route_to_sales",
            "route_to_tech"
        ])

        result = env.step(action)

        reward = result["reward"]
        total_reward += reward

        # STEP block
        print(f"[STEP] step={steps} reward={reward}", flush=True)

        done = result["done"]

    # Calculate score (simple avg reward)
    score = total_reward / steps if steps > 0 else 0

    # END block
    print(f"[END] task={task_name} score={score} steps={steps}", flush=True)


# Run when file is executed
if __name__ == "__main__":
    run_task()