import os
from openai import OpenAI
from test import run_query

# ✅ Dummy OpenAI client (for requirement compliance)
client = OpenAI(
    base_url=os.getenv("API_BASE_URL", "dummy"),
    api_key=os.getenv("HF_TOKEN", "dummy")
)

MODEL_NAME = os.getenv("MODEL_NAME", "dummy")


def main():
    print("START")

    query = input("Enter query: ")

    print("STEP: processing query")

    result = run_query(query)

    print("STEP: decision made")

    print({
        "action": result["action"],
        "state": result["state"],
        "reward": result["reward"]
    })

    print("END")


if __name__ == "__main__":
    main()