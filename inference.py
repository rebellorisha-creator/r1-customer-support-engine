
from fastapi import FastAPI
from pydantic import BaseModel
from env import CustomerSupportEnv
import random

app = FastAPI()

# Initialize environment
env = CustomerSupportEnv()


# Request model
class QueryRequest(BaseModel):
    query: str


# Root endpoint (health check)
@app.get("/")
def root():
    return {"message": "API is working"}


# Reset endpoint (MANDATORY for checker)
@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state}


# Inference endpoint
@app.post("/infer")
def infer(request: QueryRequest):

    # Reset with query
    state = env.reset(request.query)

    # Choose action (simple logic)
    action = random.choice([
        "auto_reply",
        "ask_clarification",
        "escalate",
        "route_to_sales",
        "route_to_tech"
    ])

    result = env.step(action)

    return {
        "action": action,
        "result": result
    }