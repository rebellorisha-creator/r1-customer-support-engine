#  IMPORTS
from fastapi import FastAPI
from pydantic import BaseModel
from env import CustomerSupportEnv

#  CREATE APP + ENV
app = FastAPI()
env = CustomerSupportEnv()

#  REQUEST MODEL (for /step)
class ActionRequest(BaseModel):
    decision: str


#  RESET ENDPOINT
@app.post("/reset")
def reset(task: str = "easy"):
    """
    Starts a new episode.
    Optional: pass task = easy / medium / hard
    """
    return env.reset(task)


#  STEP ENDPOINT
@app.post("/step")
def step(action: ActionRequest):
    """
    Takes one action in the environment.
    Example input:
    {
        "decision": "escalate"
    }
    """
    return env.step(action.decision)


#  STATE ENDPOINT 
@app.get("/state")
def get_state():
    """
    Returns current environment state
    """
    return env.state


# ROOT 
@app.get("/")
def home():
    return {"message": "Customer Support RL Environment is running "}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)