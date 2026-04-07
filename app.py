from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from env import PersonalFinanceEnv
from models import Action
from tasks import TASKS

app = FastAPI(title="Personal Finance OpenEnv")

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Stateful memory for the active simulation
running_envs = {}

class AuthRequest(BaseModel):
    email: str
    password: str
    name: str = None  # Reused for register

@app.post("/login")
def login(req: AuthRequest):
    # Mock auth check
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Invalid password")
    return {"token": "mock-jwt-token", "user": {"email": req.email}}

@app.post("/register")
def register(req: AuthRequest):
    # Mock auto check
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Password too short")
    return {"token": "mock-jwt-token", "user": {"email": req.email, "name": req.name}}

@app.get("/")
def index():
    return {"message": "Personal Finance OpenEnv is running.", "version": "1.0.0", "tasks": list(TASKS.keys())}

@app.get("/health")
def health():
    """Liveness probe for HF Spaces and Docker HEALTHCHECK."""
    return {"status": "ok"}

@app.get("/tasks")
def tasks_endpoint():
    return list(TASKS.keys())

@app.post("/reset")
@app.post("/reset/{task_id}")
def reset_endpoint(task_id: str = "easy"):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    env = PersonalFinanceEnv(task_id=task_id)
    obs = env.reset()
    running_envs[task_id] = env
    
    return {
        "observation": obs.model_dump(),
        "info": {
            "message": f"Environment reset successful. Starting balance: {obs.checking_balance}"
        }
    }

@app.get("/state/{task_id}")
def state_endpoint(task_id: str):
    if task_id not in running_envs:
        raise HTTPException(status_code=404, detail="Simulation not started")
    env = running_envs[task_id]
    return env.state().model_dump()

@app.post("/step")
@app.post("/step/{task_id}")
def step_endpoint(action: Action, task_id: str = "easy"):
    if task_id not in running_envs:
        raise HTTPException(status_code=404, detail="Simulation not started. Call /start first.")
    
    env = running_envs[task_id]
    
    try:
        obs, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info.model_dump() if info else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
