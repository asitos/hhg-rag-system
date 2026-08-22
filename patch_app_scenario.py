import re

with open("app.py", "r") as f:
    content = f.read()

scenario_endpoint = """
from pydantic import BaseModel
class ScenarioRequest(BaseModel):
    scenario: str

@app.post("/api/v1/scenario")
async def set_scenario(req: ScenarioRequest):
    settings.demo_scenario = req.scenario
    return {"status": "ok", "scenario": settings.demo_scenario}

@app.get("/api/v1/scenario")
async def get_scenario():
    return {"scenario": settings.demo_scenario}
"""

if "/api/v1/scenario" not in content:
    content = content.replace('from fastapi import FastAPI, UploadFile, File, Form, HTTPException', 
                              'from fastapi import FastAPI, UploadFile, File, Form, HTTPException\n' + scenario_endpoint)

with open("app.py", "w") as f:
    f.write(content)
