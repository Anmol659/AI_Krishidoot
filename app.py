# app.py
from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
import uvicorn
from backend.EcoAdvisior.EcoAdvisior import main_handler


app = FastAPI(title="EcoAdvisor API")

class QueryRequest(BaseModel):
    query: str
    has_image: Optional[bool] = False
    has_audio: Optional[bool] = False

@app.post("/query")
def process_query(req: QueryRequest):
    try:
        # Capture output by modifying main_handler to return the string instead of printing
        from io import StringIO
        import sys
        buffer = StringIO()
        sys_stdout = sys.stdout
        sys.stdout = buffer

        main_handler(req.query, has_image=req.has_image, has_audio=req.has_audio)

        sys.stdout = sys_stdout
        output = buffer.getvalue()

        return {"success": True, "response": output}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/")
def root():
    return {"message": "EcoAdvisor is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
