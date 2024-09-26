import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# FastAPI app
app = FastAPI()

@app.get("/api")
def read_root():
    return JSONResponse(content={"message": "Hello from FastAPI!"})

@app.get("/api_1")
def read_root_1():
    return JSONResponse(content={"message": "Hello from FastAPI!_gergre"})

def run_fastapi():
    port = 8001
    uvicorn.run(app, host="127.0.0.1", port=port)

# Run FastAPI in a separate thread
api_thread = threading.Thread(target=run_fastapi, daemon=True)
api_thread.start()
