from fastapi import FastAPI
from routes import question
from services.qg_service import QuestionGenerator
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(title="Mentora PDF Question Generator")

@app.on_event("startup")
async def startup_event():
    # Directories for local model cache
    model_dir = "models/t5"
    evaluator_dir = "models/bert"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(evaluator_dir, exist_ok=True)

    # Load the generator using local cache
    app.state.qg = QuestionGenerator(
        model_dir=model_dir,
        evaluator_dir=evaluator_dir
    )

app.include_router(question.router)

@app.get("/")
@app.head("/")
def root():
    return {"message": "Mentora Question Generator API is running."}
