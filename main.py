from fastapi import FastAPI
from routes import question
from services.qg_service import QuestionGenerator
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(title="Mentora PDF Question Generator")

# ✅ Load model once and reuse
@app.on_event("startup")
async def startup_event():
    app.state.qg = QuestionGenerator()

app.include_router(question.router)

@app.get("/")
@app.head("/")
def root():
    return {"message": "Mentora Question Generator API is running."}
