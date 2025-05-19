from fastapi import FastAPI
from routes import question
from questiongenerator import QuestionGenerator  # ✅ import the generator

import logging

# ✅ Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ✅ Initialize FastAPI app
app = FastAPI(title="Mentora PDF Question Generator")

# ✅ Load model once at startup
qg = QuestionGenerator()
app.state.qg = qg  # <- attach to global app state

# ✅ Register routes
app.include_router(question.router)

@app.get("/")
def read_root():
    return {"message": "Mentora Question Generator API is running."}
