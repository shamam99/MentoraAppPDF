from fastapi import FastAPI
from routes import question
import logging

# ✅ Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ✅ Initialize FastAPI app
app = FastAPI(title="Mentora PDF Question Generator")

# ✅ Register route
app.include_router(question.router)

@app.get("/")
def read_root():
    return {"message": "Mentora Question Generator API is running."}
