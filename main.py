from fastapi import FastAPI
from routes import question
import logging

# ✅ Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(title="Mentora PDF Question Generator")

# ✅ Include your main route
app.include_router(question.router)

# ✅ Allow HEAD requests for health check
@app.get("/")
@app.head("/")
def root():
    return {"message": "Mentora Question Generator API is running."}
