from fastapi import FastAPI
from routes import question
from questiongenerator import QuestionGenerator
import spacy.cli
import logging

# ✅ 1. Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ✅ 2. Download spaCy model (will skip if already installed)
spacy.cli.download("en_core_web_sm")

# ✅ 3. Initialize FastAPI
app = FastAPI(title="Mentora PDF Question Generator")

# ✅ 4. Load HuggingFace models once at startup
qg = QuestionGenerator()
app.state.qg = qg

# ✅ 5. Include your routes
app.include_router(question.router)

@app.get("/")
def read_root():
    return {"message": "Mentora Question Generator API is running."}
