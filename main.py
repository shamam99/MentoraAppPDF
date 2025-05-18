from fastapi import FastAPI
from routes import question

app = FastAPI(title="Mentora PDF Question Generator")

# Register route
app.include_router(question.router)

@app.get("/")
def read_root():
    return {"message": "Mentora Question Generator API is running."}
