from fastapi import APIRouter, HTTPException, Request
from schemas.question import QuestionInput
from services.qg_service import generate_questions_from_text
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/question", tags=["Question Generation"])

@router.post("/generate")
def generate_questions(payload: QuestionInput, request: Request):
    try:
        qg = request.app.state.qg
        questions = generate_questions_from_text(qg, payload.text, payload.mode)
        logger.info("Generated %s questions", len(questions))
        return {"questions": questions}
    except Exception as e:
        logger.error("INTERNAL BACKEND ERROR", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", tags=["Health"])
@router.head("/health", include_in_schema=False)
def health_check():
    return {"status": "ok"}
