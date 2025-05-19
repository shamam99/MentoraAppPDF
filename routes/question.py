from fastapi import APIRouter, HTTPException, Request
from schemas.question import QuestionInput
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/question", tags=["Question Generation"])

@router.post("/generate")
def generate_questions(request: Request, payload: QuestionInput):
    try:
        # Access preloaded model from app state
        qg = request.app.state.qg
        questions = qg.generate(payload.text, use_evaluator=True, answer_style=payload.mode)

        logger.info(" Generated %s questions", len(questions))
        return {"questions": questions}
    except Exception as e:
        logger.error(" INTERNAL BACKEND ERROR", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
