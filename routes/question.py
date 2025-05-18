from fastapi import APIRouter, HTTPException
from schemas.question import QuestionInput
from services.qg_service import generate_questions_from_text
import pprint

router = APIRouter(prefix="/api/question", tags=["Question Generation"])

@router.post("/generate")
def generate_questions(payload: QuestionInput):
    try:
        questions = generate_questions_from_text(payload.text, payload.mode)
        pprint.pprint({"questions": questions})
        return {"questions": questions}
    except Exception as e:
        print("❌ INTERNAL BACKEND ERROR:")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
