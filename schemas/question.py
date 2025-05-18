from pydantic import BaseModel
from typing import Literal

class QuestionInput(BaseModel):
    text: str
    mode: Literal["tf", "mcq", "both"] = "both"
