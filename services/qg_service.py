from questiongenerator import QuestionGenerator

def generate_questions_from_text(text: str, mode: str = "both") -> list:
    qg = QuestionGenerator()  # ✅ Load here to avoid RAM limit on startup
    if mode == "tf":
        return qg.generate(text, use_evaluator=True, answer_style="sentences")
    elif mode == "mcq":
        return qg.generate(text, use_evaluator=True, answer_style="multiple_choice")
    elif mode == "both":
        return qg.generate(text, use_evaluator=True, answer_style="all")
    else:
        raise ValueError("Invalid mode. Choose from: tf, mcq, both.")
