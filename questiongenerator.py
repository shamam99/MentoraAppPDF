import spacy
import json
import numpy as np
import random
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from typing import Any, List, Mapping, Tuple


class QuestionGenerator:
    """A transformer-based NLP system for generating reading comprehension-style questions from
    texts. It can generate full sentence questions, multiple choice questions, or a mix of the
    two styles.

    To filter out low quality questions, questions are assigned a score and ranked once they have
    been generated. Only the top k questions will be returned. This behaviour can be turned off
    by setting use_evaluator=False.
    """

    def __init__(self, model_dir=None, evaluator_dir=None) -> None:
        QG_PRETRAINED = model_dir or "iarfmoose/t5-base-question-generator"

        self.ANSWER_TOKEN = "<answer>"
        self.CONTEXT_TOKEN = "<context>"
        self.SEQ_LENGTH = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load T5 model
        try:
            self.qg_tokenizer = AutoTokenizer.from_pretrained(QG_PRETRAINED, use_fast=False)
        except Exception as e:
            print(f"❌ Failed loading tokenizer from {QG_PRETRAINED} — {e}")
            raise

        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)
        self.qg_model.eval()

        # Load Evaluator using directory
        self.qa_evaluator = QAEvaluator(evaluator_dir=evaluator_dir)


    def generate(
        self,
        article: str,
        use_evaluator: bool = True,
        num_questions: bool = None,
        answer_style: str = "all"
    ) -> List:
        """Takes an article and generates a set of question and answer pairs. If use_evaluator
        is True then QA pairs will be ranked and filtered based on their quality. answer_style
        should selected from ["all", "sentences", "multiple_choice"].
        """

        print("Generating questions...\n")

        qg_inputs, qg_answers = self.generate_qg_inputs(article, answer_style)
        generated_questions = self.generate_questions_from_inputs(qg_inputs)

        message = "{} questions doesn't match {} answers".format(
            len(generated_questions), len(qg_answers)
        )
        assert len(generated_questions) == len(qg_answers), message

        if use_evaluator:
            print("Evaluating QA pairs...\n")
            encoded_qa_pairs = self.qa_evaluator.encode_qa_pairs(
                generated_questions, qg_answers
            )
            scores = self.qa_evaluator.get_scores(encoded_qa_pairs)

            if num_questions:
                qa_list = self._get_ranked_qa_pairs(
                    generated_questions, qg_answers, scores, num_questions
                )
            else:
                qa_list = self._get_ranked_qa_pairs(
                    generated_questions, qg_answers, scores
                )

        else:
            print("Skipping evaluation step.\n")
            qa_list = self._get_all_qa_pairs(generated_questions, qg_answers)

        return qa_list
    
    def generate_false_statements(self, sentences: List[str], exclude_set=None) -> List[str]:
        nlp = spacy.load("en_core_web_sm")
        false_statements = []

        for sent in sentences:
            doc = nlp(sent)
            tokens = []
            negated = False

            for token in doc:
                if token.dep_ == "aux" and not negated:
                    tokens.append(token.text)
                    tokens.append("not")
                    negated = True
                elif token.dep_ == "ROOT" and token.pos_ == "VERB" and not negated:
                    tokens.append("did not")
                    negated = True
                    continue
                else:
                    tokens.append(token.text)

            new_sent = " ".join(tokens)
            if negated and new_sent.strip().lower() not in (exclude_set or set()):
                false_statements.append(new_sent)

        return false_statements



    def generate_qg_inputs(self, text: str, answer_style: str) -> Tuple[List[str], List[Any]]:
        """Given a text, returns a list of model inputs and a list of corresponding answers.
        Model inputs take the form "answer_token <answer text> context_token <context text>" where
        the answer is a string extracted from the text, and the context is the wider text.
        """

        VALID_ANSWER_STYLES = ["all", "sentences", "multiple_choice"]
        if answer_style not in VALID_ANSWER_STYLES:
            raise ValueError(f"Invalid answer style {answer_style}. Please choose from {VALID_ANSWER_STYLES}")

        inputs, answers = [], []
        used_statements = set()

        if answer_style in ["sentences", "all"]:
            segments = self._split_into_segments(text)
            for segment in segments:
                sentences = self._split_text(segment)
                prepped_inputs, prepped_answers = self._prepare_qg_inputs(sentences, segment)
                for inp, ans in zip(prepped_inputs, prepped_answers):
                    norm = ans.strip().lower()
                    if norm not in used_statements:
                        inputs.append(inp)
                        answers.append({"statement": ans, "is_true": True})
                        used_statements.add(norm)

                false_statements = self.generate_false_statements(sentences, used_statements)
                false_inputs, false_answers = self._prepare_qg_inputs(false_statements, segment)

                for inp, ans in zip(false_inputs, false_answers):
                    norm = ans.strip().lower()
                    if norm not in used_statements:
                        inputs.append(inp)
                        answers.append({"statement": ans, "is_true": False})
                        used_statements.add(norm)

            # 🔁 Ensure at least 3 false statements
            inputs, answers = self._ensure_minimum_false(inputs, answers, min_false=3)

        if answer_style in ["multiple_choice", "all"]:
            sentences = self._split_text(text)
            prepped_inputs, prepped_answers = self._prepare_qg_inputs_MC(sentences)
            inputs.extend(prepped_inputs)
            answers.extend(prepped_answers)

        return inputs, answers


    def generate_questions_from_inputs(self, qg_inputs: List) -> List[str]:
        """Given a list of concatenated answers and contexts, with the form:
        "answer_token <answer text> context_token <context text>", generates a list of 
        questions.
        """
        generated_questions = []

        for qg_input in qg_inputs:
            question = self._generate_question(qg_input)
            generated_questions.append(question)

        return generated_questions

    def _split_text(self, text: str) -> List[str]:
        """Splits the text into sentences, and attempts to split or truncate long sentences."""
        MAX_SENTENCE_LEN = 128
        sentences = re.findall(".*?[.!\?]", text)
        cut_sentences = []

        for sentence in sentences:
            if len(sentence) > MAX_SENTENCE_LEN:
                cut_sentences.extend(re.split("[,;:)]", sentence))

        # remove useless post-quote sentence fragments
        cut_sentences = [s for s in sentences if len(s.split(" ")) > 5]
        sentences = sentences + cut_sentences

        return list(set([s.strip(" ") for s in sentences]))

    def _split_into_segments(self, text: str) -> List[str]:
        """Splits a long text into segments short enough to be input into the transformer network.
        Segments are used as context for question generation.
        """
        MAX_TOKENS = 490
        paragraphs = text.split("\n")
        tokenized_paragraphs = [
            self.qg_tokenizer(p)["input_ids"] for p in paragraphs if len(p) > 0
        ]
        segments = []

        while len(tokenized_paragraphs) > 0:
            segment = []

            while len(segment) < MAX_TOKENS and len(tokenized_paragraphs) > 0:
                paragraph = tokenized_paragraphs.pop(0)
                segment.extend(paragraph)
            segments.append(segment)

        return [self.qg_tokenizer.decode(s, skip_special_tokens=True) for s in segments]

    def _prepare_qg_inputs(
        self,
        sentences: List[str],
        text: str
    ) -> Tuple[List[str], List[str]]:
        """Uses sentences as answers and the text as context. Returns a tuple of (model inputs, answers).
        Model inputs are "answer_token <answer text> context_token <context text>" 
        """
        inputs = []
        answers = []

        for sentence in sentences:
            qg_input = f"{self.ANSWER_TOKEN} {sentence} {self.CONTEXT_TOKEN} {text}"
            inputs.append(qg_input)
            answers.append(sentence)

        return inputs, answers

    def _prepare_qg_inputs_MC(self, sentences: List[str]) -> Tuple[List[str], List[str]]:
        """Performs NER on the text, and uses extracted entities are candidate answers for multiple-choice
        questions. Sentences are used as context, and entities as answers. Returns a tuple of (model inputs, answers). 
        Model inputs are "answer_token <answer text> context_token <context text>"
        """
        spacy_nlp = spacy.load("en_core_web_sm")
        docs = list(spacy_nlp.pipe(sentences, disable=["parser"]))
        inputs_from_text = []
        answers_from_text = []

        for doc, sentence in zip(docs, sentences):
            entities = doc.ents
            if entities:

                for entity in entities:
                    qg_input = f"{self.ANSWER_TOKEN} {entity} {self.CONTEXT_TOKEN} {sentence}"
                    answers = self._get_MC_answers(entity, docs)
                    inputs_from_text.append(qg_input)
                    answers_from_text.append(answers)

        return inputs_from_text, answers_from_text

    def _get_MC_answers(self, correct_answer: Any, docs: Any) -> List[Mapping[str, Any]]:
        """Finds a set of alternative answers for a multiple-choice question. Will attempt to find
        alternatives of the same entity type as correct_answer if possible.
        """
        entities = []

        for doc in docs:
            entities.extend([{"text": e.text, "label_": e.label_}
                            for e in doc.ents])

        # remove duplicate elements
        entities_json = [json.dumps(kv) for kv in entities]
        pool = set(entities_json)
        num_choices = (
            min(4, len(pool)) - 1
        )  # -1 because we already have the correct answer

        # add the correct answer
        final_choices = []
        correct_label = correct_answer.label_
        final_choices.append({"answer": correct_answer.text, "correct": True})
        pool.remove(
            json.dumps({"text": correct_answer.text,
                       "label_": correct_answer.label_})
        )

        # find answers with the same NER label
        matches = [e for e in pool if correct_label in e and "the three" not in e and "PGP" not in e and "S/MIME" not in e]

        # if we don't have enough then add some other random answers
        if len(matches) < num_choices:
            choices = matches
            pool = pool.difference(set(choices))
            choices.extend(random.sample(pool, num_choices - len(choices)))
        else:
            choices = random.sample(matches, num_choices)

        choices = [json.loads(s) for s in choices if all(x not in s for x in ["the three", "PGP", "S/MIME"])]

        for choice in choices:
            final_choices.append({"answer": choice["text"], "correct": False})

        random.shuffle(final_choices)
        return final_choices

    @torch.no_grad()
    def _generate_question(self, qg_input: str) -> str:
        """Takes qg_input which is the concatenated answer and context, and uses it to generate
        a question sentence. The generated question is decoded and then returned.
        """
        encoded_input = self._encode_qg_input(qg_input)
        output = self.qg_model.generate(input_ids=encoded_input["input_ids"])
        question = self.qg_tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
        return question

    def _encode_qg_input(self, qg_input: str) -> torch.tensor:
        """Tokenizes a string and returns a tensor of input ids corresponding to indices of tokens in 
        the vocab.
        """
        return self.qg_tokenizer(
            qg_input,
            padding='max_length',
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _get_ranked_qa_pairs(
        self, generated_questions: List[str], qg_answers: List[Any], scores, num_questions: int = 10
    ) -> List[Mapping[str, Any]]:
        """Ranks and formats the top N question-answer pairs (MCQ and T/F)."""

        if num_questions > len(scores):
            num_questions = len(scores)
            print((f"\nWas only able to generate {num_questions} questions.",
                "For more questions, please input a longer text."))

        structured_qa_list = []

        for i in range(num_questions):
            index = scores[i]
            question_text = generated_questions[index].strip().split("?")[0] + "?"
            answer_data = qg_answers[index]

            if isinstance(answer_data, list):  # MCQ
                correct = next((a["answer"] for a in answer_data if a["correct"]), None)
                structured_qa_list.append({
                    "type": "multiple_choice",
                    "question": question_text,
                    "choices": answer_data,
                    "correct_answer": correct
                })
            elif isinstance(answer_data, dict) and "statement" in answer_data:  # True/False
                structured_qa_list.append({
                    "type": "true_false",
                    "statement": answer_data["statement"],
                    "is_true": answer_data["is_true"]
                })
            else:  # fallback
                structured_qa_list.append({
                    "type": "true_false",
                    "statement": question_text,
                    "is_true": False 
                })

        return structured_qa_list
    
    def _deduplicate_statements(self, statements: List[str], threshold: float = 0.88) -> List[str]:
        import difflib
        unique = []
        for s in statements:
            if not any(difflib.SequenceMatcher(None, s.lower(), u.lower()).ratio() > threshold for u in unique):
                unique.append(s)
        return unique



    def _get_all_qa_pairs(
        self, generated_questions: List[str], qg_answers: List[Any]
    ) -> List[Mapping[str, Any]]:
        """Formats all generated question-answer pairs with type flags."""

        structured_qa_list = []

        for question, answer_data in zip(generated_questions, qg_answers):
            question_text = question.strip().split("?")[0] + "?"

            if isinstance(answer_data, list):  # MCQ
                correct = next((a["answer"] for a in answer_data if a["correct"]), None)
                structured_qa_list.append({
                    "type": "multiple_choice",
                    "question": question_text,
                    "choices": answer_data,
                    "correct_answer": correct
                })
            elif isinstance(answer_data, dict) and "statement" in answer_data:  # T/F
                structured_qa_list.append({
                    "type": "true_false",
                    "statement": question_text,
                    "is_true": answer_data["is_true"]
                })
            else:  # fallback
                structured_qa_list.append({
                    "type": "true_false",
                    "statement": question_text,
                    "is_true": True
                })

        return structured_qa_list
    
    def _ensure_minimum_false(self, inputs: List[str], answers: List[dict], min_false: int = 3) -> Tuple[List[str], List[dict]]:
        """Ensure at least `min_false` false statements in the final answer set."""
        true_items = [(inp, ans) for inp, ans in zip(inputs, answers) if ans["is_true"]]
        false_items = [(inp, ans) for inp, ans in zip(inputs, answers) if not ans["is_true"]]

        if len(false_items) >= min_false:
            return inputs, answers

        to_convert = min(min_false - len(false_items), len(true_items))
        if to_convert == 0:
            return inputs, answers

        sampled = random.sample(true_items, to_convert)
        for inp, ans in sampled:
            new_ans = ans.copy()
            new_ans["is_true"] = False
            false_items.append((inp, new_ans))
            true_items.remove((inp, ans))

        combined = true_items + false_items
        random.shuffle(combined)

        balanced_inputs, balanced_answers = zip(*combined)
        return list(balanced_inputs), list(balanced_answers)




class QAEvaluator:
    """Wrapper for a transformer model which evaluates the quality of question-answer pairs.
    Given a QA pair, the model will generate a score. Scores can be used to rank and filter
    QA pairs.
    """

    def __init__(self, evaluator_dir=None) -> None:
        QAE_PRETRAINED = evaluator_dir or "iarfmoose/bert-base-cased-qa-evaluator"
        self.SEQ_LENGTH = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qae_tokenizer = AutoTokenizer.from_pretrained(QAE_PRETRAINED)
        self.qae_model = AutoModelForSequenceClassification.from_pretrained(QAE_PRETRAINED)
        self.qae_model.to(self.device)
        self.qae_model.eval()

    def encode_qa_pairs(self, questions: List[str], answers: List[str]) -> List[torch.tensor]:
        """Takes a list of questions and a list of answers and encodes them as a list of tensors."""
        encoded_pairs = []

        for question, answer in zip(questions, answers):
            encoded_qa = self._encode_qa(question, answer)
            encoded_pairs.append(encoded_qa.to(self.device))

        return encoded_pairs

    def get_scores(self, encoded_qa_pairs: List[torch.tensor]) -> List[float]:
        """Generates scores for a list of encoded QA pairs."""
        scores = {}

        for i in range(len(encoded_qa_pairs)):
            scores[i] = self._evaluate_qa(encoded_qa_pairs[i])

        return [
            k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]

    def _encode_qa(self, question: str, answer: Any) -> torch.tensor:
        """Safely encode QA pair, ensuring both question and answer are strings."""
        if isinstance(answer, list):
            # MCQ style
            for a in answer:
                if a.get("correct"):
                    correct_answer = a.get("answer", "")
                    break
            else:
                correct_answer = ""
        elif isinstance(answer, dict):
            correct_answer = answer.get("statement", "")
        elif isinstance(answer, str):
            correct_answer = answer
        else:
            correct_answer = ""

        return self.qae_tokenizer(
            text=question if isinstance(question, str) else "",
            text_pair=correct_answer,
            padding="max_length",
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        )


    @torch.no_grad()
    def _evaluate_qa(self, encoded_qa_pair: torch.tensor) -> float:
        """Takes an encoded QA pair and returns a score."""
        output = self.qae_model(**encoded_qa_pair)
        return output[0][0][1]


def print_qa(qa_list: List[Mapping[str, str]], show_answers: bool = True) -> None:
    """Formats and prints a list of generated questions and answers."""

    for i in range(len(qa_list)):
        # wider space for 2 digit q nums
        space = " " * int(np.where(i < 9, 3, 4))

        print(f"{i + 1}) Q: {qa_list[i]['question']}")

        answer = qa_list[i]["answer"]

        # print a list of multiple choice answers
        if type(answer) is list:

            if show_answers:
                print(
                    f"{space}A: 1. {answer[0]['answer']} "
                    f"{np.where(answer[0]['correct'], '(correct)', '')}"
                )
                for j in range(1, len(answer)):
                    print(
                        f"{space + '   '}{j + 1}. {answer[j]['answer']} "
                        f"{np.where(answer[j]['correct']==True,'(correct)', '')}"
                    )

            else:
                print(f"{space}A: 1. {answer[0]['answer']}")
                for j in range(1, len(answer)):
                    print(f"{space + '   '}{j + 1}. {answer[j]['answer']}")

            print("")

        # print full sentence answers
        else:
            if show_answers:
                print(f"{space}A: {answer}\n")
