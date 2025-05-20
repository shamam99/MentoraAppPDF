import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

# Create models folders if they don't exist
os.makedirs("models/t5", exist_ok=True)
os.makedirs("models/bert", exist_ok=True)

print("📦 Downloading T5 model...")
AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator").save_pretrained("models/t5")
AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator").save_pretrained("models/t5")

print("📦 Downloading BERT evaluator...")
AutoTokenizer.from_pretrained("iarfmoose/bert-base-cased-qa-evaluator").save_pretrained("models/bert")
AutoModelForSequenceClassification.from_pretrained("iarfmoose/bert-base-cased-qa-evaluator").save_pretrained("models/bert")

print("✅ Models downloaded successfully.")
