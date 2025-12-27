from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded successfully")

def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence, label_id = torch.max(probs, dim=1)

    return {
        "label_id": label_id.item(),
        "confidence": confidence.item()
    }
