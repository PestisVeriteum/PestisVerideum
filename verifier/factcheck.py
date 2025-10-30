# verifier/factcheck.py
# =============================================
# 🧠 PestisVeriteum - Claim Verifier Core
# Uses real Natural Language Inference (NLI) model
# =============================================

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load a strong open-access model trained for FEVER fact verification
MODEL_NAME = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

# Load model and tokenizer once
print("🔍 Loading FEVER model... please wait...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label mapping for NLI tasks
label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

def verify_claim(claim: str) -> dict:
    """
    Verifies whether a claim is likely TRUE or FALSE.
    Uses a general world knowledge context and NLI reasoning.
    """

    # Some neutral context to compare against claim
    context = "According to general world knowledge, verified by Wikipedia and reliable sources."

    # Encode the context and claim pair
    inputs = tokenizer(context, claim, return_tensors="pt", truncation=True, padding=True).to(device)

    # Run through model
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1)[0][pred].item()

    label = label_map[pred]

    # Map to True/False/Unclear
    if label == "entailment":
        verdict = "True"
    elif label == "contradiction":
        verdict = "False"
    else:
        verdict = "Unclear"

    return {
        "claim": claim,
        "label": verdict,
        "confidence": round(confidence, 3)
    }
