# verifier/verifier.py

from transformers import pipeline

# Load the FEVER fine-tuned BERT model (use a public one)
model_name = "mrm8488/bert-tiny-uncased-finetuned-fever"

print("Loading verifier model...")
verifier = pipeline("text-classification", model=model_name)

def verify_claim(claim: str) -> dict:
    """
    Verifies a claim using the fine-tuned FEVER model.
    Returns a dictionary with label and confidence score.
    """
    result = verifier(claim)[0]
    label = result["label"]
    score = round(float(result["score"]), 3)

    # Convert model output to simple truth statement
    if label.lower() in ["support", "true"]:
        verdict = "True"
    elif label.lower() in ["refute", "false"]:
        verdict = "False"
    else:
        verdict = "Unclear"

    return {"claim": claim, "label": verdict, "confidence": score}
