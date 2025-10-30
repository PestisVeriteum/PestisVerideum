# verifier/verifier.py
import wikipedia
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the FEVER-compatible model
tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def verify_claim(claim, max_sentences=5):
    """
    Search Wikipedia for evidence and classify the claim as true, false, or unclear.
    """
    try:
        search_results = wikipedia.search(claim, results=3)
        evidence_texts = []
        for title in search_results:
            try:
                summary = wikipedia.summary(title, sentences=max_sentences)
                evidence_texts.append(summary)
            except:
                continue

        if not evidence_texts:
            return {"final_label": "unclear", "avg_score": 0.0}

        entail_scores, contra_scores = [], []
        for evidence in evidence_texts:
            inputs = tokenizer(claim, evidence, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                entail_scores.append(probs[0][0].item())  # entailment
                contra_scores.append(probs[0][2].item())  # contradiction

        avg_entail = sum(entail_scores) / len(entail_scores)
        avg_contra = sum(contra_scores) / len(contra_scores)

        if avg_entail > 0.6 and avg_entail > avg_contra:
            label = "true"
        elif avg_contra > 0.6 and avg_contra > avg_entail:
            label = "false"
        else:
            label = "unclear"

        return {"final_label": label, "avg_entail": avg_entail, "avg_contra": avg_contra}

    except Exception as e:
        return {"final_label": "unclear", "error": str(e)}
