# =============================================
# 🧠 PestisVeriteum Fact Verification Engine
# =============================================

import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import wikipedia
import numpy as np

# ------------------------------
# Load models only once
# ------------------------------
print("Loading models...")

nli_model_name = "facebook/bart-large-mnli"
retriever_model_name = "sentence-transformers/all-MiniLM-L6-v2"

nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
retriever_model = SentenceTransformer(retriever_model_name)

print("✅ Models loaded successfully.")

# ------------------------------
# Helper: retrieve top paragraphs from Wikipedia
# ------------------------------
def get_evidence_wikipedia(claim, top_k=5):
    try:
        results = wikipedia.search(claim, results=top_k)
        pages = []
        for title in results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                paragraphs = [p for p in page.content.split("\n") if len(p.strip()) > 100]
                pages.append({"title": title, "paragraphs": paragraphs})
            except:
                continue
        return pages
    except Exception as e:
        print("❌ Wikipedia error:", e)
        return []

# ------------------------------
# Helper: NLI classification between claim and evidence
# ------------------------------
def check_entailment(claim, evidence):
    inputs = nli_tokenizer(
        claim, evidence, return_tensors="pt", truncation=True, padding=True
    )
    with torch.no_grad():
        outputs = nli_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_id = torch.argmax(probs).item()
    labels = ["entailment", "neutral", "contradiction"]
    return {"label": labels[label_id], "score": probs[0][label_id].item()}

# ------------------------------
# Main verification function
# ------------------------------
def verify_claim(claim: str, top_pages=3, top_paragraphs=150, top_k_evidence=6):
    # 1️⃣ Retrieve candidate pages
    pages = get_evidence_wikipedia(claim, top_k=top_pages)
    all_paragraphs = []
    for p in pages:
        for para in p["paragraphs"]:
            all_paragraphs.append({"title": p["title"], "text": para})

    if not all_paragraphs:
        return {"final_label": "unknown", "confidence": 0, "evidences": []}

    # 2️⃣ Embed and select most relevant paragraphs
    claim_emb = retriever_model.encode(claim, convert_to_tensor=True)
    para_embs = retriever_model.encode([p["text"] for p in all_paragraphs], convert_to_tensor=True)
    cos_scores = util.cos_sim(claim_emb, para_embs)[0]
    top_results = torch.topk(cos_scores, k=min(top_k_evidence, len(all_paragraphs)))

    evidences = []
    for idx in top_results.indices:
        p = all_paragraphs[idx]
        score = cos_scores[idx].item()
        evidences.append({
            "title": p["title"],
            "paragraph": p["text"],
            "similarity": score,
            "nli": check_entailment(claim, p["text"])
        })

    # 3️⃣ Aggregate NLI results
    label_scores = {"entailment": 0, "contradiction": 0, "neutral": 0}
    for ev in evidences:
        lbl = ev["nli"]["label"]
        label_scores[lbl] += ev["nli"]["score"]

    # Pick final label
    final_label = max(label_scores, key=label_scores.get)
    confidence = label_scores[final_label] / sum(label_scores.values())

    if final_label == "entailment":
        final_label = "true"
    elif final_label == "contradiction":
        final_label = "false"
    else:
        final_label = "uncertain"

    return {
        "claim": claim,
        "final_label": final_label,
        "confidence": confidence,
        "evidences": evidences
    }

# ------------------------------
# Quick test (only if run directly)
# ------------------------------
if __name__ == "__main__":
    test_claim = "Canada is located in Africa"
    res = verify_claim(test_claim)
    print("✅ Result:", res["final_label"], "Confidence:", res["confidence"])
