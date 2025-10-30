# test_verifier.py

from verifier.verifier import verify_claim

# Try some example claims
examples = [
    "Canada is located in Africa.",
    "Water boils at 100 degrees Celsius.",
    "The Earth is flat."
]

for claim in examples:
    result = verify_claim(claim)
    print(f"Claim: {result['claim']}")
    print(f"Result: {result['label']} (Confidence: {result['confidence']})\n")
