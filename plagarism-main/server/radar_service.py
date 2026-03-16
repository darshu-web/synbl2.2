import json
import sys
import torch
import torch.nn.functional as F
import transformers

def emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()

def main():
    model_id = "TrustSafeAI/RADAR-Vicuna-7B"
    # Fallback to CPU if MPS or CUDA is not available.
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    try:
        # Load the RADAR model and tokenizer
        detector = transformers.AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        detector.eval()
        detector.to(device)

        emit({
            "type": "ready",
            "model": model_id,
            "status": "loaded",
            "device": device
        })
    except Exception as error:
        emit({"type": "fatal", "error": str(error), "model": model_id})
        return 1

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue

        req_id = None
        try:
            payload = json.loads(raw)
            req_id = payload.get("id")
            text = payload.get("text", "")
            
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()

            if not text:
                emit({
                    "type": "result",
                    "id": req_id,
                    "ok": True,
                    "fake_probability": 0.0,
                    "real_probability": 1.0,
                    "confidence": 0,
                    "signed_score": 0.0,
                    "votes": {"ai": 0, "human": 0},
                    "engines": [],
                })
                continue

            with torch.no_grad():
                inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                output_probs = F.log_softmax(detector(**inputs).logits, -1)[:, 0].exp().tolist()
                
            ai_prob = output_probs[0] # Probability of AI
            human_prob = 1.0 - ai_prob
            
            # Signed score: negative => AI, positive => Human
            signed = -(ai_prob * 100) if ai_prob >= 0.5 else (human_prob * 100)
            
            confidence = abs(ai_prob - 0.5) * 200 # Scale 0.5->0 to 0->100
            
            emit({
                "type": "result",
                "id": req_id,
                "ok": True,
                "fake_probability": ai_prob,
                "real_probability": human_prob,
                "confidence": round(confidence),
                "signed_score": signed,
                "votes": {"ai": 1 if ai_prob >= 0.5 else 0, "human": 1 if ai_prob < 0.5 else 0},
                "engines": [{
                    "engine": "radar",
                    "determination": "AI" if ai_prob >= 0.5 else "Human",
                    "score": abs(signed),
                    "signed_score": signed,
                    "ai_probability": ai_prob
                }],
            })
        except Exception as error:
            emit({
                "type": "result",
                "id": req_id,
                "ok": False,
                "error": str(error),
            })

    return 0

if __name__ == "__main__":
    sys.exit(main())
