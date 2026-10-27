
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "xlmr_punjabi_baseline"


def main():
    if not MODEL_DIR.exists():
        print("Model not found. Run py src/train.py first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )
    model.eval()

    print("Punjabi Humor Classifier")
    print("Enter 'quit' to exit.")

    while True:
        text = input("\nEnter text: ").strip()

        if text.lower() == "quit":
            break

        if not text:
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(
                outputs.logits, dim=-1
            )[0]

        predicted_id = int(probabilities.argmax().item())
        label = model.config.id2label[predicted_id]
        confidence = probabilities[predicted_id].item()

        print(f"Predicted category: {label}")
        print(f"Model confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
