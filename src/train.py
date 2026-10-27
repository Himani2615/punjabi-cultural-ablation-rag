
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

from data_loader import load_data
from baseline import PunjabiBaseline


MODEL_NAME = "xlm-roberta-base"
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "models" / "xlmr_punjabi_baseline"


def main():
    train_df, test_df, label2id, id2label = load_data()

    baseline = PunjabiBaseline(
        model_name=MODEL_NAME,
        label2id=label2id,
        id2label=id2label,
    )

    def prepare_dataset(df):
        dataset = Dataset.from_dict({
            "text": df["text"].tolist(),
            "labels": df["labels"].astype(int).tolist(),
        })

        def tokenize_batch(batch):
            return baseline.tokenize(batch["text"])

        dataset = dataset.map(tokenize_batch, batched=True)
        dataset = dataset.remove_columns(["text"])
        return dataset

    train_dataset = prepare_dataset(train_df)
    test_dataset = prepare_dataset(test_df)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(
                labels, predictions, average="macro", zero_division=0
            ),
        }

    args = TrainingArguments(
        output_dir=str(ROOT / "outputs" / "training"),
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        seed=42,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=baseline.tokenizer
    )

    trainer = Trainer(
        model=baseline.model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    print("\nTest-set results:")
    metrics = trainer.evaluate()
    print(metrics)

    predictions = trainer.predict(test_dataset)
    predicted_ids = np.argmax(predictions.predictions, axis=-1)

    print("\nClassification report:")
    print(classification_report(
        test_df["labels"].astype(int),
        predicted_ids,
        labels=list(range(len(label2id))),
        target_names=[id2label[i] for i in range(len(id2label))],
        zero_division=0,
    ))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    baseline.tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"\nModel saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
