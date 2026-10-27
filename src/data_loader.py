from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "annotated" / "sample_punjabi_humor_dataset.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    required = ["text", "primary_label"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["primary_label"] = df["primary_label"].astype(str).str.strip()
    df = df[df["text"] != ""]

    # Build a consistent mapping between labels and numeric IDs.
    labels = sorted(df["primary_label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    df["labels"] = df["primary_label"].map(label2id)

    if df["primary_label"].value_counts().min() < 2:
        raise ValueError("Each class needs at least two examples for this split.")

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["primary_label"],
    )

    print(f"Total examples: {len(df)}")
    print(f"Training examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")
    print("Class counts:\n", df["primary_label"].value_counts())

    return train_df, test_df, label2id, id2label
