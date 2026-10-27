
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class PunjabiBaseline:
    def __init__(self, model_name, label2id, id2label):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
        )

    def tokenize(self, texts):
        return self.tokenizer(
            list(texts),
            truncation=True,
            max_length=128,
        )
