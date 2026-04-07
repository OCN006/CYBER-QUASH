# import pandas as pd

# df = pd.read_csv("data/processed/multilingual_clean.csv")

# print("\n🔎 Checking label distribution...")
# print(df["label"].value_counts(dropna=False))

# print("\n🟥 Rows with missing labels:")
# print(df[df["label"].isna()].head())

# print("\nTotal missing:", df["label"].isna().sum())


# from transformers import XLMRobertaForSequenceClassification

# model = XLMRobertaForSequenceClassification.from_pretrained(
#     "models/xlm_roberta_multilingual"
# )

# print(model.config.id2label)
# print(model.config.label2id)


import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

MODEL = "models/xlm_roberta_multilingual"

tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL)
model.eval()

def test(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    out = model(**enc)
    pred = torch.argmax(out.logits, dim=1).item()
    print(f"TEXT: {text}")
    print("Predicted Label:", pred, "\n")

# SAFE
test("Kill all of them, they don't deserve to live.")

# OFFENSIVE
test("These people should be eliminated completely.")

# HATE
test("We must destroy that entire community.")


