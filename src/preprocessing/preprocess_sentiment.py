import os
import json
import pandas as pd
from datasets import load_dataset

RAW_DIR = "data/raw/sentiment"
OUT_FILE = "data/processed/sentiment_multilingual.csv"

os.makedirs("data/processed", exist_ok=True)


# -----------------------------------------------------------
# Universal Label Mapper (3 classes for all languages)
# -----------------------------------------------------------
def normalize_label(label):
    label = str(label).lower().strip()

    NEG = {"neg", "negative", "0", "sad", "anger", "fear", "disgust"}
    NEU = {"neu", "neutral", "1", "mixed", "unknown"}
    POS = {"pos", "positive", "2", "joy", "love", "happy"}

    if label in NEG:
        return 0
    if label in NEU:
        return 1
    if label in POS:
        return 2

    return 1  # fallback neutral


# -----------------------------------------------------------
# LOAD SENTIMENT140 (ENGLISH)
# -----------------------------------------------------------
def load_english_sentiment():
    print("📥 Loading Sentiment140 (English CSV)...")

    path = os.path.join(RAW_DIR, "english.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("❌ English sentiment dataset missing. Put english.csv in data/raw/sentiment")

    df = pd.read_csv(path, encoding="latin1", header=None)

    # Sentiment140 columns:
    # 0 = polarity (0=neg, 2=neutral, 4=positive)
    # 5 = text
    df = df[[0, 5]]
    df.columns = ["label", "text"]

    df["label"] = df["label"].map({0: 0, 2: 1, 4: 2})
    df["lang"] = "en"

    return df[["text", "label", "lang"]]

# -----------------------------------------------------------
# LOAD INDIC LANG JSON FILES
# -----------------------------------------------------------
def load_json_file(path, lang):
    if not os.path.exists(path):
        print(f"⚠ Missing file: {path}")
        return pd.DataFrame(columns=["text", "label", "lang"])

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            print(f"⚠ Fixing malformed JSON: {path}")
            txt = f.read()
            txt = "[" + txt.replace("}\n{", "},\n{") + "]"
            data = json.loads(txt)

    df = pd.DataFrame(data)

    text_col = "text" if "text" in df else df.columns[0]
    label_col = "label" if "label" in df else df.columns[1]

    df["text"] = df[text_col]
    df["label"] = df[label_col].apply(normalize_label)
    df["lang"] = lang

    return df[["text", "label", "lang"]]


# -----------------------------------------------------------
# LOAD INDIC LANGUAGES (HI, BN, TA, ML, KN)
# -----------------------------------------------------------
def load_indic():
    langs = ["hi", "bn", "ta", "ml", "kn"]
    all_dfs = []

    for lang in langs:
        test_file = os.path.join(RAW_DIR, "test", f"{lang}.json")
        val_file = os.path.join(RAW_DIR, "validation", f"{lang}.json")

        print(f"📥 Loading {lang} test...")
        df_test = load_json_file(test_file, lang)

        print(f"📥 Loading {lang} validation...")
        df_val = load_json_file(val_file, lang)

        all_dfs.append(df_test)
        all_dfs.append(df_val)

    return pd.concat(all_dfs, ignore_index=True)


# -----------------------------------------------------------
# MERGE + SAVE FINAL SENTIMENT DATASET
# -----------------------------------------------------------
def build_dataset():
    print("🔥 Building Multilingual Sentiment Dataset...")

    en = load_english_sentiment()
    indic = load_indic()

    final_df = pd.concat([en, indic], ignore_index=True)

    final_df = final_df.dropna(subset=["text"]).reset_index(drop=True)

    final_df.to_csv(OUT_FILE, index=False)

    print(f"📊 Final dataset shape: {final_df.shape}")
    print(f"✅ Saved → {OUT_FILE}")


if __name__ == "__main__":
    build_dataset()
