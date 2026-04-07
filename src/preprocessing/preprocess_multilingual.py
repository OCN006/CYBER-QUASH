import os
import pandas as pd

RAW_DIR = "data/raw"


# ---------------------------------------------------
# SAFE CSV READER (handles UTF-8, latin1, cp1252)
# ---------------------------------------------------
def safe_read(path):
    try:
        # Try UTF-8
        return pd.read_csv(path, engine="python", on_bad_lines='skip')
    except:
        # Fallback for messy CSVs (Hindi)
        return pd.read_csv(path, engine="python", encoding="latin1", on_bad_lines='skip')


# ---------------------------------------------------
# ENGLISH – toxic_dataset.csv
# ---------------------------------------------------
def load_english():
    path = os.path.join(RAW_DIR, "english", "toxic_dataset.csv")
    df = safe_read(path)

    df["text"] = df["comment_text"]

    df["label"] = 0
    df.loc[(df["toxic"] == 1) | (df["insult"] == 1) | (df["obscene"] == 1), "label"] = 1
    df.loc[(df["severe_toxic"] == 1) | (df["threat"] == 1) | (df["identity_hate"] == 1), "label"] = 2

    df["lang"] = "en"
    return df[["text", "label", "lang"]]


# ---------------------------------------------------
# HINDI – Indo-HateSpeech_Dataset.csv
# ---------------------------------------------------
def load_hindi():
    path = os.path.join(RAW_DIR, "hindi", "Indo-HateSpeech_Dataset.csv")

    df = safe_read(path)   # <--- now safe for broken CSV
    
    # Detect the text column
    possible_cols = ["text", "comment", "tweet", "Sentence", "Text"]
    for col in possible_cols:
        if col in df.columns:
            df["text"] = df[col]
            break
    else:
        df["text"] = df.iloc[:, 0]  # fallback first column

    # Detect label column
    possible_label_cols = ["label", "category", "Class", "Annotation"]
    for col in possible_label_cols:
        if col in df.columns:
            df["label"] = df[col]
            break
    else:
        df["label"] = "NONE"

    # Normalize label values
    df["label"] = df["label"].astype(str).str.lower().map({
        "not": 0,
        "none": 0,
        "offensive": 1,
        "off": 1,
        "hate": 2,
        "hateful": 2
    })

    df["label"] = df["label"].fillna(0).astype(int)
    df["lang"] = "hi"

    return df[["text", "label", "lang"]]

# ---------------------------------------------------
# KANNADA – train/dev/test
# ---------------------------------------------------
def load_kannada():
    path = os.path.join(RAW_DIR, "kannada")

    files = [
        "kannada_offensive_train.csv",
        "kannada_offensive_dev.csv",
        "kannada_offensive_test.csv"
    ]

    dfs = []
    for f in files:
        df = safe_read(os.path.join(path, f))

        # --- Detect text column ---
        text_cols = ["text", "comment", "sentence", "tweet"]
        for col in text_cols:
            if col in df.columns:
                df["text"] = df[col]
                break
        else:
            df["text"] = df.iloc[:, 0]   # fallback first column

        # --- Detect label column ---
        label_cols = ["label", "category", "class", "annotation"]
        for col in label_cols:
            if col in df.columns:
                df["label"] = df[col]
                break
        else:
            df["label"] = "not_offensive"  # fallback if no label present

        # --- Normalize labels ---
        df["label"] = df["label"].astype(str).str.lower().map({
            "not_offensive": 0,
            "not-offensive": 0,
            "normal": 0,

            "offensive": 1,
            "abusive": 1,
            "off": 1,

            "hate": 2,
            "hateful": 2
        })

        df["label"] = df["label"].fillna(0).astype(int)  # just in case

        df["lang"] = "kn"

        dfs.append(df[["text", "label", "lang"]])

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------
# MALAYALAM – train/dev/test
# ---------------------------------------------------
def load_malayalam():
    path = os.path.join(RAW_DIR, "malayalam")

    files = [
        "mal_full_offensive_train.csv",
        "mal_full_offensive_dev.csv",
        "mal_full_offensive_test.csv"
    ]

    dfs = []
    for f in files:
        df = safe_read(os.path.join(path, f))

        # --- Detect TEXT column ---
        text_cols = ["text", "comment", "sentence", "tweet"]
        for col in text_cols:
            if col in df.columns:
                df["text"] = df[col]
                break
        else:
            df["text"] = df.iloc[:, 0]

        # --- Detect LABEL column ---
        label_cols = ["label", "category", "class", "annotation", "task_1"]
        for col in label_cols:
            if col in df.columns:
                df["label"] = df[col]
                break
        else:
            # No label column → fallback: SAFE
            df["label"] = "not_offensive"

        # Normalize labels
        df["label"] = df["label"].astype(str).str.lower().map({
            "not_offensive": 0,
            "not-offensive": 0,
            "normal": 0,
            "non-offensive": 0,

            "offensive": 1,
            "abusive": 1,
            "off": 1,

            "hate": 2,
            "hateful": 2
        })

        # Any unknown label → make it harmless
        df["label"] = df["label"].fillna(0).astype(int)

        df["lang"] = "ml"

        dfs.append(df[["text", "label", "lang"]])

    return pd.concat(dfs, ignore_index=True)



# ---------------------------------------------------
# TAMIL – Offensive version (train/dev/test)
# ---------------------------------------------------
def load_tamil():
    path = os.path.join(RAW_DIR, "tamil")

    files = [
        "tamil_offensive_full_train.csv",
        "tamil_offensive_full_dev.csv",
        "tamil_offensive_full_test.csv"
    ]

    dfs = []
    for f in files:
        df = safe_read(os.path.join(path, f))

        # ---- Detect TEXT COLUMN ----
        text_cols = ["text", "comment", "tweet", "sentence", "comment_text"]
        for col in text_cols:
            if col in df.columns:
                df["text"] = df[col]
                break
        else:
            # if no known text column → take first column
            df["text"] = df.iloc[:, 0]

        # ---- Detect LABEL COLUMN ----
        label_cols = ["label", "category", "class", "annotation", "task_1"]
        for col in label_cols:
            if col in df.columns:
                df["label"] = df[col]
                break
        else:
            # no label found → mark all safe
            df["label"] = "not_offensive"

        # ---- Normalize label values ----
        df["label"] = (
            df["label"]
            .astype(str)
            .str.lower()
            .map({
                "not_offensive": 0,
                "not-offensive": 0,
                "normal": 0,
                "non-offensive": 0,

                "offensive": 1,
                "abusive": 1,

                "hate": 2,
                "hateful": 2
            })
        )

        # unknown → safe
        df["label"] = df["label"].fillna(0).astype(int)

        df["lang"] = "ta"

        dfs.append(df[["text", "label", "lang"]])

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------
# BENGALI – bengali_hate_v2.0.csv
# ---------------------------------------------------
def load_bengali():
    path = os.path.join(RAW_DIR, "bengali", "bengali_hate.csv")

    df = safe_read(path)

    # Detect text column
    text_cols = ["text", "comment", "sentence", "tweet", "post"]
    for col in text_cols:
        if col in df.columns:
            df["text"] = df[col]
            break
    else:
        df["text"] = df.iloc[:, 0]   # first column fallback

    # 🟩 IMPORTANT: Bengali dataset is PURE HATE → assign label = 2
    df["label"] = 2  

    df["lang"] = "bn"
    return df[["text", "label", "lang"]]

# ---------------------------------------------------
# MERGE ALL
# ---------------------------------------------------
def build_multilingual():
    print("🔄 Loading all datasets...")

    en = load_english()
    hi = load_hindi()
    kn = load_kannada()
    ml = load_malayalam()
    ta = load_tamil()
    bn = load_bengali()   # fixed Bengali loader

    print("✔ All datasets loaded!")

    # Merge all
    df = pd.concat([en, hi, kn, ml, ta, bn], ignore_index=True)

    # VERY IMPORTANT: Remove all NaN labels (fallback = hate = 2)
    df["label"] = df["label"].fillna(2).astype(int)

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/multilingual_clean.csv"

    df.to_csv(out_path, index=False)

    print(f"✅ Saved → {out_path}")
    print(f"📊 Total samples = {len(df)}")
    print("🟢 NaN labels fixed and dataset is fully clean now!")


if __name__ == "__main__":
    build_multilingual()
