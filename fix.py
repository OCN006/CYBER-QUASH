import os
import json

BASE = "data/raw/sentiment"

def fix_json(path):
    print("Fixing:", path)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    # If file is already a single JSON list, skip
    try:
        test_json = json.loads("\n".join(lines))
        print("Already valid JSON:", path)
        return
    except:
        pass

    # Convert multiple JSON objects → list
    fixed = []
    for line in lines:
        if line.strip() == "":
            continue
        try:
            fixed.append(json.loads(line))
        except:
            print("❌ Bad line in:", path)
            continue

    # Save valid JSON array
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print("✔ Fixed", path)


# Fix all JSON files inside test & validation
for folder in ["test", "validation"]:
    folder_path = os.path.join(BASE, folder)
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            fix_json(os.path.join(folder_path, file))

print("🎉 All JSON files fixed!")
