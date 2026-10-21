
import json
from collections import Counter

with open("document_types.json", encoding="utf-8") as f:
    document_types = json.load(f)

for doc in document_types:
    fields = doc.get("fields", [])
    names = [field.get("name") for field in fields]

    duplicates = [
        name for name, count in Counter(names).items()
        if count > 1
    ]

    invalid = [
        field.get("name", "(unnamed)")
        for field in fields
        if not field.get("name")
        or not isinstance(field.get("schema"), dict)
    ]

    print(f"\n{doc.get('name', '(unnamed)')}")
    print("  Field count:", len(fields))
    print("  Duplicate field names:", duplicates or "none")
    print(
        "  Missing names or invalid schema objects:",
        invalid or "none"
    )
