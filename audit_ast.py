import json
import re
from pathlib import Path

import pymupdf


PDF_PATH = Path("samples/AST nt 2.pdf")
JSON_PATH = Path("output_ast.json")


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def main():
    if not PDF_PATH.is_file():
        raise SystemExit(f"PDF not found: {PDF_PATH}")
    if not JSON_PATH.is_file():
        raise SystemExit(f"JSON output not found: {JSON_PATH}")

    result = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    with pymupdf.open(PDF_PATH) as pdf:
        pages = [(i + 1, page.get_text("text")) for i, page in enumerate(pdf)]

    print("LOCAL ACCURACY REVIEW")
    print("=" * 65)
    print(f"PDF: {PDF_PATH}")
    print(f"JSON: {JSON_PATH}")
    print(f"PDF pages: {len(pages)}")
    print("API calls: 0")

    fields = ["effective_date", "property_address", "landlord_name", "tenant_name"]

    for key in fields:
        value = result.get(key)
        print(f"\n{key}: {value!r}")

        if not isinstance(value, str) or not value.strip():
            print("  REVIEW: Missing or not a text value.")
            continue

        matches = [
            page_no
            for page_no, text in pages
            if normalize(value) in normalize(text)
        ]

        if matches:
            print(f"  Exact text match on page(s): {matches}")
        else:
            print("  No exact match; inspect the source wording manually.")
            words = [
                word for word in re.findall(r"[A-Za-z0-9]+", value)
                if len(word) >= 4
            ]
            shown = 0
            for page_no, text in pages:
                for line in text.splitlines():
                    if any(word.casefold() in line.casefold() for word in words):
                        print(f"  Page {page_no}: {line.strip()[:240]}")
                        shown += 1
                        if shown >= 5:
                            break
                if shown >= 5:
                    break

    print("\nKEY CLAIMS TO VERIFY")
    print("=" * 65)

    checks = [
        ("Rent amount", ["2300", "2,300"]),
        ("Rent frequency", ["monthly", "per month"]),
        ("Tenancy start", ["1 October 2024", "1 Oct 2024"]),
        ("Tenancy end", ["30 September 2025", "30 Sep 2025"]),
        ("Rent review", ["Retail Price Index", "RPI"]),
        ("Break clause", ["break clause", "six months", "6 months"]),
    ]

    all_text = "\n".join(text for _, text in pages).casefold()
    for label, terms in checks:
        found = [term for term in terms if term.casefold() in all_text]
        status = f"FOUND {found!r}" if found else "NOT FOUND - review manually"
        print(f"{label}: {status}")

    print("\nNESTED EXTRACTION DATA")
    print("=" * 65)
    for key in ("consideration", "consideration_review", "key_dates"):
        print(f"\n{key}:")
        print(json.dumps(result.get(key), indent=2, ensure_ascii=False)[:1800])

    print("\nNote: a text match is evidence to inspect, not proof that the extracted value is correct.")
    print("Date formatting differences can cause exact-match checks to miss valid values.")


if __name__ == "__main__":
    main()
