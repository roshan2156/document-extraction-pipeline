"""
Benchmark batch extraction against one-request-per-field extraction.

Run from the project root:
  python src\benchmark.py --pdf "samples\AST nt 2.pdf" --document-type "Lease: Residential (AST)"

This sends real API requests and may incur charges. It does not use the extraction cache.
It writes benchmark_report.json. It measures token usage and schema validity; it does
not claim semantic accuracy without human review against the PDF.
"""
from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from jsonschema import Draft202012Validator

import extract as pipeline


def get_usage(body: dict[str, Any]) -> dict[str, Any]:
    usage = body.get("usage") or {}
    return {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "cached_input_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
    }


def request_json(
    request_body: dict[str, Any], base_url: str, api_key: str, timeout: int, label: str
) -> tuple[dict[str, Any], dict[str, Any], float]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    started = time.monotonic()
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=request_body,
        timeout=(15, timeout),
    )
    elapsed = time.monotonic() - started
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    parsed = pipeline._parse_json_object(content, label)
    return parsed, get_usage(body), elapsed


def build_single_field_type(doc_type: dict[str, Any], field: dict[str, Any]) -> dict[str, Any]:
    # Keep the original document type metadata but request only one field.
    return {**doc_type, "fields": [field]}


def run_request(
    pages: list[dict[str, Any]],
    doc_type: dict[str, Any],
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    schema = pipeline.build_response_schema(doc_type)
    request_body, _ = pipeline.build_extraction_request(
        pages=pages, doc_type=doc_type, schema=schema, model=model
    )
    result, usage, elapsed = request_json(
        request_body, base_url, api_key, timeout, "benchmark request"
    )
    repair_usage = None
    repair_elapsed = 0.0
    valid = not pipeline.validation_error_messages(result, schema)
    errors = pipeline.validation_error_messages(result, schema)
    if errors:
        repaired, repair_usage, repair_elapsed = request_json(
            {
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": pipeline.REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": pipeline._dumps({
                        "task": "Repair the output to satisfy the schema errors.",
                        "validation_errors": errors,
                        "output_json_schema": schema,
                        "invalid_output": result,
                    })},
                ],
            },
            base_url, api_key, timeout, "benchmark schema repair"
        )
        result = repaired
        valid = not pipeline.validation_error_messages(result, schema)
    return {
        "result": result,
        "usage": usage,
        "repair_usage": repair_usage,
        "seconds": round(elapsed + repair_elapsed, 2),
        "request_seconds": round(elapsed, 2),
        "repair_seconds": round(repair_elapsed, 2),
        "repair_used": repair_usage is not None,
        "schema_valid": valid,
    }


def add_usage(a: dict[str, Any], b: dict[str, Any] | None) -> dict[str, int]:
    keys = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens")
    out = {}
    for key in keys:
        vals = [x.get(key) for x in (a, b) if x and isinstance(x.get(key), (int, float))]
        out[key] = int(sum(vals)) if vals else 0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare batch vs per-field extraction cost.")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--document-type", required=True)
    parser.add_argument("--types", type=Path, default=Path("document_types.json"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", type=Path, default=Path("benchmark_report.json"))
    args = parser.parse_args()

    api_key = os.getenv("NEURALWATT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        parser.error("Set NEURALWATT_API_KEY or OPENAI_API_KEY first.")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.neuralwatt.com/v1")
    model = os.getenv("OPENAI_MODEL", "glm-5.2")

    types = pipeline.load_document_types(args.types)
    matches = [x for x in types if x.get("name", "").casefold() == args.document_type.casefold()]
    if not matches:
        parser.error(f"Document type not found: {args.document_type}")
    doc_type = matches[0]
    pages = pipeline.extract_pages(args.pdf)
    pages = pipeline.select_pages(pages, doc_type, max_pages=24)

    print("\nWARNING: this benchmark makes real API calls and may incur charges.")
    print(f"Model: {model}; document type: {doc_type.get('name')}; fields: {len(doc_type.get('fields', []))}")
    print("\n[1/2] Running batch extraction...")
    batch = run_request(pages, doc_type, model, base_url, api_key, args.timeout)

    print("\n[2/2] Running separate extraction for each field...")
    separate_fields = []
    for index, field in enumerate(doc_type.get("fields", []), 1):
        name = str(field.get("name", f"field_{index}"))
        print(f"  Field {index}/{len(doc_type['fields'])}: {name}")
        one_type = build_single_field_type(doc_type, field)
        one = run_request(pages, one_type, model, base_url, api_key, args.timeout)
        separate_fields.append({"field": name, **one})

    merged = {}
    for item in separate_fields:
        if item["result"]:
            merged.update(item["result"])
    full_schema = pipeline.build_response_schema(doc_type)
    merged_errors = pipeline.validation_error_messages(merged, full_schema)
    separate_valid = not merged_errors

    batch_usage = add_usage(batch["usage"], batch["repair_usage"])
    separate_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0}
    for item in separate_fields:
        u = add_usage(item["usage"], item["repair_usage"])
        for key in separate_usage:
            separate_usage[key] += u[key]

    report = {
        "model": model,
        "document_type": doc_type.get("name"),
        "pdf": str(args.pdf),
        "field_count": len(doc_type.get("fields", [])),
        "rates_usd_per_million_tokens": {
            "fresh_input": None,
            "cached_input": None,
            "output": None,
        },
        "batch": {
            "requests": 1 + int(batch["repair_used"]),
            "seconds": batch["seconds"],
            "schema_valid": batch["schema_valid"],
            "usage_including_repair": batch_usage,
            "result": batch["result"],
        },
        "separate": {
            "requests": sum(1 + int(x["repair_used"]) for x in separate_fields),
            "seconds": round(sum(x["seconds"] for x in separate_fields), 2),
            "schema_valid": separate_valid,
            "schema_errors": merged_errors,
            "usage_including_repairs": separate_usage,
            "fields": separate_fields,
            "merged_result": merged,
        },
        "notes": [
            "Token counts are provider-reported. Cost is not calculated until rates are entered.",
            "Cached input tokens may have a different rate; use provider billing details.",
            "Schema validity does not prove factual accuracy. Review values against the PDF.",
            "This is one trial; run several trials for a more reliable comparison.",
        ],
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(f"Batch:    requests={report['batch']['requests']}, seconds={report['batch']['seconds']}, schema_valid={report['batch']['schema_valid']}, tokens={batch_usage['total_tokens']:,}")
    print(f"Separate: requests={report['separate']['requests']}, seconds={report['separate']['seconds']}, schema_valid={separate_valid}, tokens={separate_usage['total_tokens']:,}")
    print(f"Report saved to: {args.out}")
    print("Dollar cost is intentionally left blank until you confirm the billing rates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
