# Document Extraction Pipeline

A Python take-home implementation for extracting structured JSON from property lease PDFs. The pipeline is designed to reduce API overhead, produce schema-valid output, and support consistent results through deterministic settings and caching.

## Table of Contents

- [Overview](#overview)
- [Design and Architecture](#design-and-architecture)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Testing Without an API Call](#testing-without-an-api-call)
- [Cost Optimization](#cost-optimization)
- [Benchmark and Comparison](#benchmark-and-comparison)
- [Consistency and Validation](#consistency-and-validation)
- [Accuracy and Evaluation](#accuracy-and-evaluation)
- [Trade-offs and Limitations](#trade-offs-and-limitations)
- [Future Improvements](#future-improvements)
- [Security](#security)

## Overview

The pipeline extracts structured information from property-related PDF documents and saves the results as JSON.

It supports three document types defined in `document_types.json`:

- `Lease: Commercial`
- `Lease: Residential (AST)`
- `Notice: Rent Change`

Each document type defines the fields to extract, their descriptions, and the expected output schema. The extraction engine is shared across document types, allowing the configuration to define what information is required without duplicating the pipeline.

## Design and Architecture

The pipeline is organized around the following stages:

1. **Local PDF text extraction:** PyMuPDF extracts selectable text directly from the PDF instead of rendering every page as an image.
2. **Document-type configuration:** The pipeline loads the selected document type and its field definitions from `document_types.json`.
3. **Page selection:** A lightweight lexical ranker selects relevant pages using field vocabulary and common lease-related terms. The first and last text-bearing pages are retained where the configured page limit permits.
4. **Request construction:** The selected page text, field descriptions, and output JSON Schema are combined into a single extraction request.
5. **LLM extraction:** The configured OpenAI-compatible Chat Completions endpoint generates the structured result.
6. **Schema validation:** The response is parsed and validated using `jsonschema.Draft202012Validator`.
7. **Controlled repair:** If the output fails schema validation, the pipeline makes at most one repair request. The PDF text is not resent in that repair request.
8. **Caching:** Successfully validated output is saved to a local cache and reused when the extraction inputs and relevant settings match.

This design separates document-specific configuration from common extraction logic, reduces repeated input submission, and provides explicit validation before accepting results.

## Requirements

- Python 3.10 or later recommended
- An API key for live extraction
- Dependencies listed in `requirements.txt`

The pipeline uses PyMuPDF for PDF text extraction, `requests` for HTTP requests, and `jsonschema` for output validation.

## Setup

From the project root, create and activate a virtual environment.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

The pipeline uses environment variables to configure the API connection.

| Variable | Purpose | Default |
|---|---|---|
| `NEURALWATT_API_KEY` | Preferred API authentication key | None |
| `OPENAI_BASE_URL` | OpenAI-compatible API base URL | `https://api.neuralwatt.com/v1` |
| `OPENAI_MODEL` | Model identifier | `glm-5.2` |

Although the code supports the `OPENAI_` variable names for compatibility, the default provider is Neuralwatt. The configured provider must support the expected Chat Completions API format.

Set the API key and configuration in your current shell session.

### Windows PowerShell

```powershell
$env:NEURALWATT_API_KEY = "YOUR_API_KEY"
$env:OPENAI_BASE_URL = "https://api.neuralwatt.com/v1"
$env:OPENAI_MODEL = "glm-5.2"
```

### macOS or Linux

```bash
export NEURALWATT_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://api.neuralwatt.com/v1"
export OPENAI_MODEL="glm-5.2"
```

Replace the placeholder with your own API key. Do not commit real credentials to version control.

The script reads these environment variables directly. A `.env` file is not automatically loaded unless additional loading logic is configured.

## Running the Pipeline

Run commands from the project root with the virtual environment activated.

### Residential tenancy agreement

```powershell
python src/extract.py `
  --pdf "samples/AST nt 2.pdf" `
  --document-type "Lease: Residential (AST)" `
  --output output_ast.json
```

### Rent-change notice

```powershell
python src/extract.py `
  --pdf "samples/rent increase 2025-2026 NT2.pdf" `
  --document-type "Notice: Rent Change" `
  --output output_rent_change.json
```

### Commercial lease

```powershell
python src/extract.py `
  --pdf "samples/2025.08.22 - 4th Fl, RSH - Lease - Signed by Landlord.pdf" `
  --document-type "Lease: Commercial" `
  --output output_commercial.json `
  --max-pages 70
```

Use the actual filenames available in your `samples` directory if they differ from these examples.

### Command-line options

| Option | Description |
|---|---|
| `--pdf` | Input PDF path; required |
| `--document-type` | Exact document type name from the configuration; required |
| `--types` | Configuration path; defaults to `document_types.json` |
| `--output` | Output JSON path; defaults to `output.json` |
| `--cache-dir` | Cache directory; defaults to `.cache` |
| `--max-pages` | Maximum number of pages sent to the model; defaults to 24 |
| `--timeout` | API read timeout in seconds; defaults to 180 |
| `--compare` | Optional existing JSON output to compare against |
| `--dry-run` | Build and inspect the request without making an API call |

To inspect all supported options:

```powershell
python src/extract.py --help
```

### Dry run

A dry run extracts PDF text, selects pages, builds the request, and reports request-size information without calling the model.

```powershell
python src/extract.py `
  --pdf "samples/AST nt 2.pdf" `
  --document-type "Lease: Residential (AST)" `
  --dry-run
```

This is useful for checking configuration and estimating request size without incurring API charges. Character counts are not a substitute for provider-reported token usage.

## Testing Without an API Call

Run the automated test suite from the project root:

```powershell
python -m pytest
```

The repository also contains local configuration and evidence-checking utilities, including `src/check_config.py` and `audit_ast.py`.

These checks can validate configuration, output structure, and selected evidence from the PDF without making additional model requests. Confirm that each test is configured for offline operation before running it.

## Cost Optimization

The original per-key baseline sends the full document and a separate prompt/schema for each requested field. If there are \(N\) fields, that approach can send the document content \(N\) times.

This pipeline uses several mechanisms to reduce repeated input and unnecessary API calls.

### 1. Batch extraction

All fields for the selected document type are included in one extraction request. The document text is therefore submitted once per uncached extraction rather than being resent separately for every field.

For an eight-field document, the baseline would make eight extraction requests while the batch approach makes one extraction request, provided no repair request is needed.

This represents an 87.5% reduction in extraction request count in that example. It is **not** evidence of an 87.5% reduction in total tokens or API cost. The total cost also depends on prompt and schema size, output length, image-versus-text tokenization in the baseline, and repair requests.

### 2. Local text extraction

PyMuPDF extracts selectable PDF text locally. The model receives text rather than image payloads for every page, which can reduce input costs compared with a baseline that sends page images.

The actual saving depends on the baseline's image handling and the provider's pricing.

### 3. Page selection

The page ranker selects a bounded subset of text-bearing pages. The default maximum is 24 pages, and the limit can be increased with `--max-pages`.

For example, `--max-pages 1000` can be used to include all text-bearing pages of a document with fewer than 1,000 such pages. Increasing the limit may improve recall but also increases the amount of text sent to the model.

Page selection is a heuristic, not a guarantee that every relevant clause is retained.

### 4. Content-addressed caching

The cache key incorporates the PDF content hash, document-type configuration, model, API base URL, page limit, shared prompt text, and cache version.

After a successful extraction, the validated JSON is cached. A later request with the same cache key can reuse that result without another model call.

A cache hit can therefore avoid the model cost of a repeated extraction. It does not make the initial extraction free, and changes to the PDF, configuration, model, URL, page limit, or cache version can require a new call.

### 5. Compact request serialization

The pipeline serializes JSON without unnecessary whitespace. This reduces payload size slightly while preserving JSON meaning.

### 6. Bounded repair

If the generated result fails schema validation, the pipeline allows one repair request. The repair payload includes the validation errors, schema, and invalid output but does not resend the PDF text.

Repair requests still incur API cost. The limit prevents unbounded repair loops.

### Measuring cost

The pipeline logs provider-reported token usage when the response includes supported usage fields. Exact dollar cost requires the applicable pricing for the configured provider and model.

A proper benchmark should compare the per-key baseline and this pipeline using the same PDFs and document types. Record:

- Total input and output tokens
- Number of extraction and repair requests
- Cache hits and misses
- Total API cost using verified pricing
- Elapsed time
- Schema validation success
- Field-level extraction accuracy

A controlled benchmark has to be run before claiming a measured percentage reduction in monetary cost.

## Benchmark and Comparison

### Purpose

The benchmark is intended to compare the original per-key baseline with the optimized batch pipeline using the same representative PDFs and document types. The comparison should measure API usage, cost, latency, schema validity, and field-level accuracy.

### Design comparison

| Metric | Per-key baseline | Optimized pipeline |
|---|---|---|
| Extraction requests | One request per field | One request for all configured fields, unless a repair request is needed |
| Document text submitted | May be resent for every field | Selected text submitted once per extraction request |
| Repeated identical extraction | Depends on baseline implementation | Validated output can be reused from the local cache |
| Output validation | Depends on baseline implementation | JSON Schema validation, with at most one repair attempt |
| Token usage | To be measured | To be measured from provider usage metadata, when available |
| Monetary cost | To be calculated using verified pricing | To be calculated using verified pricing |
| Latency | To be measured | To be measured |
| Field-level accuracy | To be evaluated against manually verified values | To be evaluated against the same reference values |

For an eight-field document, the batch pipeline is designed to use one extraction request rather than eight, assuming no repair request is required. This is an expected request-count comparison, not a measured result and not proof of an equivalent percentage reduction in tokens or cost.

### Benchmark execution status

**The live API benchmark was not run.** The API client displayed a warning about credit usage, so execution was stopped to avoid unexpected charges. As a result, this README does not claim measured token savings, dollar savings, latency improvements, or live-model accuracy improvements.

The comparison above describes the implementation and the intended evaluation. The performance figures remain unmeasured.

### Safe local checks

Local checks such as configuration validation, JSON/schema validation of existing output, PDF text inspection, dry runs, and offline tests can be used without making a live model request. Confirm that tests are configured for offline operation before running them. Mocked tests can verify request construction and cache behavior, but they cannot establish real provider token usage, monetary cost, live-model accuracy, or fresh-run consistency.

A dry run can be performed with:

```powershell
python src/extract.py `
  --pdf "samples/AST nt 2.pdf" `
  --document-type "Lease: Residential (AST)" `
  --dry-run
```

### Measurements to collect when live benchmarking is approved

When API credits and permission to incur charges are available, run both approaches against the same PDFs and record:

- Input and output tokens reported by the provider
- Number of extraction and repair requests
- Cache hits and misses
- Total cost calculated from verified provider/model pricing
- Elapsed time
- Schema validation pass rate
- Field-level accuracy against manually verified reference values
- Fresh-run consistency with caching disabled or isolated

Report the sample documents, model, provider, page limit, cache state, and any failed requests alongside the results so the comparison is reproducible.

## Consistency and Validation

The pipeline uses several controls to improve consistency.

### Deterministic generation settings

The extraction request sets `temperature=0` and instructs the model to return one JSON object matching the supplied schema. These settings reduce sampling variability but do not guarantee identical results across all runs or provider deployments.

### JSON Schema validation

The response is parsed as JSON and checked against a schema generated for the selected document type. The schema requires the configured fields and disallows additional top-level properties.

Invalid JSON or schema-invalid output is rejected. A single repair request may be attempted, after which the result must pass validation before it is cached or returned.

### Cache-based repeatability

Once a valid result has been cached, identical inputs and settings can reuse the stored JSON. This gives stable results for cache hits, but it is distinct from proving that fresh model calls return identical outputs.

### How to verify consistency

A useful evaluation should distinguish two tests:

1. **Cache test:** Run the same extraction twice with the same cache directory. Confirm that the second run uses the cache and makes no additional extraction request.
2. **Fresh-run consistency test:** Run the same extraction multiple times with caching bypassed or separate cache directories. Compare the JSON outputs and record exact-match rates and field-level differences.

The second test requires actual model calls unless it is only testing the pipeline with a mock. Mocked tests verify request handling and cache behavior, not real model consistency.

## Accuracy and Evaluation

Schema validation ensures that output follows the expected structure and types. It does not guarantee that the extracted facts are correct.

Accuracy should be evaluated against manually verified reference values from the source documents.

For each field, record:

- Expected value
- Extracted value
- Whether the values match
- Supporting PDF page and quote
- Whether the value is missing, unsupported, or ambiguous

Important fields to verify include property addresses, landlord and tenant names, effective dates, rent amounts, payment frequency, lease terms, review clauses, and break conditions.

A useful evaluation dataset should include examples from all three supported document types. Report field-level exact match, schema pass rate, missing-value rate, and unsupported-claim rate.

The existing local evidence audit can help inspect selected extracted values, but it is not a complete accuracy benchmark.

## Trade-offs and Limitations

### Batch size versus response complexity

Batching avoids repeated document input but produces a larger combined response. Very large schemas or long documents may exceed the provider's context limits. The pipeline surfaces request failures rather than automatically splitting the request into smaller field groups.

### Page selection versus recall

Lexical ranking is inexpensive and easy to inspect, but it can miss clauses that use unusual wording or terms not represented in the field descriptions and anchor list. Retaining more pages improves the chance of including relevant evidence but increases input size.

### Structural validity versus factual correctness

Schema validation catches missing fields, extra top-level keys, and type or constraint violations. It cannot prove that an extracted value is supported by the source PDF.

### Caching versus freshness

Caching reduces repeat costs and provides stable results for identical cache keys. It also means a cache hit does not independently verify the source or produce a fresh model result. Cache invalidation depends on the key and versioning strategy.

### Text extraction versus scanned documents

PyMuPDF extracts selectable text. Scanned PDFs without a text layer are not supported by the current workflow unless OCR is added.

### Provider dependence

The client expects an OpenAI-compatible Chat Completions response structure. Other providers may differ in usage metadata, structured-output support, pricing, and context limits.

## Future Improvements

With additional time, I would:

1. Run a controlled batch-versus-per-key benchmark and report actual token and dollar costs.
2. Build a manually labelled ground-truth dataset for all supported document types.
3. Test fresh-run consistency independently of caching.
4. Improve page selection with a local clause index or BM25 retrieval and evaluate its recall.
5. Add optional OCR for scanned PDFs.
6. Record per-field provenance, including source page and supporting text.
7. Add retries with backoff for transient provider errors, while carefully bounding additional requests and costs.
8. Add regression tests for representative PDFs, schemas, cache invalidation, and malformed API responses.
9. Add verified pricing configuration for automatic dollar-cost estimates.

## Security

- API keys are read from environment variables.
- Never commit `.env` files or real API credentials.
- Treat PDF content as untrusted input; the extraction prompt explicitly instructs the model not to follow instructions embedded in the document.
- The local cache may contain confidential lease information. Keep `.cache/` and sensitive outputs out of source control.
- Review generated JSON and logs before sharing them, as they may contain personal or commercially sensitive information.
- This pipeline is a technical extraction aid, not legal advice.
