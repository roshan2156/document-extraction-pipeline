from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import extract
from extract import build_response_schema, load_document_types, select_pages, validate_output


def test_document_types_load():
    types = load_document_types(Path(__file__).resolve().parents[1] / "document_types.json")
    assert {x["name"] for x in types} >= {
        "Lease: Commercial", "Lease: Residential (AST)", "Notice: Rent Change"
    }


def test_schema_requires_all_fields():
    types = load_document_types(Path(__file__).resolve().parents[1] / "document_types.json")
    dt = next(x for x in types if x["name"] == "Lease: Residential (AST)")
    schema = build_response_schema(dt)
    assert "effective_date" in schema["required"]
    assert schema["additionalProperties"] is False


def test_page_selection_preserves_order_and_limit():
    pages = [{"page": i, "text": f"Page {i} rent tenant property"} for i in range(1, 50)]
    doc_type = {"fields": [{"name": "rent", "description": "rent payment"}]}
    selected = select_pages(pages, doc_type, max_pages=8)
    assert len(selected) == 8
    assert [p["page"] for p in selected] == sorted(p["page"] for p in selected)


def test_validate_output_rejects_missing_required():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False}
    try:
        validate_output({}, schema)
        assert False, "should fail"
    except ValueError:
        pass


def test_call_llm_sends_all_fields_in_one_request(monkeypatch):
    doc_type = {
        "name": "Test type",
        "fields": [
            {"name": "first", "description": "First value"},
            {"name": "second", "description": "Second value"},
        ],
    }
    calls = []

    def fake_post_chat(**kwargs):
        calls.append(kwargs)
        return {"first": "one", "second": "two"}

    monkeypatch.setattr(extract, "_post_chat", fake_post_chat)

    result = extract.call_llm(
        pages=[{"page": 1, "text": "Document text"}],
        doc_type=doc_type,
        schema=build_response_schema(doc_type),
        model="test-model",
        base_url="https://example.test/v1",
        api_key="test-key",
    )

    assert result == {"first": "one", "second": "two"}
    assert len(calls) == 1
    request_body = calls[0]["request_body"]
    user_payload = extract.json.loads(
        request_body["messages"][1]["content"]
    )
    assert [field["key"] for field in user_payload["fields"]] == [
        "first",
        "second",
    ]
    assert set(user_payload["output_json_schema"]["properties"]) == {
        "first",
        "second",
    }


def test_llm_request_uses_bearer_api_key(monkeypatch):
    captured = {}

    class FakeResponse:
        ok = True
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"value": "ok"}'}}]}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        return FakeResponse()

    monkeypatch.setattr(extract.requests, "post", fake_post)

    result = extract._post_chat(
        request_body={
            "model": "test-model",
            "messages": [{"role": "user", "content": "{}"}],
        },
        base_url="https://example.test/v1",
        api_key="test-key",
        timeout=180,
        label="Test request",
    )

    assert result == {"value": "ok"}
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_llm_request_logs_provider_token_usage(monkeypatch, capsys):
    class FakeResponse:
        ok = True
        status_code = 200

        @staticmethod
        def json():
            return {
                "choices": [{"message": {"content": '{"value": "ok"}'}}],
                "usage": {
                    "prompt_tokens": 123,
                    "completion_tokens": 45,
                    "total_tokens": 168,
                },
            }

    monkeypatch.setattr(
        extract.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    result = extract._post_chat(
        request_body={
            "model": "test-model",
            "messages": [{"role": "user", "content": "{}"}],
        },
        base_url="https://example.test/v1",
        api_key="test-key",
        timeout=180,
        label="Test request",
    )

    assert result == {"value": "ok"}
    assert "input=123, output=45, total=168" in capsys.readouterr().out


def test_llm_request_reports_missing_provider_usage(monkeypatch, capsys):
    class FakeResponse:
        ok = True
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"value": "ok"}'}}]}

    monkeypatch.setattr(
        extract.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    result = extract._post_chat(
        request_body={
            "model": "test-model",
            "messages": [{"role": "user", "content": "{}"}],
        },
        base_url="https://example.test/v1",
        api_key="test-key",
        timeout=180,
        label="Test request",
    )

    assert result == {"value": "ok"}
    assert "provider did not return token usage" in capsys.readouterr().out


def test_cache_key_includes_provider_and_page_limit(tmp_path):
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"pdf bytes")
    doc_type = {"name": "Test type"}

    original = extract.cache_key(
        pdf_path, doc_type, "test-model", "https://one.test/v1", 24
    )

    assert original == extract.cache_key(
        pdf_path, doc_type, "test-model", "https://one.test/v1/", 24
    )
    assert original != extract.cache_key(
        pdf_path, doc_type, "test-model", "https://two.test/v1", 24
    )
    assert original != extract.cache_key(
        pdf_path, doc_type, "test-model", "https://one.test/v1", 8
    )
