from lean_treepo.lean_client import KiminaLeanClient, parse_kimina_response


def test_parse_success_response() -> None:
    parsed = parse_kimina_response({"results": [{"success": True}]})
    assert parsed.success
    assert "succeeded" in parsed.feedback


def test_parse_failure_response() -> None:
    parsed = parse_kimina_response({"results": [{"success": False, "messages": ["type mismatch"]}]})
    assert not parsed.success
    assert "type mismatch" in parsed.feedback


def test_auth_header_is_bearer() -> None:
    client = KiminaLeanClient("https://kimina.example", api_key="abc")
    assert client.headers()["Authorization"] == "Bearer abc"


def test_verify_posts_expected_payload(monkeypatch) -> None:
    calls = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"results": [{"success": True}]}

    def fake_post(url, headers, json, timeout):  # noqa: A002
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        calls["timeout"] = timeout
        return Response()

    monkeypatch.setattr("lean_treepo.lean_client.requests.post", fake_post)
    result = KiminaLeanClient("https://kimina.example/", "key", 3.0).verify("#check Nat", "id-1")
    assert result.success
    assert calls["url"] == "https://kimina.example/verify"
    assert calls["json"]["codes"][0] == {"custom_id": "id-1", "proof": "#check Nat"}
    assert calls["headers"]["Authorization"] == "Bearer key"
