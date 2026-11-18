from backend.app.adapters import openai_translate
from backend.app.adapters.openai_client import normalize_openai_base_url, opencode_go_options


def test_normalize_openai_base_url_strips_chat_completions_suffix():
    assert (
        normalize_openai_base_url("https://api.example.com/v1/chat/completions")
        == "https://api.example.com/v1"
    )


def test_normalize_openai_base_url_keeps_standard_v1_root():
    assert normalize_openai_base_url("https://api.openai.com/v1/") == "https://api.openai.com/v1"


def test_opencode_go_headers_are_scoped_to_its_own_api():
    for base_url in ("https://opencode.ai/inference/go/v1", "https://opencode.ai/inference/go/openai/v1",
                     "https://opencode.ai/zen/go/v1"):
        client = openai_translate._client(base_url, "test", session_id="task-session")
        try:
            assert client.default_headers["x-opencode-session"] == "task-session"
            assert client.default_headers["User-Agent"] == "YouDub-webui/1.0"
        finally:
            client.close()
    assert not opencode_go_options("https://opencode.ai/inference/v1", "task-session")
    assert not opencode_go_options("https://opencode.ai.example/inference/go/v1", "task-session")


def test_openai_client_initializes_with_socks_proxy_environment(monkeypatch):
    for key in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,::1")

    client = openai_translate._client("http://localhost:11434/v1", "sk-test")
    try:
        assert client.base_url == "http://localhost:11434/v1/"
    finally:
        client.close()
