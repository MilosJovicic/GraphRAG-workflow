from qa_agent.config import Settings, get_settings


def test_settings_reads_required_fields(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "neo4j://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setenv("TEMPORAL_HOST", "localhost:7233")
    monkeypatch.setenv("TEMPORAL_NAMESPACE", "default")
    monkeypatch.setenv("QA_TASK_QUEUE", "qa-agent-tq")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding")
    monkeypatch.setenv("COHERE_API_KEY", "ck")
    monkeypatch.setenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    get_settings.cache_clear()
    s = Settings()
    assert s.neo4j_uri == "neo4j://localhost:7687"
    assert s.qa_task_queue == "qa-agent-tq"
    assert s.cohere_api_key == "ck"
    assert s.openai_api_key == "ok"
    assert s.openai_model == "gpt-5.4-mini"
    assert s.bm25_top_k == 50
    assert s.rrf_top_n == 40


def test_get_settings_caches(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "neo4j://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
