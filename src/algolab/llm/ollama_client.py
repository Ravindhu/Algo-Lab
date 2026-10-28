"""Ollama client for a remote machine. Uses a base URL and an optional API key.
httpx is imported lazily, so modules that only need the prompt constants do not require it."""


class OllamaClient:
    def __init__(self, url: str, api_key: str = "", model: str = "llama3.1:8b",
                 timeout: float = 60.0):
        if not url:
            raise ValueError("Ollama URL is empty. Set OLLAMA_URL in .env")
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0.1}}
        if json_mode:
            body["format"] = "json"
        import httpx
        r = httpx.post(f"{self.url}/api/chat", json=body,
                       headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]
