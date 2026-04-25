import time

import ollama

from app.config import Settings
from app.generation.prompt import SYSTEM_PROMPT


class LocalLLMAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = ollama.Client(host=settings.ollama_base_url)

    def list_models(self) -> dict[str, str]:
        return self.settings.local_models

    def generate(self, model_key: str, user_prompt: str) -> dict:
        model_name = self.settings.local_models.get(model_key)
        if not model_name:
            allowed = ", ".join(sorted(self.settings.local_models))
            raise ValueError(f"Unknown model_key '{model_key}'. Allowed: {allowed}")

        started = time.perf_counter()
        response = self.client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.1,
                "num_ctx": 4096,
            },
        )
        elapsed = time.perf_counter() - started
        return {
            "content": response["message"]["content"],
            "model_name": model_name,
            "latency_seconds": elapsed,
        }
