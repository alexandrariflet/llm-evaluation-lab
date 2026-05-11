import requests
import time
from typing import Optional, List


class OllamaClient:
    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434/api/generate",
        timeout: int = 300,      # ← increased
        max_retries: int = 3,
        retry_delay: float = 3.0,
    ):
        self.model = model
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
        num_predict: int = 100,   # ← reduced default
        stop: Optional[List[str]] = None,
    ) -> str:

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "num_predict": num_predict,
            },
        }

        if seed is not None:
            payload["options"]["seed"] = seed

        if stop is not None:
            payload["options"]["stop"] = stop

        for attempt in range(self.max_retries):
            try:
                start = time.time()

                response = requests.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                elapsed = time.time() - start

                data = response.json()
                output = data.get("response", "").strip()

                # Optional debug (VERY useful)
                print(f"[Ollama] {elapsed:.2f}s | temp={temperature}")
                print("Output: ", output)

                return output

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Ollama request failed: {e}")

                print(f"[Retry {attempt+1}] {e}")
                time.sleep(self.retry_delay)