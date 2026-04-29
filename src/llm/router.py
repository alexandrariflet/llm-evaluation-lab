def get_model(provider):
    if provider == "claude":
        return ClaudeClient()
    if provider == "gemini":
        return GeminiClient()
    if provider == "ollama":
        return OllamaClient()