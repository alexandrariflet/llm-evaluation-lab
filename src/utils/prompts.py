from src.utils.config import PROMPTS


def build_prompt(lang: str) -> str:
    item = PROMPTS[lang]
    return f"<question>{item['question']}</question><answer>{item['answer']}</answer>"