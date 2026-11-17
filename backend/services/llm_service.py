
from groq import Groq

from backend.config import GROQ_API_KEY, GROQ_MODEL


class LLMService:
    def __init__(self):
        if not GROQ_API_KEY or GROQ_API_KEY == "PASTE_YOUR_GROQ_API_KEY_HERE":
            raise ValueError(
                "Groq API key is missing. Configure your .env file."
            )

        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=800
        )

        return response.choices[0].message.content or ""


